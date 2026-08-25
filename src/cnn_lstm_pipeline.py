"""Train the frozen CNN--LSTM backbone and export probability streams.

This module stops at probabilities.  Static/Online decisions and the manuscript
event-level F1 are implemented separately; the Online condition uses the
SAOCP-inspired implementation in ``src/saocp.py``.  This separation makes the
frozen-backbone boundary explicit and prevents test-set threshold tuning inside
model training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


SEED = 42


def set_seed(seed: int = SEED) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.clip_by_value(
            tf.cast(y_pred, tf.float32),
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon(),
        )
        alpha_t = y_true_f * alpha + (1.0 - y_true_f) * (1.0 - alpha)
        p_t = y_true_f * y_pred_f + (1.0 - y_true_f) * (1.0 - y_pred_f)
        return tf.reduce_mean(-alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t))

    return loss


def build_model(window: int = 64, feature_count: int = 34) -> tf.keras.Model:
    layers = tf.keras.layers
    inputs = layers.Input((window, feature_count), name="solar_wind")
    x = inputs
    filters = (64, 64, 128, 128, 256, 256, 128, 128, 64, 64)
    for index, width in enumerate(filters, start=1):
        x = layers.Conv1D(
            width, 3, padding="same", activation="relu", name=f"conv_{index}"
        )(x)
    x = layers.LSTM(64, return_sequences=True, name="lstm_features")(x)
    probability = layers.TimeDistributed(
        layers.Dense(1, activation="sigmoid"), name="icme_probability"
    )(x)
    return tf.keras.Model(inputs, probability, name="cnn_lstm_icme")


class BalancedWindowSequence(tf.keras.utils.Sequence):
    """Deterministic balanced windows backed by memory-mapped arrays."""

    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        window: int,
        batch_size: int,
        steps: int,
        positive_ratio: float,
        feature_count: int,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if len(self.x) != len(self.y):
            raise ValueError("Feature/label row-count mismatch")
        if feature_count > self.x.shape[1]:
            raise ValueError("Requested feature count exceeds stored channels")
        self.window = int(window)
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.positive_ratio = float(positive_ratio)
        self.feature_count = int(feature_count)
        self.seed = int(seed)
        self.epoch = 0
        half = self.window // 2
        valid = np.arange(half, len(self.y) - (self.window - half), dtype=np.int64)
        labels = np.asarray(self.y[valid, 0])
        self.positive_centers = valid[labels == 1]
        self.negative_centers = valid[labels == 0]
        if not len(self.positive_centers) or not len(self.negative_centers):
            raise ValueError("Training labels must contain both classes")

    def __len__(self) -> int:
        return self.steps

    def on_epoch_end(self) -> None:
        self.epoch += 1

    def __getitem__(self, batch_index: int):
        rng = np.random.default_rng(self.seed + self.epoch * self.steps + batch_index)
        positive_count = int(round(self.batch_size * self.positive_ratio))
        centers = np.concatenate(
            [
                rng.choice(self.positive_centers, positive_count, replace=True),
                rng.choice(
                    self.negative_centers,
                    self.batch_size - positive_count,
                    replace=True,
                ),
            ]
        )
        rng.shuffle(centers)
        x_batch = np.empty(
            (self.batch_size, self.window, self.feature_count), dtype=np.float32
        )
        y_batch = np.empty((self.batch_size, self.window, 1), dtype=np.float32)
        half = self.window // 2
        for row, center in enumerate(centers):
            start = int(center) - half
            x_batch[row] = self.x[start : start + self.window, : self.feature_count]
            y_batch[row] = self.y[start : start + self.window]
        return x_batch, y_batch


def validation_dataset(
    x_path: Path,
    y_path: Path,
    window: int,
    batch_size: int,
    feature_count: int,
) -> tf.data.Dataset:
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    if len(x) != len(y):
        raise ValueError("Validation feature/label row-count mismatch")

    def generator():
        for start in range(0, len(x), window):
            end = min(start + window, len(x))
            x_window = np.zeros((window, feature_count), dtype=np.float32)
            y_window = np.zeros((window, 1), dtype=np.float32)
            x_window[: end - start] = x[start:end, :feature_count]
            y_window[: end - start] = y[start:end]
            yield x_window, y_window

    signature = (
        tf.TensorSpec((window, feature_count), tf.float32),
        tf.TensorSpec((window, 1), tf.float32),
    )
    count = int(math.ceil(len(x) / window))
    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(count))
    return dataset.batch(batch_size).prefetch(1)


def train(
    data_dir: Path,
    run_dir: Path,
    window: int = 64,
    feature_count: int = 34,
    batch_size: int = 16,
    positive_ratio: float = 0.7,
    epochs: int = 30,
    steps_per_epoch: int = 100,
    validation_batch_size: int = 32,
    loss_name: str = "bce",
) -> Path:
    set_seed(SEED)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(window, feature_count)
    if loss_name == "bce":
        training_loss = tf.keras.losses.BinaryCrossentropy()
        loss_config = {"name": "binary_crossentropy"}
    elif loss_name == "focal":
        training_loss = focal_loss(alpha=0.25, gamma=2.0)
        loss_config = {"name": "focal", "alpha": 0.25, "gamma": 2.0}
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=training_loss,
        metrics=[tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    train_sequence = BalancedWindowSequence(
        data_dir / "X_train_origin_1.npy",
        data_dir / "Y_train_aligned.npy",
        window,
        batch_size,
        steps_per_epoch,
        positive_ratio,
        feature_count,
    )
    validation = validation_dataset(
        data_dir / "X_val_origin_1.npy",
        data_dir / "Y_val_aligned.npy",
        window,
        validation_batch_size,
        feature_count,
    )
    weights = run_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            weights,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, mode="min"
        ),
        tf.keras.callbacks.CSVLogger(run_dir / "training_history.csv"),
    ]
    model.fit(
        train_sequence,
        validation_data=validation,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model.save_weights(weights)
    config = {
        "model": "cnn_lstm",
        "window": int(window),
        "feature_count": int(feature_count),
        "seed": SEED,
        "loss": loss_config,
        "optimizer": {"name": "Adam", "learning_rate": 1e-4},
        "epochs_requested": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "batch_size": int(batch_size),
        "validation_batch_size": int(validation_batch_size),
        "positive_ratio": float(positive_ratio),
    }
    (run_dir / "model_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return weights


def load_model(run_dir: Path) -> tuple[tf.keras.Model, dict]:
    config = json.loads((run_dir / "model_config.json").read_text(encoding="utf-8"))
    model = build_model(config["window"], config["feature_count"])
    model.load_weights(run_dir / "best.weights.h5")
    return model, config


def predict_overlap_add(
    model: tf.keras.Model,
    x_path: Path,
    window: int,
    stride: int,
    batch_size: int = 128,
) -> np.ndarray:
    x = np.load(x_path, mmap_mode="r")
    length = len(x)
    starts = list(range(0, max(length - window + 1, 1), stride))
    final_start = max(0, length - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    total = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.uint16)
    feature_count = int(model.input_shape[-1])
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        batch = np.zeros((len(batch_starts), window, feature_count), dtype=np.float32)
        lengths: list[int] = []
        for row, start in enumerate(batch_starts):
            end = min(start + window, length)
            lengths.append(end - start)
            batch[row, : end - start] = x[start:end, :feature_count]
        probabilities = np.asarray(model(batch, training=False)).reshape(len(batch_starts), window)
        for start, observed, values in zip(batch_starts, lengths, probabilities):
            total[start : start + observed] += values[:observed]
            count[start : start + observed] += 1
    if np.any(count == 0):
        raise RuntimeError("Overlap-add inference left uncovered rows")
    return (total / count).astype(np.float32)


def export_probabilities(data_dir: Path, run_dir: Path, split: str) -> Path:
    model, config = load_model(run_dir)
    window = int(config["window"])
    probability = predict_overlap_add(
        model,
        data_dir / f"X_{split}_origin_1.npy",
        window=window,
        stride=max(1, window // 2),
    )
    output = run_dir / f"probability_{split}.npy"
    np.save(output, probability)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "predict-val", "predict-test", "all"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--feature-count", type=int, default=34)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--positive-ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument(
        "--loss",
        choices=("bce", "focal"),
        default="bce",
        help="Training loss; manuscript Tables 4--6 use BCE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    if args.stage in {"train", "all"}:
        print(
            train(
                data_dir,
                run_dir,
                args.window,
                args.feature_count,
                args.batch_size,
                args.positive_ratio,
                args.epochs,
                args.steps_per_epoch,
                args.validation_batch_size,
                args.loss,
            )
        )
    if args.stage in {"predict-val", "all"}:
        print(export_probabilities(data_dir, run_dir, "val"))
    if args.stage in {"predict-test", "all"}:
        print(export_probabilities(data_dir, run_dir, "test"))


if __name__ == "__main__":
    main()
