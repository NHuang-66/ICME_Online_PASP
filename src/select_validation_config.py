"""Select Static thresholds and one shared Online config on validation only.

This executable deliberately knows only the three ``*_val.npy`` filenames.  It
does not contain, search for, hash, or open a test array.  It writes a lock file
that a separate test evaluator must verify before test evaluation can start.
The Online condition uses the study-specific SAOCP-inspired implementation in
``src/saocp.py``; internal ``saocp`` identifiers are retained for traceability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .event_f1 import (
        Event,
        EventProtocol,
        construct_events,
        event_f1_one_to_one,
        maximum_cardinality_matches,
        overlap_hours,
        read_catalog_events,
    )
    from .saocp import (
        block_nonconformity_scores,
        block_slices,
        make_calibrator,
        online_predict,
    )
except ImportError:
    from event_f1 import (
        Event,
        EventProtocol,
        construct_events,
        event_f1_one_to_one,
        maximum_cardinality_matches,
        overlap_hours,
        read_catalog_events,
    )
    from saocp import (
        block_nonconformity_scores,
        block_slices,
        make_calibrator,
        online_predict,
    )


VALIDATION_END = pd.Timestamp("1998-01-01")
BCE_BACKBONES = ("lstm", "cnn_lstm", "unet", "runet")
STATIC_GRID = tuple(float(value) for value in np.round(np.arange(0.10, 0.901, 0.02), 2))
SAOCP_COVERAGE_GRID = tuple(
    float(value) for value in np.round(np.arange(0.30, 0.951, 0.05), 2)
)
SAOCP_LIFETIME_GRID = (4, 8, 16, 32)
SAOCP_POLICY = "positive_singleton"
BLOCK_SIZE = 64
WARMUP_BLOCKS = 128


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run requires both NAME and PATH")
    return name.strip(), Path(path.strip())


def load_validation_stream(name: str, run_dir: Path) -> dict:
    """Load validation arrays only; no test filename exists in this function."""

    files = {
        "probability": run_dir / "probability_val.npy",
        "labels": run_dir / "Y_val_aligned.npy",
        "time": run_dir / "time_val.npy",
    }
    missing = [path.name for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing {', '.join(missing)}")
    probability = np.load(files["probability"]).reshape(-1)
    labels = np.load(files["labels"]).reshape(-1)
    time_index = pd.DatetimeIndex(np.load(files["time"]))
    if not (len(probability) == len(labels) == len(time_index)):
        raise ValueError(f"{name}: validation arrays are not aligned")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{name}: validation probabilities contain NaN/Inf")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError(f"{name}: validation probabilities lie outside [0,1]")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError(f"{name}: validation labels are not binary")
    if not time_index.is_monotonic_increasing:
        raise ValueError(f"{name}: validation time is not monotonic")
    return {
        "name": name,
        "run_dir": run_dir,
        "files": files,
        "probability": probability,
        "labels": labels.astype(np.uint8),
        "time": time_index,
    }


def metric_row(
    name: str,
    method: str,
    interval: str,
    events: Sequence[Event],
    truth: Sequence[Event],
    **settings,
) -> dict:
    return {
        "run": name,
        "method": method,
        "validation_interval": interval,
        "matching": "maximum_cardinality_one_to_one_any_positive_overlap",
        "unmatched_prediction_rule": "every_unmatched_constructed_event_is_fp",
        "truth_events": len(truth),
        "predicted_events": len(events),
        **settings,
        **event_f1_one_to_one(events, truth),
    }


def static_key(row: dict) -> tuple[float, ...]:
    """Locked tie-break: F1, P, R, |tau-.5|, larger tau."""

    threshold = float(row["threshold"])
    return (
        float(row["f1"]),
        float(row["precision"]),
        float(row["recall"]),
        -abs(threshold - 0.5),
        threshold,
    )


def select_static(
    stream: dict, truth: Sequence[Event], protocol: EventProtocol
) -> tuple[dict, list[dict], list[Event]]:
    rows: list[dict] = []
    events_by_threshold: dict[float, list[Event]] = {}
    interval = f"{stream['time'][0].isoformat()}--{VALIDATION_END.isoformat()}"
    for threshold in STATIC_GRID:
        labels = np.asarray(stream["probability"] >= threshold, dtype=np.uint8)
        events = construct_events(labels, stream["time"], protocol)
        events_by_threshold[threshold] = events
        rows.append(
            metric_row(
                stream["name"],
                "static",
                interval,
                events,
                truth,
                threshold=threshold,
            )
        )
    selected = max(rows, key=static_key).copy()
    selected.update(
        {
            "selection_split": "validation_only",
            "selection_rule": "F1,precision,recall,-abs(threshold-0.5),larger_threshold",
            "test_used_for_selection": False,
        }
    )
    return selected, rows, events_by_threshold[float(selected["threshold"])]


def prepare_saocp_validation(stream: dict, catalog_path: Path) -> dict:
    slices = block_slices(len(stream["probability"]), BLOCK_SIZE)
    if len(slices) <= WARMUP_BLOCKS:
        raise ValueError(
            f"{stream['name']}: {len(slices)} validation blocks cannot support "
            f"the locked {WARMUP_BLOCKS}-block warm-up"
        )
    eval_start = slices[WARMUP_BLOCKS][0]
    time_eval = stream["time"][eval_start:]
    truth = read_catalog_events(catalog_path, pd.Timestamp(time_eval[0]), VALIDATION_END)
    return {
        **stream,
        "eval_start": eval_start,
        "probability_warmup": stream["probability"][:eval_start],
        "labels_warmup": stream["labels"][:eval_start],
        "probability_eval": stream["probability"][eval_start:],
        "labels_eval": stream["labels"][eval_start:],
        "time_eval": time_eval,
        "truth_eval": truth,
    }


def evaluate_saocp_candidate(
    stream: dict,
    coverage: float,
    lifetime: int,
    protocol: EventProtocol,
    keep_series: bool = False,
) -> tuple[dict, list[Event], np.ndarray, np.ndarray]:
    calibration_scores = block_nonconformity_scores(
        stream["probability_warmup"],
        stream["labels_warmup"],
        coverage,
        BLOCK_SIZE,
    )
    calibrator = make_calibrator(calibration_scores, coverage, lifetime)
    labels, thresholds, radii = online_predict(
        stream["probability_eval"],
        stream["labels_eval"],
        calibrator,
        coverage=coverage,
        block_size=BLOCK_SIZE,
        policy=SAOCP_POLICY,
    )
    events = construct_events(labels, stream["time_eval"], protocol)
    candidate_id = (
        f"coverage={coverage:0.2f}|lifetime={lifetime:02d}|"
        f"policy={SAOCP_POLICY}|block={BLOCK_SIZE:03d}"
    )
    interval = f"{stream['time_eval'][0].isoformat()}--{VALIDATION_END.isoformat()}"
    row = metric_row(
        stream["name"],
        "saocp",
        interval,
        events,
        stream["truth_eval"],
        candidate_id=candidate_id,
        coverage=coverage,
        lifetime=lifetime,
        policy=SAOCP_POLICY,
        block_size=BLOCK_SIZE,
        warmup_blocks=WARMUP_BLOCKS,
        mean_threshold=float(np.mean(thresholds)),
        std_threshold=float(np.std(thresholds)),
        minimum_threshold=float(np.min(thresholds)),
        maximum_threshold=float(np.max(thresholds)),
    )
    if not keep_series:
        thresholds = np.empty(0, dtype=np.float32)
        radii = np.empty(0, dtype=np.float32)
    return row, events, thresholds, radii


def macro_row(rows: Sequence[dict]) -> dict:
    ordered = sorted(rows, key=lambda row: str(row["run"]))
    return {
        "candidate_id": str(ordered[0]["candidate_id"]),
        "coverage": float(ordered[0]["coverage"]),
        "lifetime": int(ordered[0]["lifetime"]),
        "policy": SAOCP_POLICY,
        "block_size": BLOCK_SIZE,
        "models": len(ordered),
        "macro_f1": float(np.mean([float(row["f1"]) for row in ordered])),
        "minimum_backbone_f1": float(min(float(row["f1"]) for row in ordered)),
        "macro_precision": float(
            np.mean([float(row["precision"]) for row in ordered])
        ),
        "macro_recall": float(np.mean([float(row["recall"]) for row in ordered])),
        **{f"{row['run']}_f1": float(row["f1"]) for row in ordered},
    }


def saocp_macro_key(row: dict) -> tuple[float, ...]:
    """Metric portion of the locked key; lexical ties keep the first row."""

    return (
        float(row["macro_f1"]),
        float(row["minimum_backbone_f1"]),
        float(row["macro_precision"]),
        float(row["macro_recall"]),
    )


def matching_rows(
    run: str,
    method: str,
    events: Sequence[Event],
    truth: Sequence[Event],
    interval: str,
) -> list[dict]:
    pairs = maximum_cardinality_matches(events, truth)
    matched_predictions = {prediction for prediction, _ in pairs}
    matched_truth = {target for _, target in pairs}
    rows: list[dict] = []
    for prediction, target in pairs:
        rows.append(
            {
                "run": run,
                "method": method,
                "validation_interval": interval,
                "status": "matched",
                "prediction_index": prediction,
                "prediction_begin": events[prediction].begin.isoformat(),
                "prediction_end": events[prediction].end.isoformat(),
                "prediction_duration_hours": events[prediction].duration_hours,
                "catalog_index": target,
                "catalog_begin": truth[target].begin.isoformat(),
                "catalog_end": truth[target].end.isoformat(),
                "overlap_hours": overlap_hours(events[prediction], truth[target]),
            }
        )
    for prediction, event in enumerate(events):
        if prediction not in matched_predictions:
            rows.append(
                {
                    "run": run,
                    "method": method,
                    "validation_interval": interval,
                    "status": "unmatched_prediction_fp",
                    "prediction_index": prediction,
                    "prediction_begin": event.begin.isoformat(),
                    "prediction_end": event.end.isoformat(),
                    "prediction_duration_hours": event.duration_hours,
                    "catalog_index": "",
                    "catalog_begin": "",
                    "catalog_end": "",
                    "overlap_hours": 0.0,
                }
            )
    for target, event in enumerate(truth):
        if target not in matched_truth:
            rows.append(
                {
                    "run": run,
                    "method": method,
                    "validation_interval": interval,
                    "status": "unmatched_catalog_fn",
                    "prediction_index": "",
                    "prediction_begin": "",
                    "prediction_end": "",
                    "prediction_duration_hours": "",
                    "catalog_index": target,
                    "catalog_begin": event.begin.isoformat(),
                    "catalog_end": event.end.isoformat(),
                    "overlap_hours": 0.0,
                }
            )
    return rows


def threshold_block_rows(
    stream: dict, coverage: float, lifetime: int, thresholds: np.ndarray, radii: np.ndarray
) -> list[dict]:
    rows: list[dict] = []
    for block_index, (start, end) in enumerate(
        block_slices(len(stream["probability_eval"]), BLOCK_SIZE)
    ):
        rows.append(
            {
                "run": stream["name"],
                "split": "post_warmup_validation",
                "block_index": block_index,
                "row_start": start,
                "row_end_exclusive": end,
                "time_begin": stream["time_eval"][start].isoformat(),
                "time_end": stream["time_eval"][end - 1].isoformat(),
                "coverage": coverage,
                "lifetime": lifetime,
                "radius": float(radii[block_index]),
                "threshold": float(thresholds[start]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("At least one --run NAME=PATH is required")
    run_map = dict(args.run)
    if len(run_map) != len(args.run):
        raise ValueError("Run names must be unique")
    missing_backbones = sorted(set(BCE_BACKBONES).difference(run_map))
    if missing_backbones:
        raise ValueError(f"Missing BCE backbones: {missing_backbones}")

    catalog_path = args.catalog.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = EventProtocol()
    streams = {
        name: load_validation_stream(name, path.resolve())
        for name, path in args.run
    }

    static_grid_rows: list[dict] = []
    selected_static_rows: list[dict] = []
    validation_metrics: list[dict] = []
    match_rows: list[dict] = []
    selected_static_events: dict[str, list[Event]] = {}
    static_truth: dict[str, list[Event]] = {}
    for name, stream in streams.items():
        truth = read_catalog_events(
            catalog_path, pd.Timestamp(stream["time"][0]), VALIDATION_END
        )
        selected, sweep, events = select_static(stream, truth, protocol)
        static_grid_rows.extend(sweep)
        selected_static_rows.append(selected)
        validation_metrics.append(selected)
        selected_static_events[name] = events
        static_truth[name] = truth
        match_rows.extend(
            matching_rows(
                name,
                "static",
                events,
                truth,
                str(selected["validation_interval"]),
            )
        )

    saocp_streams = {
        name: prepare_saocp_validation(stream, catalog_path)
        for name, stream in streams.items()
    }
    per_model_grid: list[dict] = []
    macro_grid: list[dict] = []
    # Candidate rows are generated in ascending candidate-id order.  `max`
    # returns the first row on an exact metric tie, hence the lexicographically
    # smallest candidate is the locked final tie-break.
    candidates = sorted(
        (
            (
                f"coverage={coverage:0.2f}|lifetime={lifetime:02d}|"
                f"policy={SAOCP_POLICY}|block={BLOCK_SIZE:03d}",
                coverage,
                lifetime,
            )
            for coverage in SAOCP_COVERAGE_GRID
            for lifetime in SAOCP_LIFETIME_GRID
        ),
        key=lambda item: item[0],
    )
    for _, coverage, lifetime in candidates:
        rows = [
            evaluate_saocp_candidate(
                saocp_streams[name], coverage, lifetime, protocol
            )[0]
            for name in BCE_BACKBONES
        ]
        per_model_grid.extend(rows)
        macro_grid.append(macro_row(rows))
    selected_saocp = max(macro_grid, key=saocp_macro_key).copy()
    selected_saocp.update(
        {
            "selection_split": "four_BCE_validation_streams_after_128_block_warmup",
            "selection_rule": (
                "macro_F1,minimum_backbone_F1,macro_precision,macro_recall,"
                "lexicographically_smallest_candidate_id"
            ),
            "test_used_for_selection": False,
        }
    )
    coverage = float(selected_saocp["coverage"])
    lifetime = int(selected_saocp["lifetime"])

    threshold_rows: list[dict] = []
    series_dir = output_dir / "validation_threshold_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    for name, stream in saocp_streams.items():
        row, events, thresholds, radii = evaluate_saocp_candidate(
            stream, coverage, lifetime, protocol, keep_series=True
        )
        row.update(
            {
                "selection_split": "validation_only",
                "shared_config_selected_from": "four_BCE_backbones",
                "test_used_for_selection": False,
            }
        )
        validation_metrics.append(row)
        match_rows.extend(
            matching_rows(
                name,
                "saocp",
                events,
                stream["truth_eval"],
                str(row["validation_interval"]),
            )
        )
        threshold_rows.extend(
            threshold_block_rows(stream, coverage, lifetime, thresholds, radii)
        )
        np.save(series_dir / f"{name}_thresholds.npy", thresholds)
        np.save(series_dir / f"{name}_radii.npy", radii)

    write_csv(output_dir / "static_validation_grid.csv", static_grid_rows)
    write_csv(
        output_dir / "selected_static_thresholds_validation_only.csv",
        selected_static_rows,
    )
    write_csv(output_dir / "saocp_validation_grid_per_model.csv", per_model_grid)
    write_csv(output_dir / "saocp_validation_grid_macro.csv", macro_grid)
    write_csv(output_dir / "validation_selected_metrics.csv", validation_metrics)
    write_csv(output_dir / "validation_event_matching.csv", match_rows)
    write_csv(output_dir / "validation_threshold_blocks.csv", threshold_rows)

    input_rows = [
        {
            "role": "catalog",
            "run": "",
            "filename": catalog_path.name,
            "bytes": catalog_path.stat().st_size,
            "sha256": sha256_file(catalog_path),
        }
    ]
    for name, stream in streams.items():
        for role, path in stream["files"].items():
            input_rows.append(
                {
                    "role": f"validation_{role}",
                    "run": name,
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_csv(output_dir / "validation_input_hashes.csv", input_rows)

    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VALIDATION_SELECTION_COMPLETE_TEST_NOT_OPENED",
        "test_files_opened": False,
        "matching": "deterministic maximum-cardinality one-to-one any-positive-overlap",
        "false_positive_rule": "every unmatched constructed prediction is FP",
        "event_protocol": asdict(protocol),
        "legacy_comparator_only": {
            "false_positive_min_hours": 2.5,
            "used_by_primary_metric": False,
        },
        "static_grid": list(STATIC_GRID),
        "static_selection_rule": (
            "F1, precision, recall, distance to 0.5, larger threshold"
        ),
        "selected_static_thresholds": {
            row["run"]: row["threshold"] for row in selected_static_rows
        },
        "saocp_grid": {
            "coverage": list(SAOCP_COVERAGE_GRID),
            "lifetime": list(SAOCP_LIFETIME_GRID),
            "policy": SAOCP_POLICY,
            "block_size": BLOCK_SIZE,
            "warmup_blocks": WARMUP_BLOCKS,
        },
        "saocp_selection_rule": selected_saocp["selection_rule"],
        "selected_saocp": selected_saocp,
        "validation_input_hashes_file": "validation_input_hashes.csv",
    }
    lock_path = output_dir / "VALIDATION_SELECTION_COMPLETE.json"
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")

    generated = sorted(
        [
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "validation_output_hashes.csv"
        ],
        key=lambda path: str(path.relative_to(output_dir)).lower(),
    )
    write_csv(
        output_dir / "validation_output_hashes.csv",
        [
            {
                "file": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in generated
        ],
    )
    print(
        json.dumps(
            {
                "status": lock["status"],
                "selected_static_thresholds": lock["selected_static_thresholds"],
                "selected_saocp": {
                    "coverage": coverage,
                    "lifetime": lifetime,
                    "macro_validation_f1": selected_saocp["macro_f1"],
                    "minimum_backbone_f1": selected_saocp["minimum_backbone_f1"],
                },
                "test_files_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
