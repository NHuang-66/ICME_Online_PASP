"""Online decision calibration for frozen probability streams.

``Online`` is the manuscript-facing decision-method label.  This module is the
study-specific, SAOCP-inspired blockwise quantile-expert implementation used for
that condition; it is not an upstream SAOCP reference release.  For every
non-overlapping block, a threshold is emitted before that block's labels are
supplied.  The revealed labels update the expert mixture only for the next
block.  Consequently this module changes decisions, not backbone probabilities
or network weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def pinball_loss(y: float, prediction: float, quantile: float) -> float:
    error = y - prediction
    return float(quantile * error if error >= 0 else (quantile - 1.0) * error)


def pinball_gradient(y: float, prediction: float, quantile: float) -> float:
    if y > prediction:
        return -float(quantile)
    if y < prediction:
        return float(1.0 - quantile)
    return 0.0


@dataclass
class _Expert:
    start: int
    scale: float
    coverage: float
    estimate: float
    lifetime_multiplier: int

    def __post_init__(self) -> None:
        self.base_learning_rate = self.scale / math.sqrt(3.0)
        value = int(self.start)
        power = 0
        while value > 0 and value % 2 == 0:
            value //= 2
            power += 1
        self.lifetime = int(self.lifetime_multiplier * (2**power))
        self.z = 0.0
        self.weighted_z = 0.0
        self.age = 0
        self.gradient_norm = 0.0
        self.prior_weight = 1.0 / (
            self.start**2 * (1.0 + math.floor(math.log2(self.start)))
        )

    @property
    def expired(self) -> bool:
        return self.age > self.lifetime

    @property
    def weight(self) -> float:
        return 0.0 if self.age == 0 else self.z / self.age * (1.0 + self.weighted_z)

    def loss(self, score: float) -> float:
        return pinball_loss(score, self.estimate, self.coverage)

    def update(self, score: float, mixture_loss: float) -> None:
        weight = self.weight
        denominator = self.scale * max(self.coverage, 1.0 - self.coverage)
        gradient = (mixture_loss - self.loss(score)) / max(denominator, 1e-12)
        gradient = min(1.0, max(-1.0 if weight > 0 else 0.0, gradient))
        self.z += float(gradient)
        self.weighted_z += float(gradient) * weight
        self.age += 1

        loss_gradient = pinball_gradient(score, self.estimate, self.coverage)
        self.gradient_norm += loss_gradient**2
        if self.gradient_norm > 0:
            self.estimate = max(
                0.0,
                self.estimate
                - self.base_learning_rate / math.sqrt(self.gradient_norm) * loss_gradient,
            )


class SAOCPQuantile:
    """Study-specific SAOCP-inspired expert mixture for scores in [0, 1]."""

    def __init__(
        self,
        coverage: float,
        scale: float,
        initial_radius: float,
        lifetime: int = 8,
    ) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie in (0, 1)")
        self.coverage = float(coverage)
        self.scale = max(float(scale), 1e-6)
        self.initial_radius = float(np.clip(initial_radius, 0.0, 1.0))
        self.lifetime = int(lifetime)
        self.time = 1
        self.experts: dict[int, _Expert] = {}

    def predict(self) -> float:
        if not self.experts:
            return self.initial_radius
        prior_total = 0.0
        prior_estimate = 0.0
        weighted_total = 0.0
        weighted_estimate = 0.0
        for expert in self.experts.values():
            prior = expert.prior_weight
            prior_total += prior
            prior_estimate += prior * expert.estimate
            weighted = prior * max(0.0, expert.weight)
            weighted_total += weighted
            weighted_estimate += weighted * expert.estimate
        estimate = (
            weighted_estimate / weighted_total
            if weighted_total > 0
            else prior_estimate / prior_total
        )
        return float(np.clip(estimate, 0.0, 1.0))

    def update(self, score: float) -> None:
        score = float(np.clip(score, 0.0, 1.0))
        estimate_before_update = self.predict()
        for start in [key for key, expert in self.experts.items() if expert.expired]:
            self.experts.pop(start)
        self.experts[self.time] = _Expert(
            start=self.time,
            scale=self.scale,
            coverage=self.coverage,
            estimate=estimate_before_update,
            lifetime_multiplier=self.lifetime,
        )
        mixture_loss = pinball_loss(score, self.predict(), self.coverage)
        for expert in self.experts.values():
            expert.update(score, mixture_loss)
        self.time += 1


def block_slices(length: int, block_size: int) -> list[tuple[int, int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [
        (start, min(start + block_size, length))
        for start in range(0, length, block_size)
    ]


def nonconformity_scores(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if len(probability) != len(labels):
        raise ValueError("Probability/label length mismatch")
    return np.abs(labels - probability)


def block_nonconformity_scores(
    probability: np.ndarray,
    labels: np.ndarray,
    coverage: float,
    block_size: int,
) -> np.ndarray:
    residual = nonconformity_scores(probability, labels)
    return np.asarray(
        [
            np.quantile(residual[start:end], coverage)
            for start, end in block_slices(len(residual), block_size)
        ],
        dtype=float,
    )


def make_calibrator(
    calibration_scores: np.ndarray, coverage: float, lifetime: int
) -> SAOCPQuantile:
    scores = np.asarray(calibration_scores, dtype=float).reshape(-1)
    if len(scores) == 0:
        raise ValueError("At least one calibration score is required")
    calibrator = SAOCPQuantile(
        coverage=coverage,
        scale=max(float(np.max(np.abs(scores))) * math.sqrt(3.0), 1e-3),
        initial_radius=float(np.quantile(scores, coverage)),
        lifetime=lifetime,
    )
    for score in scores:
        calibrator.update(float(score))
    return calibrator


def policy_threshold(radius: float, policy: str) -> float:
    if policy == "positive_inclusion":
        return float(np.clip(1.0 - radius, 0.0, 1.0))
    if policy == "positive_singleton":
        return float(np.clip(max(radius, 1.0 - radius), 0.0, 1.0))
    raise ValueError(f"Unknown policy: {policy}")


def online_predict(
    probability: np.ndarray,
    delayed_labels: np.ndarray,
    calibrator: SAOCPQuantile,
    coverage: float = 0.85,
    block_size: int = 64,
    policy: str = "positive_singleton",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict a complete block before using any label from that block."""

    probability = np.asarray(probability, dtype=float).reshape(-1)
    labels = np.asarray(delayed_labels).reshape(-1)
    if len(probability) != len(labels):
        raise ValueError("Probability/label length mismatch")
    decisions = np.zeros(len(probability), dtype=np.uint8)
    thresholds = np.zeros(len(probability), dtype=np.float32)
    radii: list[float] = []
    for start, end in block_slices(len(probability), block_size):
        radius = calibrator.predict()
        threshold = policy_threshold(radius, policy)
        decisions[start:end] = probability[start:end] >= threshold
        thresholds[start:end] = threshold
        radii.append(radius)
        # This update happens only after every decision in [start, end) exists.
        residual = np.abs(labels[start:end].astype(float) - probability[start:end])
        calibrator.update(float(np.quantile(residual, coverage)))
    return decisions, thresholds, np.asarray(radii, dtype=np.float32)


def correct_short_runs(labels: np.ndarray, maximum_points: int) -> np.ndarray:
    """Apply the documented two-row correction without crossing array bounds."""

    original = np.asarray(labels, dtype=np.uint8).reshape(-1)
    corrected = original.copy()
    if len(original) < 2:
        return corrected
    boundaries = np.r_[
        0,
        np.flatnonzero(original[1:] != original[:-1]) + 1,
        len(original),
    ]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start <= maximum_points and end < len(original):
            corrected[start:end] = original[end]
    return corrected
