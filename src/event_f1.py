"""Event-level F1 evaluators for ICME event detection.

``event_f1_one_to_one`` is the robust evaluator: any positive temporal overlap
forms an admissible edge, and maximum-cardinality one-to-one matching determines
TP, FP, and FN.  It prevents a single all-period prediction from detecting every
catalog event.

``event_f1`` retains the earlier Li-style many-to-many counting rule solely for
historical-result comparison and internal audit.  It must not be mistaken for a
one-to-one detector score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import deque
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .saocp import correct_short_runs
except ImportError:  # Allow `python src/event_f1.py ...`.
    from saocp import correct_short_runs


@dataclass(frozen=True)
class Event:
    begin: pd.Timestamp
    end: pd.Timestamp

    @property
    def duration_hours(self) -> float:
        return (self.end - self.begin).total_seconds() / 3600.0


@dataclass(frozen=True)
class EventProtocol:
    correction_points: int = 2
    minimum_points: int = 48
    merge_hours: float = 12.0
    max_observation_gap_minutes: float = 30.0


LEGACY_FALSE_POSITIVE_MIN_HOURS = 2.5


def overlap_hours(first: Event, second: Event) -> float:
    return max(
        0.0,
        (min(first.end, second.end) - max(first.begin, second.begin)).total_seconds()
        / 3600.0,
    )


def read_catalog_events(
    path: Path, start: pd.Timestamp, end: pd.Timestamp
) -> list[Event]:
    frame = pd.read_csv(path)
    frame["begin"] = pd.to_datetime(frame["begin"], format="mixed").dt.tz_localize(None)
    frame["end"] = pd.to_datetime(frame["end"], format="mixed").dt.tz_localize(None)
    frame = frame[(frame["end"] > start) & (frame["begin"] < end)]
    frame = frame.sort_values("begin", kind="stable")
    return [
        Event(max(row.begin, start), min(row.end, end))
        for row in frame.itertuples(index=False)
    ]


def _continuous_segments(
    time_index: pd.DatetimeIndex, maximum_gap_minutes: float
) -> list[tuple[int, int]]:
    if len(time_index) == 0:
        return []
    if not time_index.is_monotonic_increasing:
        raise ValueError("Time index must be monotonic nondecreasing")
    values = time_index.values.astype("datetime64[ns]").astype(np.int64)
    differences = np.diff(values)
    limit = int(maximum_gap_minutes * 60.0 * 1e9)
    starts = np.r_[0, np.flatnonzero(differences > limit) + 1]
    ends = np.r_[starts[1:], len(time_index)]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _merge(events: Sequence[Event], merge_hours: float) -> list[Event]:
    if not events:
        return []
    merged = [events[0]]
    limit = pd.Timedelta(hours=merge_hours)
    for event in events[1:]:
        previous = merged[-1]
        if event.begin - previous.end < limit:
            merged[-1] = Event(previous.begin, max(previous.end, event.end))
        else:
            merged.append(event)
    return merged


def construct_events(
    labels: np.ndarray,
    time_index: pd.DatetimeIndex,
    protocol: EventProtocol = EventProtocol(),
) -> list[Event]:
    """Construct events without correcting or merging across observation gaps."""

    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if len(labels) != len(time_index):
        raise ValueError("Label/time-index length mismatch")
    all_events: list[Event] = []
    for segment_start, segment_end in _continuous_segments(
        time_index, protocol.max_observation_gap_minutes
    ):
        corrected = correct_short_runs(
            labels[segment_start:segment_end], protocol.correction_points
        )
        boundaries = np.r_[
            0,
            np.flatnonzero(corrected[1:] != corrected[:-1]) + 1,
            len(corrected),
        ]
        segment_events: list[Event] = []
        for local_start, local_end in zip(boundaries[:-1], boundaries[1:]):
            # The established team constructor uses a strict `> minimum_points` rule.
            if corrected[local_start] != 1 or local_end - local_start <= protocol.minimum_points:
                continue
            global_start = segment_start + int(local_start)
            global_end = segment_start + int(local_end) - 1
            segment_events.append(
                Event(
                    pd.Timestamp(time_index[global_start]),
                    pd.Timestamp(time_index[global_end]),
                )
            )
        all_events.extend(_merge(segment_events, protocol.merge_hours))
    return all_events


def event_f1(
    predicted: Sequence[Event],
    truth: Sequence[Event],
    false_positive_min_hours: float = 2.5,
) -> dict[str, float | int]:
    """Return the legacy Li-style many-to-many event score.

    A catalog event is detected by any positive overlap.  Completely unmatched
    predictions shorter than ``false_positive_min_hours`` are ignored.  Because
    one prediction may detect multiple catalog events, this function is kept as
    a legacy comparator/internal audit only.
    """

    detected_truth = [
        any(overlap_hours(prediction, target) > 0 for prediction in predicted)
        for target in truth
    ]
    tp = int(sum(detected_truth))
    fn = int(len(truth) - tp)
    fp = int(
        sum(
            prediction.duration_hours >= false_positive_min_hours
            and not any(overlap_hours(prediction, target) > 0 for target in truth)
            for prediction in predicted
        )
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def event_f1_one_to_one(
    predicted: Sequence[Event], truth: Sequence[Event]
) -> dict[str, float | int]:
    """Score maximum-cardinality one-to-one matches with positive overlap.

    Each predicted event and each catalog event can contribute to at most one
    TP.  Every unmatched prediction is an FP and every unmatched catalog event
    is an FN.  No boundary-overlap magnitude threshold is imposed.
    """

    matches = maximum_cardinality_matches(predicted, truth)
    tp = len(matches)
    fp = int(len(predicted) - tp)
    fn = int(len(truth) - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def overlap_adjacency(
    predicted: Sequence[Event], truth: Sequence[Event]
) -> list[list[int]]:
    """Return sorted truth neighbours for every prediction."""

    return [
        [
            truth_index
            for truth_index, target in enumerate(truth)
            if overlap_hours(prediction, target) > 0.0
        ]
        for prediction in predicted
    ]


def _hopcroft_karp(
    adjacency: Sequence[Sequence[int]], truth_count: int
) -> list[tuple[int, int]]:
    """Deterministic maximum-cardinality bipartite matching."""

    prediction_count = len(adjacency)
    pair_prediction = [-1] * prediction_count
    pair_truth = [-1] * truth_count
    distance = [-1] * prediction_count

    def breadth_first_search() -> bool:
        queue: deque[int] = deque()
        for prediction in range(prediction_count):
            if pair_prediction[prediction] == -1:
                distance[prediction] = 0
                queue.append(prediction)
            else:
                distance[prediction] = -1
        augmenting_path_exists = False
        while queue:
            prediction = queue.popleft()
            for target in adjacency[prediction]:
                paired_prediction = pair_truth[target]
                if paired_prediction == -1:
                    augmenting_path_exists = True
                elif distance[paired_prediction] == -1:
                    distance[paired_prediction] = distance[prediction] + 1
                    queue.append(paired_prediction)
        return augmenting_path_exists

    def depth_first_search(prediction: int) -> bool:
        for target in adjacency[prediction]:
            paired_prediction = pair_truth[target]
            if paired_prediction == -1 or (
                distance[paired_prediction] == distance[prediction] + 1
                and depth_first_search(paired_prediction)
            ):
                pair_prediction[prediction] = target
                pair_truth[target] = prediction
                return True
        distance[prediction] = -1
        return False

    while breadth_first_search():
        for prediction in range(prediction_count):
            if pair_prediction[prediction] == -1:
                depth_first_search(prediction)
    return [
        (prediction, target)
        for prediction, target in enumerate(pair_prediction)
        if target != -1
    ]


def _kuhn_cardinality(
    adjacency: Sequence[Sequence[int]], truth_count: int
) -> int:
    """Independent augmenting-path cross-check of matching cardinality."""

    matched_truth = [-1] * truth_count

    def augment(prediction: int, seen: list[bool]) -> bool:
        for target in adjacency[prediction]:
            if seen[target]:
                continue
            seen[target] = True
            if matched_truth[target] == -1 or augment(matched_truth[target], seen):
                matched_truth[target] = prediction
                return True
        return False

    return sum(
        augment(prediction, [False] * truth_count)
        for prediction in range(len(adjacency))
    )


def maximum_cardinality_matches(
    predicted: Sequence[Event], truth: Sequence[Event]
) -> list[tuple[int, int]]:
    """Return audited deterministic `(prediction_index, truth_index)` pairs."""

    adjacency = overlap_adjacency(predicted, truth)
    matches = _hopcroft_karp(adjacency, len(truth))
    reference_cardinality = _kuhn_cardinality(adjacency, len(truth))
    if len(matches) != reference_cardinality:
        raise AssertionError(
            "Independent matching implementations disagree: "
            f"{len(matches)} != {reference_cardinality}"
        )
    if len({prediction for prediction, _ in matches}) != len(matches):
        raise AssertionError("A predicted event was matched more than once")
    if len({target for _, target in matches}) != len(matches):
        raise AssertionError("A catalog event was matched more than once")
    return matches


def evaluate_labels(
    labels: np.ndarray,
    time_index: pd.DatetimeIndex,
    truth: Sequence[Event],
    protocol: EventProtocol = EventProtocol(),
) -> tuple[list[Event], dict[str, float | int]]:
    """Construct events and apply the legacy Li-style evaluator."""

    events = construct_events(labels, time_index, protocol)
    metrics = event_f1(events, truth, LEGACY_FALSE_POSITIVE_MIN_HOURS)
    return events, metrics


def evaluate_labels_one_to_one(
    labels: np.ndarray,
    time_index: pd.DatetimeIndex,
    truth: Sequence[Event],
    protocol: EventProtocol = EventProtocol(),
) -> tuple[list[Event], dict[str, float | int]]:
    """Construct events and apply the positive-overlap one-to-one evaluator."""

    events = construct_events(labels, time_index, protocol)
    return events, event_f1_one_to_one(events, truth)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--time", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--matching",
        choices=("one_to_one", "legacy_li"),
        default="one_to_one",
        help="one_to_one is robust; legacy_li is for historical audit only",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    labels = np.load(args.labels)
    time_index = pd.DatetimeIndex(np.load(args.time))
    truth = read_catalog_events(args.catalog, start, end)
    if args.matching == "one_to_one":
        events, metrics = evaluate_labels_one_to_one(labels, time_index, truth)
    else:
        events, metrics = evaluate_labels(labels, time_index, truth)
    result = {
        "protocol": asdict(EventProtocol()),
        "matching": args.matching,
        "truth_events": len(truth),
        "predicted_events": len(events),
        **metrics,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
