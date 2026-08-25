"""Evaluate the validation-locked one-to-one protocol exactly once on test.

The script refuses to start unless the separate validation selector has written
``VALIDATION_SELECTION_COMPLETE.json`` and all validation input hashes still
match.  It does not search or rank any test result.
"""

from __future__ import annotations

import argparse
import csv
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
    from .select_validation_config import sha256_file
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
    from select_validation_config import sha256_file


TEST_START = pd.Timestamp("2010-01-01")
TEST_END = pd.Timestamp("2016-01-01")
BCE_BACKBONES = ("lstm", "cnn_lstm", "unet", "runet")


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


def verify_validation_lock(lock_dir: Path, run_map: dict[str, Path], catalog: Path) -> dict:
    lock_path = lock_dir / "VALIDATION_SELECTION_COMPLETE.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing validation lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VALIDATION_SELECTION_COMPLETE_TEST_NOT_OPENED":
        raise ValueError("Validation lock has an unexpected status")
    if lock.get("test_files_opened") is not False:
        raise ValueError("Validation selector did not certify test isolation")

    hash_rows = list(csv.DictReader((lock_dir / "validation_input_hashes.csv").open()))
    expected = {(row["role"], row["run"]): row for row in hash_rows}
    catalog_row = expected.get(("catalog", ""))
    if catalog_row is None or sha256_file(catalog) != catalog_row["sha256"]:
        raise ValueError("Catalog hash no longer matches the validation lock")
    role_to_filename = {
        "validation_probability": "probability_val.npy",
        "validation_labels": "Y_val_aligned.npy",
        "validation_time": "time_val.npy",
    }
    for name, run_dir in run_map.items():
        for role, filename in role_to_filename.items():
            row = expected.get((role, name))
            path = run_dir / filename
            if row is None or not path.is_file() or sha256_file(path) != row["sha256"]:
                raise ValueError(f"{name}: {filename} differs from the validation lock")
    return lock


def load_locked_arrays(name: str, run_dir: Path) -> dict:
    files = {
        "probability_val": run_dir / "probability_val.npy",
        "labels_val": run_dir / "Y_val_aligned.npy",
        "probability_test": run_dir / "probability_test.npy",
        "labels_test": run_dir / "Y_test_aligned.npy",
        "time_test": run_dir / "time_test.npy",
    }
    missing = [path.name for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: missing {', '.join(missing)}")
    arrays = {key: np.load(path).reshape(-1) for key, path in files.items()}
    arrays["time_test"] = pd.DatetimeIndex(arrays["time_test"])
    if len(arrays["probability_val"]) != len(arrays["labels_val"]):
        raise ValueError(f"{name}: validation probability/label mismatch")
    if not (
        len(arrays["probability_test"])
        == len(arrays["labels_test"])
        == len(arrays["time_test"])
    ):
        raise ValueError(f"{name}: test probability/label/time mismatch")
    for split in ("val", "test"):
        probability = arrays[f"probability_{split}"]
        labels = arrays[f"labels_{split}"]
        if not np.all(np.isfinite(probability)):
            raise ValueError(f"{name}: {split} probability contains NaN/Inf")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError(f"{name}: {split} probability lies outside [0,1]")
        if not set(np.unique(labels).tolist()).issubset({0, 1}):
            raise ValueError(f"{name}: {split} labels are not binary")
    return {**arrays, "files": files}


def metric_row(
    run: str,
    method: str,
    events: Sequence[Event],
    truth: Sequence[Event],
    **settings,
) -> dict:
    return {
        "run": run,
        "method": method,
        "split": "locked_test",
        "status": "locked_one_to_one",
        "matching": "maximum_cardinality_one_to_one_any_positive_overlap",
        "unmatched_prediction_rule": "every_unmatched_constructed_event_is_fp",
        "truth_events": len(truth),
        "predicted_events": len(events),
        **settings,
        **event_f1_one_to_one(events, truth),
    }


def matching_rows(
    run: str, method: str, events: Sequence[Event], truth: Sequence[Event]
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
                "status": "matched_tp",
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


def constructed_event_rows(
    run: str, method: str, events: Sequence[Event], truth: Sequence[Event]
) -> list[dict]:
    pairs = maximum_cardinality_matches(events, truth)
    matched = {prediction: target for prediction, target in pairs}
    return [
        {
            "run": run,
            "method": method,
            "event_index": index,
            "begin": event.begin.isoformat(),
            "end": event.end.isoformat(),
            "duration_hours": event.duration_hours,
            "matching_status": "matched_tp" if index in matched else "unmatched_fp",
            "matched_catalog_index": matched.get(index, ""),
            "matched_overlap_hours": (
                overlap_hours(event, truth[matched[index]]) if index in matched else 0.0
            ),
        }
        for index, event in enumerate(events)
    ]


def threshold_block_rows(
    run: str,
    time_index: pd.DatetimeIndex,
    coverage: float,
    lifetime: int,
    thresholds: np.ndarray,
    radii: np.ndarray,
    block_size: int,
) -> list[dict]:
    rows: list[dict] = []
    for block_index, (start, end) in enumerate(block_slices(len(time_index), block_size)):
        rows.append(
            {
                "run": run,
                "split": "locked_test",
                "block_index": block_index,
                "row_start": start,
                "row_end_exclusive": end,
                "time_begin": time_index[start].isoformat(),
                "time_end": time_index[end - 1].isoformat(),
                "coverage": coverage,
                "lifetime": lifetime,
                "radius": float(radii[block_index]),
                "threshold": float(thresholds[start]),
            }
        )
    return rows


def all_one_regression(
    time_index: pd.DatetimeIndex,
    truth: Sequence[Event],
    protocol: EventProtocol,
) -> list[dict]:
    constructed = construct_events(np.ones(len(time_index)), time_index, protocol)
    spanning = [Event(pd.Timestamp(time_index[0]), pd.Timestamp(time_index[-1]))]
    rows: list[dict] = []
    for case, events in (
        ("all_one_labels_after_gap_aware_constructor", constructed),
        ("single_prediction_spanning_test_interval", spanning),
    ):
        metrics = event_f1_one_to_one(events, truth)
        if int(metrics["tp"]) > len(events):
            raise AssertionError(f"{case}: one prediction received multiple TP credits")
        rows.append(
            {
                "case": case,
                "truth_events": len(truth),
                "predicted_events": len(events),
                **metrics,
                "assert_tp_not_greater_than_predictions": True,
                "passed": True,
            }
        )
    if int(rows[1]["tp"]) != 1:
        raise AssertionError("The single spanning prediction must receive exactly one TP")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--validation-lock-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_map = dict(args.run)
    if len(run_map) != len(args.run) or not run_map:
        raise ValueError("At least one uniquely named run is required")
    missing_backbones = sorted(set(BCE_BACKBONES).difference(run_map))
    if missing_backbones:
        raise ValueError(f"Missing BCE backbones: {missing_backbones}")
    run_map = {name: path.resolve() for name, path in run_map.items()}
    catalog_path = args.catalog.resolve()
    lock_dir = args.validation_lock_dir.resolve()

    # The validation hash lock is verified before the first test array is read.
    lock = verify_validation_lock(lock_dir, run_map, catalog_path)
    static_thresholds = {
        str(name): float(value)
        for name, value in lock["selected_static_thresholds"].items()
    }
    selected_saocp = lock["selected_saocp"]
    coverage = float(selected_saocp["coverage"])
    lifetime = int(selected_saocp["lifetime"])
    policy = str(selected_saocp["policy"])
    block_size = int(selected_saocp["block_size"])
    protocol = EventProtocol(**lock["event_protocol"])

    arrays = {name: load_locked_arrays(name, path) for name, path in run_map.items()}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    match_rows: list[dict] = []
    threshold_rows: list[dict] = []

    decision_root = output_dir / "locked_decisions"
    decision_root.mkdir(parents=True, exist_ok=True)

    for name, run_arrays in arrays.items():
        truth = read_catalog_events(catalog_path, TEST_START, TEST_END)
        static_threshold = static_thresholds[name]
        static_labels = np.asarray(
            run_arrays["probability_test"] >= static_threshold, dtype=np.uint8
        )
        static_events = construct_events(static_labels, run_arrays["time_test"], protocol)
        result_rows.append(
            metric_row(
                name,
                "static",
                static_events,
                truth,
                static_threshold=static_threshold,
            )
        )
        match_rows.extend(matching_rows(name, "static", static_events, truth))

        validation_scores = block_nonconformity_scores(
            run_arrays["probability_val"],
            run_arrays["labels_val"],
            coverage,
            block_size,
        )
        calibrator = make_calibrator(validation_scores, coverage, lifetime)
        saocp_labels, thresholds, radii = online_predict(
            run_arrays["probability_test"],
            run_arrays["labels_test"],
            calibrator,
            coverage=coverage,
            block_size=block_size,
            policy=policy,
        )
        saocp_events = construct_events(saocp_labels, run_arrays["time_test"], protocol)
        result_rows.append(
            metric_row(
                name,
                "saocp",
                saocp_events,
                truth,
                saocp_coverage=coverage,
                saocp_lifetime=lifetime,
                saocp_policy=policy,
                block_size=block_size,
                mean_threshold=float(np.mean(thresholds)),
                std_threshold=float(np.std(thresholds)),
                minimum_threshold=float(np.min(thresholds)),
                maximum_threshold=float(np.max(thresholds)),
            )
        )
        match_rows.extend(matching_rows(name, "saocp", saocp_events, truth))
        threshold_rows.extend(
            threshold_block_rows(
                name,
                run_arrays["time_test"],
                coverage,
                lifetime,
                thresholds,
                radii,
                block_size,
            )
        )
        run_output = decision_root / name
        run_output.mkdir(parents=True, exist_ok=True)
        np.save(run_output / "static_labels.npy", static_labels)
        np.save(run_output / "saocp_labels.npy", saocp_labels)
        np.save(run_output / "saocp_threshold_series.npy", thresholds)
        np.save(run_output / "saocp_block_radii.npy", radii)
        write_csv(
            run_output / "constructed_events.csv",
            constructed_event_rows(name, "static", static_events, truth)
            + constructed_event_rows(name, "saocp", saocp_events, truth),
        )

    truth = read_catalog_events(catalog_path, TEST_START, TEST_END)
    all_one_rows = all_one_regression(arrays[BCE_BACKBONES[0]]["time_test"], truth, protocol)

    write_csv(output_dir / "test_event_metrics.csv", result_rows)
    write_csv(
        output_dir / "backbone_event_f1.csv",
        [row for row in result_rows if row["run"] in BCE_BACKBONES],
    )
    write_csv(
        output_dir / "pm_window_ablation_event_f1.csv",
        [row for row in result_rows if row["run"] not in BCE_BACKBONES],
    )
    write_csv(output_dir / "test_event_matching.csv", match_rows)
    write_csv(output_dir / "test_saocp_threshold_blocks.csv", threshold_rows)
    write_csv(output_dir / "all_one_regression.csv", all_one_rows)

    input_rows: list[dict] = [
        {
            "role": "validation_lock",
            "run": "",
            "filename": "VALIDATION_SELECTION_COMPLETE.json",
            "bytes": (lock_dir / "VALIDATION_SELECTION_COMPLETE.json").stat().st_size,
            "sha256": sha256_file(lock_dir / "VALIDATION_SELECTION_COMPLETE.json"),
        },
        {
            "role": "catalog",
            "run": "",
            "filename": catalog_path.name,
            "bytes": catalog_path.stat().st_size,
            "sha256": sha256_file(catalog_path),
        },
    ]
    for name, run_arrays in arrays.items():
        for role, path in run_arrays["files"].items():
            input_rows.append(
                {
                    "role": role,
                    "run": name,
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_csv(output_dir / "test_input_hashes.csv", input_rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_TEST_EVALUATION_COMPLETE_NO_TEST_SELECTION",
        "validation_lock_sha256": sha256_file(
            lock_dir / "VALIDATION_SELECTION_COMPLETE.json"
        ),
        "configuration_reselected_after_test": False,
        "matching": "deterministic maximum-cardinality one-to-one any-positive-overlap",
        "false_positive_rule": "every unmatched constructed prediction is FP",
        "event_protocol": asdict(protocol),
        "legacy_comparator_only": {
            "false_positive_min_hours": 2.5,
            "used_by_primary_metric": False,
        },
        "static_thresholds": static_thresholds,
        "saocp": {
            "coverage": coverage,
            "lifetime": lifetime,
            "policy": policy,
            "block_size": block_size,
            "selection_source": "validation lock only",
        },
        "test_results_used_for_selection": False,
        "matching_cardinality_independently_cross_checked_on_every_evaluation": True,
        "all_one_regression_passed": True,
    }
    manifest_path = output_dir / "LOCKED_TEST_AUDIT.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generated = sorted(
        [
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "test_output_hashes.csv"
        ],
        key=lambda path: str(path.relative_to(output_dir)).lower(),
    )
    write_csv(
        output_dir / "test_output_hashes.csv",
        [
            {
                "file": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in generated
        ],
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(result_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
