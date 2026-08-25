"""Prepare a self-contained source-data bundle for the locked case figure.

This script is an audit utility, not a training or threshold-tuning script.  It
reads the already locked test probabilities, thresholds, labels, event
intervals, and catalog.  It applies the documented event-selection rule, then
copies only the selected display window into CSV files consumed by
``plot_event28_one_to_one.py``.

Example
-------
python prepare_locked_selected_event_source.py \
    --experiments-dir PATH_TO_FROZEN_PROBABILITIES \
    --decisions-dir PATH_TO_LOCKED_DECISIONS \
    --thresholds-csv PATH_TO_VALIDATION_LOCKED_THRESHOLDS \
    --catalog PATH_TO_TEST_CATALOG_CSV
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


TEST_START = pd.Timestamp("2010-01-01 00:00:00")
TEST_END = pd.Timestamp("2016-01-01 00:00:00")
ISOLATION_HOURS = 48.0
DISPLAY_MARGIN_HOURS = 12.0
MIN_SAOCP_COVERAGE = 0.5

MODEL_SPECS = {
    "lstm": ("lstm_bce_f34_w64", "LSTM"),
    "cnn_lstm": ("cnn_lstm_bce_f34_w64", "CNN_LSTM"),
    "unet": ("unet_bce_f34_w64", "UNet"),
    "runet": ("runet_bce_f34_w64", "RUNet"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], format="mixed", errors="raise")
        if getattr(frame[column].dt, "tz", None) is not None:
            frame[column] = frame[column].dt.tz_localize(None)
    return frame


def read_catalog(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"begin", "end"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Catalog must contain {sorted(required)}")
    frame = parse_time_columns(frame, ("begin", "end"))
    frame = frame[(frame["end"] > TEST_START) & (frame["begin"] < TEST_END)].copy()
    frame["begin"] = frame["begin"].clip(lower=TEST_START)
    frame["end"] = frame["end"].clip(upper=TEST_END)
    frame = frame.sort_values(["begin", "end"], kind="stable").reset_index(drop=True)
    if "catalog_event_id" in frame.columns:
        frame = frame.drop(columns=["catalog_event_id"])
    frame.insert(0, "catalog_event_id", np.arange(1, len(frame) + 1))
    if len(frame) != 230:
        raise ValueError(f"Expected 230 locked test catalog events, found {len(frame)}")
    if (frame["end"] <= frame["begin"]).any():
        raise ValueError("Catalog contains a non-positive event duration")
    return frame


def nearest_other_gap_hours(catalog: pd.DataFrame, index: int) -> float:
    target = catalog.iloc[index]
    gaps: list[float] = []
    for other_index, other in catalog.iterrows():
        if index == other_index:
            continue
        if other["end"] <= target["begin"]:
            gap = (target["begin"] - other["end"]).total_seconds() / 3600.0
        elif other["begin"] >= target["end"]:
            gap = (other["begin"] - target["end"]).total_seconds() / 3600.0
        else:
            gap = -min(
                (target["end"] - target["begin"]).total_seconds() / 3600.0,
                (other["end"] - other["begin"]).total_seconds() / 3600.0,
            )
        gaps.append(float(gap))
    return min(gaps) if gaps else float("inf")


def overlap_hours(
    begin_a: pd.Timestamp,
    end_a: pd.Timestamp,
    begin_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> float:
    left = max(begin_a, begin_b)
    right = min(end_a, end_b)
    return max(0.0, (right - left).total_seconds() / 3600.0)


def best_catalog_coverage(
    intervals: pd.DataFrame,
    model: str,
    method: str,
    begin: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    candidates = intervals[(intervals["run"] == model) & (intervals["method"] == method)]
    duration = (end - begin).total_seconds() / 3600.0
    if duration <= 0:
        raise ValueError("Catalog duration must be positive")
    if candidates.empty:
        return 0.0
    overlaps = [
        overlap_hours(begin, end, row.begin, row.end)
        for row in candidates.itertuples(index=False)
    ]
    return float(max(overlaps, default=0.0) / duration)


def count_unrelated_display_events(
    intervals: pd.DataFrame,
    begin: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    display_begin = begin - pd.Timedelta(hours=DISPLAY_MARGIN_HOURS)
    display_end = end + pd.Timedelta(hours=DISPLAY_MARGIN_HOURS)
    visible = intervals[
        (intervals["end"] >= display_begin) & (intervals["begin"] <= display_end)
    ]
    unrelated = [
        overlap_hours(begin, end, row.begin, row.end) == 0.0
        for row in visible.itertuples(index=False)
    ]
    return int(sum(unrelated))


def build_selection_audit(catalog: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    audit = catalog.copy()
    audit["duration_hours"] = (
        audit["end"] - audit["begin"]
    ).dt.total_seconds() / 3600.0
    audit["nearest_other_catalog_gap_hours"] = [
        nearest_other_gap_hours(audit, index) for index in range(len(audit))
    ]
    audit["isolated_at_48_hours"] = (
        audit["nearest_other_catalog_gap_hours"] >= ISOLATION_HOURS
    )

    for model in MODEL_SPECS:
        static_values: list[float] = []
        saocp_values: list[float] = []
        unrelated_values: list[int] = []
        model_intervals = intervals[intervals["run"] == model]
        for row in audit.itertuples(index=False):
            static_values.append(
                best_catalog_coverage(
                    intervals, model, "static", row.begin, row.end
                )
            )
            saocp_values.append(
                best_catalog_coverage(
                    intervals, model, "saocp", row.begin, row.end
                )
            )
            unrelated_values.append(
                count_unrelated_display_events(model_intervals, row.begin, row.end)
            )
        audit[f"{model}_static_catalog_coverage"] = static_values
        audit[f"{model}_saocp_catalog_coverage"] = saocp_values
        audit[f"{model}_coverage_gain"] = (
            audit[f"{model}_saocp_catalog_coverage"]
            - audit[f"{model}_static_catalog_coverage"]
        )
        audit[f"{model}_unrelated_events_in_display"] = unrelated_values

    saocp_cols = [f"{model}_saocp_catalog_coverage" for model in MODEL_SPECS]
    gain_cols = [f"{model}_coverage_gain" for model in MODEL_SPECS]
    unrelated_cols = [f"{model}_unrelated_events_in_display" for model in MODEL_SPECS]
    audit["minimum_saocp_catalog_coverage"] = audit[saocp_cols].min(axis=1)
    audit["minimum_coverage_gain"] = audit[gain_cols].min(axis=1)
    audit["mean_coverage_gain"] = audit[gain_cols].mean(axis=1)
    audit["all_saocp_coverages_at_least_0_5"] = (
        audit["minimum_saocp_catalog_coverage"] >= MIN_SAOCP_COVERAGE
    )
    audit["all_saocp_coverages_improve"] = audit["minimum_coverage_gain"] > 0.0
    audit["unrelated_display_events_total"] = audit[unrelated_cols].sum(axis=1)
    audit["clean_fixed_12h_display"] = audit["unrelated_display_events_total"] == 0
    audit["eligible_before_display_filter"] = (
        audit["isolated_at_48_hours"]
        & audit["all_saocp_coverages_at_least_0_5"]
        & audit["all_saocp_coverages_improve"]
    )
    audit["eligible"] = (
        audit["eligible_before_display_filter"] & audit["clean_fixed_12h_display"]
    )
    audit["selection_rank"] = pd.Series(pd.NA, index=audit.index, dtype="Int64")
    eligible = audit[audit["eligible"]].sort_values(
        [
            "minimum_coverage_gain",
            "minimum_saocp_catalog_coverage",
            "mean_coverage_gain",
            "begin",
            "catalog_event_id",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    )
    for rank, index in enumerate(eligible.index, start=1):
        audit.loc[index, "selection_rank"] = rank
    audit["selected"] = (audit["selection_rank"] == 1).fillna(False).astype(bool)
    return audit


def load_thresholds(path: Path) -> dict[str, float]:
    table = pd.read_csv(path)
    required = {"run", "threshold"}
    if not required.issubset(table.columns):
        raise ValueError("Static-threshold table lacks run/threshold columns")
    values = {str(row.run): float(row.threshold) for row in table.itertuples(index=False)}
    missing = set(MODEL_SPECS) - set(values)
    if missing:
        raise ValueError(f"Missing validation-selected thresholds: {sorted(missing)}")
    return values


def extract_model_window(
    experiments_dir: Path,
    decisions_dir: Path,
    model: str,
    experiment: str,
    output_name: str,
    static_threshold: float,
    display_begin: pd.Timestamp,
    display_end: pd.Timestamp,
    event_begin: pd.Timestamp,
    event_end: pd.Timestamp,
    output_dir: Path,
) -> list[dict[str, str]]:
    experiment_dir = experiments_dir / experiment
    prediction_dir = decisions_dir / model
    input_paths = {
        "probability_test.npy": experiment_dir / "probability_test.npy",
        "time_test.npy": experiment_dir / "time_test.npy",
        "saocp_threshold_series.npy": prediction_dir / "saocp_threshold_series.npy",
        "static_labels.npy": prediction_dir / "static_labels.npy",
        "saocp_labels.npy": prediction_dir / "saocp_labels.npy",
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    probability = np.load(input_paths["probability_test.npy"], mmap_mode="r").reshape(-1)
    time = pd.DatetimeIndex(np.load(input_paths["time_test.npy"], mmap_mode="r").reshape(-1))
    saocp_threshold = np.load(
        input_paths["saocp_threshold_series.npy"], mmap_mode="r"
    ).reshape(-1)
    static_labels = np.load(input_paths["static_labels.npy"], mmap_mode="r").reshape(-1)
    saocp_labels = np.load(input_paths["saocp_labels.npy"], mmap_mode="r").reshape(-1)
    lengths = {
        len(probability),
        len(time),
        len(saocp_threshold),
        len(static_labels),
        len(saocp_labels),
    }
    if len(lengths) != 1:
        raise ValueError(f"{model}: locked arrays have different lengths")
    if not time.is_monotonic_increasing:
        raise ValueError(f"{model}: time index is not monotonic")
    mask = (time >= display_begin) & (time <= display_end)
    if not mask.any():
        raise ValueError(f"{model}: no observations in the selected window")
    selected_time = time[mask]
    frame = pd.DataFrame(
        {
            "time_utc": selected_time,
            "backbone_probability": np.asarray(probability[mask], dtype=float),
            "saocp_threshold": np.asarray(saocp_threshold[mask], dtype=float),
            "static_threshold": float(static_threshold),
            "catalog_active": (
                (selected_time >= event_begin) & (selected_time <= event_end)
            ).astype(np.uint8),
            "static_point_label": np.asarray(static_labels[mask], dtype=np.uint8),
            "saocp_point_label": np.asarray(saocp_labels[mask], dtype=np.uint8),
        }
    )
    frame.to_csv(
        output_dir / "source_data" / f"{output_name}_window.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    return [
        {
            "logical_input": f"{model}/{name}",
            "sha256": sha256(path),
            "n_bytes": str(path.stat().st_size),
        }
        for name, path in input_paths.items()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        required=True,
        help="Directory containing the four locked experiment subdirectories.",
    )
    parser.add_argument(
        "--decisions-dir",
        type=Path,
        required=True,
        help="Directory containing locked_decisions/{run}/ arrays and event CSVs.",
    )
    parser.add_argument(
        "--thresholds-csv",
        type=Path,
        required=True,
        help="Validation-only selected static-threshold audit CSV.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Locked test-catalog CSV with begin/end columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--expected-event-id",
        type=int,
        default=None,
        help="Optional audit guard; fail if the deterministic rank-1 event differs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments_dir = args.experiments_dir.resolve()
    decisions_dir = args.decisions_dir.resolve()
    threshold_path = args.thresholds_csv.resolve()
    catalog_path = args.catalog.resolve()
    output_dir = args.output_dir.resolve()
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    interval_paths = [
        decisions_dir / model / "constructed_events.csv" for model in MODEL_SPECS
    ]
    for path in interval_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    intervals = pd.concat(
        [pd.read_csv(path) for path in interval_paths], ignore_index=True
    )
    intervals = parse_time_columns(intervals, ("begin", "end"))
    intervals = intervals.rename(columns={"event_index": "predicted_event_id"})
    catalog = read_catalog(catalog_path)
    audit = build_selection_audit(catalog, intervals)
    selected = audit[audit["selected"]]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one rank-1 event; found {len(selected)}")
    selected = selected.iloc[0]
    if (
        args.expected_event_id is not None
        and int(selected["catalog_event_id"]) != args.expected_event_id
    ):
        raise RuntimeError(
            f"Rank-1 event is {int(selected['catalog_event_id'])}, not the expected "
            f"event {args.expected_event_id}; refusing to overwrite the source data."
        )

    display_begin = selected["begin"] - pd.Timedelta(hours=DISPLAY_MARGIN_HOURS)
    display_end = selected["end"] + pd.Timedelta(hours=DISPLAY_MARGIN_HOURS)
    audit.to_csv(
        output_dir / "selection_audit.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    catalog.to_csv(
        source_dir / "catalog_test_2010_2015.csv",
        index=False,
        encoding="utf-8-sig",
    )

    case_tag = f"event{int(selected['catalog_event_id'])}"
    selected_record = selected.to_dict()
    selected_record.update(
        {
            "case_tag": case_tag,
            "display_begin_utc": display_begin,
            "display_end_utc": display_end,
            "selection_rule": (
                "nearest catalog gap >=48 h; all four SAOCP catalog coverages >=0.5; "
                "all four gains >0; no unrelated constructed event in the fixed +/-12 h display; "
                "rank by minimum gain, minimum SAOCP coverage, mean gain, then earliest UTC"
            ),
        }
    )
    pd.DataFrame([selected_record]).to_csv(
        source_dir / "selected_event.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    visible_intervals = intervals[
        (intervals["end"] >= display_begin) & (intervals["begin"] <= display_end)
    ].copy()
    visible_intervals["display_begin"] = visible_intervals["begin"].clip(
        lower=display_begin
    )
    visible_intervals["display_end"] = visible_intervals["end"].clip(upper=display_end)
    visible_intervals = visible_intervals[
        [
            "run",
            "method",
            "predicted_event_id",
            "begin",
            "end",
            "display_begin",
            "display_end",
            "duration_hours",
        ]
    ]
    visible_intervals.to_csv(
        source_dir / "display_event_intervals.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    static_thresholds = load_thresholds(threshold_path)
    hash_rows = [
        {
            "logical_input": "catalog_test_2010_2015.csv",
            "sha256": sha256(catalog_path),
            "n_bytes": str(catalog_path.stat().st_size),
        },
        {
            "logical_input": "selected_static_thresholds_validation_only.csv",
            "sha256": sha256(threshold_path),
            "n_bytes": str(threshold_path.stat().st_size),
        },
    ]
    hash_rows.extend(
        {
            "logical_input": f"locked_decisions/{path.parent.name}/constructed_events.csv",
            "sha256": sha256(path),
            "n_bytes": str(path.stat().st_size),
        }
        for path in interval_paths
    )
    for model, (experiment, output_name) in MODEL_SPECS.items():
        hash_rows.extend(
            extract_model_window(
                experiments_dir,
                decisions_dir,
                model,
                experiment,
                f"{output_name}_{case_tag}",
                static_thresholds[model],
                display_begin,
                display_end,
                selected["begin"],
                selected["end"],
                output_dir,
            )
        )
    pd.DataFrame(hash_rows).to_csv(
        source_dir / "locked_input_hashes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eligible_before = int(audit["eligible_before_display_filter"].sum())
    eligible_after = int(audit["eligible"].sum())
    print(
        f"Selected event {int(selected['catalog_event_id'])}: "
        f"{selected['begin']} to {selected['end']} UTC; "
        f"eligible before/after display filter = {eligible_before}/{eligible_after}; "
        f"window rows written to {source_dir}"
    )


if __name__ == "__main__":
    main()
