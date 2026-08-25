"""Shared schema normalization and integrity checks for one-to-one event metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METHOD_ORDER = ("static", "saocp")
METRIC_COLUMNS = ("precision", "recall", "f1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("‐", "-")
        .replace("‑", "-")
        .replace("‒", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("_", "-")
    )


def validate_manifest(path: Path) -> dict:
    """Require an explicit maximum-cardinality one-to-one positive-overlap contract."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    matching = _normalized_text(manifest.get("matching", ""))
    required_terms = ("maximum", "cardinality", "one-to-one", "positive", "overlap")
    if any(term not in matching for term in required_terms):
        raise ValueError(
            "Manifest must state maximum-cardinality one-to-one matching over positive-duration overlap"
        )
    cross_check = _normalized_text(manifest.get("matching_cross_check", ""))
    if cross_check and "cardinality" not in cross_check:
        raise ValueError("Manifest matching cross-check is not a cardinality audit")
    return manifest


def _wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "family",
        "run",
        "static_tp",
        "static_fp",
        "static_fn",
        "static_precision",
        "static_recall",
        "static_f1",
        "saocp_tp",
        "saocp_fp",
        "saocp_fn",
        "saocp_precision",
        "saocp_recall",
        "saocp_f1",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Wide metric table lacks columns: {sorted(missing)}")
    rows: list[dict] = []
    passthrough = [
        column
        for column in ("fp_convention", "matching", "split", "evaluation_split")
        if column in frame.columns
    ]
    for source in frame.to_dict(orient="records"):
        for method in METHOD_ORDER:
            row = {
                "family": source["family"],
                "run": source["run"],
                "method": method,
                "tp": source[f"{method}_tp"],
                "fp": source[f"{method}_fp"],
                "fn": source[f"{method}_fn"],
                "precision": source[f"{method}_precision"],
                "recall": source[f"{method}_recall"],
                "f1": source[f"{method}_f1"],
            }
            for column in passthrough:
                row[column] = source[column]
            rows.append(row)
    return pd.DataFrame(rows)


def _normalize_long(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "decision" in frame.columns and "method" not in frame.columns:
        frame = frame.rename(columns={"decision": "method"})
    if "evaluation_split" in frame.columns and "split" not in frame.columns:
        frame = frame.rename(columns={"evaluation_split": "split"})
    if "unmatched_prediction_rule" in frame.columns and "fp_convention" not in frame.columns:
        frame = frame.rename(columns={"unmatched_prediction_rule": "fp_convention"})
    if "family" not in frame.columns and "run" in frame.columns:
        backbone_runs = {"lstm", "cnn_lstm", "unet", "runet"}
        frame["family"] = np.where(
            frame["run"].astype(str).isin(backbone_runs),
            "bce_backbones",
            np.where(
                frame["run"].astype(str).str.startswith("cnn_lstm_f"),
                "bce_ablation",
                "unknown",
            ),
        )
        if (frame["family"] == "unknown").any():
            unknown = sorted(frame.loc[frame["family"] == "unknown", "run"].unique())
            raise ValueError(f"Cannot infer result family for runs: {unknown}")
    required = {"family", "run", "method", "tp", "fp", "fn", *METRIC_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Long metric table lacks columns: {sorted(missing)}")
    frame["method"] = frame["method"].astype(str).str.strip().str.lower()
    return frame


def load_one_to_one_metrics(
    csv_path: Path,
    manifest_path: Path,
    fp_convention: str,
    split: str | None = None,
    expected_runs: dict[str, Iterable[str]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read wide or long results and return a validated long-form test table."""
    manifest = validate_manifest(manifest_path)
    raw = pd.read_csv(csv_path)
    if {"static_tp", "saocp_tp"}.issubset(raw.columns):
        frame = _wide_to_long(raw)
    else:
        frame = _normalize_long(raw)

    if "fp_convention" in frame.columns:
        available = sorted(frame["fp_convention"].dropna().astype(str).unique())
        frame = frame[frame["fp_convention"].astype(str) == fp_convention].copy()
        if frame.empty:
            raise ValueError(
                f"Requested fp_convention={fp_convention!r}; available={available}"
            )
    else:
        frame["fp_convention"] = fp_convention

    if "split" in frame.columns:
        available_splits = sorted(frame["split"].dropna().astype(str).unique())
        if len(available_splits) > 1 and split is None:
            raise ValueError(
                f"Metric CSV contains multiple splits {available_splits}; pass --split explicitly"
            )
        if split is not None:
            frame = frame[frame["split"].astype(str) == split].copy()
            if frame.empty:
                raise ValueError(f"Requested split={split!r}; available={available_splits}")
    elif split is not None:
        frame["split"] = split

    if "matching" in frame.columns:
        bad = [
            value
            for value in frame["matching"].dropna().unique()
            if not (
                "one-to-one" in _normalized_text(value)
                and "positive" in _normalized_text(value)
                and "overlap" in _normalized_text(value)
            )
        ]
        if bad:
            raise ValueError(f"Metric rows contain a non-one-to-one matching rule: {bad}")
    else:
        frame["matching"] = manifest["matching"]

    for column in ("tp", "fp", "fn"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if (frame[column] < 0).any():
            raise ValueError(f"Negative count in {column}")
    for column in METRIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not frame[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} leaves [0, 1]")

    calculated_precision = np.divide(
        frame["tp"],
        frame["tp"] + frame["fp"],
        out=np.zeros(len(frame), dtype=float),
        where=(frame["tp"] + frame["fp"]).to_numpy() != 0,
    )
    calculated_recall = np.divide(
        frame["tp"],
        frame["tp"] + frame["fn"],
        out=np.zeros(len(frame), dtype=float),
        where=(frame["tp"] + frame["fn"]).to_numpy() != 0,
    )
    calculated_f1 = np.divide(
        2 * frame["tp"],
        2 * frame["tp"] + frame["fp"] + frame["fn"],
        out=np.zeros(len(frame), dtype=float),
        where=(2 * frame["tp"] + frame["fp"] + frame["fn"]).to_numpy() != 0,
    )
    for column, calculated in (
        ("precision", calculated_precision),
        ("recall", calculated_recall),
        ("f1", calculated_f1),
    ):
        if not np.allclose(frame[column].to_numpy(), calculated, rtol=0, atol=5e-12):
            raise ValueError(f"Reported {column} is inconsistent with TP/FP/FN")

    frame["truth_events"] = frame["tp"] + frame["fn"]
    if "predicted_events" in frame.columns:
        predicted = pd.to_numeric(frame["predicted_events"], errors="raise").astype(int)
        if (frame["tp"] > predicted).any():
            raise ValueError("One-to-one TP exceeds the number of predicted events")
        frame["predicted_events"] = predicted

    duplicated = frame.duplicated(["family", "run", "method"], keep=False)
    if duplicated.any():
        rows = frame.loc[duplicated, ["family", "run", "method"]].to_dict("records")
        raise ValueError(f"Duplicate metric rows after filtering: {rows}")
    if not set(frame["method"]).issubset(set(METHOD_ORDER)):
        raise ValueError(f"Unexpected decision methods: {sorted(frame['method'].unique())}")

    for (_family, _run), group in frame.groupby(["family", "run"]):
        if set(group["method"]) != set(METHOD_ORDER):
            raise ValueError(f"Each run must contain one static and one SAOCP row: {_family}/{_run}")
        if group["truth_events"].nunique() != 1:
            raise ValueError(f"Static and SAOCP truth counts differ for {_family}/{_run}")

    if expected_runs:
        for family, runs in expected_runs.items():
            actual = set(frame.loc[frame["family"] == family, "run"])
            expected = set(runs)
            if actual != expected:
                raise ValueError(
                    f"{family} run set differs: expected={sorted(expected)}, actual={sorted(actual)}"
                )

    frame = frame.sort_values(["family", "run", "method"], kind="stable").reset_index(drop=True)
    return frame, manifest


def load_thresholds(path: Path, expected_runs: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"run", "threshold"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Threshold table lacks {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="raise")
    if not frame["threshold"].between(0.0, 1.0).all():
        raise ValueError("Static threshold leaves [0, 1]")
    expected = set(expected_runs)
    frame = frame[frame["run"].isin(expected)].copy()
    if set(frame["run"]) != expected or frame["run"].duplicated().any():
        raise ValueError("Threshold table does not contain exactly one row per requested run")
    return frame


def thresholds_from_metrics(
    metrics: pd.DataFrame, family: str, expected_runs: Iterable[str]
) -> pd.DataFrame:
    """Extract the frozen static threshold from the locked test-result rows."""
    if "static_threshold" not in metrics.columns:
        raise ValueError("Metric CSV lacks static_threshold; provide a threshold CSV")
    selected = metrics[
        (metrics["family"] == family) & (metrics["method"] == "static")
    ][["run", "static_threshold"]].rename(columns={"static_threshold": "threshold"})
    selected["threshold"] = pd.to_numeric(selected["threshold"], errors="raise")
    expected = set(expected_runs)
    if set(selected["run"]) != expected or selected["run"].duplicated().any():
        raise ValueError(f"Locked test rows do not contain one static threshold per {family} run")
    if not selected["threshold"].between(0.0, 1.0).all():
        raise ValueError("Static threshold leaves [0, 1]")
    return selected.reset_index(drop=True)


def validate_submission_lock(
    metrics: pd.DataFrame,
    validation_lock_path: Path,
    locked_test_audit_path: Path,
) -> tuple[dict, dict]:
    """Cross-check that test rows use the pre-test validation lock unchanged."""
    validation_lock = json.loads(validation_lock_path.read_text(encoding="utf-8"))
    test_audit = json.loads(locked_test_audit_path.read_text(encoding="utf-8"))

    if validation_lock.get("status") != "VALIDATION_SELECTION_COMPLETE_TEST_NOT_OPENED":
        raise ValueError("Validation-selection lock is not in the pre-test completed state")
    if validation_lock.get("test_files_opened") is not False:
        raise ValueError("Validation-selection lock says test files were opened")
    if test_audit.get("status") != "LOCKED_TEST_EVALUATION_COMPLETE_NO_TEST_SELECTION":
        raise ValueError("Locked-test audit status is not final")
    if test_audit.get("configuration_reselected_after_test") is not False:
        raise ValueError("Locked-test audit reports post-test configuration reselection")
    if test_audit.get("test_results_used_for_selection") is not False:
        raise ValueError("Locked-test audit reports test-based selection")
    if test_audit.get("all_one_regression_passed") is not True:
        raise ValueError("All-positive regression guard did not pass")
    if test_audit.get("validation_lock_sha256") != sha256(validation_lock_path):
        raise ValueError("Locked-test audit does not hash the supplied validation lock")

    for record in (validation_lock, test_audit):
        matching = _normalized_text(record.get("matching", ""))
        for term in ("maximum", "cardinality", "one-to-one", "positive", "overlap"):
            if term not in matching:
                raise ValueError("Lock/audit matching contract is incomplete")
        fp_rule = _normalized_text(
            record.get("false_positive_rule", record.get("unmatched_prediction_rule", ""))
        )
        for term in ("every", "unmatched", "constructed", "prediction", "fp"):
            if term not in fp_rule:
                raise ValueError("Lock/audit false-positive rule is incomplete")

    validation_thresholds = {
        str(run): float(value)
        for run, value in validation_lock["selected_static_thresholds"].items()
    }
    audit_thresholds = {
        str(run): float(value) for run, value in test_audit["static_thresholds"].items()
    }
    if validation_thresholds != audit_thresholds:
        raise ValueError("Static thresholds differ between validation lock and test audit")
    static_rows = metrics[metrics["method"] == "static"]
    if "static_threshold" not in static_rows.columns:
        raise ValueError("Locked test metrics do not expose static_threshold")
    observed_thresholds = {
        str(row.run): float(row.static_threshold)
        for row in static_rows.itertuples(index=False)
    }
    if observed_thresholds != validation_thresholds:
        raise ValueError("Locked test rows do not use the validation-selected static thresholds")

    selected_saocp = validation_lock["selected_saocp"]
    audit_saocp = test_audit["saocp"]
    expected_saocp = {
        "coverage": float(selected_saocp["coverage"]),
        "lifetime": int(selected_saocp["lifetime"]),
        "policy": str(selected_saocp["policy"]),
        "block_size": int(selected_saocp["block_size"]),
    }
    observed_audit_saocp = {
        "coverage": float(audit_saocp["coverage"]),
        "lifetime": int(audit_saocp["lifetime"]),
        "policy": str(audit_saocp["policy"]),
        "block_size": int(audit_saocp["block_size"]),
    }
    if expected_saocp != observed_audit_saocp:
        raise ValueError("SAOCP settings differ between validation lock and test audit")
    if selected_saocp.get("test_used_for_selection") is not False:
        raise ValueError("Validation lock reports test use in SAOCP selection")

    saocp_rows = metrics[metrics["method"] == "saocp"]
    metric_columns = {
        "coverage": "saocp_coverage",
        "lifetime": "saocp_lifetime",
        "policy": "saocp_policy",
        "block_size": "block_size",
    }
    for key, column in metric_columns.items():
        if column not in saocp_rows.columns:
            raise ValueError(f"Locked test metrics lack {column}")
        values = saocp_rows[column].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"Locked test metrics contain multiple {column} values")
        expected = expected_saocp[key]
        observed = values[0]
        if isinstance(expected, float):
            if not np.isclose(float(observed), expected, rtol=0, atol=1e-12):
                raise ValueError(f"Locked test {column} differs from validation selection")
        elif str(observed) != str(expected):
            if not (isinstance(expected, int) and int(observed) == expected):
                raise ValueError(f"Locked test {column} differs from validation selection")
    return validation_lock, test_audit
