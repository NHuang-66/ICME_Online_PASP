"""Verify the arithmetic and schema of the compact locked-result tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


REQUIRED = {
    "run",
    "method",
    "truth_events",
    "predicted_events",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
}


def verify(path: Path, tolerance: float = 1e-12) -> int:
    frame = pd.read_csv(path)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    for row in frame.itertuples(index=False):
        tp, fp, fn = int(row.tp), int(row.fp), int(row.fn)
        if tp + fn != int(row.truth_events):
            raise AssertionError(f"{path}:{row.run}/{row.method}: TP + FN mismatch")
        if tp + fp != int(row.predicted_events):
            raise AssertionError(
                f"{path}:{row.run}/{row.method}: TP + FP mismatch; "
                "the primary protocol counts every unmatched prediction"
            )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        for label, observed, expected in (
            ("precision", float(row.precision), precision),
            ("recall", float(row.recall), recall),
            ("f1", float(row.f1), f1),
        ):
            if not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance):
                raise AssertionError(
                    f"{path}:{row.run}/{row.method}: {label} {observed} != {expected}"
                )
    return len(frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    total = 0
    for table in parse_args().tables:
        count = verify(table)
        total += count
        print(f"verified {count} rows: {table}")
    print(f"verified {total} result rows in total")


if __name__ == "__main__":
    main()
