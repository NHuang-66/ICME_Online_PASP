"""Prepare the row-aligned 34-channel ICME arrays used by the manuscript.

The script reads the source Parquet table and ICME catalog, appends the four
derived channels in the documented order (Beta, Pdyn, RmsBob, Pm), applies the
chronological split, and fits the standardization parameters on training data
only.  It never consults test labels while computing feature transformations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import constants
from sklearn.preprocessing import StandardScaler


START = pd.Timestamp("1997-10-01")
TRAIN_START = pd.Timestamp("1998-01-01")
TEST_START = pd.Timestamp("2010-01-01")
END = pd.Timestamp("2016-01-01")
DERIVED_FEATURES = ("Beta", "Pdyn", "RmsBob", "Pm")


def _naive_datetime_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the Parquet row index as timezone-naive timestamps."""

    if isinstance(frame.index, pd.DatetimeIndex):
        index = frame.index
    elif "__index_level_0__" in frame.columns:
        index = pd.DatetimeIndex(frame.pop("__index_level_0__"))
    else:
        raise ValueError("Parquet input has no datetime index column")
    if index.tz is not None:
        index = index.tz_localize(None)
    if not index.is_monotonic_increasing:
        raise ValueError("Input timestamps must be monotonic nondecreasing")
    return index


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the four derived channels exactly as used in the study."""

    required = {"B", "Bx_rms", "By_rms", "Bz_rms", "Np", "V", "Vth"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing source features: {', '.join(missing)}")

    result = frame.copy()
    result["Beta"] = (
        1e6
        * result["Vth"] ** 2
        * constants.m_p
        * result["Np"]
        * 1e6
        * constants.mu_0
        / (1e-18 * result["B"] ** 2)
    )
    result["Pdyn"] = 1e12 * constants.m_p * result["Np"] * result["V"] ** 2
    result["RmsBob"] = np.sqrt(
        result["Bx_rms"] ** 2 + result["By_rms"] ** 2 + result["Bz_rms"] ** 2
    ) / result["B"]
    # B is stored in nT.  The factor 1e-18 converts B^2 from nT^2 to T^2.
    result["Pm"] = 1e-18 * result["B"] ** 2 / (2.0 * constants.mu_0)
    return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def read_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path)
    required = {"begin", "end"}
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise KeyError(f"Catalog is missing columns: {', '.join(missing)}")
    catalog["begin"] = pd.to_datetime(catalog["begin"], format="mixed").dt.tz_localize(None)
    catalog["end"] = pd.to_datetime(catalog["end"], format="mixed").dt.tz_localize(None)
    catalog = catalog.sort_values("begin", kind="stable").reset_index(drop=True)
    if (catalog["end"] <= catalog["begin"]).any():
        raise ValueError("Every catalog end time must be after its begin time")
    return catalog


def labels_for_index(index: pd.DatetimeIndex, catalog: pd.DataFrame) -> np.ndarray:
    """Build labels in row order; duplicate timestamps remain duplicate rows."""

    labels = np.zeros(len(index), dtype=np.uint8)
    values = index.values
    for event in catalog.itertuples(index=False):
        left = int(np.searchsorted(values, np.datetime64(event.begin), side="right"))
        right = int(np.searchsorted(values, np.datetime64(event.end), side="right"))
        labels[left:right] = 1
    return labels.reshape(-1, 1)


def split_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "train": np.asarray((index >= TRAIN_START) & (index < TEST_START)),
        "val": np.asarray((index >= START) & (index < TRAIN_START)),
        "test": np.asarray((index >= TEST_START) & (index < END)),
    }


def prepare(parquet_path: Path, catalog_path: Path, output_dir: Path) -> dict:
    """Create standardized features, row-aligned labels, and split timestamps."""

    table = pq.read_table(parquet_path)
    frame = table.to_pandas()
    frame.index = _naive_datetime_index(frame)
    frame = frame[(frame.index > START) & (frame.index < END)]
    frame = add_derived_features(frame.fillna(0.0))
    if frame.shape[1] != 34:
        raise ValueError(
            f"Expected 30 observed plus 4 derived channels, found {frame.shape[1]}"
        )

    catalog = read_catalog(catalog_path)
    masks = split_masks(frame.index)
    scaler = StandardScaler()
    scaler.fit(frame.loc[masks["train"]].to_numpy(dtype=np.float64))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "feature_scaler_train_only.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_names=np.asarray(frame.columns, dtype=str),
    )

    report: dict[str, object] = {
        "source_parquet": parquet_path.name,
        "source_catalog": catalog_path.name,
        "feature_names": list(frame.columns),
        "feature_count": int(frame.shape[1]),
        "derived_feature_order": list(DERIVED_FEATURES),
        "pm_formula": "Pm[Pa] = 1e-18 * B[nT]^2 / (2 * mu0)",
        "mu0_N_per_A2": float(constants.mu_0),
        "standardization_fit_split": "train only (1998-01-01 <= t < 2010-01-01)",
        "splits": {},
    }
    for split, mask in masks.items():
        split_frame = frame.loc[mask]
        features = scaler.transform(split_frame.to_numpy(dtype=np.float64)).astype(np.float32)
        labels = labels_for_index(split_frame.index, catalog)
        np.save(output_dir / f"X_{split}_origin_1.npy", features)
        np.save(output_dir / f"Y_{split}_aligned.npy", labels)
        np.save(output_dir / f"time_{split}.npy", split_frame.index.values.astype("datetime64[ns]"))
        report["splits"][split] = {
            "rows": int(len(split_frame)),
            "positive_rows": int(labels.sum()),
            "duplicate_timestamps": int(split_frame.index.duplicated().sum()),
        }

    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare(args.parquet.resolve(), args.catalog.resolve(), args.output_dir.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
