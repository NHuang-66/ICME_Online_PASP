#!/usr/bin/env python3
"""Validate and render one-to-one event-metric manuscript figures.

The program is intentionally a presentation layer.  It never trains a model,
selects a threshold, changes an event interval, or recomputes event matching.
It accepts a locked result CSV plus its audit manifest, independently verifies
all reported P/R/F1 values from TP/FP/FN, and then draws the requested figures.

Use ``--validate-only`` while the final validation-locked/test results are not
yet frozen.  That mode performs schema/provenance checks and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from one_to_one_metrics import (
    load_one_to_one_metrics,
    load_thresholds,
    thresholds_from_metrics,
    validate_submission_lock,
)


MM_PER_INCH = 25.4
WIDTH_MM = 183.0
BACKBONE_HEIGHT_MM = 109.0
ABLATION_HEIGHT_MM = 115.0
RASTER_DPI = 600

BACKBONE_RUNS = ("lstm", "cnn_lstm", "unet", "runet")
BACKBONE_LABELS = {
    "lstm": "LSTM",
    "cnn_lstm": "CNN-LSTM",
    "unet": "U-net",
    "runet": "RU-net",
}
ABLATION_RUNS = (
    "cnn_lstm_f33_w64",
    "cnn_lstm_f34_w32",
    "cnn_lstm_f34_w64",
    "cnn_lstm_f34_w128",
)
FEATURE_RUNS = ("cnn_lstm_f33_w64", "cnn_lstm_f34_w64")
FEATURE_LABELS = {
    "cnn_lstm_f33_w64": "33 features\nwithout $P_m$",
    "cnn_lstm_f34_w64": "34 features\nwith $P_m$",
}
WINDOW_RUNS = ("cnn_lstm_f34_w32", "cnn_lstm_f34_w64", "cnn_lstm_f34_w128")
WINDOW_LABELS = {
    "cnn_lstm_f34_w32": "32 observations",
    "cnn_lstm_f34_w64": "64 observations",
    "cnn_lstm_f34_w128": "128 observations",
}
EXPECTED_RUNS = {
    "bce_backbones": BACKBONE_RUNS,
    "bce_ablation": ABLATION_RUNS,
}
METHOD_ORDER = ("static", "saocp")
METHOD_LABELS = {"static": "Static", "saocp": "Online"}
METHOD_COLORS = {"static": "#777E8C", "saocp": "#245B9E"}
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "one_to_one_blue",
    ["#F7F9FC", "#DFE8F2", "#B5CAE0", "#739BC4", "#315F8C"],
)
SCORE_NORM = Normalize(vmin=0.45, vmax=1.00)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.4,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_matrix(
    frame: pd.DataFrame,
    family: str,
    runs: tuple[str, ...],
) -> tuple[np.ndarray, list[str], list[str], list[float]]:
    rows: list[list[float]] = []
    row_runs: list[str] = []
    row_methods: list[str] = []
    deltas: list[float] = []
    family_frame = frame[frame["family"] == family]
    for run in runs:
        run_frame = family_frame[family_frame["run"] == run].set_index("method")
        for method in METHOD_ORDER:
            row = run_frame.loc[method]
            rows.append([float(row["precision"]), float(row["recall"]), float(row["f1"])])
            row_runs.append(run)
            row_methods.append(method)
        deltas.append(float(run_frame.loc["saocp", "f1"] - run_frame.loc["static", "f1"]))
    return np.asarray(rows), row_runs, row_methods, deltas


def draw_matrix(
    ax: mpl.axes.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    row_methods: list[str],
    deltas: list[float],
    title: str,
    panel: str | None,
) -> mpl.image.AxesImage:
    im = ax.imshow(
        matrix,
        cmap=SCORE_CMAP,
        norm=SCORE_NORM,
        aspect="auto",
        interpolation="nearest",
    )
    n_rows = matrix.shape[0]
    ax.set_xticks(range(3), ["Precision", "Recall", "F1"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=False, labeltop=True, pad=3)
    ax.set_yticks(range(n_rows), row_labels)
    ax.tick_params(axis="y", left=False, pad=3)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    for pair in range(1, n_rows // 2):
        ax.axhline(pair * 2 - 0.5, color="#626B78", linewidth=0.75)

    for row_index in range(n_rows):
        ax.add_patch(
            Rectangle(
                (-0.47, row_index - 0.34),
                0.075,
                0.68,
                facecolor=METHOD_COLORS[row_methods[row_index]],
                edgecolor="white",
                linewidth=0.3,
                clip_on=True,
                zorder=4,
            )
        )
        for column_index in range(3):
            value = float(matrix[row_index, column_index])
            rgba = SCORE_CMAP(SCORE_NORM(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.15,
                fontweight="bold" if column_index == 2 else "normal",
                color="white" if luminance < 0.54 else "#1D2731",
            )
    for pair_index, delta in enumerate(deltas):
        ax.text(
            2.60,
            pair_index * 2 + 1,
            rf"$\Delta$F1 {delta:+.3f}",
            ha="left",
            va="center",
            fontsize=5.9,
            fontweight="bold",
            color=METHOD_COLORS["saocp"],
            clip_on=False,
        )
    ax.set_xlim(-0.5, 3.55)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_title(title, loc="left", fontsize=7.7, fontweight="bold", pad=9)
    if panel:
        ax.text(
            -0.18,
            1.06,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=METHOD_COLORS["static"], edgecolor="none", label="Static decision"),
        Patch(facecolor=METHOD_COLORS["saocp"], edgecolor="none", label="Online decision"),
    ]


def build_backbone_figure(metrics: pd.DataFrame) -> mpl.figure.Figure:
    matrix, runs, methods, deltas = score_matrix(metrics, "bce_backbones", BACKBONE_RUNS)
    labels = [
        f"{BACKBONE_LABELS[run]} - {METHOD_LABELS[method]}"
        for run, method in zip(runs, methods)
    ]
    fig, ax = plt.subplots(
        figsize=(WIDTH_MM / MM_PER_INCH, BACKBONE_HEIGHT_MM / MM_PER_INCH)
    )
    fig.subplots_adjust(left=0.235, right=0.835, top=0.80, bottom=0.20)
    im = draw_matrix(
        ax,
        matrix,
        labels,
        methods,
        deltas,
        "Static versus Online decisions across frozen backbones",
        None,
    )
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.55, 0.965),
        ncol=2,
        handlelength=1.35,
        columnspacing=1.7,
    )
    cax = fig.add_axes([0.405, 0.09, 0.22, 0.018])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.5, 0.75, 1.0])
    cbar.set_label("Event score", fontsize=6.0, labelpad=1.2)
    cbar.ax.tick_params(labelsize=5.6, length=2, pad=1)
    cbar.outline.set_linewidth(0.45)
    return fig


def build_ablation_figure(metrics: pd.DataFrame) -> mpl.figure.Figure:
    feature_matrix, feature_runs, feature_methods, feature_deltas = score_matrix(
        metrics, "bce_ablation", FEATURE_RUNS
    )
    window_matrix, window_runs, window_methods, window_deltas = score_matrix(
        metrics, "bce_ablation", WINDOW_RUNS
    )
    feature_labels = [
        f"{FEATURE_LABELS[run]} - {METHOD_LABELS[method]}"
        for run, method in zip(feature_runs, feature_methods)
    ]
    window_labels = [
        f"{WINDOW_LABELS[run]} - {METHOD_LABELS[method]}"
        for run, method in zip(window_runs, window_methods)
    ]
    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, ABLATION_HEIGHT_MM / MM_PER_INCH))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.23])
    ax_feature = fig.add_subplot(grid[0, 0])
    ax_window = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.19, right=0.965, top=0.77, bottom=0.23, wspace=0.43)
    im = draw_matrix(
        ax_feature,
        feature_matrix,
        feature_labels,
        feature_methods,
        feature_deltas,
        "Feature ablation (window = 64)",
        "a",
    )
    draw_matrix(
        ax_window,
        window_matrix,
        window_labels,
        window_methods,
        window_deltas,
        "Sequence-length ablation (34 features)",
        "b",
    )
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.965),
        ncol=2,
        handlelength=1.35,
        columnspacing=1.7,
    )
    cax = fig.add_axes([0.405, 0.105, 0.22, 0.018])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.5, 0.75, 1.0])
    cbar.set_label("Event score", fontsize=6.0, labelpad=1.2)
    cbar.ax.tick_params(labelsize=5.6, length=2, pad=1)
    cbar.outline.set_linewidth(0.45)
    return fig


def export_figure(
    figure: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
    height_mm: float,
) -> tuple[dict[str, Path], dict]:
    paths = {
        suffix: output_dir / f"{stem}.{suffix}"
        for suffix in ("pdf", "svg", "png", "tiff")
    }
    figure.savefig(
        paths["pdf"],
        format="pdf",
        metadata={"Creator": "Python/matplotlib; one-to-one locked metrics only"},
    )
    figure.savefig(paths["svg"], format="svg")
    figure.savefig(paths["png"], format="png", dpi=RASTER_DPI)
    # Save the TIFF from the validated PNG pixels.  This avoids backend-specific
    # libtiff failures while preserving the same 600-dpi raster dimensions.
    with Image.open(paths["png"]) as png_image:
        tifffile.imwrite(
            paths["tiff"],
            np.asarray(png_image.convert("RGB")),
            resolution=(RASTER_DPI, RASTER_DPI),
            resolutionunit="INCH",
            compression="zlib",
        )
    plt.close(figure)

    expected = (
        int(WIDTH_MM / MM_PER_INCH * RASTER_DPI),
        int(height_mm / MM_PER_INCH * RASTER_DPI),
    )
    png_size = Image.open(paths["png"]).size
    tiff_size = Image.open(paths["tiff"]).size
    svg_text = paths["svg"].read_text(encoding="utf-8")
    qa = {
        "size_mm": [WIDTH_MM, height_mm],
        "raster_dpi": RASTER_DPI,
        "expected_pixels": list(expected),
        "png_pixels": list(png_size),
        "tiff_pixels": list(tiff_size),
        "raster_dimensions_match": png_size == expected and tiff_size == expected,
        "svg_contains_editable_text": "<text" in svg_text,
    }
    if not qa["raster_dimensions_match"] or not qa["svg_contains_editable_text"]:
        raise RuntimeError(f"Export QA failed for {stem}: {qa}")
    return paths, qa


def compact_table(
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    family: str,
    runs: tuple[str, ...],
) -> pd.DataFrame:
    threshold_map = thresholds.set_index("run")["threshold"].to_dict()
    rows: list[dict] = []
    family_metrics = metrics[metrics["family"] == family]
    for run in runs:
        selected = family_metrics[family_metrics["run"] == run].set_index("method")
        row: dict[str, object] = {
            "family": family,
            "run": run,
            "static_threshold_validation_locked": float(threshold_map[run]),
            "truth_events": int(selected.loc["static", "truth_events"]),
        }
        for method in METHOD_ORDER:
            for field in ("tp", "fp", "fn", "precision", "recall", "f1"):
                row[f"{method}_{field}"] = selected.loc[method, field]
        row["delta_f1_saocp_minus_static"] = (
            float(selected.loc["saocp", "f1"]) - float(selected.loc["static", "f1"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def copy_source_data(
    metrics_path: Path,
    manifest_path: Path,
    validation_lock_path: Path,
    metrics: pd.DataFrame,
    backbone_thresholds: pd.DataFrame,
    ablation_thresholds: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    code_dir = output_dir / "python_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = source_dir / "one_to_one_metrics_long.csv"
    metrics.to_csv(metrics_out, index=False, float_format="%.12g", encoding="utf-8-sig")
    backbone_compact = source_dir / "backbone_one_to_one_metrics_compact.csv"
    compact_table(metrics, backbone_thresholds, "bce_backbones", BACKBONE_RUNS).to_csv(
        backbone_compact, index=False, float_format="%.12g", encoding="utf-8-sig"
    )
    ablation_compact = source_dir / "ablation_one_to_one_metrics_compact.csv"
    compact_table(metrics, ablation_thresholds, "bce_ablation", ABLATION_RUNS).to_csv(
        ablation_compact, index=False, float_format="%.12g", encoding="utf-8-sig"
    )
    backbone_threshold_out = source_dir / "backbone_validation_locked_thresholds.csv"
    ablation_threshold_out = source_dir / "ablation_validation_locked_thresholds.csv"
    backbone_thresholds.to_csv(backbone_threshold_out, index=False, encoding="utf-8-sig")
    ablation_thresholds.to_csv(ablation_threshold_out, index=False, encoding="utf-8-sig")
    manifest_out = source_dir / "one_to_one_audit_manifest.json"
    shutil.copy2(manifest_path, manifest_out)
    validation_lock_out = source_dir / "VALIDATION_SELECTION_COMPLETE.json"
    shutil.copy2(validation_lock_path, validation_lock_out)
    copied_audit_manifests: list[Path] = []
    for audit_name in ("test_input_hashes.csv", "test_output_hashes.csv"):
        audit_path = manifest_path.parent / audit_name
        if audit_path.exists():
            audit_copy = source_dir / audit_name
            shutil.copy2(audit_path, audit_copy)
            copied_audit_manifests.append(audit_copy)
    provenance = {
        "input_metric_csv": "source_data/one_to_one_metrics_long.csv",
        "input_metric_csv_sha256": sha256(metrics_out),
        "input_manifest": "source_data/one_to_one_audit_manifest.json",
        "input_manifest_sha256": sha256(manifest_path),
        "validation_lock": "source_data/VALIDATION_SELECTION_COMPLETE.json",
        "validation_lock_sha256": sha256(validation_lock_path),
        "presentation_only": True,
        "value_transformation": "none; P/R/F1 checked from integer TP/FP/FN",
    }
    provenance_out = source_dir / "source_provenance.json"
    provenance_out.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    generator_copy = code_dir / Path(__file__).name
    loader_copy = code_dir / "one_to_one_metrics.py"
    shutil.copy2(Path(__file__), generator_copy)
    shutil.copy2(Path(__file__).with_name("one_to_one_metrics.py"), loader_copy)
    contract_copy = code_dir / "FIGURE_CONTRACT.md"
    shutil.copy2(Path(__file__).with_name("FIGURE_CONTRACT.md"), contract_copy)
    return [
        metrics_out,
        backbone_compact,
        ablation_compact,
        backbone_threshold_out,
        ablation_threshold_out,
        manifest_out,
        validation_lock_out,
        *copied_audit_manifests,
        provenance_out,
        generator_copy,
        loader_copy,
        contract_copy,
    ]


def write_caption_and_qa(
    output_dir: Path,
    fp_convention: str,
    split: str | None,
    outputs: dict[str, Path],
    source_paths: list[Path],
    qa: dict,
) -> None:
    caption = (
        "Suggested caption. Event-level precision, recall, and F1 for Static and Online "
        "decisions. A maximum-cardinality one-to-one assignment is formed over all "
        "positive-duration overlaps: each predicted interval and each catalog interval "
        "can be used at most once. Consequently, one overlong prediction cannot be "
        "counted as multiple true-positive events. Static thresholds and all Online "
        "settings are fixed before test evaluation. Values are shown directly in each cell."
    )
    (output_dir / "SUGGESTED_CAPTIONS_ONE_TO_ONE.md").write_text(
        caption + "\n", encoding="utf-8"
    )
    hash_rows = []
    for role, paths in (("output", outputs.values()), ("source", source_paths)):
        for path in paths:
            hash_rows.append(
                {
                    "role": role,
                    "file": str(path.relative_to(output_dir)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(hash_rows).to_csv(
        output_dir / "file_hashes.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "matching_contract": (
            "maximum-cardinality one-to-one matching over positive-duration overlap edges"
        ),
        "interpretation": (
            "each prediction and each catalog event participates in at most one TP"
        ),
        "fp_convention": fp_convention,
        "split": split,
        "render_backend": "Python/matplotlib only",
        "no_training_or_threshold_selection": True,
        "exports": qa,
        "manual_visual_qa_required": True,
    }
    (output_dir / "qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--validation-lock-json", type=Path, required=True)
    parser.add_argument("--fp-convention", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--backbone-thresholds-csv", type=Path)
    parser.add_argument("--ablation-thresholds-csv", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "rendered"
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, manifest = load_one_to_one_metrics(
        args.metrics_csv,
        args.manifest_json,
        args.fp_convention,
        args.split,
        EXPECTED_RUNS,
    )
    validation_lock, test_audit = validate_submission_lock(
        metrics,
        args.validation_lock_json,
        args.manifest_json,
    )
    backbone_thresholds = None
    ablation_thresholds = None
    if args.backbone_thresholds_csv:
        backbone_thresholds = load_thresholds(args.backbone_thresholds_csv, BACKBONE_RUNS)
    else:
        backbone_thresholds = thresholds_from_metrics(
            metrics, "bce_backbones", BACKBONE_RUNS
        )
    if args.ablation_thresholds_csv:
        ablation_thresholds = load_thresholds(args.ablation_thresholds_csv, ABLATION_RUNS)
    else:
        ablation_thresholds = thresholds_from_metrics(
            metrics, "bce_ablation", ABLATION_RUNS
        )

    summary = {
        "status": "validated_only" if args.validate_only else "rendered",
        "matching": manifest["matching"],
        "fp_convention": args.fp_convention,
        "split": args.split,
        "metric_rows": int(len(metrics)),
        "families": sorted(metrics["family"].unique()),
        "runs": sorted(metrics["run"].unique()),
        "f1_range": [float(metrics["f1"].min()), float(metrics["f1"].max())],
        "validation_lock_status": validation_lock["status"],
        "test_audit_status": test_audit["status"],
        "validation_lock_sha256": sha256(args.validation_lock_json),
    }
    if args.validate_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    configure_matplotlib()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_fig = build_backbone_figure(metrics)
    backbone_outputs, backbone_qa = export_figure(
        backbone_fig,
        output_dir,
        "Figure_backbones_static_vs_online_one_to_one",
        BACKBONE_HEIGHT_MM,
    )
    ablation_fig = build_ablation_figure(metrics)
    ablation_outputs, ablation_qa = export_figure(
        ablation_fig,
        output_dir,
        "Figure_ablation_Pm_window_one_to_one",
        ABLATION_HEIGHT_MM,
    )
    source_paths = copy_source_data(
        args.metrics_csv,
        args.manifest_json,
        args.validation_lock_json,
        metrics,
        backbone_thresholds,
        ablation_thresholds,
        output_dir,
    )
    all_outputs = {**backbone_outputs, **{f"ablation_{k}": v for k, v in ablation_outputs.items()}}
    write_caption_and_qa(
        output_dir,
        args.fp_convention,
        args.split,
        all_outputs,
        source_paths,
        {"backbone": backbone_qa, "ablation": ablation_qa},
    )
    summary["outputs"] = {key: str(path) for key, path in all_outputs.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
