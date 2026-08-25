"""Render the self-contained four-backbone event-28 figure.

The plotting path reads only the CSV files in ``source_data``.  It does not
load a neural network, tune a threshold, select an event, or alter values.
All figure drawing, exports, and preview generation use Python/matplotlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

from one_to_one_metrics import load_one_to_one_metrics, validate_submission_lock


MM_PER_INCH = 25.4
WIDTH_MM = 183.0
HEIGHT_MM = 145.0
RASTER_DPI = 600

MODEL_SPECS = [
    ("LSTM", "LSTM", "lstm"),
    ("CNN–LSTM", "CNN_LSTM", "cnn_lstm"),
    ("U-net", "UNet", "unet"),
    ("RU-net", "RUNet", "runet"),
]

EXPECTED_RUNS = {"bce_backbones": ("lstm", "cnn_lstm", "unet", "runet")}

COLORS = {
    "probability": "#2C73B9",
    "saocp_threshold": "#C66FA2",
    "static_threshold": "#626B7F",
    "catalog_fill": "#EAB735",
    "catalog_shade": "#F6E7AD",
    "static_event": "#858C99",
    "saocp_event": "#2C73B9",
    "axis": "#1C2430",
    "muted": "#667085",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "axes.titlesize": 8.1,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 6.1,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.6,
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


def read_selected_event(source_dir: Path) -> pd.Series:
    path = source_dir / "selected_event.csv"
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError("selected_event.csv must contain exactly one row")
    row = frame.iloc[0].copy()
    for column in ("begin", "end", "display_begin_utc", "display_end_utc"):
        row[column] = pd.Timestamp(row[column])
    if "case_tag" not in row.index or not str(row["case_tag"]).startswith("event"):
        raise ValueError("selected_event.csv lacks a valid case_tag")
    if row["end"] <= row["begin"]:
        raise ValueError("Selected catalog event has non-positive duration")
    return row


def read_intervals(source_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_dir / "display_event_intervals.csv")
    required = {"run", "method", "begin", "end", "display_begin", "display_end"}
    if not required.issubset(frame.columns):
        raise ValueError(f"display_event_intervals.csv lacks {sorted(required - set(frame))}")
    for column in ("begin", "end", "display_begin", "display_end"):
        frame[column] = pd.to_datetime(frame[column], format="mixed", errors="raise")
    if (frame["display_end"] < frame["display_begin"]).any():
        raise ValueError("An event interval has negative display duration")
    return frame


def read_model_source(source_dir: Path, file_stem: str) -> pd.DataFrame:
    path = source_dir / f"{file_stem}_window.csv"
    frame = pd.read_csv(path)
    required = {
        "time_utc",
        "backbone_probability",
        "saocp_threshold",
        "static_threshold",
        "catalog_active",
        "static_point_label",
        "saocp_point_label",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{path.name} lacks {sorted(required - set(frame))}")
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], format="mixed", errors="raise")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    if not frame["time_utc"].is_monotonic_increasing:
        raise ValueError(f"{path.name}: timestamps are not monotonic")
    if frame[list(required - {"time_utc"})].isna().any().any():
        raise ValueError(f"{path.name}: missing values are not allowed")
    for column in ("backbone_probability", "saocp_threshold", "static_threshold"):
        if not frame[column].between(0.0, 1.0).all():
            raise ValueError(f"{path.name}: {column} leaves [0, 1]")
    if frame["static_threshold"].nunique() != 1:
        raise ValueError(f"{path.name}: static threshold is not constant")
    for column in ("catalog_active", "static_point_label", "saocp_point_label"):
        if not set(frame[column].astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"{path.name}: {column} is not binary")
    return frame


def interval_spans(intervals: pd.DataFrame, model: str, method: str) -> list[tuple[float, float]]:
    selected = intervals[(intervals["run"] == model) & (intervals["method"] == method)]
    spans: list[tuple[float, float]] = []
    for row in selected.itertuples(index=False):
        left = mdates.date2num(pd.Timestamp(row.display_begin).to_pydatetime())
        right = mdates.date2num(pd.Timestamp(row.display_end).to_pydatetime())
        spans.append((left, max(right - left, 1.0 / (24.0 * 60.0))))
    return spans


def draw_panel(
    ax_signal: mpl.axes.Axes,
    ax_events: mpl.axes.Axes,
    display_name: str,
    model_key: str,
    letter: str,
    frame: pd.DataFrame,
    intervals: pd.DataFrame,
    selected: pd.Series,
    metric_rows: pd.DataFrame,
    show_signal_ylabel: bool,
) -> None:
    display_begin = selected["display_begin_utc"]
    display_end = selected["display_end_utc"]
    event_begin = selected["begin"]
    event_end = selected["end"]
    static_threshold = float(frame["static_threshold"].iloc[0])

    ax_signal.axvspan(
        event_begin,
        event_end,
        facecolor=COLORS["catalog_shade"],
        alpha=0.42,
        edgecolor="none",
        zorder=0,
    )
    ax_signal.plot(
        frame["time_utc"],
        frame["backbone_probability"],
        color=COLORS["probability"],
        linewidth=0.82,
        zorder=3,
    )
    ax_signal.step(
        frame["time_utc"],
        frame["saocp_threshold"],
        where="post",
        color=COLORS["saocp_threshold"],
        linestyle=(0, (2.2, 1.4)),
        linewidth=1.10,
        zorder=4,
    )
    ax_signal.axhline(
        static_threshold,
        color=COLORS["static_threshold"],
        linestyle=(0, (4.0, 2.6)),
        linewidth=0.90,
        zorder=2,
    )
    ax_signal.set_xlim(display_begin, display_end)
    ax_signal.set_ylim(0.0, 1.02)
    ax_signal.set_yticks([0.0, 0.5, 1.0])
    ax_signal.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_signal.margins(x=0)
    if show_signal_ylabel:
        ax_signal.set_ylabel("Probability / threshold", labelpad=4)
    else:
        ax_signal.set_ylabel("")
    ax_signal.text(
        -0.105,
        1.045,
        letter,
        transform=ax_signal.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=COLORS["axis"],
        clip_on=False,
    )
    ax_signal.text(
        -0.015,
        1.045,
        display_name,
        transform=ax_signal.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["axis"],
        clip_on=False,
    )
    ax_signal.text(
        0.995,
        0.965,
        f"Static threshold = {static_threshold:.2f}",
        transform=ax_signal.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        zorder=5,
    )

    test_metrics = metric_rows.set_index("method")
    if set(test_metrics.index) != {"static", "saocp"}:
        raise ValueError(f"Expected Static and SAOCP metric rows for {model_key}")
    metric_text = (
        "Test event P / R / F1\n"
        f"Static  {test_metrics.loc['static', 'precision']:.3f} / "
        f"{test_metrics.loc['static', 'recall']:.3f} / "
        f"{test_metrics.loc['static', 'f1']:.3f}\n"
        f"Online  {test_metrics.loc['saocp', 'precision']:.3f} / "
        f"{test_metrics.loc['saocp', 'recall']:.3f} / "
        f"{test_metrics.loc['saocp', 'f1']:.3f}"
    )
    ax_signal.text(
        0.995,
        0.055,
        metric_text,
        transform=ax_signal.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.1,
        linespacing=1.18,
        color=COLORS["axis"],
        bbox={
            "facecolor": "white",
            "edgecolor": "#D5D9E0",
            "linewidth": 0.45,
            "alpha": 0.88,
            "pad": 1.35,
        },
        zorder=6,
    )

    bar_y = {"Catalog": 2.15, "Static": 1.15, "Online": 0.15}
    bar_height = 0.52
    catalog_left = mdates.date2num(event_begin.to_pydatetime())
    catalog_right = mdates.date2num(event_end.to_pydatetime())
    ax_events.broken_barh(
        [(catalog_left, catalog_right - catalog_left)],
        (bar_y["Catalog"], bar_height),
        facecolors=COLORS["catalog_fill"],
        edgecolors="none",
        zorder=3,
    )
    for method, label, color in (
        ("static", "Static", COLORS["static_event"]),
        ("saocp", "Online", COLORS["saocp_event"]),
    ):
        spans = interval_spans(intervals, model_key, method)
        if spans:
            ax_events.broken_barh(
                spans,
                (bar_y[label], bar_height),
                facecolors=color,
                edgecolors="none",
                zorder=3,
            )
    ax_events.set_xlim(display_begin, display_end)
    ax_events.set_ylim(-0.02, 2.82)
    ax_events.set_yticks([2.41, 1.41, 0.41])
    ax_events.set_yticklabels(["Catalog", "Static", "Online"])
    ax_events.tick_params(axis="y", length=0, pad=3)
    ax_events.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax_events.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax_events.tick_params(axis="x", pad=2)
    ax_events.set_xlabel("Time (UTC)", labelpad=2)
    ax_events.margins(x=0)


def build_figure(
    source_dir: Path,
    selected: pd.Series,
    intervals: pd.DataFrame,
    model_frames: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
) -> mpl.figure.Figure:
    figure = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH))
    outer = figure.add_gridspec(
        2,
        2,
        left=0.092,
        right=0.988,
        bottom=0.085,
        top=0.905,
        wspace=0.205,
        hspace=0.345,
    )
    letters = ("a", "b", "c", "d")
    for index, ((display_name, _file_stem, model_key), letter) in enumerate(
        zip(MODEL_SPECS, letters)
    ):
        inner = outer[index // 2, index % 2].subgridspec(
            2, 1, height_ratios=[3.25, 1.0], hspace=0.045
        )
        ax_signal = figure.add_subplot(inner[0])
        ax_events = figure.add_subplot(inner[1], sharex=ax_signal)
        draw_panel(
            ax_signal,
            ax_events,
            display_name,
            model_key,
            letter,
            model_frames[model_key],
            intervals,
            selected,
            metrics[
                (metrics["family"] == "bce_backbones")
                & (metrics["run"] == model_key)
            ],
            show_signal_ylabel=index % 2 == 0,
        )

    handles = [
        Line2D(
            [0], [0], color=COLORS["probability"], lw=1.15, label="Backbone probability"
        ),
        Line2D(
            [0], [0], color=COLORS["saocp_threshold"], lw=1.35,
            ls=(0, (2.2, 1.4)), label="Online threshold"
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["static_threshold"],
            lw=1.05,
            ls=(0, (4.0, 2.6)),
            label="Validation-selected Static threshold",
        ),
        Patch(
            facecolor=COLORS["catalog_shade"],
            edgecolor="none",
            alpha=0.65,
            label="Catalog interval",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=4,
        handlelength=2.2,
        columnspacing=1.35,
        handletextpad=0.45,
    )
    return figure


def export_bundle(
    figure: mpl.figure.Figure, output_dir: Path, output_stem: str
) -> dict[str, Path]:
    stem = output_dir / output_stem
    paths = {
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    common_metadata = {
        "Creator": "Python/matplotlib; plot_event28_one_to_one.py; locked data only"
    }
    figure.savefig(paths["pdf"], format="pdf", metadata=common_metadata)
    figure.savefig(paths["svg"], format="svg")
    figure.savefig(paths["png"], format="png", dpi=RASTER_DPI)
    figure.savefig(
        paths["tiff"],
        format="tiff",
        dpi=RASTER_DPI,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    return paths


def write_qa(
    output_dir: Path,
    source_dir: Path,
    outputs: dict[str, Path],
    model_frames: dict[str, pd.DataFrame],
    selected: pd.Series,
) -> None:
    png = Image.open(outputs["png"])
    tiff = Image.open(outputs["tiff"])
    # Matplotlib's raster canvas truncates fractional output pixels.
    expected_pixels = (
        int(WIDTH_MM / MM_PER_INCH * RASTER_DPI),
        int(HEIGHT_MM / MM_PER_INCH * RASTER_DPI),
    )
    svg_text = outputs["svg"].read_text(encoding="utf-8")
    source_paths = (
        sorted(source_dir.glob("*.csv"))
        + sorted(source_dir.glob("*.json"))
        + sorted((output_dir / "python_code").glob("*.py"))
        + [output_dir / "selection_audit.csv"]
    )
    output_hashes = pd.DataFrame(
        [
            {
                "file": path.name,
                "sha256": sha256(path),
                "n_bytes": path.stat().st_size,
            }
            for path in outputs.values()
        ]
    )
    output_hashes.to_csv(output_dir / "output_hashes.csv", index=False, encoding="utf-8-sig")
    source_hashes = pd.DataFrame(
        [
            {
                "file": path.name,
                "sha256": sha256(path),
                "n_bytes": path.stat().st_size,
            }
            for path in source_paths
        ]
    )
    source_hashes.to_csv(output_dir / "source_data_hashes.csv", index=False, encoding="utf-8-sig")

    time_reference = next(iter(model_frames.values()))["time_utc"].to_numpy()
    qa = {
        "figure_contract": {
            "core_conclusion": (
                "For a transparently selected isolated catalog event, Online calibration changes only the "
                "decision threshold; accompanying test P/R/F1 use one-to-one event assignment."
            ),
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib only",
            "final_size_mm": [WIDTH_MM, HEIGHT_MM],
            "raster_dpi": RASTER_DPI,
        },
        "locked_case": {
            "catalog_event_id": int(selected["catalog_event_id"]),
            "catalog_begin_utc": selected["begin"].isoformat(),
            "catalog_end_utc": selected["end"].isoformat(),
            "display_begin_utc": selected["display_begin_utc"].isoformat(),
            "display_end_utc": selected["display_end_utc"].isoformat(),
        },
        "data_checks": {
            "models": list(model_frames),
            "rows_per_model": {key: int(len(frame)) for key, frame in model_frames.items()},
            "identical_time_index": all(
                np.array_equal(time_reference, frame["time_utc"].to_numpy())
                for frame in model_frames.values()
            ),
            "all_probabilities_and_thresholds_in_0_1": True,
            "no_value_transformation_or_smoothing": True,
        },
        "evaluation_contract": {
            "matching": (
                "maximum-cardinality one-to-one over positive-duration overlap edges"
            ),
            "reuse_prevented": (
                "each prediction and catalog event can contribute to at most one TP"
            ),
            "metric_annotation_scope": "full locked test set, not the displayed case alone",
        },
        "display_selection_disclosure": (
            "post hoc display selection using the prespecified high-contrast rule after "
            "test results were locked; not used for parameter selection or global metrics"
        ),
        "export_checks": {
            "png_pixels": list(png.size),
            "tiff_pixels": list(tiff.size),
            "expected_600dpi_pixels": list(expected_pixels),
            "raster_dimensions_match": png.size == expected_pixels and tiff.size == expected_pixels,
            "svg_contains_editable_text": "<text" in svg_text,
            "pdf_true_type_text_requested": True,
        },
        "manual_visual_qa_required": True,
    }
    (output_dir / "qa_report.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        required=True,
        help="Existing audited event28 package containing source_data/.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="Locked one-to-one event-metric CSV (wide or long).",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        required=True,
        help="Audit manifest declaring one-to-one positive-overlap matching.",
    )
    parser.add_argument("--validation-lock-json", type=Path, required=True)
    parser.add_argument("--fp-convention", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "rendered_event28",
        help="New self-contained output directory; the input package is never overwritten.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional output basename without extension; default follows selected case_tag.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate metrics and event28 source CSVs without writing or rendering anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    source_dir = package_dir / "source_data"
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
    selected = read_selected_event(source_dir)
    case_tag = str(selected["case_tag"])
    intervals = read_intervals(source_dir)
    model_frames = {
        model_key: read_model_source(source_dir, f"{file_stem}_{case_tag}")
        for _display, file_stem, model_key in MODEL_SPECS
    }
    reference_time = next(iter(model_frames.values()))["time_utc"].to_numpy()
    if not all(
        np.array_equal(reference_time, frame["time_utc"].to_numpy())
        for frame in model_frames.values()
    ):
        raise ValueError("The four window CSVs do not share the same timestamp index")

    validation_summary = {
        "status": "validated_only" if args.validate_only else "rendered",
        "matching": manifest["matching"],
        "fp_convention": args.fp_convention,
        "split": args.split,
        "catalog_event_id": int(selected["catalog_event_id"]),
        "rows_per_model": {key: int(len(frame)) for key, frame in model_frames.items()},
        "metric_rows": int(len(metrics)),
        "validation_lock_status": validation_lock["status"],
        "test_audit_status": test_audit["status"],
        "validation_lock_sha256": sha256(args.validation_lock_json),
    }
    if args.validate_only:
        print(json.dumps(validation_summary, ensure_ascii=False, indent=2))
        return

    configure_matplotlib()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_source_dir = output_dir / "source_data"
    shutil.copytree(source_dir, copied_source_dir)
    selection_audit = package_dir / "selection_audit.csv"
    if not selection_audit.exists():
        raise FileNotFoundError(f"Missing required selection audit: {selection_audit}")
    shutil.copy2(selection_audit, output_dir / "selection_audit.csv")
    metrics.to_csv(
        copied_source_dir / "one_to_one_test_metrics_long.csv",
        index=False,
        float_format="%.12g",
        encoding="utf-8-sig",
    )
    shutil.copy2(args.manifest_json, copied_source_dir / "one_to_one_audit_manifest.json")
    shutil.copy2(
        args.validation_lock_json,
        copied_source_dir / "VALIDATION_SELECTION_COMPLETE.json",
    )
    for audit_name in ("test_input_hashes.csv", "test_output_hashes.csv"):
        audit_path = args.manifest_json.parent / audit_name
        if audit_path.exists():
            shutil.copy2(audit_path, copied_source_dir / audit_name)
    code_dir = output_dir / "python_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    shutil.copy2(
        Path(__file__).with_name("one_to_one_metrics.py"),
        code_dir / "one_to_one_metrics.py",
    )
    for supporting_name in (
        "prepare_locked_selected_event_source.py",
        "FIGURE_CONTRACT.md",
    ):
        supporting_path = Path(__file__).with_name(supporting_name)
        if supporting_path.exists():
            shutil.copy2(supporting_path, code_dir / supporting_name)

    figure = build_figure(copied_source_dir, selected, intervals, model_frames, metrics)
    output_stem = args.output_stem or f"Figure_four_models_{case_tag}"
    outputs = export_bundle(figure, output_dir, output_stem)
    write_qa(output_dir, copied_source_dir, outputs, model_frames, selected)
    caption = (
        "Suggested caption. Frozen-backbone probabilities (blue), the validation-selected "
        "Static thresholds (gray dashed), and Online thresholds (magenta dashed) around "
        f"catalog event {int(selected['catalog_event_id'])}; event bars show the corresponding "
        "decisions. This window was selected post hoc for display, after the test results "
        "were locked, using the prespecified high-contrast rule documented in the complete "
        "selection audit. It is not a random or typical case and was not used for parameter "
        "selection or global performance calculation. Insets report "
        "full-test precision, recall, and F1, not scores for this single displayed event. "
        "Events are evaluated by maximum-cardinality one-to-one assignment over positive-"
        "duration temporal overlaps, so each prediction and each catalog event can be used "
        "at most once and one overlong prediction cannot contribute multiple true positives."
    )
    (output_dir / "SUGGESTED_CAPTION_ONE_TO_ONE.md").write_text(
        caption + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                **validation_summary,
                "outputs": {key: str(path) for key, path in outputs.items()},
                "size_mm": [WIDTH_MM, HEIGHT_MM],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
