#!/usr/bin/env python3
"""
Create a PDF report comparing evaluated ManyDepth runs.

This script reads evaluation artifacts saved by manydepth/evaluate_depth_mda.py,
for example:

  outs/mdp/models/weights_1/multi_eigen_benchmark_split.npy
  outs/mdp/models/weights_1/multi_eigen_benchmark_errors.npy

It does not read TensorBoard logs and it does not report training losses.

The PDF includes:
  - run/checkpoint overview
  - config differences against a base config/run
  - evaluation metric tables: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
  - depth visualizations from the saved disparity/depth arrays
  - difference maps against the base run for samples with the largest depth changes

Examples:
  python make_outs_report.py --outs outs --output outs/eval_report.pdf --base-run mdp
  python make_outs_report.py --outs /outs --eval-split eigen_benchmark --method-tag multi
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import math
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/manydepth_matplotlib")

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


METRIC_NAMES = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
LOWER_IS_BETTER = {"abs_rel", "sq_rel", "rmse", "rmse_log"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAGE_SIZE = (11.0, 8.5)
VISUAL_PAGE_SIZE = (14.0, 8.5)
PDF_DPI = 300
plt.rcParams["savefig.dpi"] = PDF_DPI
plt.rcParams["figure.dpi"] = 150


@dataclass
class EvalArtifact:
    run_name: str
    display_name: str
    run_path: Path
    checkpoint_path: Path
    prediction_path: Path
    errors_path: Optional[Path]
    prefix: str
    method_tag: str
    eval_split: str
    config: Dict[str, Any] = field(default_factory=dict)
    prediction_shape: Tuple[int, ...] = field(default_factory=tuple)
    sample_count: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    weight_number: int = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PDF comparing evaluated runs in outs/."
    )
    parser.add_argument("--outs", type=Path, default=Path("outs"), help="Runs directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outs/eval_report.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--base-run",
        default=None,
        help="Run name to use as the visual/config base. Defaults to mdp if present.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Config file used for config diffs. If missing, the base run config is used.",
    )
    parser.add_argument(
        "--runs",
        default=None,
        help="Comma-separated run names to include. Defaults to all evaluated runs under --outs.",
    )
    parser.add_argument(
        "--method-tag",
        default="multi",
        help="Prediction prefix to compare, e.g. multi or teacher. Use '*' to include any prefix.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Evaluation split to compare, e.g. eigen_benchmark. Defaults to the first split found for the base run.",
    )
    parser.add_argument(
        "--checkpoints",
        choices=("latest", "all"),
        default="latest",
        help="Use the latest evaluated checkpoint per run, or include all evaluated checkpoints.",
    )
    parser.add_argument(
        "--prediction-kind",
        choices=("auto", "disparity", "depth"),
        default="auto",
        help="How to interpret *_split.npy files. Evaluation split files are usually disparities.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=1e-3,
        help="Minimum depth used when converting disparity to depth for visualization.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=80.0,
        help="Maximum depth used for visualization clipping.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=6,
        help="Number of highest-difference samples to visualize.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="Deprecated; visual samples are selected randomly now.",
    )
    parser.add_argument(
        "--runs-per-page",
        type=int,
        default=2,
        help="Maximum non-base runs shown on one visual page.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1,
        help="Seed used for random visual sample selection.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional folder of RGB example images. Used before dataset lookup.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional KITTI data root override for loading RGB eval images. "
        "If omitted, the script reads data_path from configs and tries nearby dataset roots.",
    )
    parser.add_argument(
        "--max-config-diffs",
        type=int,
        default=22,
        help="Maximum config differences shown per run.",
    )
    return parser.parse_args()


def read_yaml_or_simple(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}

    if yaml is not None:
        try:
            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

    data: Dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    return data


def read_run_config(run_path: Path) -> Dict[str, Any]:
    for candidate in (run_path / "config.yaml", run_path / "models" / "config.yaml"):
        config = read_yaml_or_simple(candidate)
        if config:
            return config
    return {}


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def short_text(value: Any, max_len: int = 58) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def fmt_float(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1000.0 or (0.0 < abs_value < 0.001):
        return f"{value:.3e}"
    return f"{value:.5g}"


def parse_prediction_name(path: Path) -> Tuple[str, str, str]:
    stem = path.stem
    if not stem.endswith("_split"):
        return stem, stem, ""
    prefix = stem[: -len("_split")]
    for known_tag in ("multi", "teacher", "student", "mono"):
        marker = known_tag + "_"
        if prefix.startswith(marker):
            return prefix, known_tag, prefix[len(marker) :]
    if "_" in prefix:
        tag, eval_split = prefix.split("_", 1)
        return prefix, tag, eval_split
    return prefix, prefix, ""


def weight_number_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"weights_(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def checkpoint_path_for_prediction(path: Path) -> Path:
    for parent in [path.parent] + list(path.parents):
        if re.fullmatch(r"weights_\d+", parent.name):
            return parent
    return path.parent


def load_prediction_shape(path: Path) -> Tuple[Tuple[int, ...], int]:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = tuple(int(dim) for dim in array.shape)
    except Exception:
        return (), 0

    squeezed_shape = tuple(dim for dim in shape if dim != 1)
    if len(squeezed_shape) <= 2:
        return shape, 1
    return shape, int(squeezed_shape[0])


def load_eval_metrics(errors_path: Optional[Path]) -> Dict[str, float]:
    if errors_path is None or not errors_path.exists():
        return {}
    try:
        errors = np.asarray(np.load(errors_path, allow_pickle=False), dtype=np.float64)
    except Exception:
        return {}

    errors = np.squeeze(errors)
    if errors.size == 0:
        return {}
    if errors.ndim == 1:
        if errors.shape[0] < len(METRIC_NAMES):
            return {}
        mean_errors = errors[: len(METRIC_NAMES)]
    else:
        if errors.shape[-1] < len(METRIC_NAMES):
            return {}
        errors = errors.reshape((-1, errors.shape[-1]))
        mean_errors = np.nanmean(errors[:, : len(METRIC_NAMES)], axis=0)

    return {name: float(value) for name, value in zip(METRIC_NAMES, mean_errors)}


def discover_artifacts(args: argparse.Namespace) -> List[EvalArtifact]:
    if not args.outs.exists():
        raise FileNotFoundError(f"Could not find outs directory: {args.outs}")

    include_runs = None
    if args.runs:
        include_runs = {name.strip() for name in args.runs.split(",") if name.strip()}

    artifacts: List[EvalArtifact] = []
    for run_path in sorted(path for path in args.outs.iterdir() if path.is_dir()):
        if include_runs and run_path.name not in include_runs:
            continue
        config = read_run_config(run_path)
        for prediction_path in sorted(run_path.rglob("*_split.npy")):
            prefix, method_tag, eval_split = parse_prediction_name(prediction_path)
            if args.method_tag != "*" and method_tag != args.method_tag:
                continue
            if args.eval_split and eval_split != args.eval_split:
                continue

            errors_path = prediction_path.with_name(prefix + "_errors.npy")
            if not errors_path.exists():
                errors_path = None

            prediction_shape, sample_count = load_prediction_shape(prediction_path)
            checkpoint_path = checkpoint_path_for_prediction(prediction_path)
            weight_number = weight_number_from_path(prediction_path)
            display_name = run_path.name
            if args.checkpoints == "all":
                display_name = f"{run_path.name}/{checkpoint_path.name}"

            artifacts.append(
                EvalArtifact(
                    run_name=run_path.name,
                    display_name=display_name,
                    run_path=run_path,
                    checkpoint_path=checkpoint_path,
                    prediction_path=prediction_path,
                    errors_path=errors_path,
                    prefix=prefix,
                    method_tag=method_tag,
                    eval_split=eval_split,
                    config=config,
                    prediction_shape=prediction_shape,
                    sample_count=sample_count,
                    metrics=load_eval_metrics(errors_path),
                    weight_number=weight_number,
                )
            )

    if args.checkpoints == "latest":
        by_run: Dict[Tuple[str, str, str], EvalArtifact] = {}
        for artifact in artifacts:
            key = (artifact.run_name, artifact.method_tag, artifact.eval_split)
            previous = by_run.get(key)
            if previous is None:
                by_run[key] = artifact
                continue
            previous_key = (previous.weight_number, str(previous.prediction_path))
            current_key = (artifact.weight_number, str(artifact.prediction_path))
            if current_key > previous_key:
                by_run[key] = artifact
        artifacts = sorted(by_run.values(), key=lambda item: item.run_name)

    return artifacts


def choose_base_artifact(artifacts: Sequence[EvalArtifact], requested: Optional[str]) -> EvalArtifact:
    if not artifacts:
        raise ValueError("No evaluated prediction files were found.")
    if requested:
        candidates = [item for item in artifacts if item.run_name == requested or item.display_name == requested]
        if candidates:
            return sorted(candidates, key=lambda item: (item.weight_number, item.display_name))[-1]
        raise ValueError(f"--base-run {requested!r} was not found among evaluated artifacts.")
    for preferred in ("mdp", "base", "baseline"):
        candidates = [item for item in artifacts if item.run_name == preferred]
        if candidates:
            return sorted(candidates, key=lambda item: (item.weight_number, item.display_name))[-1]
    return artifacts[0]


def filter_to_base_split(artifacts: Sequence[EvalArtifact], base: EvalArtifact, args: argparse.Namespace) -> List[EvalArtifact]:
    filtered = [
        item
        for item in artifacts
        if item.eval_split == base.eval_split and item.method_tag == base.method_tag
    ]
    if args.eval_split:
        filtered = [item for item in filtered if item.eval_split == args.eval_split]
    return sorted(filtered, key=lambda item: item.display_name)


def add_text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.93, title, fontsize=20, weight="bold", va="top")
    y = 0.86
    for line in lines:
        wrapped = textwrap.wrap(str(line), width=122) or [""]
        for part in wrapped:
            fig.text(0.06, y, part, fontsize=10.2, va="top")
            y -= 0.035
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight", dpi=PDF_DPI)
                plt.close(fig)
                fig = plt.figure(figsize=PAGE_SIZE)
                fig.patch.set_facecolor("white")
                y = 0.93
    pdf.savefig(fig, bbox_inches="tight", dpi=PDF_DPI)
    plt.close(fig)


def add_table_pages(
    pdf: PdfPages,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    rows_per_page: int = 22,
    font_size: float = 7.2,
) -> None:
    if not rows:
        add_text_page(pdf, title, ["No data found."])
        return

    for page_idx, start in enumerate(range(0, len(rows), rows_per_page)):
        chunk = rows[start : start + rows_per_page]
        fig, ax = plt.subplots(figsize=PAGE_SIZE)
        fig.patch.set_facecolor("white")
        ax.axis("off")
        suffix = f" ({page_idx + 1})" if len(rows) > rows_per_page else ""
        ax.set_title(title + suffix, loc="left", fontsize=16, weight="bold", pad=16)
        table = ax.table(cellText=chunk, colLabels=headers, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.25)
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#e9eef6")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#ffffff" if row % 2 else "#f7f7f7")
        pdf.savefig(fig, bbox_inches="tight", dpi=PDF_DPI)
        plt.close(fig)


def add_overview_page(pdf: PdfPages, artifacts: Sequence[EvalArtifact], base: EvalArtifact) -> None:
    rows = []
    for item in artifacts:
        rows.append(
            [
                item.display_name + ("  [base]" if item is base else ""),
                item.method_tag,
                item.eval_split,
                item.checkpoint_path.name,
                "x".join(str(dim) for dim in item.prediction_shape),
                str(item.sample_count),
                short_text(item.prediction_path.relative_to(item.run_path)),
                short_text(item.errors_path.relative_to(item.run_path) if item.errors_path else ""),
            ]
        )
    add_table_pages(
        pdf,
        "Evaluated Checkpoints",
        ["run", "method", "split", "checkpoint", "prediction shape", "samples", "prediction file", "errors file"],
        rows,
        rows_per_page=18,
        font_size=6.7,
    )


def add_config_diff_pages(
    pdf: PdfPages,
    artifacts: Sequence[EvalArtifact],
    base: EvalArtifact,
    base_config: Dict[str, Any],
    max_diffs: int,
) -> None:
    base_flat = flatten_dict(base_config or base.config)
    important_tokens = (
        "model_name",
        "learning_rate",
        "batch_size",
        "height",
        "width",
        "lora",
        "split",
        "eval",
        "data",
    )
    rows: List[List[str]] = []
    for item in artifacts:
        if item.display_name == base.display_name and not base_config:
            continue
        flat = flatten_dict(item.config)
        keys = sorted(set(base_flat) | set(flat))

        def score_key(key: str) -> Tuple[int, str]:
            low = key.lower()
            return (0 if any(token in low for token in important_tokens) else 1, key)

        diffs = [key for key in keys if base_flat.get(key) != flat.get(key)]
        diffs.sort(key=score_key)
        for key in diffs[:max_diffs]:
            rows.append(
                [
                    item.display_name,
                    key,
                    short_text(base_flat.get(key, "<missing>")),
                    short_text(flat.get(key, "<missing>")),
                ]
            )
        if len(diffs) > max_diffs:
            rows.append([item.display_name, f"... {len(diffs) - max_diffs} more differences", "", ""])

    add_table_pages(
        pdf,
        "Config Differences Against Base",
        ["run", "key", "base value", "run value"],
        rows,
        rows_per_page=24,
        font_size=6.9,
    )


def metric_sort_key(item: EvalArtifact) -> Tuple[float, str]:
    abs_rel = item.metrics.get("abs_rel")
    if abs_rel is None or not math.isfinite(abs_rel):
        return (float("inf"), item.display_name)
    return (abs_rel, item.display_name)


def add_eval_metric_pages(pdf: PdfPages, artifacts: Sequence[EvalArtifact], base: EvalArtifact) -> None:
    sorted_items = sorted(artifacts, key=metric_sort_key)
    rows: List[List[str]] = []
    base_metrics = base.metrics
    for rank, item in enumerate(sorted_items, start=1):
        row = [str(rank), item.display_name, item.checkpoint_path.name]
        for metric in METRIC_NAMES:
            row.append(fmt_float(item.metrics.get(metric)))
        delta = None
        if "abs_rel" in item.metrics and "abs_rel" in base_metrics:
            delta = item.metrics["abs_rel"] - base_metrics["abs_rel"]
        row.append(fmt_float(delta))
        rows.append(row)

    add_table_pages(
        pdf,
        "Evaluation Metrics",
        ["rank", "run", "checkpoint", *METRIC_NAMES, "abs_rel - base"],
        rows,
        rows_per_page=22,
        font_size=6.6,
    )


def add_metric_bar_pages(pdf: PdfPages, artifacts: Sequence[EvalArtifact]) -> None:
    metrics_with_data = [
        metric for metric in METRIC_NAMES if any(metric in item.metrics for item in artifacts)
    ]
    if not metrics_with_data:
        return

    for start in range(0, len(metrics_with_data), 4):
        chunk = metrics_with_data[start : start + 4]
        fig, axes = plt.subplots(2, 2, figsize=PAGE_SIZE, squeeze=False)
        fig.patch.set_facecolor("white")
        fig.suptitle("Evaluation Metric Comparison", fontsize=16, weight="bold", x=0.04, ha="left")
        for ax, metric in zip(axes.ravel(), chunk):
            values = [item.metrics.get(metric, np.nan) for item in artifacts]
            names = [item.display_name for item in artifacts]
            colors = ["#5577aa" if metric in LOWER_IS_BETTER else "#5f9f6e"] * len(values)
            ax.barh(names, values, color=colors)
            ax.set_title(f"{metric} ({'lower' if metric in LOWER_IS_BETTER else 'higher'} is better)", fontsize=10)
            ax.grid(axis="x", linewidth=0.4, alpha=0.35)
            ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", labelsize=7)
        for ax in axes.ravel()[len(chunk) :]:
            ax.axis("off")
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))
        pdf.savefig(fig, bbox_inches="tight", dpi=PDF_DPI)
        plt.close(fig)


def load_prediction_array(path: Path) -> Optional[np.ndarray]:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception:
        return None


def squeeze_prediction_sample(array: np.ndarray, sample_index: int) -> Optional[np.ndarray]:
    if array is None:
        return None
    squeezed = np.squeeze(array)
    if squeezed.ndim == 2:
        sample = squeezed
    elif squeezed.ndim >= 3:
        if sample_index < 0 or sample_index >= squeezed.shape[0]:
            return None
        sample = np.squeeze(squeezed[sample_index])
    else:
        return None

    if sample.ndim == 3 and sample.shape[0] in (1, 3, 4):
        sample = sample[0]
    elif sample.ndim == 3 and sample.shape[-1] in (1, 3, 4):
        sample = sample[..., 0]
    if sample.ndim != 2:
        return None
    return np.asarray(sample, dtype=np.float32)


def finite_2d(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def resize_2d(array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    array = finite_2d(array)
    if array.shape == target_shape:
        return array
    if Image is not None:
        image = Image.fromarray(array)
        image = image.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32)

    src_h, src_w = array.shape
    dst_h, dst_w = target_shape
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, dst_h)).astype(int), 0, src_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, dst_w)).astype(int), 0, src_w - 1)
    return array[y_idx][:, x_idx]


def resize_mask_2d(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == target_shape:
        return mask
    if Image is not None:
        image = Image.fromarray(mask.astype(np.uint8) * 255)
        image = image.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        return np.asarray(image) > 127

    src_h, src_w = mask.shape
    dst_h, dst_w = target_shape
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, dst_h)).astype(int), 0, src_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, dst_w)).astype(int), 0, src_w - 1)
    return mask[y_idx][:, x_idx]


def prediction_is_depth(item: EvalArtifact, prediction_kind: str) -> bool:
    if prediction_kind == "depth":
        return True
    if prediction_kind == "disparity":
        return False
    low = item.prediction_path.name.lower()
    if "depth" in low and "disp" not in low and not low.endswith("_split.npy"):
        return True
    return False


def sample_to_depth(
    sample: np.ndarray,
    item: EvalArtifact,
    prediction_kind: str,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    sample = finite_2d(sample)
    if prediction_is_depth(item, prediction_kind):
        depth = sample
    else:
        disp = np.maximum(sample, 1.0 / max_depth if max_depth > 0 else 1e-6)
        depth = 1.0 / disp
    return np.clip(depth, min_depth, max_depth).astype(np.float32, copy=False)


def sample_to_unclipped_depth(
    sample: np.ndarray,
    item: EvalArtifact,
    prediction_kind: str,
) -> np.ndarray:
    sample = finite_2d(sample)
    if prediction_is_depth(item, prediction_kind):
        return sample.astype(np.float32, copy=False)
    disp = np.asarray(sample, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = 1.0 / disp
    return depth.astype(np.float32, copy=False)


def sample_to_disparity(
    sample: np.ndarray,
    item: EvalArtifact,
    prediction_kind: str,
) -> np.ndarray:
    sample = finite_2d(sample)
    if prediction_is_depth(item, prediction_kind):
        depth = np.maximum(sample, 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = 1.0 / depth
        return finite_2d(disp)
    return sample.astype(np.float32, copy=False)


def valid_depth_mask(
    sample: np.ndarray,
    item: EvalArtifact,
    prediction_kind: str,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    depth = sample_to_unclipped_depth(sample, item, prediction_kind)
    return np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)


def robust_limits(arrays: Sequence[np.ndarray], low: float = 2.0, high: float = 98.0) -> Tuple[float, float]:
    values = []
    for array in arrays:
        finite = np.asarray(array)[np.isfinite(array)]
        if finite.size:
            values.append(finite.reshape(-1))
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    vmin, vmax = np.percentile(merged, [low, high])
    if not math.isfinite(float(vmin)) or not math.isfinite(float(vmax)) or abs(vmax - vmin) < 1e-12:
        vmin = float(np.nanmin(merged))
        vmax = float(np.nanmax(merged))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def normalize_with_limits(array: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
    vmin, vmax = limits
    return np.clip((array - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)


def depth_difference_score(base_depth: np.ndarray, other_depth: np.ndarray) -> float:
    other_depth = resize_2d(other_depth, base_depth.shape)
    limits = robust_limits([base_depth, other_depth])
    base_norm = normalize_with_limits(base_depth, limits)
    other_norm = normalize_with_limits(other_depth, limits)
    return float(np.mean(np.abs(other_norm - base_norm)))


def scan_indices(sample_count: int, scan_limit: int) -> List[int]:
    if sample_count <= 0:
        return []
    if scan_limit <= 0 or sample_count <= scan_limit:
        return list(range(sample_count))
    return sorted(set(int(round(x)) for x in np.linspace(0, sample_count - 1, scan_limit)))


def choose_visual_samples(
    artifacts: Sequence[EvalArtifact],
    base: EvalArtifact,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> List[Tuple[int, float, str]]:
    available_counts = [
        item.sample_count
        for item in artifacts
        if arrays.get(item.display_name) is not None and item.sample_count > 0
    ]
    if not available_counts:
        return []
    common_count = min(available_counts)
    if common_count <= 0:
        return []
    count = min(max(args.max_examples, 0), common_count)
    if count <= 0:
        return []
    rng = np.random.default_rng(args.random_seed)
    selected = rng.choice(common_count, size=count, replace=False)
    return [(int(idx), 0.0, "random") for idx in sorted(selected.tolist())]


def load_image_from_path(path: Path, target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    if Image is None or not path.exists():
        return None
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if target_shape is not None:
                image = image.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
            return np.asarray(image)
    except Exception:
        return None


def sorted_image_files(image_dir: Optional[Path]) -> List[Path]:
    if image_dir is None or not image_dir.exists():
        return []
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_split_lines(eval_split: str) -> List[str]:
    path = Path("splits") / eval_split / "test_files.txt"
    if not path.exists():
        return []
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def bool_from_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def candidate_kitti_image_paths(
    data_path: Path,
    split_line: str,
    config: Dict[str, Any],
) -> List[Path]:
    parts = split_line.split()
    if len(parts) < 2:
        return []
    folder = parts[0]
    frame = parts[1]
    side = parts[2] if len(parts) > 2 else "l"
    camera = "image_02" if side in {"l", "2", "02"} else "image_03"

    ext = ".png" if bool_from_config(config.get("png", False)) else ".jpg"
    frame_names = [frame]
    try:
        frame_names.append(f"{int(frame):010d}")
    except Exception:
        pass
    frame_names = list(dict.fromkeys(frame_names))

    paths: List[Path] = []
    for frame_name in frame_names:
        paths.append(data_path / folder / camera / "data" / f"{frame_name}{ext}")
        paths.append(data_path / folder / camera / f"{frame_name}{ext}")
        paths.append(data_path / folder / f"{frame_name}{ext}")
    for frame_name in frame_names:
        for any_ext in (".jpg", ".png", ".jpeg"):
            paths.append(data_path / folder / camera / "data" / f"{frame_name}{any_ext}")
            paths.append(data_path / folder / camera / f"{frame_name}{any_ext}")
    return list(dict.fromkeys(paths))


def path_from_config_value(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def path_has_split_sample(data_path: Optional[Path], split_lines: Sequence[str], config: Dict[str, Any]) -> bool:
    if data_path is None or not data_path.exists() or not split_lines:
        return False
    for split_line in split_lines[: min(len(split_lines), 20)]:
        for candidate in candidate_kitti_image_paths(data_path, split_line, config):
            if candidate.exists():
                return True
    return False


def dataset_root_candidates(artifacts: Sequence[EvalArtifact], base: EvalArtifact, base_config: Dict[str, Any]) -> List[Path]:
    candidates: List[Path] = []
    for config in [base.config, base_config, *(item.config for item in artifacts)]:
        for key in ("data_path", "dataset_path", "kitti_path", "data_root", "root_dir"):
            candidate = path_from_config_value(config.get(key) if config else None)
            if candidate is not None:
                candidates.append(candidate)

    repo_root = Path(__file__).resolve().parent
    home_root = repo_root.parent
    candidates.extend(
        [
            repo_root / "kitti_data",
            repo_root / "data",
            repo_root / "datasets" / "kitti",
            repo_root / "datasets" / "KITTI",
            home_root / "kitti_data",
            home_root / "data",
            home_root / "datasets" / "kitti",
            home_root / "datasets" / "KITTI",
            Path("/data"),
            Path("/datasets/kitti"),
            Path("/datasets/KITTI"),
            Path("/work/gn21/h62001/data"),
            Path("/work/gn21/h62001/kitti_data"),
            Path("/work/gn21/h62001/datasets/kitti"),
            Path("/work/gn21/h62001/datasets/KITTI"),
        ]
    )

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser()
        except Exception:
            continue
        key = str(resolved)
        if key not in seen:
            deduped.append(resolved)
            seen.add(key)
    return deduped


def resolve_data_path(
    cli_data_path: Optional[Path],
    artifacts: Sequence[EvalArtifact],
    base: EvalArtifact,
    base_config: Dict[str, Any],
    split_lines: Sequence[str],
) -> Optional[Path]:
    if cli_data_path is not None:
        return cli_data_path.expanduser()

    lookup_config = dict(base_config)
    lookup_config.update(base.config)
    for candidate in dataset_root_candidates(artifacts, base, base_config):
        if path_has_split_sample(candidate, split_lines, lookup_config):
            return candidate
    return None


def find_rgb_image(
    sample_index: int,
    target_shape: Tuple[int, int],
    image_files: Sequence[Path],
    split_lines: Sequence[str],
    data_path: Optional[Path],
    base_config: Dict[str, Any],
) -> Optional[np.ndarray]:
    if image_files:
        if sample_index < len(image_files):
            return load_image_from_path(image_files[sample_index], target_shape)
        key = str(sample_index)
        for path in image_files:
            if key in path.stem:
                return load_image_from_path(path, target_shape)

    if data_path is None:
        config_data_path = base_config.get("data_path")
        data_path = Path(config_data_path).expanduser() if config_data_path else None
    if data_path is None or sample_index >= len(split_lines):
        return None

    for candidate in candidate_kitti_image_paths(data_path, split_lines[sample_index], base_config):
        image = load_image_from_path(candidate, target_shape)
        if image is not None:
            return image
    return None


def load_sample_depth(
    item: EvalArtifact,
    sample_index: int,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Optional[np.ndarray]:
    array = arrays.get(item.display_name)
    sample = squeeze_prediction_sample(array, sample_index)
    if sample is None:
        return None
    return sample_to_depth(sample, item, args.prediction_kind, args.min_depth, args.max_depth)


def load_sample_disparity(
    item: EvalArtifact,
    sample_index: int,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Optional[np.ndarray]:
    array = arrays.get(item.display_name)
    sample = squeeze_prediction_sample(array, sample_index)
    if sample is None:
        return None
    return sample_to_disparity(sample, item, args.prediction_kind)


def load_sample_valid_mask(
    item: EvalArtifact,
    sample_index: int,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Optional[np.ndarray]:
    array = arrays.get(item.display_name)
    sample = squeeze_prediction_sample(array, sample_index)
    if sample is None:
        return None
    return valid_depth_mask(sample, item, args.prediction_kind, args.min_depth, args.max_depth)


def sample_error_metrics(item: EvalArtifact, sample_index: int) -> Dict[str, float]:
    if item.errors_path is None or not item.errors_path.exists():
        return {}
    try:
        errors = np.asarray(np.load(item.errors_path, mmap_mode="r", allow_pickle=False), dtype=np.float64)
    except Exception:
        return {}
    errors = np.squeeze(errors)
    if errors.ndim == 1:
        if errors.shape[0] < len(METRIC_NAMES):
            return {}
        values = errors[: len(METRIC_NAMES)]
    elif sample_index < errors.shape[0] and errors.shape[-1] >= len(METRIC_NAMES):
        values = np.asarray(errors[sample_index]).reshape(-1)[: len(METRIC_NAMES)]
    else:
        return {}
    return {name: float(value) for name, value in zip(METRIC_NAMES, values)}


def add_depth_visual_pages(
    pdf: PdfPages,
    artifacts: Sequence[EvalArtifact],
    base: EvalArtifact,
    arrays: Dict[str, np.ndarray],
    selected_samples: Sequence[Tuple[int, float, str]],
    image_files: Sequence[Path],
    split_lines: Sequence[str],
    args: argparse.Namespace,
) -> bool:
    if not selected_samples:
        return False

    comparison_runs = [item for item in artifacts if item.display_name != base.display_name]
    if not comparison_runs:
        return False

    made_page = False
    runs_per_page = max(1, args.runs_per_page)
    diff_cmap = plt.get_cmap("coolwarm").copy()
    diff_cmap.set_bad(color="#ffffff")

    for sample_index, _score, _selection_kind in selected_samples:
        base_depth = load_sample_depth(base, sample_index, arrays, args)
        base_disp = load_sample_disparity(base, sample_index, arrays, args)
        base_mask = load_sample_valid_mask(base, sample_index, arrays, args)
        if base_depth is None or base_disp is None or base_mask is None:
            continue
        base_disp = resize_2d(base_disp, base_depth.shape)
        base_mask = resize_mask_2d(base_mask, base_depth.shape)

        for group_start in range(0, len(comparison_runs), runs_per_page):
            group = comparison_runs[group_start : group_start + runs_per_page]
            valid_group = []
            depths = [base_depth]
            for item in group:
                depth = load_sample_depth(item, sample_index, arrays, args)
                disp = load_sample_disparity(item, sample_index, arrays, args)
                mask = load_sample_valid_mask(item, sample_index, arrays, args)
                if depth is None or disp is None or mask is None:
                    continue
                depth = resize_2d(depth, base_depth.shape)
                disp = resize_2d(disp, base_depth.shape)
                mask = resize_mask_2d(mask, base_depth.shape)
                valid_group.append((item, depth, disp, mask))
                depths.append(depth)
            if not valid_group:
                continue

            cols = 2 + len(valid_group)
            fig, axes = plt.subplots(2, cols, figsize=VISUAL_PAGE_SIZE, squeeze=False)
            fig.patch.set_facecolor("white")
            title = (
                f"Sample {sample_index}: Random Evaluation Example"
                f" | disparity diff vs {base.display_name}"
            )
            fig.suptitle(title, fontsize=14.5, weight="bold", x=0.03, ha="left")

            rgb = find_rgb_image(
                sample_index,
                base_depth.shape,
                image_files,
                split_lines,
                args.data_path,
                base.config,
            )
            if rgb is not None:
                axes[0, 0].imshow(rgb, interpolation="lanczos")
                axes[0, 0].set_title("example image", fontsize=9)
            else:
                axes[0, 0].text(0.5, 0.5, "RGB image not found", ha="center", va="center", fontsize=9)
                axes[0, 0].set_title("example image", fontsize=9)
            axes[0, 0].axis("off")
            axes[1, 0].axis("off")

            depth_limits = robust_limits(depths)
            axes[0, 1].imshow(
                base_depth,
                cmap="plasma",
                vmin=depth_limits[0],
                vmax=depth_limits[1],
                interpolation="lanczos",
            )
            axes[0, 1].set_title(f"{base.display_name}\ndepth", fontsize=8.4)
            axes[0, 1].axis("off")
            axes[1, 1].axis("off")
            base_sample_metrics = sample_error_metrics(base, sample_index)
            if base_sample_metrics:
                axes[1, 1].text(
                    0.02,
                    0.7,
                    f"abs_rel {fmt_float(base_sample_metrics.get('abs_rel'))}\nrmse {fmt_float(base_sample_metrics.get('rmse'))}\na1 {fmt_float(base_sample_metrics.get('a1'))}",
                    fontsize=8,
                    va="top",
                )

            for col, (item, depth, disp, mask) in enumerate(valid_group, start=2):
                axes[0, col].imshow(
                    depth,
                    cmap="plasma",
                    vmin=depth_limits[0],
                    vmax=depth_limits[1],
                    interpolation="lanczos",
                )
                axes[0, col].set_title(f"{item.display_name}\ndepth", fontsize=8.4)
                axes[0, col].axis("off")

                valid = base_mask & mask & np.isfinite(base_disp) & np.isfinite(disp)
                diff = np.abs(disp - base_disp)
                valid_values = diff[valid]
                if valid_values.size:
                    diff_vmax = max(float(np.percentile(valid_values, 99)), 1e-12)
                    mean_diff = float(np.mean(valid_values))
                else:
                    diff_vmax = 1.0
                    mean_diff = float("nan")
                masked_diff = np.ma.array(diff, mask=~valid)
                axes[1, col].imshow(
                    masked_diff,
                    cmap=diff_cmap,
                    vmin=0.0,
                    vmax=diff_vmax,
                    interpolation="nearest",
                )
                axes[1, col].set_title("disparity abs diff\nblue low, red high", fontsize=8.4)
                axes[1, col].axis("off")
                metrics = sample_error_metrics(item, sample_index)
                text = f"mean disp diff {fmt_float(mean_diff)}\nmasked depth > {args.max_depth:g}m"
                if metrics:
                    text += (
                        f"\nabs_rel {fmt_float(metrics.get('abs_rel'))}"
                        f"\nrmse {fmt_float(metrics.get('rmse'))}"
                        f"\na1 {fmt_float(metrics.get('a1'))}"
                    )
                axes[1, col].text(0.02, -0.08, text, transform=axes[1, col].transAxes, fontsize=7.2, va="top")

            fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
            pdf.savefig(fig, bbox_inches="tight", dpi=PDF_DPI)
            plt.close(fig)
            made_page = True
    return made_page


def add_notes_page(
    pdf: PdfPages,
    artifacts: Sequence[EvalArtifact],
    base: EvalArtifact,
    visual_pages: bool,
    args: argparse.Namespace,
) -> None:
    missing_errors = [item.display_name for item in artifacts if not item.metrics]
    notes = [
        "Report notes:",
        "  - TensorBoard event files are not read.",
        "  - Metrics come only from evaluation errors files, such as multi_eigen_benchmark_errors.npy.",
        "  - Metric columns are exactly: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3.",
        "  - Depth pages are built from saved split arrays, such as multi_eigen_benchmark_split.npy.",
        "  - By default *_split.npy is treated as disparity and visualized as depth = 1 / disparity.",
        f"  - Visual samples are selected randomly with seed {args.random_seed}; largest differences are not searched.",
        "  - Difference maps use absolute disparity differences, not depth differences.",
        f"  - Pixels whose implied depth is greater than {args.max_depth:g}m are masked out of difference maps.",
        "  - Difference maps use blue for low difference and red for high difference.",
        f"  - Base artifact: {base.display_name} ({base.prediction_path})",
    ]
    if args.prediction_kind == "auto":
        notes.append("  - --prediction-kind auto treats *_split.npy files as disparity unless the filename clearly says depth.")
    if missing_errors:
        notes.append("  - Missing eval errors for: " + ", ".join(missing_errors))
    if not visual_pages:
        notes.append("  - No visual comparison pages were created; check that base and comparison arrays share sample indices.")
    if args.image_dir is None and args.data_path is None:
        notes.append("  - Original RGB images are optional. Pass --image-dir or --data-path if automatic lookup cannot find them.")
    add_text_page(pdf, "Notes", notes)


def main() -> None:
    args = parse_args()
    artifacts = discover_artifacts(args)
    if not artifacts:
        raise SystemExit(
            f"No evaluated prediction files matching method={args.method_tag!r} "
            f"and split={args.eval_split!r} were found under {args.outs}"
        )

    base = choose_base_artifact(artifacts, args.base_run)
    artifacts = filter_to_base_split(artifacts, base, args)
    base = choose_base_artifact(artifacts, base.display_name)

    base_config = read_yaml_or_simple(args.base_config) if args.base_config else {}
    image_files = sorted_image_files(args.image_dir)
    split_lines = load_split_lines(base.eval_split)
    args.data_path = resolve_data_path(args.data_path, artifacts, base, base_config, split_lines)
    arrays = {
        item.display_name: load_prediction_array(item.prediction_path)
        for item in artifacts
    }
    selected_samples = choose_visual_samples(artifacts, base, arrays, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    created_at = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with PdfPages(args.output) as pdf:
        add_text_page(
            pdf,
            "ManyDepth Evaluation Report",
            [
                f"Created: {created_at}",
                f"Runs directory: {args.outs}",
                f"Method tag: {base.method_tag}",
                f"Evaluation split: {base.eval_split}",
                f"Artifacts included: {', '.join(item.display_name for item in artifacts)}",
                f"Base for visual differences: {base.display_name}",
                f"Visual samples: random, seed {args.random_seed}",
                f"RGB data path: {args.data_path if args.data_path is not None else 'not found'}",
                f"Output PDF: {args.output}",
            ],
        )
        add_overview_page(pdf, artifacts, base)
        add_config_diff_pages(pdf, artifacts, base, base_config, args.max_config_diffs)
        add_eval_metric_pages(pdf, artifacts, base)
        add_metric_bar_pages(pdf, artifacts)
        visual_pages = add_depth_visual_pages(
            pdf,
            artifacts,
            base,
            arrays,
            selected_samples,
            image_files,
            split_lines,
            args,
        )
        add_notes_page(pdf, artifacts, base, visual_pages, args)

    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
