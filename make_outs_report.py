#!/usr/bin/env python3
"""
Create a single self-contained HTML report comparing evaluated ManyDepth runs.

This script reads evaluation artifacts saved by manydepth/evaluate_depth_mda.py,
for example:

  outs/mdp/models/weights_1/multi_eigen_benchmark_split.npy
  outs/mdp/models/weights_1/multi_eigen_benchmark_errors.npy

It does not read TensorBoard logs and it does not report training losses.

There is no "base" run: every evaluated run is shown on equal footing. The
report includes:
  - run/checkpoint overview
  - evaluation metric tables and bar charts: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
  - depth visualizations for a random sample of evaluation images, with every
    run's predicted depth placed side by side (sharing a color scale) so they
    can be compared directly

The output is a single .html file with all charts and depth maps embedded as
base64 PNGs, so it can be opened or shared without any extra files.

Examples:
  python make_outs_report.py --outs outs --output outs/eval_report.html
  python make_outs_report.py --outs /outs --eval-split eigen_benchmark --num-samples 60
"""

from __future__ import annotations

import argparse
import base64
import datetime as _datetime
import io
import math
import os
import re
from dataclasses import dataclass, field
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/manydepth_matplotlib")

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
CHART_FIGSIZE = (14.0, 9.0)
SAMPLE_PANEL_SIZE = (3.8, 4.2)  # inches per RGB/depth panel: (width, height)
REPORT_DPI = 180
plt.rcParams["figure.dpi"] = 150

REPORT_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 24px; color: #1c1f24; background: #fbfbfd; }
h1 { margin-bottom: 6px; }
h2 { margin-top: 36px; border-bottom: 1px solid #d8dde3; padding-bottom: 6px; }
ul.meta { list-style: none; padding: 0; color: #50565e; font-size: 0.92em; }
ul.meta li { margin: 2px 0; }
table.data-table { border-collapse: collapse; width: 100%; font-size: 0.86em; margin-top: 10px; }
table.data-table th, table.data-table td { border: 1px solid #d8dde3; padding: 5px 8px; text-align: left; }
table.data-table th { background: #e9eef6; }
table.data-table tr:nth-child(even) td { background: #f7f8fa; }
img.chart { max-width: 100%; display: block; margin: 14px 0; cursor: zoom-in; }
div.sample-card { border: 1px solid #d8dde3; border-radius: 8px; padding: 12px 16px;
                  margin: 16px 0; background: #ffffff; }
div.sample-card h3 { margin: 0 0 8px 0; font-size: 1.0em; }
img.sample-image { max-width: 100%; height: auto; display: block; cursor: zoom-in; }
#lb { display:none; position:fixed; top:0; left:0; width:100%; height:100%;
      background:rgba(0,0,0,0.88); z-index:9999; align-items:center;
      justify-content:center; cursor:zoom-out; }
#lb-img { max-width:95vw; max-height:95vh; object-fit:contain;
          box-shadow:0 4px 32px rgba(0,0,0,0.7); }
p.sample-metrics { font-size: 0.82em; color: #4a525c; margin: 8px 0 0 0; }
ul.notes { font-size: 0.9em; color: #3b4149; }
"""


@dataclass
class EvalArtifact:
    run_name: str
    display_name: str
    run_path: Path
    checkpoint_path: Path
    prediction_path: Path
    errors_path: Optional[Path]
    metric_errors_path: Optional[Path]
    prefix: str
    method_tag: str
    eval_split: str
    config: Dict[str, Any] = field(default_factory=dict)
    prediction_shape: Tuple[int, ...] = field(default_factory=tuple)
    sample_count: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    metrics_metric: Dict[str, float] = field(default_factory=dict)
    weight_number: int = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HTML report comparing evaluated runs in outs/, side by side."
    )
    parser.add_argument("--outs", type=Path, default=Path("outs"), help="Runs directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outs/eval_report.html"),
        help="Output HTML report path (single self-contained file).",
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
        help="Evaluation split to compare, e.g. eigen_benchmark. Defaults to the most common split found.",
    )
    parser.add_argument(
        "--checkpoints",
        default="latest",
        help=(
            "Which checkpoints to include. 'latest' (default): best (highest) per run. "
            "'all': every checkpoint. Or a comma-separated list of epoch numbers, e.g. '3,4'."
        ),
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
        "--num-samples",
        type=int,
        default=40,
        help="Number of evaluation images to visualize side by side (at least this many when available).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1,
        help="Seed used for random sample selection.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Comma-separated subset of metrics to show in tables and charts, e.g. 'abs_rel,sq_rel'. "
            "Defaults to all seven metrics. Shorthand: 'abs'→abs_rel, 'sq'/'seq'→sq_rel, "
            "'rmse_log'→rmse_log, 'a1'/'a2'/'a3' as-is."
        ),
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

            metric_errors_path = prediction_path.with_name(prefix + "_metric_errors.npy")
            if not metric_errors_path.exists():
                metric_errors_path = None

            prediction_shape, sample_count = load_prediction_shape(prediction_path)
            checkpoint_path = checkpoint_path_for_prediction(prediction_path)
            weight_number = weight_number_from_path(prediction_path)
            display_name = run_path.name
            if args.checkpoints != "latest":
                display_name = f"{run_path.name}/{checkpoint_path.name}"

            artifacts.append(
                EvalArtifact(
                    run_name=run_path.name,
                    display_name=display_name,
                    run_path=run_path,
                    checkpoint_path=checkpoint_path,
                    prediction_path=prediction_path,
                    errors_path=errors_path,
                    metric_errors_path=metric_errors_path,
                    prefix=prefix,
                    method_tag=method_tag,
                    eval_split=eval_split,
                    config=config,
                    prediction_shape=prediction_shape,
                    sample_count=sample_count,
                    metrics=load_eval_metrics(errors_path),
                    metrics_metric=load_eval_metrics(metric_errors_path),
                    weight_number=weight_number,
                )
            )

    checkpoints_arg = args.checkpoints
    selected_weights: Optional[set] = None
    if checkpoints_arg not in ("latest", "all"):
        try:
            selected_weights = {int(n.strip()) for n in checkpoints_arg.split(",") if n.strip()}
        except ValueError:
            raise SystemExit(f"--checkpoints must be 'latest', 'all', or comma-separated integers, got: {checkpoints_arg!r}")

    if selected_weights is not None:
        artifacts = [a for a in artifacts if a.weight_number in selected_weights]
    elif checkpoints_arg == "latest":
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


def most_common_value(values: Iterable[str]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    return sorted(key for key, count in counts.items() if count == best)[0]


def select_comparison_group(artifacts: Sequence[EvalArtifact], args: argparse.Namespace) -> Tuple[List[EvalArtifact], str, str]:
    eval_split = args.eval_split or most_common_value(item.eval_split for item in artifacts)
    method_tag = None if args.method_tag == "*" else args.method_tag
    if method_tag is None:
        method_tag = most_common_value(item.method_tag for item in artifacts)
    selected = sorted(
        (item for item in artifacts if item.eval_split == eval_split and item.method_tag == method_tag),
        key=lambda item: item.display_name,
    )
    return selected, str(eval_split or ""), str(method_tag or "")


def metric_sort_key(item: EvalArtifact) -> Tuple[float, str]:
    abs_rel = item.metrics.get("abs_rel")
    if abs_rel is None or not math.isfinite(abs_rel):
        return (float("inf"), item.display_name)
    return (abs_rel, item.display_name)


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


def choose_visual_samples(
    artifacts: Sequence[EvalArtifact],
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> List[int]:
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
    count = min(max(args.num_samples, 0), common_count)
    if count <= 0:
        return []
    rng = np.random.default_rng(args.random_seed)
    selected = rng.choice(common_count, size=count, replace=False)
    return sorted(int(idx) for idx in selected.tolist())


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


def dataset_root_candidates(artifacts: Sequence[EvalArtifact]) -> List[Path]:
    candidates: List[Path] = []
    for item in artifacts:
        for key in ("data_path", "dataset_path", "kitti_path", "data_root", "root_dir"):
            candidate = path_from_config_value(item.config.get(key) if item.config else None)
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
    lookup_config: Dict[str, Any],
    split_lines: Sequence[str],
) -> Optional[Path]:
    if cli_data_path is not None:
        return cli_data_path.expanduser()

    for candidate in dataset_root_candidates(artifacts):
        if path_has_split_sample(candidate, split_lines, lookup_config):
            return candidate
    return None


def find_rgb_image(
    sample_index: int,
    target_shape: Tuple[int, int],
    image_files: Sequence[Path],
    split_lines: Sequence[str],
    data_path: Optional[Path],
    lookup_config: Dict[str, Any],
) -> Optional[np.ndarray]:
    if image_files:
        if sample_index < len(image_files):
            return load_image_from_path(image_files[sample_index], target_shape)
        key = str(sample_index)
        for path in image_files:
            if key in path.stem:
                return load_image_from_path(path, target_shape)

    if data_path is None:
        config_data_path = lookup_config.get("data_path")
        data_path = Path(config_data_path).expanduser() if config_data_path else None
    if data_path is None or sample_index >= len(split_lines):
        return None

    for candidate in candidate_kitti_image_paths(data_path, split_lines[sample_index], lookup_config):
        image = load_image_from_path(candidate, target_shape)
        if image is not None:
            return image
    return None


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


def fig_to_data_uri(fig: "plt.Figure", dpi: int = REPORT_DPI) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html_escape(str(value))}</th>" for value in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html_escape(str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    if not body_rows:
        body_rows.append(f'<tr><td colspan="{len(headers)}">No data found.</td></tr>')
    return (
        '<table class="data-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def build_html_document(title: str, meta_lines: Sequence[str], sections: Sequence[str]) -> str:
    meta_html = "".join(f"<li>{html_escape(line)}</li>" for line in meta_lines)
    body = "\n".join(sections)
    lightbox_html = (
        '<div id="lb"><img id="lb-img" alt="zoomed"></div>\n'
        "<script>\n"
        "(function(){\n"
        "  var lb=document.getElementById('lb');\n"
        "  var lbImg=document.getElementById('lb-img');\n"
        "  function open(src){lbImg.src=src;lb.style.display='flex';}\n"
        "  function close(){lb.style.display='none';lbImg.src='';}\n"
        "  document.querySelectorAll('img.chart,img.sample-image').forEach(function(img){\n"
        "    img.addEventListener('click',function(){open(this.src);});\n"
        "  });\n"
        "  lb.addEventListener('click',close);\n"
        "  document.addEventListener('keydown',function(e){if(e.key==='Escape')close();});\n"
        "})();\n"
        "</script>\n"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{html_escape(title)}</title>\n"
        f"<style>{REPORT_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{html_escape(title)}</h1>\n"
        f'<ul class="meta">{meta_html}</ul>\n'
        f"{body}\n"
        f"{lightbox_html}"
        "</body>\n"
        "</html>\n"
    )


def render_overview_section(artifacts: Sequence[EvalArtifact]) -> str:
    rows = []
    for item in artifacts:
        rows.append(
            [
                item.display_name,
                item.method_tag,
                item.eval_split,
                item.checkpoint_path.name,
                "x".join(str(dim) for dim in item.prediction_shape),
                item.sample_count,
                short_text(item.prediction_path.relative_to(item.run_path)),
                short_text(item.errors_path.relative_to(item.run_path) if item.errors_path else ""),
            ]
        )
    table = html_table(
        ["run", "method", "split", "checkpoint", "prediction shape", "samples", "prediction file", "errors file"],
        rows,
    )
    return f"<section><h2>Evaluated Runs ({len(artifacts)})</h2>{table}</section>"


def render_metrics_section(artifacts: Sequence[EvalArtifact], metrics: Tuple[str, ...] = METRIC_NAMES) -> str:
    sorted_items = sorted(artifacts, key=metric_sort_key)
    rows = []
    for rank, item in enumerate(sorted_items, start=1):
        row = [rank, item.display_name, item.checkpoint_path.name]
        for metric in metrics:
            row.append(fmt_float(item.metrics.get(metric)))
        rows.append(row)
    table = html_table(["rank (by abs_rel)", "run", "checkpoint", *metrics], rows)

    has_metric = any(item.metrics_metric for item in artifacts)
    metric_table_html = ""
    if has_metric:
        sorted_metric = sorted(artifacts, key=lambda item: (
            item.metrics_metric.get("abs_rel", float("inf")), item.display_name))
        metric_rows = []
        for rank, item in enumerate(sorted_metric, start=1):
            row = [rank, item.display_name, item.checkpoint_path.name]
            for metric in metrics:
                row.append(fmt_float(item.metrics_metric.get(metric)))
            metric_rows.append(row)
        metric_table_html = (
            "<h3>Metric-depth (unscaled / g2s)</h3>"
            + html_table(["rank (by abs_rel)", "run", "checkpoint", *metrics], metric_rows)
        )

    return (
        f"<section><h2>Evaluation Metrics</h2>"
        f"<h3>Scaled with ground-truth median</h3>{table}"
        f"{metric_table_html}</section>"
    )


def _bar_chart_images(artifacts: Sequence[EvalArtifact], metrics: Tuple[str, ...], get_val) -> List[str]:
    metrics_with_data = [m for m in metrics if any(m in get_val(item) for item in artifacts)]
    if not metrics_with_data:
        return []
    images = []
    for start in range(0, len(metrics_with_data), 4):
        chunk = metrics_with_data[start : start + 4]
        fig, axes = plt.subplots(2, 2, figsize=CHART_FIGSIZE, squeeze=False)
        fig.patch.set_facecolor("white")
        for ax, metric in zip(axes.ravel(), chunk):
            values = [get_val(item).get(metric, np.nan) for item in artifacts]
            names = [item.display_name for item in artifacts]
            color = "#5577aa" if metric in LOWER_IS_BETTER else "#5f9f6e"
            ax.barh(names, values, color=color)
            ax.set_title(f"{metric} ({'lower' if metric in LOWER_IS_BETTER else 'higher'} is better)", fontsize=10)
            ax.grid(axis="x", linewidth=0.4, alpha=0.35)
            ax.tick_params(axis="both", labelsize=7)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        fig.tight_layout()
        images.append(fig_to_data_uri(fig))
        plt.close(fig)
    return images


def render_metric_bar_section(artifacts: Sequence[EvalArtifact], metrics: Tuple[str, ...] = METRIC_NAMES) -> str:
    images = _bar_chart_images(artifacts, metrics, lambda item: item.metrics)
    if not images:
        return ""
    img_tags = "".join(f'<img class="chart" src="{uri}" alt="metric comparison chart">' for uri in images)

    metric_html = ""
    if any(item.metrics_metric for item in artifacts):
        metric_images = _bar_chart_images(artifacts, metrics, lambda item: item.metrics_metric)
        if metric_images:
            metric_img_tags = "".join(
                f'<img class="chart" src="{uri}" alt="metric-depth comparison chart">' for uri in metric_images
            )
            metric_html = f"<h3>Metric-depth (unscaled / g2s)</h3>{metric_img_tags}"

    return (
        f"<section><h2>Metric Comparison Charts</h2>"
        f"<h3>Scaled with ground-truth median</h3>{img_tags}"
        f"{metric_html}</section>"
    )


def render_depth_sample_figure(
    sample_index: int,
    artifacts: Sequence[EvalArtifact],
    arrays: Dict[str, np.ndarray],
    image_files: Sequence[Path],
    split_lines: Sequence[str],
    lookup_config: Dict[str, Any],
    args: argparse.Namespace,
) -> Optional[Tuple[str, List[Tuple[str, Dict[str, float]]]]]:
    valid_items: List[Tuple[EvalArtifact, np.ndarray]] = []
    target_shape: Optional[Tuple[int, int]] = None
    for item in artifacts:
        depth = load_sample_depth(item, sample_index, arrays, args)
        if depth is None:
            continue
        if target_shape is None:
            target_shape = depth.shape
        else:
            depth = resize_2d(depth, target_shape)
        valid_items.append((item, depth))
    if not valid_items or target_shape is None:
        return None

    rgb = find_rgb_image(sample_index, target_shape, image_files, split_lines, args.data_path, lookup_config)
    depth_limits = robust_limits([depth for _, depth in valid_items])

    panel_w, panel_h = SAMPLE_PANEL_SIZE
    cols = len(valid_items) + (1 if rgb is not None else 0)
    fig, axes = plt.subplots(1, cols, figsize=(panel_w * cols, panel_h), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("white")

    col = 0
    if rgb is not None:
        axes[col].imshow(rgb, interpolation="lanczos")
        axes[col].set_title("RGB", fontsize=9)
        axes[col].axis("off")
        col += 1

    legend_rows: List[Tuple[str, Dict[str, float]]] = []
    for item, depth in valid_items:
        axes[col].imshow(depth, cmap="plasma", vmin=depth_limits[0], vmax=depth_limits[1], interpolation="lanczos")
        axes[col].set_title(item.display_name, fontsize=8.6)
        axes[col].axis("off")
        legend_rows.append((item.display_name, sample_error_metrics(item, sample_index)))
        col += 1

    fig.suptitle(f"Sample {sample_index}", fontsize=12, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0.01, 0.0, 0.99, 0.90))
    data_uri = fig_to_data_uri(fig)
    plt.close(fig)
    return data_uri, legend_rows


def render_depth_samples_section(
    artifacts: Sequence[EvalArtifact],
    arrays: Dict[str, np.ndarray],
    selected_samples: Sequence[int],
    image_files: Sequence[Path],
    split_lines: Sequence[str],
    lookup_config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[str, int]:
    empty = (
        "<section><h2>Depth Comparisons</h2>"
        "<p>No comparable samples were found across the evaluated runs.</p></section>"
    )
    if not selected_samples:
        return empty, 0

    cards = []
    for sample_index in selected_samples:
        result = render_depth_sample_figure(sample_index, artifacts, arrays, image_files, split_lines, lookup_config, args)
        if result is None:
            continue
        data_uri, legend_rows = result

        metrics_bits = []
        for name, metrics in legend_rows:
            if not metrics:
                continue
            metrics_bits.append(
                f"<strong>{html_escape(name)}</strong>: "
                f"abs_rel {fmt_float(metrics.get('abs_rel'))}, "
                f"rmse {fmt_float(metrics.get('rmse'))}, "
                f"a1 {fmt_float(metrics.get('a1'))}"
            )
        legend_html = ""
        if metrics_bits:
            legend_html = '<p class="sample-metrics">' + " &nbsp;|&nbsp; ".join(metrics_bits) + "</p>"

        cards.append(
            '<div class="sample-card">'
            f"<h3>Sample {sample_index}</h3>"
            f'<img class="sample-image" src="{data_uri}" alt="depth comparison for sample {sample_index}">'
            f"{legend_html}"
            "</div>"
        )

    if not cards:
        return empty, 0

    heading = (
        f"Depth Comparisons &mdash; {len(cards)} sample(s), every run shown side by side "
        "(shared color scale per sample, plasma colormap)"
    )
    return f"<section><h2>{heading}</h2>{''.join(cards)}</section>", len(cards)


def render_notes_section(artifacts: Sequence[EvalArtifact], rendered_count: int, args: argparse.Namespace) -> str:
    missing_errors = [item.display_name for item in artifacts if not item.metrics]
    notes = [
        "TensorBoard event files are not read; metrics come only from saved evaluation errors files such as multi_eigen_benchmark_errors.npy.",
        "Metric columns are exactly: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3.",
        "Depth panels are built from saved split arrays such as multi_eigen_benchmark_split.npy.",
        f"By default *_split.npy is treated as disparity and visualized as depth = 1 / disparity, clipped to [{args.min_depth:g}, {args.max_depth:g}] meters.",
        "Within each sample, every run's depth map shares the same color scale (2nd-98th percentile across runs) for a fair side-by-side comparison.",
        f"Visual samples were selected randomly with seed {args.random_seed}; {rendered_count} of {args.num_samples} requested sample(s) are shown.",
    ]
    if rendered_count < args.num_samples:
        notes.append(
            "Fewer samples were shown than requested because the smallest evaluated run "
            "has fewer overlapping evaluation samples (or RGB/depth could not be loaded for some indices)."
        )
    if args.prediction_kind == "auto":
        notes.append("--prediction-kind auto treats *_split.npy files as disparity unless the filename clearly says depth.")
    if missing_errors:
        notes.append("Missing eval errors for: " + ", ".join(missing_errors))
    if args.image_dir is None and args.data_path is None:
        notes.append("Original RGB images are optional. Pass --image-dir or --data-path if automatic lookup cannot find them.")

    items = "".join(f"<li>{html_escape(note)}</li>" for note in notes)
    return f'<section><h2>Notes</h2><ul class="notes">{items}</ul></section>'


_METRIC_SHORTHANDS: Dict[str, str] = {
    "abs": "abs_rel",
    "sq": "sq_rel",
    "seq": "sq_rel",
    "rmse_log": "rmse_log",
}


def resolve_metrics(raw: Optional[str]) -> Tuple[str, ...]:
    if not raw:
        return METRIC_NAMES
    resolved = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        name = _METRIC_SHORTHANDS.get(token, token)
        if name not in METRIC_NAMES:
            raise SystemExit(f"Unknown metric {token!r}. Valid names: {', '.join(METRIC_NAMES)}")
        if name not in resolved:
            resolved.append(name)
    return tuple(resolved)


def main() -> None:
    args = parse_args()
    args.metrics = resolve_metrics(args.metrics)
    artifacts = discover_artifacts(args)
    if not artifacts:
        raise SystemExit(
            f"No evaluated prediction files matching method={args.method_tag!r} "
            f"and split={args.eval_split!r} were found under {args.outs}"
        )

    artifacts, eval_split, method_tag = select_comparison_group(artifacts, args)
    if not artifacts:
        raise SystemExit(
            f"No evaluated prediction files matching method={method_tag!r} and split={eval_split!r} "
            f"were found under {args.outs}"
        )

    lookup_config: Dict[str, Any] = {}
    for item in artifacts:
        lookup_config.update(item.config)

    image_files = sorted_image_files(args.image_dir)
    split_lines = load_split_lines(eval_split)
    args.data_path = resolve_data_path(args.data_path, artifacts, lookup_config, split_lines)
    arrays = {item.display_name: load_prediction_array(item.prediction_path) for item in artifacts}
    selected_samples = choose_visual_samples(artifacts, arrays, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    created_at = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        render_overview_section(artifacts),
        render_metrics_section(artifacts, args.metrics),
        render_metric_bar_section(artifacts, args.metrics),
    ]
    sample_section, rendered_count = render_depth_samples_section(
        artifacts, arrays, selected_samples, image_files, split_lines, lookup_config, args
    )
    sections.append(sample_section)
    sections.append(render_notes_section(artifacts, rendered_count, args))

    meta_lines = [
        f"Created: {created_at}",
        f"Runs directory: {args.outs}",
        f"Method tag: {method_tag}",
        f"Evaluation split: {eval_split}",
        f"Runs included ({len(artifacts)}): " + ", ".join(item.display_name for item in artifacts),
        f"Visual samples: random, seed {args.random_seed}, requested {args.num_samples}, shown {rendered_count}",
        f"RGB data path: {args.data_path if args.data_path is not None else 'not found'}",
        f"Output report: {args.output}",
    ]
    html_doc = build_html_document("ManyDepth Evaluation Report", meta_lines, sections)
    args.output.write_text(html_doc, encoding="utf-8")

    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
