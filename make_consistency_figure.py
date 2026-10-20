#!/usr/bin/env python3
"""Render a focused qualitative ablation figure: base model vs. no consistency loss.

For a few KITTI Eigen test samples (rows), lays out the input RGB next to the
predicted depth of the full model and the variant trained without the teacher
SSI consistency (distillation) loss (columns), sharing a per-sample color scale
so the effect of the consistency loss on fine structure can be compared directly.

Output: paper_extracted/imgs/consistency_visual.png
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/manydepth_matplotlib")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUTS = Path("outs")
SPLIT = "eigen_benchmark"
DATA_PATH = Path("/work/gn21/share/datasets/kitti_data")
MIN_DEPTH, MAX_DEPTH = 1e-3, 80.0

# (run dir, column label) in display order. First is the full model.
RUNS = [
    ("mdp", "Full (base)"),
    ("mdp_no_consistency_loss", "w/o consistency loss"),
]
WEIGHTS = "weights_4"

# Hand-picked sample indices (0..651) that show informative thin structure.
SAMPLES = [12, 110, 244, 401]
OUT = Path("paper_extracted/imgs/consistency_visual.png")


def load_pred(run: str) -> np.ndarray:
    path = OUTS / run / "models" / WEIGHTS / f"multi_{SPLIT}_split.npy"
    return np.load(path, mmap_mode="r")


def clean_disp(disp: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(disp, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def split_lines() -> list[str]:
    return [l.strip() for l in (Path("splits") / SPLIT / "test_files.txt").read_text().splitlines() if l.strip()]


def load_rgb(line: str, target_hw: tuple[int, int]) -> np.ndarray | None:
    parts = line.split()
    folder, frame = parts[0], parts[1]
    side = parts[2] if len(parts) > 2 else "l"
    cam = "image_02" if side in {"l", "2", "02"} else "image_03"
    for ext in (".png", ".jpg"):
        p = DATA_PATH / folder / cam / "data" / f"{int(frame):010d}{ext}"
        if p.exists():
            with Image.open(p) as im:
                im = im.convert("RGB").resize((target_hw[1], target_hw[0]), Image.BILINEAR)
                return np.asarray(im)
    return None


def main() -> None:
    preds = {run: load_pred(run) for run, _ in RUNS}
    lines = split_lines()

    ncols = 1 + len(RUNS)
    nrows = len(SAMPLES)
    panel_w = 3.6
    sample0 = clean_disp(np.squeeze(preds[RUNS[0][0]][SAMPLES[0]]))
    panel_h = panel_w * (sample0.shape[0] / sample0.shape[1])  # match depth-map aspect ratio
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows + 0.35), squeeze=False
    )
    fig.patch.set_facecolor("white")

    for r, idx in enumerate(SAMPLES):
        disps = {run: clean_disp(np.squeeze(preds[run][idx])) for run, _ in RUNS}
        target_hw = next(iter(disps.values())).shape
        stacked = np.concatenate([d[np.isfinite(d)].ravel() for d in disps.values()])
        vmin, vmax = np.percentile(stacked, [2, 98])

        rgb = load_rgb(lines[idx], target_hw)
        ax = axes[r][0]
        if rgb is not None:
            ax.imshow(rgb, interpolation="lanczos")
        ax.axis("off")
        if r == 0:
            ax.set_title("RGB", fontsize=12)

        for c, (run, label) in enumerate(RUNS, start=1):
            ax = axes[r][c]
            ax.imshow(disps[run], cmap="magma", vmin=vmin, vmax=vmax, interpolation="lanczos")
            ax.axis("off")
            if r == 0:
                ax.set_title(label, fontsize=12)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005, wspace=0.03, hspace=0.04)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
