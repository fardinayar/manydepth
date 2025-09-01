#!/usr/bin/env python3
# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import argparse
import glob
import numpy as np
import cv2
import torch
from PIL import Image
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from pygroundsegmentation import GroundPlaneFitting  # z-axis up expected
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Add the parent directory to Python path so we can import from manydepth
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from layers import disp_to_depth
from save_pointcloud import (
    load_image,
    setup_models,
    predict_depth_student,
)


@torch.no_grad()
def infer_depth_for_sequence(
    image_files: List[str],
    encoder, depth_decoder, device: torch.device,
    height: int, width: int,
    min_depth: float, max_depth: float,
    num_matching_frames: int,
) -> List[np.ndarray]:
    depths: List[np.ndarray] = []
    print(f"-> Inferring depth for {len(image_files)} frames at {height}x{width}, matching frames: {num_matching_frames}")
    for i, path in enumerate(tqdm(image_files, desc="Depth", unit="frame")):
        print(f"[Depth] Frame {i+1}/{len(image_files)}: {os.path.basename(path)}")
        input_color = load_image(path, height, width).to(device)
        lookup_list = []
        for j in range(1, num_matching_frames + 1):
            idx = max(0, i - j)
            lookup_list.append(load_image(image_files[idx], height, width))
        lookup_frames = torch.stack(lookup_list, dim=1).to(device)

        output = predict_depth_student(encoder, depth_decoder, input_color, lookup_frames)
        output = output.sigmoid()
        _, pred_depth = disp_to_depth(output, min_depth, max_depth)
        depth_map = pred_depth.cpu().squeeze().numpy()

        # Resize depth to original image resolution
        orig = cv2.imread(path)
        oh, ow = orig.shape[:2]
        depth_resized = cv2.resize(depth_map, (ow, oh), interpolation=cv2.INTER_LINEAR)
        print(f"[Depth]   Output resized to original {oh}x{ow}")
        depths.append(depth_resized)
    return depths


def sample_ground_points_from_depth(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    roi_height_ratio: float = 0.35,
    stride: int = 8,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> np.ndarray:
    """Back-project bottom part of image to 3D and return Nx3 points (z-up).

    Coordinate frame: X right, Y forward, Z up.
    """
    h, w = depth.shape
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    y0 = int((1.0 - roi_height_ratio) * h)
    y_indices = np.arange(y0, h, stride)
    x_indices = np.arange(0, w, stride)
    xv, yv = np.meshgrid(x_indices, y_indices, indexing='xy')
    d = depth[yv, xv]
    valid = (d > min_depth) & (d < max_depth)
    xv = xv[valid].astype(np.float32)
    yv = yv[valid].astype(np.float32)
    d = d[valid].astype(np.float32)

    if d.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = (xv - cx) / fx
    y = (yv - cy) / fy
    X = x * d              # right
    Y_down = y * d         # image down
    Z_forward = d          # forward
    Z_up = -Y_down         # up as +Z
    # Provide points in (x_right, y_forward, z_up) where z is vertical up
    points = np.stack([X, Z_forward, Z_up], axis=1)
    return points


def fit_plane_ransac(points: np.ndarray, threshold: float = 0.03, max_iters: int = 200) -> Optional[Tuple[np.ndarray, float]]:
    """Deprecated: kept for reference; not used when PyGroundSegmentation is available."""
    return fit_plane_least_squares(points)


def fit_plane_least_squares(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Fit plane ax+by+cz+d=0 via least squares, return unit normal and d."""
    if points.shape[0] < 3:
        return None
    centroid = points.mean(axis=0)
    Q = points - centroid
    # SVD smallest singular vector as normal
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1, :]
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return None
    n = n / n_norm
    # ensure normal roughly points upward (positive z)
    if n[2] < 0:
        n = -n
    d = -float(n @ centroid)
    return n, d


def camera_height_from_plane(plane: Tuple[np.ndarray, float]) -> float:
    """Distance from camera origin to plane ax+by+cz+d=0 is |d| when ||n||=1."""
    n, d = plane
    return abs(d)


def constrain_normal_to_tilt(n: np.ndarray, max_tilt_deg: float) -> np.ndarray:
    """Project a unit normal onto the cone around +Z with apex angle = max_tilt_deg.

    Ensures the returned normal has tilt <= max_tilt_deg relative to +Z.
    Assumes input is approximately unit length; output is unit length.
    """
    # Ensure upward-facing normal
    if n[2] < 0:
        n = -n
    n = n.astype(np.float64)
    n /= max(np.linalg.norm(n), 1e-12)

    max_tilt_rad = np.deg2rad(max_tilt_deg)
    nz_min = float(np.cos(max_tilt_rad))

    if n[2] >= nz_min:
        return n

    # Preserve horizontal direction, set magnitude to sin(max_tilt)
    horiz_norm = float(np.linalg.norm(n[:2]))
    if horiz_norm < 1e-12:
        # Already near vertical but pointing down or numerical issues; snap to +Z within bound
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    scale = float(np.sin(max_tilt_rad) / horiz_norm)
    nx = n[0] * scale
    ny = n[1] * scale
    nz = nz_min
    n_new = np.array([nx, ny, nz], dtype=np.float64)
    n_new /= max(np.linalg.norm(n_new), 1e-12)
    return n_new


def robust_refine_plane(
    points: np.ndarray,
    max_iters: int = 3,
    k: float = 2.5,
    min_thresh: float = 0.005,
) -> Tuple[Optional[Tuple[np.ndarray, float]], np.ndarray, float]:
    """Iteratively refine plane using MAD-based inlier thresholding (per frame).

    Returns (plane, inlier_mask, final_thresh).
    """
    if points.shape[0] < 3:
        return None, np.zeros((points.shape[0],), dtype=bool), 0.0

    plane = fit_plane_least_squares(points)
    if plane is None:
        return None, np.zeros((points.shape[0],), dtype=bool), 0.0

    inlier_mask = np.ones((points.shape[0],), dtype=bool)
    final_thresh = 0.0
    for _ in range(max_iters):
        n, d = plane
        dist = np.abs(points @ n + d)
        med = np.median(dist)
        mad = np.median(np.abs(dist - med))
        scale = 1.4826 * mad if mad > 0 else (med + 1e-9)
        thresh = max(min_thresh, k * scale)
        new_inliers = dist < thresh
        if new_inliers.sum() < 3:
            break
        if np.array_equal(new_inliers, inlier_mask):
            final_thresh = float(thresh)
            break
        inlier_mask = new_inliers
        plane = fit_plane_least_squares(points[inlier_mask])
        if plane is None:
            break
        final_thresh = float(thresh)

    return plane, inlier_mask, final_thresh


def main():
    parser = argparse.ArgumentParser(description='Estimate camera height from ground plane over a folder of frames')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to folder containing input images')
    parser.add_argument('--weights_folder', type=str, required=True, help='Path to model weights (encoder.pth, depth.pth)')
    parser.add_argument('--depth_anything_encoder', type=str, choices=["vits", "vitb", "vitl", "vitg"], default="vits")
    parser.add_argument('--height', type=int, default=182, help='Model input height (ignored if stored in weights)')
    parser.add_argument('--width', type=int, default=630, help='Model input width (ignored if stored in weights)')
    parser.add_argument('--min_depth', type=float, default=0.1)
    parser.add_argument('--max_depth', type=float, default=80.0)
    parser.add_argument('--num_matching_frames', type=int, default=1, help='Number of previous frames to use for matching')
    parser.add_argument('--fx', type=float, required=True, help='Camera focal length in pixels along x')
    parser.add_argument('--fy', type=float, required=True, help='Camera focal length in pixels along y')
    parser.add_argument('--cx', type=float, default=None, help='Principal point x (pixels); default image center')
    parser.add_argument('--cy', type=float, default=None, help='Principal point y (pixels); default image center')
    parser.add_argument('--roi_height_ratio', type=float, default=0.50, help='Bottom-of-image ratio used to sample ground points')
    parser.add_argument('--stride', type=int, default=1, help='Sampling stride in pixels for ground points')
    parser.add_argument('--output_plot', type=str, default='camera_height_vs_frame.png', help='Path to save the plot image')
    parser.add_argument('--robust_iters', type=int, default=20, help='Robust plane refine iterations (per frame)')
    parser.add_argument('--robust_k', type=float, default=2.5, help='Robustness factor (k * MAD) for inliers')
    parser.add_argument('--min_inliers', type=int, default=100, help='Warn if refined inliers fewer than this')
    parser.add_argument('--max_tilt_deg', type=float, default=30.0, help='Warn if plane tilt from up exceeds this (deg)')
    parser.add_argument('--viz_first', type=int, default=0, help='Save debug 3D plots for the first N frames (0=off)')
    parser.add_argument('--viz_dir', type=str, default='debug_viz', help='Directory to save debug 3D plots')
    parser.add_argument('--viz_max_points', type=int, default=20000, help='Max points to show in debug scatter (randomly subsampled)')
    parser.add_argument('--viz_all_points', action='store_true', help='Plot all sampled points in debug viz (no subsampling)')
    parser.add_argument('--enforce_tilt_deg', type=float, default=None, help='If set, constrain plane normal tilt to <= this many degrees')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Collect images
    image_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    files: List[str] = []
    for ext in image_exts:
        files.extend(glob.glob(os.path.join(args.image_folder, ext)))
        files.extend(glob.glob(os.path.join(args.image_folder, ext.upper())))
    files.sort()
    if len(files) == 0:
        raise ValueError(f"No images found in folder: {args.image_folder}")
    print(f"Found {len(files)} images in {args.image_folder}")
    print(f"Intrinsics: fx={args.fx:.3f}, fy={args.fy:.3f}, cx={'center' if args.cx is None else args.cx}, cy={'center' if args.cy is None else args.cy}")
    print(f"Ground ROI bottom {args.roi_height_ratio*100:.1f}% of image, stride={args.stride}")
    print("Ground segmentation: PyGroundSegmentation (GPF)")
    print(f"Robust refine: iters={args.robust_iters}, k={args.robust_k}, min_inliers={args.min_inliers}, max_tilt={args.max_tilt_deg} deg")
    if args.enforce_tilt_deg is not None:
        print(f"Tilt constraint: enforcing <= {args.enforce_tilt_deg} deg relative to +Z")

    # Load models
    encoder, depth_decoder, H, W = setup_models(
        args.weights_folder, args.depth_anything_encoder, args.height, args.width, device
    )

    # Infer depths
    depths = infer_depth_for_sequence(
        files, encoder, depth_decoder, device, H, W, args.min_depth, args.max_depth, args.num_matching_frames
    )

    # Prepare ground estimator (z-up expected)
    ground_estimator = GroundPlaneFitting()

    # Estimate heights
    heights: List[float] = []
    for idx, (depth, path) in enumerate(zip(depths, files)):
        h, w = depth.shape
        cx = args.cx if args.cx is not None else w / 2.0
        cy = args.cy if args.cy is not None else h / 2.0
        pts = sample_ground_points_from_depth(
            depth, fx=args.fx, fy=args.fy, cx=cx, cy=cy,
            roi_height_ratio=args.roi_height_ratio, stride=args.stride,
            min_depth=args.min_depth, max_depth=args.max_depth,
        )
        print(f"[GroundSeg] Frame {idx+1}/{len(files)}: sampled {pts.shape[0]} candidates")
        if pts.shape[0] < 3:
            heights.append(float('nan'))
            print(f"[GroundSeg]   Not enough points for estimation")
        else:
            ground_idx = ground_estimator.estimate_ground(pts)
            ground_pts = pts[ground_idx]
            print(f"[GroundSeg]   ground points: {ground_pts.shape[0]}")
            plane, inliers_mask, thr = robust_refine_plane(
                ground_pts, max_iters=args.robust_iters, k=args.robust_k
            )
            if plane is None or inliers_mask.sum() < 3:
                heights.append(float('nan'))
                print(f"[GroundSeg]   Plane refine failed")
            else:
                n, d = plane
                inlier_count = int(inliers_mask.sum())
                height_val = camera_height_from_plane(plane)
                # Compute tilt angle relative to +Z up
                nz = float(n[2])
                nz = max(min(nz, 1.0), -1.0)
                tilt_deg = float(np.degrees(np.arccos(nz)))

                # Optionally enforce a hard tilt bound by clamping the normal
                if args.enforce_tilt_deg is not None and tilt_deg > args.enforce_tilt_deg:
                    n_constrained = constrain_normal_to_tilt(n, args.enforce_tilt_deg)
                    centroid_inliers = ground_pts[inliers_mask].mean(axis=0)
                    d = -float(n_constrained @ centroid_inliers)
                    plane = (n_constrained, d)
                    n = n_constrained
                    height_val = camera_height_from_plane(plane)
                    nz = float(n[2])
                    nz = max(min(nz, 1.0), -1.0)
                    tilt_deg = float(np.degrees(np.arccos(nz)))
                    print(f"[GroundSeg]   enforced tilt <= {args.enforce_tilt_deg:.2f} deg")

                warn_inliers = " (LOW INLIERS)" if inlier_count < args.min_inliers else ""
                warn_tilt = " (HIGH TILT)" if tilt_deg > args.max_tilt_deg else ""
                print(f"[GroundSeg]   refined inliers={inlier_count}, thr={thr:.4f}{warn_inliers}")
                print(f"[GroundSeg]   normal={n}, tilt={tilt_deg:.2f} deg{warn_tilt}, d={d:.3f}, height={height_val:.3f} m")
                heights.append(height_val)

                # Optional 3D debug visualization for first N frames
                if args.viz_first > 0 and idx < args.viz_first:
                    os.makedirs(args.viz_dir, exist_ok=True)

                    # Build boolean mask of ground over all sampled points (pts)
                    if isinstance(ground_idx, np.ndarray) and ground_idx.dtype == bool and ground_idx.shape[0] == pts.shape[0]:
                        ground_mask_all = ground_idx
                    else:
                        ground_mask_all = np.zeros((pts.shape[0],), dtype=bool)
                        ground_mask_all[np.asarray(ground_idx, dtype=int)] = True

                    # Subsample for visualization unless plotting all points
                    if args.viz_all_points:
                        pts_plot = pts
                        ground_mask_plot = ground_mask_all
                    else:
                        N_all = pts.shape[0]
                        if N_all > args.viz_max_points:
                            sel = np.random.default_rng(42).choice(N_all, size=args.viz_max_points, replace=False)
                            pts_plot = pts[sel]
                            ground_mask_plot = ground_mask_all[sel]
                        else:
                            pts_plot = pts
                            ground_mask_plot = ground_mask_all

                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')

                    # Non-ground points
                    non_ground = ~ground_mask_plot
                    if non_ground.any():
                        ax.scatter(pts_plot[non_ground, 0], pts_plot[non_ground, 1], pts_plot[non_ground, 2],
                                   c='lightgray', s=1, depthshade=False, label='non-ground')
                    # Ground points
                    if ground_mask_plot.any():
                        ax.scatter(pts_plot[ground_mask_plot, 0], pts_plot[ground_mask_plot, 1], pts_plot[ground_mask_plot, 2],
                                   c='tab:green', s=2, depthshade=False, label='ground')

                    # Plane surface
                    a, b, c = n[0], n[1], n[2]
                    if abs(c) > 1e-6:
                        x_min, x_max = np.percentile(pts_plot[:, 0], [2, 98])
                        y_min, y_max = np.percentile(pts_plot[:, 1], [2, 98])
                        Xg, Yg = np.meshgrid(np.linspace(x_min, x_max, 15), np.linspace(y_min, y_max, 15))
                        Zg = -(a * Xg + b * Yg + d) / c
                        ax.plot_surface(Xg, Yg, Zg, color='tab:blue', alpha=0.3, linewidth=0, antialiased=False)

                    ax.set_xlabel('X right (m)')
                    ax.set_ylabel('Y forward (m)')
                    ax.set_zlabel('Z up (m)')
                    ax.set_title(f'Frame {idx} — Ground segmentation and plane')
                    ax.legend(loc='upper right')
                    ax.view_init(elev=30, azim=-60)
                    out_path = os.path.join(args.viz_dir, f'frame_{idx:06d}_pc_plane.png')
                    plt.tight_layout()
                    plt.savefig(out_path, dpi=150)
                    plt.close(fig)
                    print(f"[Viz] Saved {out_path}")

    # Plot
    x = np.arange(len(heights))
    plt.figure(figsize=(10, 4))
    plt.plot(x, heights, marker='o', linewidth=1)
    plt.xlabel('Frame index')
    plt.ylabel('Camera height above ground (m)')
    plt.title('Estimated camera height over frames')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_dir = os.path.dirname(args.output_plot)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output_plot, dpi=150)
    print(f"Saved plot to: {args.output_plot}")


if __name__ == '__main__':
    main()


