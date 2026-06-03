# ManyDepth scripts

Shared utilities live in `utils_scripts.py` (model loading, image preprocessing, depth prediction, point cloud conversion). Each script is described below.

---

## save_pointcloud.py

**Purpose:** Run depth inference and export depth maps, disparity maps, and/or colored 3D point clouds (PLY).

- Uses the **student (single-frame, LoRA-finetuned)** model: depth is predicted from a single target image.
- Single-image mode: `--target_image` path.
- Folder mode: `--input_folder`; each image is processed independently.
- Optional camera intrinsics (`--fx`, `--fy`) to generate PLY point clouds. Outputs can be depth PNG, disparity NPY, and PLY.

**Example:**
```bash
python manydepth/scripts/save_pointcloud.py \
  --target_image path/to/frame1.png \
  --weights_folder path/to/run/models/weights_4 \
  --output_dir out_ply \
  --fx 700 --fy 700
```

---

## save_pointcloud_dual.py

**Purpose:** Same as `save_pointcloud.py` but supports both **teacher (frozen monocular)** and **student (single-frame, LoRA-finetuned)** inference, and a flexible API (e.g. numpy arrays as input).

- Teacher mode: single-frame depth from the frozen Depth Anything teacher.
- Student mode: single-frame depth from the LoRA-finetuned model.
- Accepts image paths or in-memory RGB arrays. Returns depth, disparity, and optional point cloud. Intended for programmatic use (APIs, pipelines) as well as CLI.

**Example:**
```bash
python manydepth/scripts/save_pointcloud_dual.py \
  --target_image frame1.png \
  --weights_folder path/to/run/models/weights_4 \
  --output_dir out_dual
# Teacher mode: add --teacher_mode
```

---

## video_demo.py

**Purpose:** Build a **video** from a folder of images: original frames, depth visualizations (colorized), and optional segmentation mask overlays with distance labels.

- Runs student (single-frame) depth on the image sequence.
- Renders depth with a colormap and can overlay Grounding DINO–style masks and per-mask median distance.
- Output is a single video file (e.g. MP4) combining input and depth (and optionally masks).

**Example:**
```bash
python manydepth/scripts/video_demo.py \
  --image_folder path/to/frames \
  --weights_folder path/to/run/models/weights_4 \
  --output_video demo.mp4 \
  --fps 15
# Optional: --mask_folder path/to/mask_output for overlays
```

---

## utils_scripts.py

**Purpose:** Shared helpers used by the above scripts. Not meant to be run directly.

- **Image:** `load_image`, `load_image_from_array`, `get_original_rgb_image` (ImageNet normalization for the depth model).
- **Models:** `setup_models(weights_folder, ..., teacher_mode=False)` (loads from run’s `config.yaml` + weights).
- **Inference:** `predict_depth_teacher`, `predict_depth_student`.
- **3D:** `depth_to_pointcloud` (depth + RGB → Open3D point cloud, camera or lidar frame).

Run scripts from the **repository root** so that `manydepth` is on the path, or ensure the parent of `manydepth` is in `PYTHONPATH`.
