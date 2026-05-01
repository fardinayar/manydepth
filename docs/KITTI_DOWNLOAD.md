# KITTI data: what to download and where

The missing file  
`.../2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png`  
comes from the **KITTI depth completion** ground truth, not from the standard KITTI raw sync data.

---

## If you use **eigen** split (default)

- You only need **KITTI raw** (images + velodyne).
- **Where:** Follow [Monodepth2](https://github.com/nianticlabs/monodepth2) or use the URLs in  
  `splits/kitti_archives_to_download.txt`.
- **What:** Raw sync zips (e.g. `2011_09_26_drive_0002_sync.zip`) from  
  https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/  
  Unzip so you have e.g. `{data_path}/2011_09_26/2011_09_26_drive_0002_sync/image_02/`,  
  `velodyne_points/data/`, etc.
- Ground truth for evaluation is built from **velodyne** via `export_gt_depth.py --split eigen` (no extra download).

---

## If you use **eigen_benchmark** split

- You need the **depth completion** ground truth (the `proj_depth/groundtruth/image_02/` PNGs).
- **Where:** Official KITTI depth completion benchmark (login required):  
  **https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion**
- **What to download:**
  1. **Annotated depth maps dataset (14 GB)**  
     → Unzip into your KITTI root so you get  
     `{data_path}/2011_09_26/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png`  
     and equivalent for all drives/frames used in `splits/eigen_benchmark/test_files.txt`.
  2. Optionally: **Projected raw LiDAR scans (5 GB)** and **validation/test sets (2 GB)** if you need the full benchmark.
- You must have the same **raw sync** structure (e.g. from `kitti_archives_to_download.txt`) so that paths under `proj_depth/groundtruth/` sit next to `image_02/`, `velodyne_points/`, etc.

**Summary:** For eigen_benchmark, register at the link above and download the **“Download annotated depth maps data set (14 GB)”** file, then unzip it into your existing KITTI data root (same place as `2011_09_26_drive_0002_sync`, etc.).
