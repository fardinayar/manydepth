# The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth

[Jamie Watson](https://scholar.google.com/citations?user=5pC7fw8AAAAJ&hl=en),
[Oisin Mac Aodha](https://homepages.inf.ed.ac.uk/omacaod/),
[Victor Prisacariu](https://www.robots.ox.ac.uk/~victor/),
[Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/) and
[Michael Firman](http://www.michaelfirman.co.uk) – **CVPR 2021**

[[Link to paper]](https://arxiv.org/abs/2104.14540)

We introduce ***ManyDepth***, an adaptive approach to dense depth estimation that can make use of sequence information at test time, when it is available.

* ✅ **Self-supervised**: We train from monocular video only. No depths or poses are needed at training or test time.
* ✅ Good depths from single frames; even better depths from **short sequences**.
* ✅ **Efficient**: Only one forward pass at test time. No test-time optimization needed.
* ✅ **State-of-the-art** self-supervised monocular-trained depth estimation on KITTI and CityScapes.


<p align="center">
  <a
href="https://storage.googleapis.com/niantic-lon-static/research/manydepth/manydepth_cvpr_cc.mp4">
  <img src="assets/video_thumbnail.png" alt="5 minute CVPR presentation video link" width="400">
  </a>
</p>


## Overview

Cost volumes are commonly used for estimating depths from multiple input views:

<p align="center">
  <img src="assets/cost_volume.jpg" alt="Cost volume used for aggreagting sequences of frames" width="700" />
</p>

However, cost volumes do not easily work with self-supervised training.

<p align="center">
  <img src="assets/baseline.gif" alt="Baseline: Depth from cost volume input without our contributions" width="700" />
</p>

In our paper, we:

* Introduce an adaptive cost volume to deal with unknown scene scales
* Fix problems with moving objects
* Introduce augmentations to deal with static cameras and start-of-sequence frames

These contributions enable cost volumes to work with self-supervised training:

<p align="center">
  <img src="assets/ours.gif" alt="ManyDepth: Depth from cost volume input with our contributions" width="700" />
</p>

With our contributions, short test-time sequences give better predictions than methods which predict depth from just a single frame.

<p align="center">
  <img src="assets/manydepth_vs_monodepth2.jpg" alt="ManyDepth vs Monodepth2 depths and error maps" width="700" />
</p>

## ✏️ 📄 Citation

If you find our work useful or interesting, please cite our paper:

```latex
@inproceedings{watson2021temporal,
    author = {Jamie Watson and
              Oisin Mac Aodha and
              Victor Prisacariu and
              Gabriel Brostow and
              Michael Firman},
    title = {{The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth}},
    booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```

## 📈 Results

Our **ManyDepth** method outperforms all previous methods in all subsections across most metrics, whether or not the baselines use multiple frames at test time.
See our paper for full details.

<p align="center">
  <img src="assets/results_table.png" alt="KITTI results table" width="700" />
</p>

## 👀 Reproducing Paper Results

To recreate the results from our paper, run:

```bash
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python manydepth/train.py -c configs/base.yaml \
    --data_path <your_KITTI_path> \
    --log_dir <your_save_path>  \
    --model_name <your_model_name>
```

Depending on the size of your GPU, you may need to set `--batch_size` to be lower than 12. Additionally you can train
a high resolution model by adding `--height 320 --width 1024`.
Commands in this repository are documented using script paths, not `python -m`, because the code uses local sibling imports.

For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)

To train a CityScapes model, run:

```bash
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python manydepth/train.py -c configs/base.yaml \
    --data_path <your_preprocessed_cityscapes_path> \
    --log_dir <your_save_path>  \
    --model_name <your_model_name> \
    --dataset cityscapes_preprocessed \
    --split cityscapes_preprocessed \
    --height 192 --width 512
```

This assumes you have already preprocessed the CityScapes dataset using SfMLearner's [prepare_train_data.py](https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py) script.
We used the following command:

```bash
python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir <path_to_downloaded_cityscapes_data> \
    --dataset_name cityscapes \
    --dump_root <your_preprocessed_cityscapes_path> \
    --seq_length 3 \
    --num_threads 8
```

Note that while we use the `--img_height 512` flag, the `prepare_train_data.py` script will save images which are `1024x384` as it also crops off the bottom portion of the image.
You could probably save disk space without a loss of accuracy by preprocessing with `--img_height 256 --img_width 512` (to create `512x192` images), but this isn't what we did for our experiments.

## 💾 Pretrained weights and evaluation

You can download weights for some pretrained models here:

* [KITTI MR (640x192)](https://storage.googleapis.com/niantic-lon-static/research/manydepth/models/KITTI_MR.zip)
* [KITTI HR (1024x320)](https://storage.googleapis.com/niantic-lon-static/research/manydepth/models/KITTI_HR.zip)
* [CityScapes (512x192)](https://storage.googleapis.com/niantic-lon-static/research/manydepth/models/CityScapes_MR.zip)

To evaluate a model on KITTI, run:

```bash
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python manydepth/evaluate_depth_mda.py \
    --data_path <your_KITTI_path> \
    --load_weights_folder <your_model_path> \
    --eval_split eigen
```

Make sure you have first run `export_gt_depth.py` to extract ground truth files.

And to evaluate a model on Cityscapes, run:

```bash
CUDA_VISIBLE_DEVICES=<your_desired_GPU> \
python manydepth/evaluate_depth_mda.py \
    --data_path <your_cityscapes_path> \
    --load_weights_folder <your_model_path> \
    --eval_split cityscapes
```

During evaluation, we crop and evaluate on the middle 50% of the images.

We provide ground truth depth files [HERE](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip),
which were converted from pixel disparities using intrinsics and the known baseline. Download this and unzip into `splits/cityscapes`.


If you want to evaluate a teacher network (i.e. the monocular network used for consistency loss), then add the flag `--eval_teacher`. This will
load the weights of `mono_encoder.pth` and `mono_depth.pth`.

## 🖼 Running on your own images

The inference helpers in `manydepth/scripts/` demonstrate single-frame inference.
`save_pointcloud.py` predicts depth for a single target image.
If you provide `--fx` and `--fy`, it can also export a point cloud.

Download and unzip model weights, then run:

```bash
python manydepth/scripts/save_pointcloud.py \
    --target_image assets/test_sequence_target.jpg \
    --weights_folder path/to/weights \
    --output_dir outputs/demo
```

Depth and disparity arrays will be saved under `outputs/demo`.

## 👩‍⚖️ License

Copyright © Niantic, Inc. 2021. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.
