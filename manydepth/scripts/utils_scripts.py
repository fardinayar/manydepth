#!/usr/bin/env python3
# Copyright Niantic 2021. Patent Pending. All rights reserved.
# Shared utilities for manydepth/scripts: model setup, image loading, depth prediction, point clouds.

import os
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Add parent so we can import from manydepth
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import TrainConfig
from layers import disp_to_depth
import networks
from networks.replace_with_lora import replace_mlp_with_lora, replace_conv_with_loraconv

try:
    import open3d as o3d
except ImportError:
    o3d = None


def load_image(image_path: str, height: int, width: int) -> torch.Tensor:
    """Load and preprocess an image from path for the depth model (ImageNet normalization)."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height), Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image = (image - mean) / std
    return image


def load_image_from_array(
    image_array: np.ndarray, height: int, width: int
) -> torch.Tensor:
    """Load and preprocess a numpy RGB image (HxWx3) into model tensor (ImageNet normalization)."""
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Expected image array of shape HxWx3 (RGB)")
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    resized = cv2.resize(image_array, (width, height), interpolation=cv2.INTER_LANCZOS4)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def get_original_rgb_image(image_or_path: Union[str, np.ndarray]) -> np.ndarray:
    """Return original RGB image as numpy array HxWx3 (uint8). Path or array input."""
    if isinstance(image_or_path, str):
        bgr = cv2.imread(image_or_path)
        if bgr is None:
            raise ValueError(f"Failed to read image from path: {image_or_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if isinstance(image_or_path, np.ndarray):
        img = image_or_path
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected numpy image with shape HxWx3 (RGB)")
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        return img
    raise TypeError("image_or_path must be a str path or an np.ndarray")


def setup_models(
    weights_folder: str,
    depth_anything_encoder: str,
    height: int,
    width: int,
    device: torch.device,
    teacher_mode: bool = False,
):
    """
    Load encoder and decoder from a run folder (config.yaml + weights).
    Returns (encoder, depth_decoder, HEIGHT, WIDTH). HEIGHT/WIDTH from state dict if present.
    """
    if teacher_mode:
        encoder_path = os.path.join(weights_folder, "mono_encoder.pth")
        decoder_path = os.path.join(weights_folder, "mono_depth.pth")
        print("-> Loading teacher models (monocular, single-frame)")
    else:
        encoder_path = os.path.join(weights_folder, "encoder.pth")
        decoder_path = os.path.join(weights_folder, "depth.pth")
        print("-> Loading student models (multi-frame, no poses)")

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Model weights not found in {weights_folder}")

    print(f"-> Loading weights from {weights_folder}")
    encoder_dict = torch.load(encoder_path, map_location=device)

    try:
        HEIGHT, WIDTH = encoder_dict["height"], encoder_dict["width"]
        print(f"Using model dimensions from encoder state: {HEIGHT}x{WIDTH}")
    except KeyError:
        print('No "height" or "width" in encoder state_dict, using provided values!')
        HEIGHT, WIDTH = height, width

    saved_cfg = TrainConfig.from_saved_run_dir(weights_folder)

    if teacher_mode:
        encoder, depth_decoder = networks.get_da_encoder_decoder(
            encoder_name=depth_anything_encoder,
            checkpoint_dir=saved_cfg.depth_anything_checkpoint_dir,
        )
    else:
        config = networks.MODEL_CONFIGS[depth_anything_encoder]
        encoder = networks.ManyDepthAnythingEncoder(
            encoder_name=depth_anything_encoder,
            checkpoint_dir=saved_cfg.depth_anything_checkpoint_dir,
        )
        depth_decoder = networks.ManyDepthAnythingDecoder(
            patch_h=HEIGHT // 14,
            patch_w=WIDTH // 14,
            features=config["features"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            temporal_fusion=not saved_cfg.no_temporal_fusion,
            num_passes=saved_cfg.num_passes,
            num_register_tokens=saved_cfg.num_register_tokens,
            fusion_neighborhood_size=saved_cfg.fusion_neighborhood_size,
            fusion_num_scales=saved_cfg.fusion_num_scales,
            fusion_independent_blocks=saved_cfg.fusion_independent_blocks,
            fusion_mode=saved_cfg.fusion_mode,
            fusion_lora_rank=saved_cfg.fusion_lora_rank,
            fusion_lora_alpha=saved_cfg.fusion_lora_alpha,
            fusion_dropout=saved_cfg.fusion_dropout,
            fusion_drop_path=saved_cfg.fusion_drop_path,
            fusion_separate_norms=saved_cfg.fusion_separate_norms,
            use_cls_scale_shift=getattr(saved_cfg, "use_cls_scale_shift", False),
        )
        if not saved_cfg.no_lora:
            encoder = replace_mlp_with_lora(
                encoder,
                r=saved_cfg.lora_rank,
                lora_alpha=saved_cfg.lora_alpha,
                lora_dropout=0.0,
            )
            depth_decoder = replace_conv_with_loraconv(
                depth_decoder,
                r=saved_cfg.lora_rank,
                lora_alpha=saved_cfg.lora_alpha,
                lora_dropout=0.0,
            )
        print(
            f"  num_passes: {saved_cfg.num_passes}, no_temporal_fusion: {saved_cfg.no_temporal_fusion}"
        )

    encoder.load_state_dict(encoder_dict, strict=False)
    decoder_state = torch.load(decoder_path, map_location=device)
    missing, unexpected = depth_decoder.load_state_dict(decoder_state, strict=False)
    if missing:
        print(f"Depth decoder missing keys during load: {missing}")
    if unexpected:
        print(f"Depth decoder unexpected keys during load: {unexpected}")
    encoder.eval()
    depth_decoder.eval()
    encoder.to(device)
    depth_decoder.to(device)
    return encoder, depth_decoder, HEIGHT, WIDTH


@torch.no_grad()
def predict_depth_teacher(encoder, depth_decoder, input_color: torch.Tensor):
    """Teacher (monocular) depth prediction."""
    features = encoder.get_intermediate_layers(
        input_color, encoder.intermediate_layer_idx, return_class_token=True
    )
    patch_h = input_color.shape[-2] // 14
    patch_w = input_color.shape[-1] // 14
    output, _ = depth_decoder(features, patch_h, patch_w)
    return output


@torch.no_grad()
def predict_depth_student(
    encoder,
    depth_decoder,
    input_color: torch.Tensor,
    lookup_frames: torch.Tensor,
):
    """Student (multi-frame) depth prediction."""
    if getattr(depth_decoder, "temporal_fusion", True) and lookup_frames.shape[1] != 1:
        raise ValueError("Student inference currently supports exactly one matching frame")
    encoder_lookup_frames = lookup_frames if getattr(depth_decoder, "temporal_fusion", True) else None
    features, lookup_features = encoder(input_color, encoder_lookup_frames)
    output, _ = depth_decoder(features, lookup_features)
    return output


def postprocess_depth_output(output: torch.Tensor, max_depth: float):
    """Convert a raw model output to disparity and depth using training/eval semantics."""
    return disp_to_depth(output.relu(), max_depth)


def depth_to_pointcloud(
    depth_map: np.ndarray,
    image: np.ndarray,
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
    coordinate_system: str = "lidar",
):
    """Convert depth map and RGB image to colored Open3D point cloud."""
    if o3d is None:
        raise ImportError("open3d is required for point cloud export. pip install open3d")
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    x = (i - cx) / fx
    y = (j - cy) / fy
    z = depth_map
    x_world = x * z
    y_world = y * z
    z_world = z
    valid_mask = (z > min_depth) & (z < max_depth)
    if coordinate_system == "lidar":
        points_3d = np.stack(
            [
                z_world[valid_mask],
                -x_world[valid_mask],
                -y_world[valid_mask],
            ],
            axis=1,
        )
    else:
        points_3d = np.stack(
            [x_world[valid_mask], y_world[valid_mask], z_world[valid_mask]], axis=1
        )
    colors = image[valid_mask] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
