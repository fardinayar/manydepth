#!/usr/bin/env python3
# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple, Dict, Any, Union
import open3d as o3d

# Add the parent directory to Python path so we can import from manydepth
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from networks.replace_with_lora import replace_qkv_with_mergedlinear, replace_conv_with_loraconv
import networks
from layers import disp_to_depth

def load_image(image_path, height, width):
    """Load and preprocess an image"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height), Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    # Normalize using ImageNet statistics expected by ViT/Depth Anything encoders
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image = (image - mean) / std
    return image

def load_image_from_array(image_array: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Load and preprocess a numpy RGB image (HxWx3) into model tensor size."""
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Expected image array of shape HxWx3 (RGB)")
    # Ensure uint8 or float32
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    # Normalize to [0,1] if values appear to be in [0,255]
    if image_array.max() > 1.0:
        image_array = image_array / 255.0
    resized = cv2.resize(image_array, (width, height), interpolation=cv2.INTER_LANCZOS4)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    # Normalize using ImageNet statistics expected by ViT/Depth Anything encoders
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor

def get_original_rgb_image(image_or_path: Union[str, np.ndarray]) -> np.ndarray:
    """Return original RGB image as numpy array HxWx3 (uint8).

    - If input is a path: uses cv2 to read BGR then converts to RGB.
    - If input is an array: assumes RGB and converts to uint8 if needed.
    """
    if isinstance(image_or_path, str):
        bgr = cv2.imread(image_or_path)
        if bgr is None:
            raise ValueError(f"Failed to read image from path: {image_or_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    elif isinstance(image_or_path, np.ndarray):
        img = image_or_path
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected numpy image with shape HxWx3 (RGB)")
        if img.dtype != np.uint8:
            # If float-like, scale if values appear in [0,1]
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        return img
    else:
        raise TypeError("image_or_path must be a str path or an np.ndarray")

def setup_models(weights_folder, depth_anything_encoder, height, width, device):
    """Setup encoder and decoder models - student mode only, no poses

    Returns (encoder, depth_decoder, HEIGHT, WIDTH) where HEIGHT/WIDTH prefer values
    stored in the encoder state dict under keys 'height' and 'width' if present.
    """
    
    encoder_path = os.path.join(weights_folder, "encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth.pth")
    print("-> Loading student models (multi-frame, no poses)")

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Model weights not found in {weights_folder}")

    print(f"-> Loading weights from {weights_folder}")
    
    encoder_dict = torch.load(encoder_path, map_location=device)
    
    # Get model dimensions
    try:
        HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        print(f"Using model dimensions from encoder state: {HEIGHT}x{WIDTH}")
    except KeyError:
        print('No "height" or "width" keys found in the encoder state_dict, using provided values!')
        HEIGHT, WIDTH = height, width

    # Setup models - student mode
    
    config = networks.MODEL_CONFIGS[depth_anything_encoder]
    encoder = networks.ManyDepthAnythingEncoder(encoder_name=depth_anything_encoder)
    depth_decoder = networks.ManyDepthAnythingDecoder(
        matching_height=HEIGHT // 14, matching_width=WIDTH // 14, features=config['features'], in_channels=config['in_channels'], out_channels=config['out_channels'])

    encoder = replace_qkv_with_mergedlinear(encoder, lora_dropout=0.0)
    depth_decoder = replace_conv_with_loraconv(depth_decoder, lora_dropout=0.0)

    # Load state dicts
    encoder.load_state_dict(encoder_dict, strict=False)
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    # Move to device and set to eval mode
    encoder.eval()
    depth_decoder.eval()
    encoder.to(device)
    depth_decoder.to(device)
    
    return encoder, depth_decoder, HEIGHT, WIDTH

@torch.no_grad()
def predict_depth_student(encoder, depth_decoder, input_color, lookup_frames):
    """Student mode depth prediction - multi-frame without poses"""
    
    features, lookup_features = encoder(input_color, lookup_frames)
    patch_h, patch_w = input_color.shape[-2] // 14, input_color.shape[-1] // 14
    output, _ = depth_decoder(features, lookup_features, patch_h, patch_w)
    
    return output

def depth_to_pointcloud(depth_map: np.ndarray,
                        image: np.ndarray,
                        height: int,
                        width: int,
                        fx: float,
                        fy: float,
                        cx: Optional[float] = None,
                        cy: Optional[float] = None,
                        min_depth: float = 0.1,
                        max_depth: float = 80.0,
                        coordinate_system: str = "lidar") -> o3d.geometry.PointCloud:
    """Convert depth map and image to colored point cloud using pixel-space intrinsics.

    Parameters
    - depth_map: HxW depth (meters)
    - image: HxWx3 RGB image in [0, 255]
    - fx, fy: focal lengths in pixels (unnormalized)
    - cx, cy: principal point in pixels (defaults to image center)
    - min_depth, max_depth: clamp for valid depth range
    - coordinate_system: 'camera' (X right, Y up, Z forward) or 'lidar' (X forward, Y left, Z up)
    """

    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

    x = (i - cx) / fx
    y = (j - cy) / fy
    z = depth_map

    x_world = x * z
    y_world = y * z
    z_world = z

    valid_mask = (z > min_depth) & (z < max_depth)

    # Output coordinate frame selection
    # - camera: X right, Y up, Z forward
    # - lidar:  X forward, Y left, Z up (KITTI/Velodyne)
    if coordinate_system == "lidar":
        points_3d = np.stack([
            z_world[valid_mask],        # X forward
            -x_world[valid_mask],       # Y left
            -y_world[valid_mask],       # Z up
        ], axis=1)
    else:
        points_3d = np.stack([
            x_world[valid_mask],        # X right
            y_world[valid_mask],       # Y up
            z_world[valid_mask],        # Z forward
        ], axis=1)
    colors = image[valid_mask] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def infer_depth_disparity_and_pointcloud(
    target_image: Union[str, np.ndarray],
    lookup_frame: Union[str, np.ndarray],
    weights_folder: str,
    depth_anything_encoder: str = "vits",
    height: Optional[int] = None,
    width: Optional[int] = None,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
    fx: float = None,
    fy: float = None,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
    coordinate_system: str = "lidar",
) -> Dict[str, Any]:
    """High-level API to run inference and optionally save outputs.

    - If fx and fy are provided, a point cloud is generated and optionally saved.
    - Depth and disparity are always computed and optionally saved when output_path is provided.

    Returns a dict with keys: 'disp', 'depth', 'pointcloud' (None if no fx/fy), 'paths' (if saved).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(target_image, str) and not os.path.exists(target_image):
        raise ValueError(f"Target image not found: {target_image}")
    if isinstance(lookup_frame, str) and not os.path.exists(lookup_frame):
        raise ValueError(f"Lookup frame not found: {lookup_frame}")
    if not os.path.exists(weights_folder):
        raise ValueError(f"Weights folder not found: {weights_folder}")

    encoder, depth_decoder, H_model, W_model = setup_models(
        weights_folder, depth_anything_encoder, height or 0, width or 0, device
    )

    HEIGHT, WIDTH = H_model, W_model

    with torch.no_grad():
        if isinstance(target_image, str):
            target_color = load_image(target_image, HEIGHT, WIDTH)
        else:
            target_color = load_image_from_array(target_image, HEIGHT, WIDTH)
        target_color = target_color.to(device)

        if isinstance(lookup_frame, str):
            lookup_color = load_image(lookup_frame, HEIGHT, WIDTH)
        else:
            lookup_color = load_image_from_array(lookup_frame, HEIGHT, WIDTH)
        lookup_frames = lookup_color.unsqueeze(1).to(device)

        output = predict_depth_student(encoder, depth_decoder, target_color, lookup_frames)
        output = output.sigmoid()
        pred_disp, pred_depth = disp_to_depth(output, min_depth, max_depth)

        disp_map = pred_disp.cpu().squeeze().numpy()
        depth_map = pred_depth.cpu().squeeze().numpy()

        original_image_rgb = get_original_rgb_image(target_image)
        original_height, original_width = original_image_rgb.shape[:2]
        depth_map_original = cv2.resize(depth_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        disp_map_original = cv2.resize(disp_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        pcd = None
        if fx is not None and fy is not None:
            pcd = depth_to_pointcloud(
                depth_map_original,
                original_image_rgb,
                original_height,
                original_width,
                fx=fx,
                fy=fy,
                min_depth=min_depth,
                max_depth=max_depth,
                coordinate_system=coordinate_system,
            )

        saved_paths: Dict[str, Optional[str]] = {"ply": None, "depth": None, "disp": None}
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            if isinstance(target_image, str):
                base_name = os.path.splitext(os.path.basename(target_image))[0]
            else:
                base_name = "target"
            base = os.path.join(output_dir, base_name)
            depth_path = f"{base}_depth.npy"
            disp_path = f"{base}_disp.npy"

            np.save(depth_path, depth_map_original)
            np.save(disp_path, disp_map_original)
            ply_path = None
            if pcd is not None:
                ply_path = f"{base}.ply"
                o3d.io.write_point_cloud(ply_path, pcd)

            saved_paths = {"ply": ply_path, "depth": depth_path, "disp": disp_path}

        return {
            "disp": disp_map_original,
            "depth": depth_map_original,
            "pointcloud": pcd,
            "paths": saved_paths,
            "model_input_size": (HEIGHT, WIDTH),
        }

def infer_depths_for_folder(
    input_folder: str,
    weights_folder: str,
    depth_anything_encoder: str = "vits",
    min_depth: float = 0.1,
    max_depth: float = 80.0,
    fx: float = None,
    fy: float = None,
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    coordinate_system: str = "lidar",
) -> Dict[str, Dict[str, Any]]:
    """Process a folder of images, pairing each image with its previous one as lookup.

    - Saves depth/disp for each processed image when output_dir is provided.
    - Saves point clouds only if fx and fy are provided.
    - Returns a dict mapping target image path to result dict (same schema as single API).
    """

    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder not found or not a directory: {input_folder}")
    if not os.path.exists(weights_folder):
        raise ValueError(f"Weights folder not found: {weights_folder}")

    # List and sort images
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
    image_files = sorted([p for p in all_files if os.path.splitext(p)[1].lower() in extensions])

    if len(image_files) < 2:
        raise ValueError("Need at least two images in the folder to form (target, lookup) pairs")

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    # Iterate over images, pairing each with its previous one as lookup
    for idx in range(1, len(image_files)):
        target_path = image_files[idx]
        lookup_path = image_files[idx - 1]

        result = infer_depth_disparity_and_pointcloud(
            target_image=target_path,
            lookup_frame=lookup_path,
            weights_folder=weights_folder,
            depth_anything_encoder=depth_anything_encoder,
            height=None,
            width=None,
            min_depth=min_depth,
            max_depth=max_depth,
            fx=fx,
            fy=fy,
            device=device,
            output_dir=output_dir,
            coordinate_system=coordinate_system,
        )

        results[target_path] = result

    return results

def main():
    parser = argparse.ArgumentParser(description='Run inference and save point cloud, depth, and disparity - student mode, multi-frame, no poses')
    parser.add_argument('--target_image', type=str, default="kitti_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png",
                        help='Path to target image (main image for depth prediction)')
    parser.add_argument('--lookup_frame', type=str, default="kitti_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000004.png",
                        help='Path to lookup frame image for matching')
    parser.add_argument('--input_folder', type=str, default=None,
                        help='Folder containing images; each image is paired with the previous one as lookup')
    parser.add_argument('--output_dir', type=str, default="output_pointclouds",
                        help='Directory to save outputs into')
    parser.add_argument('--weights_folder', type=str, default="outs/kitti/base/mdp/models/weights_1",
                        help='Path to folder containing model weights (encoder.pth and depth.pth)')
    parser.add_argument('--depth_anything_encoder', type=str, 
                        choices=["vits", "vitb", "vitl", "vitg"], default="vits",
                        help='Depth Anything encoder variant')
    parser.add_argument('--height', type=int, default=None,
                        help='Input image height (used only if not present in weights)')
    parser.add_argument('--width', type=int, default=None,
                        help='Input image width (used only if not present in weights)')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Minimum depth for visualization / validity')
    parser.add_argument('--max_depth', type=float, default=80,
                        help='Maximum depth for visualization / validity')
    parser.add_argument('--fx', type=float, default=None,
                        help='Camera focal length in pixels along x (unnormalized); if provided, PLY is generated')
    parser.add_argument('--fy', type=float, default=None,
                        help='Camera focal length in pixels along y (unnormalized); if provided, PLY is generated')
    parser.add_argument('--coordinate_system', type=str, choices=['camera', 'lidar'], default='camera',
                        help='Coordinate frame for saved PLY: camera (Z forward) or lidar (Z up)')
    
    args = parser.parse_args()

    if args.input_folder is not None:
        results = infer_depths_for_folder(
            input_folder=args.input_folder,
            weights_folder=args.weights_folder,
            depth_anything_encoder=args.depth_anything_encoder,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            fx=args.fx,
            fy=args.fy,
            output_dir=args.output_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            coordinate_system=args.coordinate_system,
        )

        num_items = len(results)
        num_ply = sum(1 for r in results.values() if r["paths"]["ply"] is not None)
        print(f"Processed {num_items} images from folder: {args.input_folder}")
        print(f"Saved {num_ply} point clouds (fx/fy provided: {'yes' if args.fx is not None and args.fy is not None else 'no'})")
        print(f"Outputs saved to: {args.output_dir}")
    else:
        result = infer_depth_disparity_and_pointcloud(
            target_image=args.target_image,
            lookup_frame=args.lookup_frame,
            weights_folder=args.weights_folder,
            depth_anything_encoder=args.depth_anything_encoder,
            height=args.height,
            width=args.width,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            fx=args.fx,
            fy=args.fy,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            output_dir=args.output_dir,
            coordinate_system=args.coordinate_system,
        )

        if result["paths"]["depth"]:
            print(f"Saved depth map to {result['paths']['depth']}")
        if result["paths"]["disp"]:
            print(f"Saved disparity map to {result['paths']['disp']}")
        if result["paths"]["ply"]:
            print(f"Saved point cloud with {len(result['pointcloud'].points)} points to {result['paths']['ply']}")

if __name__ == "__main__":
    main()