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
    return image

def setup_models(weights_folder, depth_anything_encoder, height, width, device):
    """Setup encoder and decoder models - student mode only, no poses"""
    
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
        print(f"Using model dimensions: {HEIGHT}x{WIDTH}")
    except KeyError:
        print('No "height" or "width" keys found in the encoder state_dict, using provided values!')
        HEIGHT, WIDTH = height, width

    # Setup models - student mode
    encoder = networks.ManyDepthAnythingEncoder(encoder_name=depth_anything_encoder)
    depth_decoder = networks.ManyDepthAnythingDecoder(
        matching_height=HEIGHT // 14, matching_width=WIDTH // 14)

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

def predict_depth_student(encoder, depth_decoder, input_color, lookup_frames):
    """Student mode depth prediction - multi-frame without poses"""
    
    features, lookup_features = encoder(input_color, lookup_frames)
    patch_h, patch_w = input_color.shape[-2] // 14, input_color.shape[-1] // 14
    output, _ = depth_decoder(features, lookup_features, patch_h, patch_w)
    
    return output

def depth_to_pointcloud(depth_map, image, height, width, min_depth=0.1, max_depth=80):
    """Convert depth map and image to colored point cloud"""
    
    # Camera intrinsics (assuming reasonable FOV)
    focal_length_x_normalized = 876.02 / 1920  
    focal_length_y_normalized = 858.84 / 1080  
    focal_length_x = focal_length_x_normalized * width
    focal_length_y = focal_length_y_normalized * height
    cx, cy = width / 2, height / 2
    
    # Create coordinate grids
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    # Convert to normalized coordinates and create 3D points
    x = (i - cx) / focal_length_x
    y = (j - cy) / focal_length_y
    z = depth_map
    
    # Create 3D points
    x_world = x * z
    y_world = y * z
    z_world = z
    
    # Filter valid points
    valid_mask = (z > min_depth) & (z < max_depth)
    
    # Get valid points and colors
    points_3d = np.stack([x_world[valid_mask], -y_world[valid_mask], z_world[valid_mask]], axis=1)
    colors = image[valid_mask] / 255.0  # Normalize colors to [0,1]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def main():
    parser = argparse.ArgumentParser(description='Run inference and save point cloud from depth estimation - student mode, multi-frame, no poses')
    parser.add_argument('--target_image', type=str, default="kitti_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png",
                        help='Path to target image (main image for depth prediction)')
    parser.add_argument('--lookup_frame', type=str, default="kitti_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png",
                        help='Path to lookup frame image for matching')
    parser.add_argument('--output_path', type=str, default="output_pointclouds/pointcloud_0.ply",
                        help='Output path for the point cloud PLY file')
    parser.add_argument('--weights_folder', type=str, default="outs/kitti/base/mdp/models/weights_4",
                        help='Path to folder containing model weights (encoder.pth and depth.pth)')
    parser.add_argument('--depth_anything_encoder', type=str, 
                        choices=["vits", "vitb", "vitl", "vitg"], default="vits",
                        help='Depth Anything encoder variant')
    parser.add_argument('--height', type=int, default=182,
                        help='Input image height')
    parser.add_argument('--width', type=int, default=630,
                        help='Input image width')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Minimum depth for visualization')
    parser.add_argument('--max_depth', type=float, default=80,
                        help='Maximum depth for visualization')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not os.path.exists(args.target_image):
        raise ValueError(f"Target image not found: {args.target_image}")
    
    if not os.path.exists(args.lookup_frame):
        raise ValueError(f"Lookup frame not found: {args.lookup_frame}")
    
    if not os.path.exists(args.weights_folder):
        raise ValueError(f"Weights folder not found: {args.weights_folder}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Mode: Student (multi-frame, no poses)")
    print(f"Encoder: {args.depth_anything_encoder}")
    print(f"Target image: {args.target_image}")
    print(f"Lookup frame: {args.lookup_frame}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup models
    encoder, depth_decoder, HEIGHT, WIDTH = setup_models(
        args.weights_folder, args.depth_anything_encoder, args.height, args.width, device)
    
    print(f"-> Computing predictions with size {HEIGHT}x{WIDTH}")
    
    with torch.no_grad():
        # Load target frame
        target_color = load_image(args.target_image, HEIGHT, WIDTH)
        target_color = target_color.to(device)
        
        # Load lookup frame
        lookup_frame = load_image(args.lookup_frame, HEIGHT, WIDTH)
        
        # Stack lookup frame: batch x frames x 3 x h x w (single frame)
        lookup_frames = lookup_frame.unsqueeze(1).to(device)
        
        # Predict depth (student mode, multi-frame, no poses)
        output = predict_depth_student(encoder, depth_decoder, target_color, lookup_frames)
        
        # Convert to depth and disparity
        output = output.sigmoid()
        pred_disp, pred_depth = disp_to_depth(output, args.min_depth, args.max_depth)
        
        # Convert disparity and depth to numpy
        disp_map = pred_disp.cpu().squeeze().numpy()
        depth_map = pred_depth.cpu().squeeze().numpy()
        
        # Load original target image for colors
        original_image_full = cv2.imread(args.target_image)  # Keep full resolution
        original_height, original_width = original_image_full.shape[:2]
        
        # Resize depth map back to original resolution
        depth_map_original = cv2.resize(depth_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB for point cloud colors
        original_image_rgb = cv2.cvtColor(original_image_full, cv2.COLOR_BGR2RGB)
        
        # Create point cloud
        pcd = depth_to_pointcloud(depth_map_original, original_image_rgb, original_height, original_width, 
                                  args.min_depth, args.max_depth)
        
        # Save point cloud
        o3d.io.write_point_cloud(args.output_path, pcd)
        print(f"Saved point cloud with {len(pcd.points)} points to {args.output_path}")

if __name__ == "__main__":
    main()