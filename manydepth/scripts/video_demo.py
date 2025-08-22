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
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json

# Add the parent directory to Python path so we can import from manydepth
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from networks.replace_with_lora import replace_qkv_with_mergedlinear, replace_conv_with_loraconv
import networks
from layers import disp_to_depth, BackprojectDepth

def load_image(image_path, height, width):
    """Load and preprocess an image"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((width, height), Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def colorize_depth(depth, min_depth=0.1, max_depth=80):
    """Convert depth map to colorized visualization"""
    # Normalize depth to 0-1 range  
    depth_norm = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Apply colormap (using matplotlib's plasma colormap approximation)
    colormap = cv2.COLORMAP_PLASMA
    depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), colormap)
    
    return depth_colored

def load_masks_and_labels(mask_folder, image_name):
    """Load masks and labels from Grounding DINO output folder
    
    Supports the specific folder structure:
    - Images: 20250410_110218_599_lat53.35477480_lon-6.29399900_frame000000.jpg
    - Masks: output_mask/mask_data/mask_20250410_110218_599_lat53.npy
    - Labels: output_mask/json_data/mask_20250410_110218_599_lat53.json
    """
    base_name = os.path.splitext(image_name)[0]
    
    # Extract the truncated base name (up to lat53) from the full image name
    # Example: "20250410_110218_599_lat53.35477480_lon-6.29399900_frame000000" -> "20250410_110218_599_lat53"
    if '_lat53.' in base_name:
        truncated_base = base_name.split('_lat53.')[0] + '_lat53'
    elif '_lat53' in base_name:
        truncated_base = base_name.split('_lat53')[0] + '_lat53'
    else:
        # Fallback: try to find pattern with lat coordinates
        import re
        match = re.match(r'(.+_lat\d+)', base_name)
        if match:
            truncated_base = match.group(1)
        else:
            truncated_base = base_name
    
    mask_file = os.path.join(mask_folder, 'mask_data', f'mask_{truncated_base}.npy')
    label_file = os.path.join(mask_folder, 'json_data', f'mask_{truncated_base}.json')
    
    masks = None
    labels = []
    boxes = []
    
    # Load mask file
    if os.path.exists(mask_file):
        try:
            mask_data = np.load(mask_file)
            if mask_data.ndim == 3:  # Multiple masks (n, h, w)
                masks = mask_data
            elif mask_data.ndim == 2:  # Single segmentation map
                # Convert segmentation map to individual masks
                unique_ids = np.unique(mask_data)
                mask_list = []
                for uid in unique_ids:
                    if uid == 0:  # Skip background
                        continue
                    mask_list.append((mask_data == uid))
                if mask_list:
                    masks = np.stack(mask_list, axis=0)
        except Exception as e:
            print(f"Warning: Could not load mask file {mask_file}: {e}")
    
    # Load label file in your specific JSON format
    if os.path.exists(label_file):
        try:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                
                if 'labels' in label_data and isinstance(label_data['labels'], dict):
                    # Sort by instance_id to match mask order
                    sorted_labels = sorted(label_data['labels'].items(), 
                                         key=lambda x: int(x[0]))
                    
                    labels = []
                    boxes = []
                    
                    for instance_id, obj_data in sorted_labels:
                        class_name = obj_data.get('class_name', f'object_{instance_id}')
                        labels.append(class_name)
                        
                        # Extract bounding box [x1, y1, x2, y2]
                        x1 = obj_data.get('x1', 0)
                        y1 = obj_data.get('y1', 0)
                        x2 = obj_data.get('x2', 100)
                        y2 = obj_data.get('y2', 100)
                        boxes.append([x1, y1, x2, y2])
                        
        except Exception as e:
            print(f"Warning: Could not load label file {label_file}: {e}")
    
    # If no labels found, create default ones
    if masks is not None and len(labels) == 0:
        labels = [f'object_{i+1}' for i in range(len(masks))]
    
    return masks, labels, boxes

def calculate_mask_distance(mask, depth_map):
    """Calculate the median distance (not depth) for a mask region"""
    if mask.sum() == 0:  # Empty mask
        return 0.0
    
    # Get depth values in the masked region
    masked_depths = depth_map[mask]
    valid_depths = masked_depths[masked_depths > 0.1]  # Filter out invalid depths
    
    if len(valid_depths) == 0:
        return 0.0
    
    # Calculate median depth for stability
    median_depth = np.median(valid_depths)
    
    # Convert depth to distance (depth is already distance from camera)
    # For more accurate distance, could consider camera orientation/angle
    distance = median_depth
    
    return distance

def overlay_masks_with_distance(image, masks, labels, depth_map, alpha=0.3):
    """Overlay masks on image with distance labels"""
    if masks is None or len(masks) == 0:
        return image
    
    overlay = image.copy()
    image_height, image_width = image.shape[:2]
    depth_height, depth_width = depth_map.shape[:2]
    
    # Generate colors for each mask
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    
    for i, mask in enumerate(masks):
        if mask.sum() == 0:  # Skip empty masks
            continue
        
        # Ensure mask dimensions match image dimensions
        if mask.shape != (image_height, image_width):
            print(f"Warning: Mask {i} shape {mask.shape} doesn't match image shape {image.shape[:2]}")
            continue
            
        color = colors[i % len(colors)]
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        
        # Blend with original image
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        
        # Calculate distance for this mask (resize mask if depth_map has different dimensions)
        if mask.shape != (depth_height, depth_width):
            depth_mask = cv2.resize(mask.astype(np.uint8), (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
            depth_mask = depth_mask.astype(bool)
        else:
            depth_mask = mask
            
        distance = calculate_mask_distance(depth_mask, depth_map)
        
        # Get mask centroid for label placement
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            centroid_y = int(np.mean(y_coords))
            centroid_x = int(np.mean(x_coords))
            
            # Create label with distance
            label = labels[i] if i < len(labels) else f'object_{i}'
            distance_text = f'{label}: {distance:.1f}m'
            
            # Add text background for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            text_size = cv2.getTextSize(distance_text, font, font_scale, thickness)[0]
            
            # Ensure text coordinates are within image bounds
            text_x = max(5, min(centroid_x, image_width - text_size[0] - 10))
            text_y = max(text_size[1] + 10, min(centroid_y, image_height - 10))
            
            # Draw background rectangle
            cv2.rectangle(overlay, 
                         (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(overlay, distance_text, 
                       (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
    
    return overlay

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

def create_video_demo(image_folder, weights_folder, output_video, 
                     depth_anything_encoder="vits", height=288, width=512, fps=15,
                     min_depth=0.1, max_depth=80, num_matching_frames=2, max_frames=None, mask_folder=None):
    """Create a video demo with original images, depth predictions, and optional mask overlays with distance labels
    """
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup models
    encoder, depth_decoder, HEIGHT, WIDTH = setup_models(
        weights_folder, depth_anything_encoder, height, width, device)
    
    # Get list of image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
    
    image_files.sort()
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {image_folder}")
    
    # Apply max_frames limit if specified
    if max_frames is not None and max_frames > 0:
        image_files = image_files[:max_frames]
        print(f"Limited to {max_frames} frames")
    
    print(f"Processing {len(image_files)} images")
    print(f"-> Computing predictions with size {HEIGHT}x{WIDTH}")
    print(f"-> Using {num_matching_frames} matching frames")
    
    # Get original image dimensions from first image for display resolution
    first_image = cv2.imread(image_files[0])
    original_display_height, original_display_width = first_image.shape[:2]
    print(f"-> Displaying results at original resolution {original_display_height}x{original_display_width}")
    
    # Setup video writer with better codec options
    # Two panels: original image, depth map
    combined_width = original_display_width  # Just original and depth stacked vertically
    combined_height = original_display_height * 2  # Original image + depth map stacked vertically
    
    # Try different codecs for better compatibility
    if output_video.lower().endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif output_video.lower().endswith('.mp4'):
        # Try H.264 first, fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:
        # Default to XVID and change extension
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not output_video.lower().endswith('.avi'):
            output_video = os.path.splitext(output_video)[0] + '.avi'
            print(f"Changed output to: {output_video}")
    
    print(f"-> Creating video with resolution {combined_width}x{combined_height} at {fps} fps")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (combined_width, combined_height))
    
    # Check if VideoWriter was initialized successfully
    if not video_writer.isOpened():
        print("Failed to open video writer with H264, trying XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = os.path.splitext(output_video)[0] + '.avi'
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (combined_width, combined_height))
        
        if not video_writer.isOpened():
            raise RuntimeError("Failed to initialize video writer. Please check codec support.")
    
    print("Processing images and creating video...")
    
    with torch.no_grad():
        for i, image_path in enumerate(tqdm(image_files)):
            # Load current frame
            input_color = load_image(image_path, HEIGHT, WIDTH)
            input_color = input_color.to(device)
            
            # Create lookup frames (previous frames or repeated current frame if at start)
            lookup_frame_list = []
            for j in range(1, num_matching_frames + 1):
                lookup_idx = max(0, i - j)  # Use previous frames, or repeat first frame
                lookup_image_path = image_files[lookup_idx]
                lookup_frame = load_image(lookup_image_path, HEIGHT, WIDTH)
                lookup_frame_list.append(lookup_frame)
            
            # Stack lookup frames: batch x frames x 3 x h x w
            lookup_frames = torch.stack(lookup_frame_list, dim=1).to(device)

            
            # Predict depth (student mode, multi-frame, no poses)
            output = predict_depth_student(encoder, depth_decoder, input_color, lookup_frames)
            
            # Convert to depth and disparity
            output = output.sigmoid()
            pred_disp, pred_depth = disp_to_depth(output, min_depth, max_depth)
            
            # Convert disparity and depth to numpy
            disp_map = pred_disp.cpu().squeeze().numpy()
            depth_map = pred_depth.cpu().squeeze().numpy()
            
            # Load original image for display
            original_image_full = cv2.imread(image_path)  # Keep full resolution
            original_height, original_width = original_image_full.shape[:2]
            
            # Resize depth map back to original resolution for display
            depth_map_original = cv2.resize(depth_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            disp_map_original = cv2.resize(disp_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            
            # Load masks and prepare for processing at original resolution
            masks_original = None
            labels = []
            
            if mask_folder is not None:
                image_name = os.path.basename(image_path)
                masks_original, labels, boxes = load_masks_and_labels(mask_folder, image_name)
                if masks_original is not None:
                    # Overlay masks on the full resolution image with original resolution depth
                    original_image_full = overlay_masks_with_distance(original_image_full, masks_original, labels, depth_map_original)
            
            
            # Colorize disparity map at original resolution (disparity has different range than depth)
            depth_colored = colorize_depth(disp_map_original, disp_map_original.min(), disp_map_original.max())
            
            # Combine images: original and depth stacked vertically
            combined_frame = np.vstack([original_image_full, depth_colored])  # Stack original and depth
            
            
            # Ensure frame is in correct format (uint8)
            combined_frame = combined_frame.astype(np.uint8)
            
            # Write frame to video
            video_writer.write(combined_frame)
                
    
    # Clean up
    video_writer.release()
    cv2.destroyAllWindows()  # Clean up any OpenCV windows
    print(f"Video saved to: {output_video}")
    
    # Verify the video file was created and has content
    if os.path.exists(output_video):
        file_size = os.path.getsize(output_video)
        print(f"Video file size: {file_size / (1024*1024):.2f} MB")
        if file_size < 1024:  # Less than 1KB indicates potential problem
            print("Warning: Video file is very small, there might be an issue with encoding")
    else:
        print("Error: Video file was not created!")

def main():
    parser = argparse.ArgumentParser(description='Create video demo with depth estimation, and optional Grounding DINO mask overlays - student mode, multi-frame, no poses')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to folder containing input images')
    parser.add_argument('--weights_folder', type=str, required=True,
                        help='Path to folder containing model weights (encoder.pth and depth.pth)')
    parser.add_argument('--output_video', type=str, default='depth_demo.mp4',
                        help='Output video path')
    parser.add_argument('--depth_anything_encoder', type=str, 
                        choices=["vits", "vitb", "vitl", "vitg"], default="vits",
                        help='Depth Anything encoder variant')
    parser.add_argument('--height', type=int, default=182,
                        help='Input image height')
    parser.add_argument('--width', type=int, default=630,
                        help='Input image width')
    parser.add_argument('--fps', type=int, default=2,
                        help='Output video frame rate')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Minimum depth for visualization')
    parser.add_argument('--max_depth', type=float, default=80,
                        help='Maximum depth for visualization')
    parser.add_argument('--num_matching_frames', type=int, default=1,
                        help='Number of previous frames to use for matching')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process (default: process all frames)')
    parser.add_argument('--mask_folder', type=str, default=None,
                        help='Path to output_mask folder containing mask_data/ and json_data/ subdirectories (optional)')

    
    args = parser.parse_args()
    
    # Validate input arguments
    if not os.path.exists(args.image_folder):
        raise ValueError(f"Image folder not found: {args.image_folder}")
    
    if not os.path.exists(args.weights_folder):
        raise ValueError(f"Weights folder not found: {args.weights_folder}")
    
    if args.mask_folder and not os.path.exists(args.mask_folder):
        raise ValueError(f"Mask folder not found: {args.mask_folder}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Mode: Student (multi-frame, no poses)")
    print(f"Encoder: {args.depth_anything_encoder}")
    print(f"Matching frames: {args.num_matching_frames}")
    if args.max_frames:
        print(f"Max frames to process: {args.max_frames}")
    if args.mask_folder:
        print(f"Mask folder: {args.mask_folder}")
    
    # Run video demo
    create_video_demo(
        args.image_folder,
        args.weights_folder,
        args.output_video,
        args.depth_anything_encoder,
        args.height,
        args.width,
        args.fps,
        args.min_depth,
        args.max_depth,
        args.num_matching_frames,
        args.max_frames,
        args.mask_folder,
    )

if __name__ == "__main__":
    main() 