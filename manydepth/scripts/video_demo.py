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

def colorize_depth(depth, min_depth=0.1, max_depth=80):
    """Convert depth map to colorized visualization"""
    # Normalize depth to 0-1 range  
    depth_norm = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Apply colormap (using matplotlib's plasma colormap approximation)
    colormap = cv2.COLORMAP_PLASMA
    depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), colormap)
    
    return depth_colored

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
                     min_depth=0.1, max_depth=80, num_matching_frames=2):
    """Create a video demo with original images on top and depth predictions on bottom"""
    
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
    
    print(f"Found {len(image_files)} images")
    print(f"-> Computing predictions with size {HEIGHT}x{WIDTH}")
    print(f"-> Using {num_matching_frames} matching frames")
    
    # Setup video writer with better codec options
    combined_height = HEIGHT * 2  # Original image + depth map
    
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
    
    print(f"-> Creating video with resolution {WIDTH}x{combined_height} at {fps} fps")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (WIDTH, combined_height))
    
    # Check if VideoWriter was initialized successfully
    if not video_writer.isOpened():
        print("Failed to open video writer with H264, trying XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = os.path.splitext(output_video)[0] + '.avi'
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (WIDTH, combined_height))
        
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
            
            # Convert to depth
            output = output.sigmoid()
            #pred_disp, pred_depth = disp_to_depth(output, min_depth, max_depth)
            
            # Convert to numpy
            depth_map = output.cpu().squeeze().numpy()
            
            # Load original image for display
            original_image = cv2.imread(image_path)
            original_image = cv2.resize(original_image, (WIDTH, HEIGHT))
            
            # Colorize depth map
            depth_colored = colorize_depth(depth_map, min_depth, max_depth)
            
            # Combine images vertically (original on top, depth on bottom)
            combined_frame = np.vstack([original_image, depth_colored])
            
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
    parser = argparse.ArgumentParser(description='Create video demo with depth estimation - student mode, multi-frame, no poses')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to folder containing input images')
    parser.add_argument('--weights_folder', type=str, required=True,
                        help='Path to folder containing model weights (encoder.pth and depth.pth)')
    parser.add_argument('--output_video', type=str, default='depth_demo.mp4',
                        help='Output video path')
    parser.add_argument('--depth_anything_encoder', type=str, 
                        choices=["vits", "vitb", "vitl", "vitg"], default="vits",
                        help='Depth Anything encoder variant')
    parser.add_argument('--height', type=int, default=288,
                        help='Input image height')
    parser.add_argument('--width', type=int, default=512,
                        help='Input image width')
    parser.add_argument('--fps', type=int, default=15,
                        help='Output video frame rate')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Minimum depth for visualization')
    parser.add_argument('--max_depth', type=float, default=80,
                        help='Maximum depth for visualization')
    parser.add_argument('--num_matching_frames', type=int, default=1,
                        help='Number of previous frames to use for matching')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not os.path.exists(args.image_folder):
        raise ValueError(f"Image folder not found: {args.image_folder}")
    
    if not os.path.exists(args.weights_folder):
        raise ValueError(f"Weights folder not found: {args.weights_folder}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Mode: Student (multi-frame, no poses)")
    print(f"Encoder: {args.depth_anything_encoder}")
    print(f"Matching frames: {args.num_matching_frames}")
    
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
        args.num_matching_frames
    )

if __name__ == "__main__":
    main() 