#!/usr/bin/env python3
"""
Depth prediction visualization script for comparing multiple methods.
Creates side-by-side comparisons of depth maps from different models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import os
import glob
from pathlib import Path
from scipy.ndimage import zoom
import cv2

def load_original_image(sample_id, target_shape=(192, 640)):
    """Load the original target image for a sample and resize to exact target dimensions."""
    image_path = f"../test_data/sample_{sample_id}/target_{sample_id}.png"
    
    if os.path.exists(image_path):
        # Load image using cv2
        image = cv2.imread(image_path)
        if image is not None:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize to exact target shape (width, height) for cv2.resize
            image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            print(f"Original image resized to: {image.shape}")
            return image
        else:
            print(f"Warning: Could not load image from {image_path}")
            return None
    else:
        print(f"Warning: Image file not found: {image_path}")
        return None

def load_disparity_data(method, sample_id):
    """Load disparity data for a specific method and sample ID."""
    if method == 'metricdepthv2':
        # MetricDepthV2 doesn't have separate disparity files, convert depth to disparity
        depth_file = f'{method}/{sample_id}.npy'
        if os.path.exists(depth_file):
            depth = np.load(depth_file)
            # Convert depth to disparity (1/depth, handling zeros)
            disparity = np.zeros_like(depth)
            disparity[depth > 0] = 1.0 / depth[depth > 0]
            return disparity
        else:
            print(f"Warning: File not found for {method}: {depth_file}")
            return None
    elif method == 'prodepth':
        file_path = f'{method}/sample_{sample_id}_multi_disparity.npy'
    else:
        file_path = f'{method}/sample_{sample_id}_{method}_disp.npy'
    
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print(f"Warning: File not found for {method}: {file_path}")
        return None

def resize_depth_map(depth_map, target_shape=(192, 640)):
    """Resize depth map to target shape using bilinear interpolation."""
    if depth_map is None:
        return None
    
    # Calculate zoom factors
    zoom_factors = (target_shape[0] / depth_map.shape[0], target_shape[1] / depth_map.shape[1])
    
    # Resize using zoom (bilinear interpolation)
    resized = zoom(depth_map, zoom_factors, order=1)
    
    return resized

def affine_align_disparity_to_reference(disparity_maps, reference_method='prodepth'):
    """Affine align all disparity maps to match the reference method's statistics."""
    # Get reference disparity map
    reference_disparity = disparity_maps.get(reference_method)
    
    if reference_disparity is None:
        print(f"Warning: Reference method '{reference_method}' not found, using global normalization")
        return normalize_disparity_unified(disparity_maps)
    
    # Get reference statistics
    ref_mean = reference_disparity.mean()
    ref_std = reference_disparity.std()
    
    print(f"Reference ({reference_method}) statistics: mean={ref_mean:.3f}, std={ref_std:.3f}")
    
    # Affine align all disparity maps to match reference statistics
    aligned_maps = {}
    for method, disparity_map in disparity_maps.items():
        if disparity_map is not None:
            if method == reference_method:
                # Reference method: just normalize to [0, 1] for visualization
                ref_min = reference_disparity.min()
                ref_max = reference_disparity.max()
                normalized = (disparity_map - ref_min) / (ref_max - ref_min)
                normalized = np.clip(normalized, 0, 1)
                aligned_maps[method] = normalized
            else:
                # Other methods: affine transform to match reference statistics
                method_mean = disparity_map.mean()
                method_std = disparity_map.std()
                
                # Avoid division by zero
                if method_std == 0:
                    print(f"Warning: {method} has zero standard deviation, using identity transform")
                    aligned = disparity_map.copy()
                else:
                    # Affine transformation: align mean and std
                    # y = (x - method_mean) * (ref_std / method_std) + ref_mean
                    aligned = (disparity_map - method_mean) * (ref_std / method_std) + ref_mean
                
                # Normalize to [0, 1] for visualization using reference range
                ref_min = reference_disparity.min()
                ref_max = reference_disparity.max()
                normalized = (aligned - ref_min) / (ref_max - ref_min)
                normalized = np.clip(normalized, 0, 1)
                aligned_maps[method] = normalized
                
                print(f"{method}: original mean={method_mean:.3f}, std={method_std:.3f} -> aligned mean={aligned.mean():.3f}, std={aligned.std():.3f}")
        else:
            aligned_maps[method] = None
    
    return aligned_maps, ref_mean, ref_std

def normalize_disparity_unified(disparity_maps):
    """Normalize all disparity maps to the same scale for fair comparison."""
    # Collect all valid disparity maps
    valid_disparities = [d for d in disparity_maps.values() if d is not None]
    
    if not valid_disparities:
        return disparity_maps
    
    # Find global min and max across all methods
    global_min = min(d.min() for d in valid_disparities)
    global_max = max(d.max() for d in valid_disparities)
    
    print(f"Global disparity range: {global_min:.3f} - {global_max:.3f}")
    
    # Normalize all disparity maps to [0, 1] using global range
    normalized_maps = {}
    for method, disparity_map in disparity_maps.items():
        if disparity_map is not None:
            # Normalize to [0, 1] using global range
            normalized = (disparity_map - global_min) / (global_max - global_min)
            normalized = np.clip(normalized, 0, 1)
            normalized_maps[method] = normalized
        else:
            normalized_maps[method] = None
    
    return normalized_maps, global_min, global_max

def create_disparity_comparison(sample_ids, output_dir='comparisons'):
    """Create disparity comparison visualizations for multiple samples."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define methods and their display names
    methods = {
        'prodepth': 'ProDepth',
        'metricdepthv2': 'MetricDepthV2', 
        'teacher': 'Depth Anything v2',
        'student': 'Ours'
    }
    
    # Use the same colormap for all methods
    unified_colormap = 'viridis'
    
    for sample_id in sample_ids:
        print(f"Processing sample {sample_id}...")
        
        # Load disparity data for all methods
        disparity_data = {}
        for method in methods.keys():
            disparity_data[method] = load_disparity_data(method, sample_id)
        
        # Check if we have any valid data
        valid_methods = [m for m, d in disparity_data.items() if d is not None]
        if not valid_methods:
            print(f"No valid data found for sample {sample_id}")
            continue
        
        # Resize all disparity maps to 192x640
        print(f"Resizing all disparity maps to 192x640 for sample {sample_id}...")
        for method in valid_methods:
            if disparity_data[method] is not None:
                disparity_data[method] = resize_depth_map(disparity_data[method], target_shape=(192, 640))
        
        # Affine align all disparity maps to match ProDepth as reference
        aligned_data, ref_mean, ref_std = affine_align_disparity_to_reference(disparity_data, reference_method='prodepth')
        
        # Load original image
        original_image = load_original_image(sample_id, target_shape=(192, 640))
        
        # Verify original image dimensions match disparity maps
        if original_image is not None:
            expected_shape = (192, 640)
            if original_image.shape[:2] != expected_shape:
                print(f"Warning: Original image shape {original_image.shape[:2]} doesn't match expected {expected_shape}")
            else:
                print(f"Original image dimensions verified: {original_image.shape[:2]}")
        
        # Create figure with subplots (original image + disparity methods)
        n_methods = len(valid_methods)
        n_total = n_methods + (1 if original_image is not None else 0)
        fig, axes = plt.subplots(n_total, 1, figsize=(12, 3*n_total))
        if n_total == 1:
            axes = [axes]
        
        fig.suptitle(f'Disparity Prediction Comparison - Sample {sample_id}', fontsize=16, fontweight='bold')
        
        # Add original image at the top
        if original_image is not None:
            axes[0].imshow(original_image, aspect='auto')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            start_idx = 1
        else:
            start_idx = 0
        
        # Add disparity visualizations
        for idx, method in enumerate(valid_methods):
            disparity_aligned = aligned_data[method]
            if disparity_aligned is None:
                continue
            
            ax_idx = start_idx + idx
            # Create disparity visualization
            im = axes[ax_idx].imshow(disparity_aligned, cmap=unified_colormap, 
                                aspect='auto', interpolation='nearest', vmin=0, vmax=1)
            
            # Remove axis ticks for cleaner look
            axes[ax_idx].set_xticks([])
            axes[ax_idx].set_yticks([])
            axes[ax_idx].set_ylabel(methods[method], fontsize=12, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        output_path = os.path.join(output_dir, f'disparity_comparison_{sample_id}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison for sample {sample_id} to {output_path}")

def create_disparity_grid_comparison(sample_ids, output_dir='comparisons'):
    """Create a grid comparison showing all methods side by side."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    methods = {
        'prodepth': 'ProDepth',
        'metricdepthv2': 'MetricDepthV2', 
        'teacher': 'Depth Anything v2',
        'student': 'Ours'
    }
    
    # Use the same colormap for all methods
    unified_colormap = 'viridis'
    
    for sample_id in sample_ids:
        print(f"Creating grid comparison for sample {sample_id}...")
        
        # Load disparity data
        disparity_data = {}
        for method in methods.keys():
            disparity_data[method] = load_disparity_data(method, sample_id)
        
        valid_methods = [m for m, d in disparity_data.items() if d is not None]
        if not valid_methods:
            continue
        
        # Resize all disparity maps to 192x640
        for method in valid_methods:
            if disparity_data[method] is not None:
                disparity_data[method] = resize_depth_map(disparity_data[method], target_shape=(192, 640))
        
        # Affine align all disparity maps to match ProDepth as reference
        aligned_data, ref_mean, ref_std = affine_align_disparity_to_reference(disparity_data, reference_method='prodepth')
        
        # Load original image
        original_image = load_original_image(sample_id, target_shape=(192, 640))
        
        # Verify original image dimensions match disparity maps
        if original_image is not None:
            expected_shape = (192, 640)
            if original_image.shape[:2] != expected_shape:
                print(f"Warning: Original image shape {original_image.shape[:2]} doesn't match expected {expected_shape}")
            else:
                print(f"Original image dimensions verified: {original_image.shape[:2]}")
        
        # Create 2x3 grid (original image + 4 methods, with one empty slot)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        fig.suptitle(f'Disparity Prediction Grid Comparison - Sample {sample_id}', 
                    fontsize=18, fontweight='bold')
        
        # Add original image in the first position
        if original_image is not None:
            axes[0].imshow(original_image, aspect='auto')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            start_idx = 1
        else:
            start_idx = 0
        
        # Add disparity visualizations
        for idx, method in enumerate(valid_methods):
            if idx >= 4:  # Only show first 4 methods
                break
                
            disparity_aligned = aligned_data[method]
            if disparity_aligned is None:
                continue
            
            ax_idx = start_idx + idx
            im = axes[ax_idx].imshow(disparity_aligned, cmap=unified_colormap, 
                                aspect='auto', interpolation='nearest', vmin=0, vmax=1)
            
            # Add title
            axes[ax_idx].set_title(f'{methods[method]}', fontsize=12, fontweight='bold')
            axes[ax_idx].set_xticks([])
            axes[ax_idx].set_yticks([])
        
        # Hide unused subplots
        for idx in range(start_idx + len(valid_methods), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        output_path = os.path.join(output_dir, f'disparity_grid_{sample_id}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved grid comparison for sample {sample_id} to {output_path}")

def get_available_samples():
    """Get list of available sample IDs from the data."""
    sample_ids = set()
    
    # Check each method directory
    methods = ['prodepth', 'metricdepthv2', 'teacher', 'student']
    for method in methods:
        if method == 'metricdepthv2':
            files = glob.glob(f'{method}/*.npy')
            for file in files:
                sample_id = os.path.basename(file).replace('.npy', '')
                sample_ids.add(sample_id)
        else:
            files = glob.glob(f'{method}/sample_*_depth.npy')
            for file in files:
                # Extract sample ID from filename
                basename = os.path.basename(file)
                if method == 'prodepth':
                    sample_id = basename.split('_')[1]
                else:
                    sample_id = basename.split('_')[1]
                sample_ids.add(sample_id)
    
    return sorted(list(sample_ids))

def main():
    """Main function to create all visualizations."""
    print("Disparity Prediction Visualization Tool")
    print("=" * 40)
    
    # Get available samples
    sample_ids = get_available_samples()
    print(f"Found {len(sample_ids)} samples: {sample_ids}")
    
    if not sample_ids:
        print("No samples found! Please check your data directories.")
        return
    
    # Create individual comparisons
    print("\nCreating individual comparisons...")
    create_disparity_comparison(sample_ids)
    
    # Create grid comparisons
    print("\nCreating grid comparisons...")
    create_disparity_grid_comparison(sample_ids)
    
    print(f"\nVisualization complete! Check the 'comparisons' directory for results.")
    print(f"Generated visualizations for {len(sample_ids)} samples.")

if __name__ == "__main__":
    main()
