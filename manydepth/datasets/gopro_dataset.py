# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import numpy as np
import PIL.Image as pil
from pathlib import Path
from .mono_dataset import MonoDataset


class GoProDataset(MonoDataset):
    """GoPro dataset with exact same interface as KITTIRAWDataset
    """
    def __init__(self, *args, **kwargs):
        # Extract split before calling super
        self.split = kwargs.pop('split', 'train')  # 'train', 'test', 'val'
        
        super(GoProDataset, self).__init__(*args, **kwargs)

        # Estimated Camera calibration
        focal_x_px = 898.15   # Horizontal focal length
        focal_y_px = 899.20   # Vertical focal length (from vFoV)
        
        # Image dimensions for normalization
        self.img_width = 1920  # GoPro 1920p width
        self.img_height = 1080  # GoPro 1080p height
        
        # Convert to normalized coordinates (same format as KITTI)
        focal_normalized_x = focal_x_px / self.img_width
        focal_normalized_y = focal_y_px / self.img_height
        
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[focal_normalized_x, 0, 0.5, 0],
                           [0, focal_normalized_y, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        
        # Set full resolution shape (width, height) - same format as KITTI
        self.full_res_shape = (self.img_width, self.img_height)

        
        # Load video data and create filenames list (KITTI-compatible format)
        self.video_data = self._load_video_data()
        self._build_filenames_list()

    def _load_video_data(self):
        """Load metadata from all processed videos"""
        video_data = {}
        
        data = os.path.join(self.data_path, "extracted_frames")

        # Find all video subdirectories
        for video_dir in Path(data).iterdir():
            if video_dir.is_dir():
                # Generate metadata from files with GPS in filename
                frame_files = list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png'))
                frame_gps_pairs = []
                    
                for frame_file in sorted(frame_files):
                    gps_data = self.parse_gps_from_filename(frame_file.name)
                    if gps_data:
                        # Extract frame number from filename
                        frame_start = frame_file.name.find('_frame') + 6
                        frame_end = frame_file.name.find('.jpg')
                        if frame_end == -1:
                            frame_end = frame_file.name.find('.png')
                        frame_number = int(frame_file.name[frame_start:frame_end])
                        
                        frame_gps_pairs.append({
                            'frame_number': frame_number,
                            'frame_filename': frame_file.name,
                            'gps': gps_data
                        })
                
                if frame_gps_pairs:
                    metadata = {
                        'frame_gps_pairs': frame_gps_pairs,
                        'total_frames': len(frame_gps_pairs)
                    }
                    video_data[video_dir.name] = {
                        'metadata': metadata,
                        'video_dir': str(video_dir),
                        'use_filename_gps': True
                    }
        
        return video_data

    def _build_filenames_list(self):
        """Build KITTI-compatible filenames list for sequential train/test/val split"""
        filenames = []
        
        for video_name, video_info in self.video_data.items():
            metadata = video_info['metadata']
            frame_pairs = sorted(metadata['frame_gps_pairs'], key=lambda x: x['frame_number'])
            total_frames = len(frame_pairs)
            
            if total_frames == 0:
                continue
            
            # Sequential split: 70% train, 20% test, 10% val
            train_end = int(0.7 * total_frames)
            test_end = int(0.9 * total_frames)
            
            if self.split == 'train':
                selected_pairs = frame_pairs[:train_end]
            elif self.split == 'test':
                selected_pairs = frame_pairs[train_end:test_end]
            elif self.split == 'val':
                selected_pairs = frame_pairs[test_end:]
            else:
                raise ValueError(f"Unknown split: {self.split}")
            
            # Create mapping of sequential indices to actual frame numbers
            if video_name not in self.video_data:
                self.video_data[video_name] = {}
            self.video_data[video_name]['frame_mapping'] = {}
            self.video_data[video_name]['available_frames'] = [p['frame_number'] for p in frame_pairs]
            
            # Create KITTI-compatible filename strings using sequential indices
            for seq_idx, pair in enumerate(selected_pairs):
                # Map sequential index to actual frame number
                self.video_data[video_name]['frame_mapping'][seq_idx] = pair['frame_number']
                # Format: "folder sequential_index side" (same as KITTI)
                filename_str = f"{video_name} {seq_idx} l"
                filenames.append(filename_str)
        
        # Set filenames for parent class
        self.filenames = filenames

    def check_depth(self):
        """Check if depth data is available - always False for GoPro (no ground truth depth)"""
        return False

    def get_image_path(self, folder, frame_index, side):
        """Get path to image file - same interface as KITTIRAWDataset"""
        # Find the video directory and frame
        if folder not in self.video_data:
            raise ValueError(f"Video folder not found: {folder}")
        
        video_info = self.video_data[folder]
        video_dir = video_info['video_dir']
        metadata = video_info['metadata']
        
        # Convert sequential index to actual frame number
        if 'frame_mapping' in video_info and frame_index in video_info['frame_mapping']:
            actual_frame_number = video_info['frame_mapping'][frame_index]
        else:
            # Fallback: handle temporal offsets by finding nearest available frame
            available_frames = sorted([p['frame_number'] for p in metadata['frame_gps_pairs']])
            if frame_index < 0:
                # Negative index - get from end
                actual_frame_number = available_frames[max(0, len(available_frames) + frame_index)]
            elif frame_index >= len(available_frames):
                # Beyond range - get last frame
                actual_frame_number = available_frames[-1]
            else:
                # Use closest available frame
                actual_frame_number = available_frames[min(frame_index, len(available_frames) - 1)]
        
        # Find the frame with matching frame_number
        for pair in metadata['frame_gps_pairs']:
            if pair['frame_number'] == actual_frame_number:
                return os.path.join(video_dir, pair['frame_filename'])
        
        raise ValueError(f"Frame not found: {folder}, frame {frame_index} -> actual {actual_frame_number}")

    def get_color(self, folder, frame_index, side, do_flip):
        """Load color image - required by MonoDataset interface"""
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        """Get depth data - returns dummy depth since GoPro has no ground truth depth"""
        # Return dummy depth map of zeros (same size as image)
        depth_gt = np.zeros(self.full_res_shape[::-1], dtype=np.float32)  # (height, width)
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        
        return depth_gt

    def index_to_folder_and_frame_idx(self, index):
        """Convert index to folder and frame info - required by MonoDataset interface"""
        line = self.filenames[index].split()
        folder = line[0]
        frame_index = int(line[1])
        side = line[2] if len(line) > 2 else 'l'
        
        return folder, frame_index, side

    def parse_gps_from_filename(self, filename):
        """Parse GPS coordinates from filename
        
        Expected format: {timestamp}_lat{latitude}_lon{longitude}_frame{frame_number}.jpg
        Example: 20231201_143022_123_lat44.128728_lon5.427715_frame000001.jpg
        """
        # Extract latitude
        lat_start = filename.find('_lat') + 4
        lat_end = filename.find('_lon')
        latitude = float(filename[lat_start:lat_end])
        
        # Extract longitude  
        lon_start = filename.find('_lon') + 4
        lon_end = filename.find('_frame')
        longitude = float(filename[lon_start:lon_end])
        
        return {
            'latitude': latitude,
            'longitude': longitude,
            'altitude': 0.0,  # Not encoded in filename
            'timestamp': filename[:filename.find('_lat')]  # Timestamp part
        }

    def get_gps_data_for_frame(self, folder, frame_index):
        """Get GPS data for a specific frame by parsing filename - always returns data"""
        video_info = self.video_data[folder]
        metadata = video_info['metadata']
        
        # Find the frame with matching frame_number
        for pair in metadata['frame_gps_pairs']:
            if pair['frame_number'] == frame_index:
                filename = pair['frame_filename']
                # Parse GPS directly from filename
                return self.parse_gps_from_filename(filename)
        
        # If frame not found, this is an error since GPS should always be available
        raise ValueError(f"GPS data not found for {folder} frame {frame_index}")

    def gps_to_local_coordinates(self, lat, lon, ref_lat, ref_lon):
        """Convert GPS coordinates to local meters (approximate)"""
        # Simple conversion for small distances
        # 1 degree lat ≈ 111,000 meters
        # 1 degree lon ≈ 111,000 * cos(lat) meters
        
        import math
        
        lat_rad = math.radians(ref_lat)
        
        # Convert to meters relative to reference point
        y = (lat - ref_lat) * 111000  # North-South
        x = (lon - ref_lon) * 111000 * math.cos(lat_rad)  # East-West
        
        return x, y

    def get_gps_translations_for_sequence(self, folder, frame_index):
        """Get GPS translations for frame_index-1, frame_index, frame_index+1"""
        translations = []
        
        video_info = self.video_data[folder]
        
        # Convert sequential frame index to actual frame number
        if 'frame_mapping' in video_info and frame_index in video_info['frame_mapping']:
            actual_frame_number = video_info['frame_mapping'][frame_index]
        else:
            actual_frame_number = frame_index
        
        # Use the current frame as reference for local coordinates
        ref_gps = self.get_gps_data_for_frame(folder, actual_frame_number)
        ref_lat = ref_gps['latitude'] 
        ref_lon = ref_gps['longitude']
        
        # Get GPS data for 3 consecutive frames using available frames
        available_frames = video_info['available_frames']
        current_idx = available_frames.index(actual_frame_number)
        
        for offset in [-1, 0, 1]:
            target_idx = current_idx + offset
            
            # Clamp to available frame bounds
            target_idx = max(0, min(len(available_frames) - 1, target_idx))
            target_frame_number = available_frames[target_idx]
            
            gps_data = self.get_gps_data_for_frame(folder, target_frame_number)
            
            lat = gps_data['latitude']
            lon = gps_data['longitude']
            alt = gps_data['altitude']
            
            # Convert to local meters
            x, y = self.gps_to_local_coordinates(lat, lon, ref_lat, ref_lon)
            translations.append([x, y, alt])
        
        return translations

    def __getitem__(self, index):
        """Override __getitem__ to always include GPS data"""
        # Temporarily disable GPS loading to prevent file search, then call parent
        original_load_gps = self.load_gps
        self.load_gps = False
        
        # Call parent __getitem__ to get standard data (without GPS)
        inputs = super().__getitem__(index)
        
        # Restore original GPS loading setting
        self.load_gps = original_load_gps
        
        # Always add GPS data (GPS data is always available for GoPro)
        folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
        
        # Get GPS translations for consecutive frames
        translations = self.get_gps_translations_for_sequence(folder, frame_index)
        
        # Calculate norms (same as MonoDataset)
        from .mono_dataset import norm
        inputs["gps12"] = norm(translations[1], translations[0])  # Current to previous
        inputs["gps23"] = norm(translations[1], translations[2])  # Current to next
        
        return inputs

 