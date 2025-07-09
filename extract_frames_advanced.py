#!/usr/bin/env python3
"""
Advanced Video Frame Extraction Tool

This script extracts video frames corresponding to GPS data points from CSV files,
with configurable frame rates and advanced filtering options.

Features:
- Configurable frame extraction rate (Hz)
- GPS quality filtering (exclude bad GPS points)
- Batch processing of multiple video-GPS pairs
- Progress tracking with tqdm
- Organized output structure
- Memory-efficient processing
- Error handling and logging
"""

import os
import csv
import cv2
import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import glob
import re
from typing import List, Tuple, Optional, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frame_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPSFrameExtractor:
    """Advanced GPS-based frame extractor with configurable options"""
    
    def __init__(self, 
                 gps_folder: str = "our_data/extracted_gps",
                 video_folder: str = "our_data", 
                 output_folder: str = "our_data/extracted_frames",
                 frame_rate_hz: float = 1.0,
                 filter_bad_gps: bool = True,
                 image_quality: int = 95,
                 max_time_gap_seconds: float = 1.0):
        """
        Initialize the GPS Frame Extractor
        
        Args:
            gps_folder: Path to folder containing GPS CSV files
            video_folder: Path to folder containing video files
            output_folder: Path to output folder for extracted frames
            frame_rate_hz: Frame extraction rate in Hz (frames per second)
            filter_bad_gps: Whether to exclude bad GPS points (GPSFIX=0, GPSP>2000)
            image_quality: JPEG quality for saved frames (1-100)
            max_time_gap_seconds: Maximum time gap between GPS points to consider valid
        """
        self.gps_folder = Path(gps_folder)
        self.video_folder = Path(video_folder)
        self.output_folder = Path(output_folder)
        self.frame_rate_hz = frame_rate_hz
        self.filter_bad_gps = filter_bad_gps
        self.image_quality = image_quality
        self.max_time_gap_seconds = max_time_gap_seconds
        
        # Create output directory
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized GPS Frame Extractor:")
        logger.info(f"  GPS folder: {self.gps_folder}")
        logger.info(f"  Video folder: {self.video_folder}")
        logger.info(f"  Output folder: {self.output_folder}")
        logger.info(f"  Frame rate: {self.frame_rate_hz} Hz")
        logger.info(f"  Filter bad GPS: {self.filter_bad_gps}")
        logger.info(f"  Image quality: {self.image_quality}")
    
    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse ISO timestamp string to datetime object"""
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError as e:
            logger.error(f"Failed to parse timestamp: {timestamp_str} - {e}")
            raise
    
    def find_video_file(self, csv_filename: str) -> Optional[Path]:
        """Find corresponding video file for a CSV file"""
        # Extract base name from CSV (remove .csv extension)
        base_name = csv_filename.replace('.csv', '')
        
        # Common video extensions
        video_extensions = ['.MP4', '.mp4', '.MOV', '.mov', '.AVI', '.avi']
        
        # Try to find video file
        for ext in video_extensions:
            video_path = self.video_folder / f"{base_name}{ext}"
            if video_path.exists():
                return video_path
        
        logger.warning(f"No video file found for {csv_filename}")
        return None
    
    def load_and_filter_gps_data(self, csv_path: Path) -> pd.DataFrame:
        """Load GPS data from CSV and apply filtering"""
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} GPS points from {csv_path.name}")
            
            # Convert timestamp to datetime
            df['datetime'] = df['time'].apply(self.parse_timestamp)
            
            # Filter bad GPS points if enabled
            if self.filter_bad_gps:
                initial_count = len(df)
                
                # Filter out bad GPS fixes and precision
                if 'GPSFIX' in df.columns:
                    df = df[df['GPSFIX'] > 0]
                    
                if 'GPSP' in df.columns:
                    df = df[df['GPSP'] <= 2000]
                    
                filtered_count = len(df)
                logger.info(f"Filtered out {initial_count - filtered_count} bad GPS points")
            
            # Sort by timestamp
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load GPS data from {csv_path}: {e}")
            raise
    
    def select_frames_by_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select GPS points based on specified frame rate"""
        if self.frame_rate_hz <= 0:
            logger.warning("Invalid frame rate, using all GPS points")
            return df
        
        # Calculate time interval between frames
        interval_seconds = 1.0 / self.frame_rate_hz
        
        # Select frames at regular intervals
        selected_indices = []
        last_time = None
        
        for idx, row in df.iterrows():
            current_time = row['datetime']
            
            if last_time is None or (current_time - last_time).total_seconds() >= interval_seconds:
                selected_indices.append(idx)
                last_time = current_time
        
        selected_df = df.iloc[selected_indices].reset_index(drop=True)
        logger.info(f"Selected {len(selected_df)} frames at {self.frame_rate_hz} Hz from {len(df)} GPS points")
        
        return selected_df
    
    def extract_frames_from_video(self, video_path: Path, gps_df: pd.DataFrame, output_subdir: Path) -> Dict[str, int]:
        """Extract frames from video at GPS timestamps"""
        
        # Create output subdirectory
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps
        
        logger.info(f"Video properties: {fps:.2f} FPS, {total_frames} frames, {duration_seconds:.2f}s duration")
        
        # Get video start time (assuming first GPS point is close to video start)
        video_start_time = gps_df.iloc[0]['datetime']
        
        stats = {
            'total_gps_points': len(gps_df),
            'frames_extracted': 0,
            'frames_skipped': 0,
            'errors': 0
        }
        
        # Process each GPS point
        for idx, row in tqdm(gps_df.iterrows(), 
                           total=len(gps_df), 
                           desc=f"Extracting frames from {video_path.name}"):
            
            try:
                # Calculate frame number from timestamp
                time_offset = (row['datetime'] - video_start_time).total_seconds()
                frame_number = int(time_offset * fps)
                
                # Skip if frame is outside video bounds
                if frame_number < 0 or frame_number >= total_frames:
                    stats['frames_skipped'] += 1
                    continue
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Could not read frame {frame_number}")
                    stats['errors'] += 1
                    continue
                
                # Create filename with GPS info
                timestamp_str = row['datetime'].strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds
                lat, lon = row['latitude'], row['longitude']
                filename = f"{timestamp_str}_lat{lat:.6f}_lon{lon:.6f}_frame{frame_number:06d}.jpg"
                
                # Save frame
                output_path = output_subdir / filename
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
                success = cv2.imwrite(str(output_path), frame, encode_params)
                
                if success:
                    stats['frames_extracted'] += 1
                else:
                    logger.warning(f"Failed to save frame to {output_path}")
                    stats['errors'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing GPS point {idx}: {e}")
                stats['errors'] += 1
        
        cap.release()
        return stats
    
    def process_video_gps_pair(self, csv_path: Path) -> Dict[str, int]:
        """Process a single video-GPS pair"""
        logger.info(f"Processing {csv_path.name}")
        
        # Find corresponding video file
        video_path = self.find_video_file(csv_path.name)
        if not video_path:
            logger.error(f"No video file found for {csv_path.name}")
            return {'error': 1}
        
        # Load and filter GPS data
        gps_df = self.load_and_filter_gps_data(csv_path)
        if len(gps_df) == 0:
            logger.warning(f"No valid GPS data found in {csv_path.name}")
            return {'error': 1}
        
        # Select frames based on frame rate
        selected_gps_df = self.select_frames_by_rate(gps_df)
        
        # Create output subdirectory
        output_subdir = self.output_folder / csv_path.stem
        
        # Extract frames
        stats = self.extract_frames_from_video(video_path, selected_gps_df, output_subdir)
        
        logger.info(f"Completed {csv_path.name}: {stats}")
        return stats
    
    def process_all(self) -> Dict[str, int]:
        """Process all GPS-video pairs"""
        logger.info("Starting batch processing of all GPS-video pairs")
        
        # Find all CSV files
        csv_files = list(self.gps_folder.glob("*.csv"))
        
        if not csv_files:
            logger.error(f"No CSV files found in {self.gps_folder}")
            return {'error': 1}
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Process each file
        total_stats = {
            'total_gps_points': 0,
            'frames_extracted': 0,
            'frames_skipped': 0,
            'errors': 0,
            'files_processed': 0
        }
        
        for csv_path in csv_files:
            try:
                stats = self.process_video_gps_pair(csv_path)
                
                if 'error' not in stats:
                    total_stats['files_processed'] += 1
                    for key in ['total_gps_points', 'frames_extracted', 'frames_skipped', 'errors']:
                        total_stats[key] += stats.get(key, 0)
                        
            except Exception as e:
                logger.error(f"Failed to process {csv_path.name}: {e}")
                total_stats['errors'] += 1
        
        logger.info(f"Batch processing completed: {total_stats}")
        return total_stats

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Advanced GPS-based video frame extraction tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames at 1 Hz (1 frame per second)
  python extract_frames_advanced.py --frame-rate 1.0
  
  # Extract frames at 0.5 Hz (1 frame every 2 seconds)
  python extract_frames_advanced.py --frame-rate 0.5
  
  # Extract frames at 10 Hz (10 frames per second) without GPS filtering
  python extract_frames_advanced.py --frame-rate 10.0 --no-filter-gps
  
  # Custom paths and high quality
  python extract_frames_advanced.py --gps-folder my_gps --video-folder my_videos --output-folder my_frames --quality 100
        """
    )
    
    parser.add_argument('--gps-folder', 
                       default='our_data/extracted_gps',
                       help='Path to folder containing GPS CSV files')
    
    parser.add_argument('--video-folder', 
                       default='our_data',
                       help='Path to folder containing video files')
    
    parser.add_argument('--output-folder', 
                       default='our_data/extracted_frames',
                       help='Path to output folder for extracted frames')
    
    parser.add_argument('--frame-rate', 
                       type=float, 
                       default=10.0,
                       help='Frame extraction rate in Hz (frames per second). Default: 10.0')
    
    parser.add_argument('--no-filter-gps', 
                       action='store_true',
                       help='Disable GPS quality filtering (include bad GPS points)')
    
    parser.add_argument('--quality', 
                       type=int, 
                       default=95,
                       help='JPEG quality for saved frames (1-100). Default: 95')
    
    parser.add_argument('--max-time-gap', 
                       type=float, 
                       default=1.0,
                       help='Maximum time gap between GPS points (seconds). Default: 1.0')
    
    parser.add_argument('--file', 
                       help='Process only a specific CSV file (filename only)')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = GPSFrameExtractor(
        gps_folder=args.gps_folder,
        video_folder=args.video_folder,
        output_folder=args.output_folder,
        frame_rate_hz=args.frame_rate,
        filter_bad_gps=not args.no_filter_gps,
        image_quality=args.quality,
        max_time_gap_seconds=args.max_time_gap
    )
    
    # Process files
    if args.file:
        # Process single file
        csv_path = Path(args.gps_folder) / args.file
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 1
        
        stats = extractor.process_video_gps_pair(csv_path)
        
    else:
        # Process all files
        stats = extractor.process_all()
    
    logger.info("Frame extraction completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 