"""
VAPOR - Blur Processing Example Usage
Example script showing how to process videos with blur effects.
NOTE: For production use, see blur_generator.py instead.

Features:
- Interactive video selection from data/videos/original/
- Efficient single-pass cropping detection
- All blur types (6) in both low and high intensities
- Proper filename conventions (e.g., pat3_gaussian_low.mp4)
- Original video size preservation with black padding
- Configurable stride for frame processing

Usage:
    python blur/blur_example_usage.py [--stride N]
    
Examples:
    python blur/blur_example_usage.py                 # Process all frames
    python blur/blur_example_usage.py --stride 5      # Process every 5th frame
    python blur/blur_example_usage.py --stride 10     # Process every 10th frame
"""

import cv2
import numpy as np
import os
import sys
import subprocess
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import math
from scipy import signal

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import required modules from their actual locations
from blur.fx_01_blur.effects.blur_engine import EnhancedBlurEffects
from utils.core_utilities import VideoConfig, crop_to_content, pad_frame_to_size, get_image_files, find_content_bounds_diagonal
from data.scripts.video_reconstructor import VideoReconstructor


class VAPORPipeline:
    """Unified VAPOR processing pipeline using shared core modules."""
    
    def __init__(self, stride: int = 1):
        """Initialize the pipeline.
        
        Args:
            stride: Frame processing stride (1 = all frames, 2 = every 2nd frame, etc.)
        """
        # Setup project paths using shared utility
        self.paths = setup_project_paths()
        ensure_directories_exist(self.paths)
        
        # Blur types and intensities
        self.blur_types = ["gaussian", "motion_blur", "outoffocus", "average", "median", "combined"]
        self.intensities = ["low", "high"]
        
        # Processing options
        self.stride = stride  # Process every Nth frame (1 = all frames)
        
        # Components
        self.video_selector = VideoSelector(self.paths['videos_original'])
        self.mode_selector = ProcessingModeSelector()
        
        # Video properties (set during processing)
        self.video_config = None
        self.crop_bounds = None
        
    def get_python_executable(self):
        """Get the correct Python executable path for the virtual environment."""
        return "C:/Users/kesne/Documents/Webdev/MAPLE-25/VAPOR/venv/Scripts/python.exe"
    
    def select_video(self) -> Optional[Path]:
        """Interactive video selection using shared VideoSelector."""
        selected_video = self.video_selector.select_video_interactive()
        
        if selected_video:
            # Ask for processing options if stride not set via command line
            self.stride = self.mode_selector.select_stride_interactive(self.stride)
        
        return selected_video
    
    def extract_video_config(self, video_path: Path) -> Dict:
        """Extract video configuration for reconstruction using shared VideoConfig."""
        video_config = VideoConfig(video_path)
        config_dict = video_config.to_dict()
        
        print("Video configuration:")
        print(f"  Resolution: {config_dict['width']}x{config_dict['height']}")
        print(f"  FPS: {config_dict['fps']:.2f}")
        print(f"  Frame count: {config_dict['frame_count']}")
        print(f"  Stride: {self.stride} (processing every {self.stride} frame{'s' if self.stride > 1 else ''})")
        
        return config_dict
    
    def detect_crop_bounds_once(self, video_path: Path) -> Optional[Tuple]:
        """Detect crop bounds using shared image processing utilities."""
        print("\n[STEP 1] Detecting optimal crop bounds...")
        
        bounds = detect_optimal_crop_bounds(video_path, sample_count=5)
        
        if bounds:
            print(f"  [OK] Final crop bounds: {bounds}")
        else:
            print("  [WARNING] No valid crop bounds detected, using full frame")
        
        return bounds
    
    def extract_and_process_frames(self, video_path: Path, video_name: str) -> bool:
        """Extract all frames and apply all blur effects using shared utilities."""
        print("\n[STEP 2] Processing all frames with blur effects...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frames to process based on stride setting
        frame_indices = list(range(0, total_frames, self.stride))
        frames_to_process = len(frame_indices)
        
        # Create directories for original frames and each blur type and intensity
        frame_dirs = {}
        
        # Original frames directory
        original_frames_dir = self.paths['frames_original'] / video_name
        original_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Blurred frames directories
        blurred_base_dir = self.paths['frames_blurred'] / video_name
        for blur_type in self.blur_types:
            for intensity in self.intensities:
                dir_name = f"{blur_type}_{intensity}"
                frame_dir = blurred_base_dir / dir_name
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_dirs[dir_name] = frame_dir
        
        print(f"  Processing {frames_to_process} frames (every {self.stride} frame{'s' if self.stride > 1 else ''})...")
        
        processed_count = 0
        failed_count = 0
        
        for i, frame_num in enumerate(frame_indices):
            try:
                # Read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    failed_count += 1
                    continue
                
                # Save original frame first (apply cropping if available)
                frame_filename = f"{video_name}_{frame_num:06d}.png"
                original_frame_path = original_frames_dir / frame_filename
                
                # Apply crop bounds to original frame as well
                original_frame_to_save = frame
                if self.crop_bounds:
                    original_frame_to_save = crop_to_content(frame, self.crop_bounds)
                
                cv2.imwrite(str(original_frame_path), original_frame_to_save)
                
                # Use the same cropped frame for blur effects
                cropped_frame = original_frame_to_save
                
                # Apply all blur effects and intensities using shared blur engine
                for blur_type in self.blur_types:
                    for intensity in self.intensities:
                        # Apply blur using shared effect engine
                        blurred_frame = apply_blur_effect(cropped_frame, blur_type, intensity)
                        
                        # Save frame with proper numbering
                        frame_path = frame_dirs[f"{blur_type}_{intensity}"] / frame_filename
                        cv2.imwrite(str(frame_path), blurred_frame)
                
                processed_count += 1
                
                # Progress update
                if (i + 1) % 50 == 0 or i == frames_to_process - 1:
                    progress = (i + 1) / frames_to_process * 100
                    print(f"    Progress: {progress:.1f}% ({i + 1}/{frames_to_process})")
                    
            except Exception as e:
                print(f"    Error processing frame {frame_num}: {e}")
                failed_count += 1
        
        cap.release()
        
        print("  [OK] Frame processing complete!")
        print(f"    Successfully processed: {processed_count} frames")
        print(f"    Failed: {failed_count} frames")
        
        return processed_count > 0
    
    def create_videos_from_frames(self, video_name: str, original_video_path: Path) -> int:
        """Create videos from all processed frame sets using shared VideoReconstructor."""
        print("\n[STEP 3] Creating videos...")
        
        success_count = 0
        total_count = len(self.blur_types) * len(self.intensities)
        
        # Create video reconstructor with original video config
        video_config = VideoConfig(original_video_path)
        reconstructor = VideoReconstructor(video_config)
        
        for blur_type in self.blur_types:
            for intensity in self.intensities:
                dir_name = f"{blur_type}_{intensity}"
                frames_dir = self.paths['frames_blurred'] / video_name / dir_name
                
                if not frames_dir.exists():
                    print(f"    [SKIP] {dir_name}: No frames directory")
                    continue
                
                # Get frame files using shared utility
                frame_files = get_image_files(frames_dir)
                if not frame_files:
                    print(f"    [SKIP] {dir_name}: No frames found")
                    continue
                
                # Process frames to match original video size
                processed_frames_dir = frames_dir / "processed"
                processed_frames_dir.mkdir(exist_ok=True)
                
                processed_files = []
                for frame_file in frame_files:
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        continue
                    
                    # Pad frame to original video size
                    padded_frame = pad_frame_to_size(
                        frame, 
                        self.video_config['width'], 
                        self.video_config['height']
                    )
                    
                    processed_frame_path = processed_frames_dir / frame_file.name
                    cv2.imwrite(str(processed_frame_path), padded_frame)
                    processed_files.append(processed_frame_path)
                
                # Create video
                output_filename = f"{video_name}_{blur_type}_{intensity}.mp4"
                output_path = self.paths['videos_blurred'] / output_filename
                
                if reconstructor.create_video_from_frames(processed_files, output_path):
                    print(f"    [OK] {output_filename}")
                    success_count += 1
                else:
                    print(f"    [FAILED] {output_filename}")
        
        print(f"  [OK] Video creation complete: {success_count}/{total_count} videos created")
        return success_count
    
    def calculate_quality_metrics(self, video_name: str) -> bool:
        """Calculate quality metrics for all processed frames and create CSV files."""
        print("\n[STEP 4] Calculating quality metrics...")
        
        # Initialize quality metrics logger
        metrics_logger = QualityMetricsLogger(self.paths['frames_base'])
        
        # Process original frames
        original_frames_dir = self.paths['frames_original'] / video_name
        if original_frames_dir.exists():
            print("  Processing original frames...")
            metrics_logger.process_frames_directory(original_frames_dir, 'original')
        
        # Process all blurred frame sets
        blurred_base_dir = self.paths['frames_blurred'] / video_name
        if blurred_base_dir.exists():
            for blur_type in self.blur_types:
                for intensity in self.intensities:
                    method_name = f"{blur_type}_{intensity}"
                    frames_dir = blurred_base_dir / method_name
                    
                    if frames_dir.exists():
                        print(f"  Processing {method_name} frames...")
                        metrics_logger.process_frames_directory(frames_dir, method_name)
        
        # Create summary CSV with mean and std deviation
        print("  Creating summary statistics...")
        summary_csv = metrics_logger.create_summary_csv(video_name, original_frames_dir)
        
        if summary_csv:
            print("  [OK] Quality metrics analysis complete!")
            print(f"    Summary CSV: {summary_csv}")
            return True
        else:
            print("  [WARNING] No summary CSV created")
            return False
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline using shared VAPOR core modules."""
        print("VAPOR - Video Blur Processing Pipeline")
        print("=" * 60)
        
        try:
            # Step 0: Select video
            video_path = self.select_video()
            if not video_path:
                return False
            
            video_name = video_path.stem
            print(f"\nSelected video: {video_path.name}")
            
            # Extract video configuration
            self.video_config = self.extract_video_config(video_path)
            
            # Detect crop bounds once
            self.crop_bounds = self.detect_crop_bounds_once(video_path)
            
            # Process all frames with all blur effects
            if not self.extract_and_process_frames(video_path, video_name):
                print("[ERROR] Frame processing failed")
                return False
            
            # Create videos
            success_count = self.create_videos_from_frames(video_name, video_path)
            
            # Calculate quality metrics
            self.calculate_quality_metrics(video_name)
            
            # Summary
            print("\n" + "=" * 60)
            print("[SUCCESS] PIPELINE COMPLETED!")
            print("=" * 60)
            print(f"Processed video: {video_name}")
            print(f"Videos created: {success_count}")
            print(f"Output videos: {self.paths['videos_blurred']}")
            print(f"Extracted frames: {self.paths['frames_base']}")
            print(f"Stride used: {self.stride}")
            print("\nGenerated videos:")
            
            # List generated videos
            for blur_type in self.blur_types:
                for intensity in self.intensities:
                    video_file = self.paths['videos_blurred'] / f"{video_name}_{blur_type}_{intensity}.mp4"
                    if video_file.exists():
                        size_mb = video_file.stat().st_size / (1024 * 1024)
                        print(f"  - {video_file.name} ({size_mb:.1f} MB)")
            
            return True
            
        except KeyboardInterrupt:
            print("\n[CANCELLED] Pipeline cancelled by user")
            return False
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VAPOR - Unified Video Blur Processing Pipeline"
    )
    parser.add_argument(
        "--stride", 
        type=int, 
        default=1,
        help="Frame processing stride (1 = all frames, 2 = every 2nd frame, etc.). Default: 1"
    )
    
    args = parser.parse_args()
    
    # Validate stride
    if args.stride < 1 or args.stride > 50:
        print("Error: Stride must be between 1 and 50")
        return 1
    
    pipeline = VAPORPipeline(stride=args.stride)
    success = pipeline.run_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
