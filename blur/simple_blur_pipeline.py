"""
Simplified VAPOR Blur Pipeline
A working implementation that focuses on generating blurred videos and metrics.

Usage:
    python blur/simple_blur_pipeline.py --video pat3.mp4 --stride 60
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from blur.fx_01_blur.effects.blur_engine import EnhancedBlurEffects
from utils.core_utilities import VideoConfig, crop_to_content, pad_frame_to_size, get_image_files, find_content_bounds_diagonal
from data.scripts.video_reconstructor import VideoReconstructor

# Import metrics modules
from blur.metrics.no_reference import NoReferenceMetrics
from blur.metrics.full_reference import FullReferenceMetrics
from blur.metrics.sharpness import SharpnessMetrics


class SimplifiedBlurPipeline:
    """Simplified blur pipeline that works with existing modules."""
    
    def __init__(self, video_name: str, stride: int = 1):
        """Initialize the pipeline.
        
        Args:
            video_name: Name of the video file (e.g., 'pat3.mp4')
            stride: Frame processing stride (1 = all frames, 60 = every 60th frame)
        """
        self.video_name = video_name
        self.stride = stride
        
        # Setup paths manually
        self.base_dir = Path(__file__).parent.parent
        self.video_path = self.base_dir / "data" / "videos" / "original" / video_name
        
        # Output directories
        self.frames_base = self.base_dir / "data" / "frames"
        self.frames_original = self.frames_base / "original" / Path(video_name).stem
        self.frames_blurred = self.frames_base / "blurred" / Path(video_name).stem
        self.videos_blurred = self.base_dir / "data" / "videos" / "blurred"
        
        # Create directories
        self._ensure_directories_exist()
        
        # Initialize metrics calculators
        self.no_ref_calculator = NoReferenceMetrics()
        self.full_ref_calculator = FullReferenceMetrics()
        self.sharpness_calculator = SharpnessMetrics()
        
        # Initialize blur engine
        self.blur_engine = EnhancedBlurEffects()
        
        # Blur configurations - use the correct names from the blur engine
        self.blur_types = ["gaussian", "motion_blur", "defocus", "haze", "combined"]
        self.intensities = ["low", "high"]
        
        # Video properties
        self.video_config = None
        self.crop_bounds = None
        
    def _ensure_directories_exist(self):
        """Create necessary directories."""
        self.frames_base.mkdir(parents=True, exist_ok=True)
        self.frames_original.mkdir(parents=True, exist_ok=True)
        self.frames_blurred.mkdir(parents=True, exist_ok=True)
        self.videos_blurred.mkdir(parents=True, exist_ok=True)
        
    def detect_crop_bounds(self) -> Optional[Tuple]:
        """Detect crop bounds for the video."""
        print("\n[STEP 1] Detecting optimal crop bounds...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print("  [WARNING] Could not open video for crop detection")
            return None
            
        # Sample a few frames to detect crop bounds
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]
        
        all_bounds = []
        for frame_num in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                bounds = find_content_bounds_diagonal(frame)
                if bounds != (0, 0, frame.shape[1], frame.shape[0]):  # Not full frame
                    all_bounds.append(bounds)
        
        cap.release()
        
        if all_bounds:
            # Use the most restrictive bounds (smallest area that still contains content)
            min_top = max(b[0] for b in all_bounds)
            min_left = max(b[1] for b in all_bounds)
            max_bottom = min(b[2] for b in all_bounds)
            max_right = min(b[3] for b in all_bounds)
            
            bounds = (min_top, min_left, max_bottom, max_right)
            print(f"  [OK] Detected crop bounds: {bounds}")
            return bounds
        else:
            print("  [WARNING] No crop bounds detected, using full frame")
            return None
    
    def extract_and_process_frames(self) -> bool:
        """Extract frames and apply blur effects."""
        print(f"\n[STEP 2] Processing frames with stride {self.stride}...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print("  [ERROR] Could not open video")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, self.stride))
        frames_to_process = len(frame_indices)
        
        print(f"  Processing {frames_to_process} frames (every {self.stride} frame)...")
        
        # Create directories for each blur type and intensity
        blur_dirs = {}
        for blur_type in self.blur_types:
            for intensity in self.intensities:
                dir_name = f"{blur_type}_{intensity}"
                blur_dir = self.frames_blurred / dir_name
                blur_dir.mkdir(parents=True, exist_ok=True)
                blur_dirs[dir_name] = blur_dir
        
        processed_count = 0
        
        for i, frame_num in enumerate(frame_indices):
            try:
                # Read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Apply cropping if available
                if self.crop_bounds:
                    frame = crop_to_content(frame, self.crop_bounds)
                
                # Save original frame
                frame_filename = f"{Path(self.video_name).stem}_{frame_num:06d}.png"
                original_frame_path = self.frames_original / frame_filename
                cv2.imwrite(str(original_frame_path), frame)
                
                # Apply blur effects
                for blur_type in self.blur_types:
                    for intensity in self.intensities:
                        try:
                            # Apply blur using the blur engine
                            blurred_frame, _ = self.blur_engine.apply_blur_effect(frame, blur_type, intensity)
                            
                            # Save blurred frame
                            dir_name = f"{blur_type}_{intensity}"
                            blur_path = blur_dirs[dir_name] / frame_filename
                            cv2.imwrite(str(blur_path), blurred_frame)
                            
                        except Exception as e:
                            print(f"    Warning: Failed to apply {blur_type}_{intensity}: {e}")
                
                processed_count += 1
                
                # Progress update
                if (i + 1) % 10 == 0 or i == frames_to_process - 1:
                    progress = (i + 1) / frames_to_process * 100
                    print(f"    Progress: {progress:.1f}% ({i + 1}/{frames_to_process})")
                    
            except Exception as e:
                print(f"    Error processing frame {frame_num}: {e}")
        
        cap.release()
        
        print(f"  [OK] Processed {processed_count} frames")
        return processed_count > 0
    
    def create_videos(self) -> int:
        """Create videos from processed frames."""
        print("\n[STEP 3] Creating videos from frames...")
        
        success_count = 0
        total_count = len(self.blur_types) * len(self.intensities)
        
        # Create video reconstructor
        reconstructor = VideoReconstructor(str(self.video_path))
        
        for blur_type in self.blur_types:
            for intensity in self.intensities:
                dir_name = f"{blur_type}_{intensity}"
                frames_dir = self.frames_blurred / dir_name
                
                if not frames_dir.exists():
                    print(f"    [SKIP] {dir_name}: No frames directory")
                    continue
                
                # Get frame files
                frame_files = get_image_files(frames_dir)
                if not frame_files:
                    print(f"    [SKIP] {dir_name}: No frames found")
                    continue
                
                # Output video name
                output_filename = f"{Path(self.video_name).stem}_{blur_type}_{intensity}.mp4"
                output_path = self.videos_blurred / output_filename
                
                try:
                    # Create video
                    if reconstructor.create_video(str(frames_dir), str(output_path)):
                        print(f"    [OK] {output_filename}")
                        success_count += 1
                    else:
                        print(f"    [FAILED] {output_filename}")
                except Exception as e:
                    print(f"    [ERROR] {output_filename}: {e}")
        
        print(f"  [OK] Created {success_count}/{total_count} videos")
        return success_count
    
    def calculate_metrics(self) -> bool:
        """Calculate quality metrics for all frames."""
        print("\n[STEP 4] Calculating quality metrics...")
        
        try:
            # Prepare metrics data
            metrics_data = []
            
            # Process original frames
            print("  Processing original frames...")
            for frame_file in get_image_files(self.frames_original):
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue
                
                # Calculate no-reference metrics for original
                no_ref_metrics = self.no_ref_calculator.calculate_all(frame)
                sharpness_metrics = self.sharpness_calculator.calculate_all(frame)
                
                metrics_data.append({
                    'frame': frame_file.name,
                    'method': 'original',
                    'blur_type': 'none',
                    'intensity': 'none',
                    **no_ref_metrics,
                    **sharpness_metrics
                })
            
            # Process blurred frames
            print("  Processing blurred frames...")
            for blur_type in self.blur_types:
                for intensity in self.intensities:
                    dir_name = f"{blur_type}_{intensity}"
                    frames_dir = self.frames_blurred / dir_name
                    
                    if not frames_dir.exists():
                        continue
                    
                    print(f"    Processing {dir_name}...")
                    
                    for frame_file in get_image_files(frames_dir):
                        frame = cv2.imread(str(frame_file))
                        if frame is None:
                            continue
                        
                        # Find corresponding original frame
                        original_frame_path = self.frames_original / frame_file.name
                        original_frame = cv2.imread(str(original_frame_path))
                        
                        # Calculate metrics
                        no_ref_metrics = self.no_ref_calculator.calculate_all(frame)
                        sharpness_metrics = self.sharpness_calculator.calculate_all(frame)
                        
                        metrics_row = {
                            'frame': frame_file.name,
                            'method': dir_name,
                            'blur_type': blur_type,
                            'intensity': intensity,
                            **no_ref_metrics,
                            **sharpness_metrics
                        }
                        
                        # Add full-reference metrics if original frame exists
                        if original_frame is not None:
                            full_ref_metrics = self.full_ref_calculator.calculate_all(frame, original_frame)
                            metrics_row.update(full_ref_metrics)
                        
                        metrics_data.append(metrics_row)
            
            # Save metrics to CSV
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                
                # Save detailed metrics
                metrics_csv = self.frames_base / f"{Path(self.video_name).stem}_detailed_metrics.csv"
                df.to_csv(metrics_csv, index=False)
                print(f"    [OK] Detailed metrics saved: {metrics_csv}")
                
                # Save summary statistics
                summary_df = df.groupby(['method', 'blur_type', 'intensity']).agg(['mean', 'std']).round(4)
                summary_csv = self.frames_base / f"{Path(self.video_name).stem}_summary_metrics.csv"
                summary_df.to_csv(summary_csv)
                print(f"    [OK] Summary metrics saved: {summary_csv}")
                
                return True
            else:
                print("    [WARNING] No metrics data collected")
                return False
                
        except Exception as e:
            print(f"    [ERROR] Metrics calculation failed: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Run the complete blur pipeline."""
        print("VAPOR - Simplified Blur Processing Pipeline")
        print("=" * 60)
        print(f"Video: {self.video_name}")
        print(f"Stride: {self.stride}")
        
        if not self.video_path.exists():
            print(f"[ERROR] Video not found: {self.video_path}")
            return False
        
        try:
            # Extract video configuration
            self.video_config = VideoConfig(self.video_path)
            print(f"Video: {self.video_config.width}x{self.video_config.height} @ {self.video_config.fps:.2f} fps")
            
            # Detect crop bounds
            self.crop_bounds = self.detect_crop_bounds()
            
            # Process frames
            if not self.extract_and_process_frames():
                print("[ERROR] Frame processing failed")
                return False
            
            # Create videos
            video_count = self.create_videos()
            
            # Calculate metrics
            self.calculate_metrics()
            
            # Summary
            print("\n" + "=" * 60)
            print("[SUCCESS] PIPELINE COMPLETED!")
            print("=" * 60)
            print(f"Videos created: {video_count}")
            print(f"Output directory: {self.videos_blurred}")
            print(f"Frames directory: {self.frames_base}")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Simplified VAPOR Blur Processing Pipeline"
    )
    parser.add_argument(
        "--video", 
        required=True,
        help="Video filename (e.g., pat3.mp4)"
    )
    parser.add_argument(
        "--stride", 
        type=int, 
        default=1,
        help="Frame processing stride (1 = all frames, 60 = every 60th frame). Default: 1"
    )
    
    args = parser.parse_args()
    
    # Validate stride
    if args.stride < 1 or args.stride > 100:
        print("Error: Stride must be between 1 and 100")
        return 1
    
    pipeline = SimplifiedBlurPipeline(args.video, args.stride)
    success = pipeline.run_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())