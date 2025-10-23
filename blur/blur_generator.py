"""
VAPOR Blur Generator
Generates blurred frames from original video with various blur types and intensities.

Usage:
    python blur/blur_generator.py --video pat3.mp4 --stride 60
"""

import cv2
import numpy as np
import os
import sys
import argparse
import random
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


class BlurGenerator:
    """Generates blurred frames from original video with various blur effects."""
    
    def __init__(self, video_name: str, stride: int = 1, max_frames: int = None, blur_types: List[str] = None, intensities: List[str] = None, force_regenerate: bool = False, generate_videos: bool = False, start_time: float = None, duration: float = None):
        """Initialize the pipeline.
        
        Args:
            video_name: Name of the video file (e.g., 'pat3.mp4')
            stride: Frame processing stride (1 = all frames, 60 = every 60th frame)
            max_frames: Maximum number of frames to extract (None = no limit)
            blur_types: List of blur types to generate (defaults to all types if None)
            intensities: List of blur intensities to generate (defaults to ['low', 'high'] if None)
            force_regenerate: If True, delete existing blurred frames before processing
            generate_videos: If True, generate videos from processed frames (default: False)
            start_time: Start time in seconds for video cropping (None = from beginning)
            duration: Duration in seconds for video cropping (None = full video)
        """
        self.video_name = video_name
        self.stride = stride
        self.max_frames = max_frames
        self.force_regenerate = force_regenerate
        self.generate_videos = generate_videos
        self.start_time = start_time
        self.duration = duration
        
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
        
        # Clean existing blurred frames if force regenerate is enabled
        if self.force_regenerate:
            self._clean_existing_blurred_frames()
        
        # Initialize metrics calculators
        self.no_ref_calculator = NoReferenceMetrics()
        self.full_ref_calculator = FullReferenceMetrics()
        self.sharpness_calculator = SharpnessMetrics()
        
        # Initialize blur engine
        self.blur_engine = EnhancedBlurEffects()
        
        # Blur configurations - use provided or default values
        if blur_types is None:
            self.blur_types = ["gaussian", "motion_blur", "defocus", "haze", "combined"]
        else:
            self.blur_types = blur_types
            
        if intensities is None:
            self.intensities = ["low", "high"]
        else:
            self.intensities = intensities
        
        # Video properties
        self.video_config = None
        self.crop_bounds = None
        
    def _ensure_directories_exist(self):
        """Create necessary directories."""
        self.frames_base.mkdir(parents=True, exist_ok=True)
        self.frames_original.mkdir(parents=True, exist_ok=True)
        self.frames_blurred.mkdir(parents=True, exist_ok=True)
        self.videos_blurred.mkdir(parents=True, exist_ok=True)
    
    def _clean_existing_blurred_frames(self):
        """Clean existing blurred frame directories for force regenerate."""
        import shutil
        
        # Clean original frames directory when force regenerating
        if self.frames_original.exists():
            print(f"Force regenerate enabled - cleaning existing original frames: {self.frames_original}")
            for file_path in self.frames_original.iterdir():
                if file_path.is_file():
                    print(f"  Removing: {file_path.name}")
                    file_path.unlink()
                elif file_path.is_dir():
                    print(f"  Removing directory: {file_path}")
                    shutil.rmtree(file_path)
        
        if self.frames_blurred.exists():
            print(f"Force regenerate enabled - cleaning existing blurred frames: {self.frames_blurred}")
            
            # Remove all subdirectories in the blurred frames directory
            for subdir in self.frames_blurred.iterdir():
                if subdir.is_dir():
                    print(f"  Removing: {subdir}")
                    shutil.rmtree(subdir)
            
            print(f"Cleaned blurred frames directory: {self.frames_blurred}")
    
    def clean_frame_directories_before_processing(self):
        """Clean frame directories before processing to avoid data from other runs."""
        import shutil
        
        # Clean original frames directory
        if self.frames_original.exists():
            print(f"Cleaning existing original frames directory: {self.frames_original}")
            for file_path in self.frames_original.iterdir():
                if file_path.is_file():
                    print(f"  Removing: {file_path.name}")
                    file_path.unlink()
                elif file_path.is_dir():
                    print(f"  Removing directory: {file_path}")
                    shutil.rmtree(file_path)
        
        # Clean blurred frames directory
        if self.frames_blurred.exists():
            print(f"Cleaning existing blurred frames directory: {self.frames_blurred}")
            for subdir in self.frames_blurred.iterdir():
                if subdir.is_dir():
                    print(f"  Removing: {subdir}")
                    shutil.rmtree(subdir)
        
        print(f"Frame directories cleaned for fresh processing")
        
    def detect_crop_bounds(self) -> Optional[Tuple]:
        """Detect crop bounds for the video using improved frame sampling strategy."""
        print("\n[STEP 1] Detecting optimal crop bounds with enhanced sampling...")
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print("  [WARNING] Could not open video for crop detection")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the time range of interest
        if self.start_time is not None:
            start_frame = max(0, int(self.start_time * fps))
        else:
            start_frame = 0
            
        if self.duration is not None and self.start_time is not None:
            end_frame = min(total_frames - 1, int((self.start_time + self.duration) * fps))
        else:
            end_frame = total_frames - 1
            
        time_range_frames = end_frame - start_frame + 1
        
        # Double the default sample count (from 5 to 10)
        desired_samples = 10
        print(f"  Target samples: {desired_samples}")
        print(f"  Time range: frames {start_frame} to {end_frame} ({time_range_frames} frames)")
        
        # Get sample frames using improved strategy
        sample_frames = self._get_enhanced_sample_frames(
            start_frame, end_frame, total_frames, desired_samples
        )
        
        print(f"  Selected {len(sample_frames)} sample frames: {sample_frames}")
        
        # Extract frames and analyze their content quality
        valid_bounds = []
        frame_areas = []
        
        for frame_num in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                # Calculate frame brightness/contrast as quality metric
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                brightness_std = np.std(gray)
                
                bounds = find_content_bounds_diagonal(frame)
                if bounds != (0, 0, frame.shape[1], frame.shape[0]):  # Not full frame
                    # Calculate content area
                    content_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                    total_area = frame.shape[0] * frame.shape[1]
                    area_ratio = content_area / total_area
                    
                    valid_bounds.append({
                        'bounds': bounds,
                        'frame_num': frame_num,
                        'content_area': content_area,
                        'area_ratio': area_ratio,
                        'brightness': mean_brightness,
                        'contrast': brightness_std
                    })
                    frame_areas.append(content_area)
                    
                    print(f"    Frame {frame_num}: area={content_area}, ratio={area_ratio:.3f}, brightness={mean_brightness:.1f}")
        
        cap.release()
        
        if not valid_bounds:
            print("  [WARNING] No valid crop bounds detected, using full frame")
            return None
        
        # Filter out frames with significantly smaller areas (outliers)
        filtered_bounds = self._filter_outlier_frames(valid_bounds)
        print(f"  After outlier filtering: {len(filtered_bounds)} frames retained")
        
        if not filtered_bounds:
            print("  [WARNING] All frames filtered out as outliers, using original set")
            filtered_bounds = valid_bounds
        
        # Calculate final bounds using the most restrictive approach
        final_bounds = self._calculate_final_bounds(filtered_bounds)
        
        print(f"  [OK] Final crop bounds: {final_bounds}")
        return final_bounds
    
    def _get_enhanced_sample_frames(self, start_frame: int, end_frame: int, total_frames: int, desired_samples: int) -> List[int]:
        """
        Get sample frames using enhanced strategy with fallback options.
        
        Args:
            start_frame: Start frame of time range of interest
            end_frame: End frame of time range of interest  
            total_frames: Total frames in video
            desired_samples: Number of samples desired
            
        Returns:
            List of frame numbers to sample
        """
        time_range_frames = end_frame - start_frame + 1
        
        # If video has fewer frames than desired samples, use all frames
        if total_frames <= desired_samples:
            print(f"  Video has only {total_frames} frames, using all frames")
            return list(range(0, total_frames, max(1, total_frames // desired_samples)))
        
        sample_frames = []
        
        # Try to get samples from time range of interest first
        if time_range_frames >= desired_samples:
            # Enough frames in time range, sample randomly within it
            available_frames = list(range(start_frame, end_frame + 1))
            sample_frames = sorted(random.sample(available_frames, desired_samples))
            print(f"  Sampled {len(sample_frames)} frames from time range {start_frame}-{end_frame}")
        else:
            # Not enough frames in time range, get what we can and supplement
            if time_range_frames > 0:
                # Get all frames from time range
                sample_frames.extend(range(start_frame, end_frame + 1))
                remaining_samples = desired_samples - len(sample_frames)
                print(f"  Got {len(sample_frames)} frames from time range, need {remaining_samples} more")
                
                # Get remaining samples from rest of video
                other_frames = []
                if start_frame > 0:
                    other_frames.extend(range(0, start_frame))
                if end_frame < total_frames - 1:
                    other_frames.extend(range(end_frame + 1, total_frames))
                
                if other_frames and remaining_samples > 0:
                    additional_samples = min(remaining_samples, len(other_frames))
                    additional_frames = sorted(random.sample(other_frames, additional_samples))
                    sample_frames.extend(additional_frames)
                    print(f"  Added {len(additional_frames)} frames from rest of video")
            else:
                # No valid time range, sample from entire video
                print(f"  No valid time range, sampling from entire video")
                available_frames = list(range(0, total_frames))
                sample_frames = sorted(random.sample(available_frames, min(desired_samples, len(available_frames))))
        
        return sorted(sample_frames)
    
    def _filter_outlier_frames(self, bounds_data: List[Dict]) -> List[Dict]:
        """
        Filter out frames with significantly smaller content areas.
        
        Args:
            bounds_data: List of dictionaries containing bounds and metadata
            
        Returns:
            Filtered list with outliers removed
        """
        if len(bounds_data) <= 2:
            return bounds_data
        
        areas = [frame['content_area'] for frame in bounds_data]
        median_area = np.median(areas)
        
        # Remove frames with areas much smaller than median (less than 70% of median)
        threshold = 0.7 * median_area
        
        filtered = []
        for frame_data in bounds_data:
            if frame_data['content_area'] >= threshold:
                filtered.append(frame_data)
            else:
                print(f"    Filtering out frame {frame_data['frame_num']} (area {frame_data['content_area']} < {threshold:.0f})")
        
        # If we filtered out too many frames, try substitutions
        if len(filtered) < len(bounds_data) * 0.5:  # Less than 50% retained
            print(f"  Too many frames filtered ({len(filtered)}/{len(bounds_data)}), trying substitutions...")
            return self._try_frame_substitutions(bounds_data, filtered)
        
        return filtered
    
    def _try_frame_substitutions(self, original_data: List[Dict], filtered_data: List[Dict]) -> List[Dict]:
        """
        Try to substitute filtered frames with nearby frames.
        
        Args:
            original_data: Original frame data
            filtered_data: Currently filtered data
            
        Returns:
            Enhanced filtered data with substitutions
        """
        # For now, return the original data if too many were filtered
        # In a full implementation, this would try to sample nearby frames
        # and test them for better content area
        print(f"  Substitution logic would be implemented here - using original data for now")
        return original_data
    
    def _calculate_final_bounds(self, bounds_data: List[Dict]) -> Tuple[int, int, int, int]:
        """
        Calculate final crop bounds from filtered frame data.
        
        Args:
            bounds_data: List of frame data with bounds
            
        Returns:
            Final crop bounds tuple
        """
        # Use the most restrictive bounds (smallest area that still contains content)
        all_bounds = [frame['bounds'] for frame in bounds_data]
        
        min_top = max(b[0] for b in all_bounds)
        min_left = max(b[1] for b in all_bounds)
        max_bottom = min(b[2] for b in all_bounds)
        max_right = min(b[3] for b in all_bounds)
        
        return (min_top, min_left, max_bottom, max_right)
    
    def extract_and_process_frames(self) -> bool:
        """Extract frames and apply blur effects."""
        print(f"\n[STEP 2] Processing frames with stride {self.stride}...")
        
        # Always clean directories before processing to avoid data from other runs
        self.clean_frame_directories_before_processing()
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print("  [ERROR] Could not open video")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range based on time cropping
        if self.start_time is not None:
            start_frame = max(0, int(self.start_time * fps))
            print(f"  Time cropping: starting at {self.start_time}s (frame {start_frame})")
        else:
            start_frame = 0
            
        if self.duration is not None and self.start_time is not None:
            end_frame = min(total_frames - 1, int((self.start_time + self.duration) * fps))
            print(f"  Time cropping: duration {self.duration}s, ending at frame {end_frame}")
        else:
            end_frame = total_frames - 1
            
        # Generate frame indices with stride within the time range
        frame_indices = list(range(start_frame, end_frame + 1, self.stride))
        
        # Apply max_frames limit if specified
        if self.max_frames is not None and len(frame_indices) > self.max_frames:
            frame_indices = frame_indices[:self.max_frames]
            print(f"  Limited to {self.max_frames} frames (max_frames setting)")
        
        frames_to_process = len(frame_indices)
        print(f"  Processing {frames_to_process} frames (every {self.stride} frame)...")
        if frame_indices:
            print(f"  Frame range: {frame_indices[0]} to {frame_indices[-1]}")
            if self.start_time is not None:
                start_time_actual = frame_indices[0] / fps
                end_time_actual = frame_indices[-1] / fps
                print(f"  Time range: {start_time_actual:.2f}s to {end_time_actual:.2f}s")
        
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
        """Metrics calculation is now handled by the main pipeline through metrics_calculator.py"""
        print("\n[STEP 4] Metrics calculation handled by main pipeline")
        print("    [INFO] Metrics will be calculated by vapor_complete_pipeline.py using metrics_calculator.py")
        print("    [INFO] This prevents duplicate metric entries and ensures consistency")
        return True
    
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
            
            # Create videos only if enabled
            video_count = 0
            if self.generate_videos:
                video_count = self.create_videos()
            else:
                print("\n[STEP 3] Video generation disabled (generate_videos=False)")
            
            # Calculate metrics
            self.calculate_metrics()
            
            # Summary
            print("\n" + "=" * 60)
            print("[SUCCESS] PIPELINE COMPLETED!")
            print("=" * 60)
            if self.generate_videos:
                print(f"Videos created: {video_count}")
                print(f"Videos directory: {self.videos_blurred}")
            else:
                print("Video generation: Disabled")
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
        description="VAPOR Blur Generator - Generate blurred frames from original video"
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
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: no limit)"
    )
    parser.add_argument(
        "--blur-types",
        nargs='+',
        default=None,
        help="List of blur types to generate (e.g., motion_blur gaussian)"
    )
    parser.add_argument(
        "--intensities",
        nargs='+',
        default=None,
        help="List of blur intensities to generate (e.g., high low)"
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time in seconds for video cropping (default: from beginning)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds for video cropping (default: full video)"
    )
    
    args = parser.parse_args()
    
    # Validate stride
    if args.stride < 1 or args.stride > 100:
        print("Error: Stride must be between 1 and 100")
        return 1
    
    pipeline = BlurGenerator(args.video, args.stride, args.max_frames, args.blur_types, args.intensities, start_time=args.start_time, duration=args.duration)
    success = pipeline.run_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())