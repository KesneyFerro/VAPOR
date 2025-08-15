"""
VAPOR - Unified Video Blur Processing Pipeline
Processes videos through frame extraction, blur effects, and video reconstruction.

Features:
- Interactive video selection from data/videos/original/
- Efficient single-pass cropping detection
- All blur types (6) in both low and high intensities
- Proper filename conventions (e.g., pat3_gaussian_low.mp4)
- Original video size preservation with black padding
- Configurable stride for frame processing

Usage:
    python blur/main.py [--stride N]
    
Examples:
    python blur/main.py                 # Process all frames
    python blur/main.py --stride 5      # Process every 5th frame
    python blur/main.py --stride 10     # Process every 10th frame
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

# Add specularity modules to path
sys.path.append(str(Path(__file__).parent.parent))
from specularity.utils.content_detection import find_content_bounds, crop_to_content


class VAPORPipeline:
    """Unified VAPOR processing pipeline."""
    
    def __init__(self, stride: int = 1):
        """Initialize the pipeline.
        
        Args:
            stride: Frame processing stride (1 = all frames, 2 = every 2nd frame, etc.)
        """
        self.base_path = Path(__file__).parent.parent  # Go up one level since we'll be in blur folder
        self.videos_path = self.base_path / "data" / "videos" / "original"
        self.output_videos_path = self.base_path / "data" / "videos" / "blurred"
        self.frames_base_path = self.base_path / "data" / "extracted_frames"
        
        # Blur types and intensities
        self.blur_types = ["gaussian", "motion_blur", "outoffocus", "average", "median", "combined"]
        self.intensities = ["low", "high"]
        
        # Processing options
        self.stride = stride  # Process every Nth frame (1 = all frames)
        
        # Video properties (set during processing)
        self.video_config = {}
        self.crop_bounds = None
        
    def get_python_executable(self):
        """Get the correct Python executable path for the virtual environment."""
        return "C:/Users/kesne/Documents/Webdev/MAPLE-25/VAPOR/venv/Scripts/python.exe"
    
    def list_available_videos(self) -> List[Path]:
        """List all available videos in the original directory."""
        if not self.videos_path.exists():
            print(f"[ERROR] Videos directory not found: {self.videos_path}")
            return []
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        videos = []
        
        for file_path in self.videos_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                videos.append(file_path)
        
        return sorted(videos)
    
    def select_video(self) -> Optional[Path]:
        """Interactive video selection."""
        videos = self.list_available_videos()
        
        if not videos:
            print("[ERROR] No videos found in data/videos/original/")
            return None
        
        print("\nAvailable videos:")
        print("=" * 50)
        for i, video in enumerate(videos, 1):
            file_size = video.stat().st_size / (1024 * 1024)  # MB
            print(f"{i:2d}. {video.name} ({file_size:.1f} MB)")
        
        while True:
            try:
                choice = input(f"\nSelect video (1-{len(videos)}): ").strip()
                if not choice:
                    continue
                
                index = int(choice) - 1
                if 0 <= index < len(videos):
                    selected_video = videos[index]
                    
                    # Ask for processing options if stride not set via command line
                    if self.stride == 1:  # Default value, may want to change
                        print(f"\nSelected: {selected_video.name}")
                        print("\nProcessing options:")
                        print("1. Full video (all frames) - SLOW but complete")
                        print("2. Sample frames (every 10th frame) - FAST for testing")
                        print("3. Sample frames (every 5th frame) - MEDIUM")
                        print("4. Custom stride - specify your own")
                        
                        while True:
                            mode = input("\nChoose processing mode (1-4): ").strip()
                            if mode in ['1', '2', '3', '4']:
                                if mode == '1':
                                    self.stride = 1  # Process all frames
                                elif mode == '2':
                                    self.stride = 10  # Every 10th frame
                                elif mode == '3':
                                    self.stride = 5   # Every 5th frame
                                else:  # mode == '4'
                                    while True:
                                        try:
                                            custom_stride = int(input("Enter custom stride (1-50): "))
                                            if 1 <= custom_stride <= 50:
                                                self.stride = custom_stride
                                                break
                                            else:
                                                print("Please enter a number between 1 and 50")
                                        except ValueError:
                                            print("Please enter a valid number")
                                break
                            print("Please enter 1, 2, 3, or 4")
                    
                    return selected_video
                else:
                    print(f"[ERROR] Please enter a number between 1 and {len(videos)}")
            except ValueError:
                print("[ERROR] Please enter a valid number")
            except KeyboardInterrupt:
                print("\n[CANCELLED] Operation cancelled by user")
                return None
    
    def extract_video_config(self, video_path: Path) -> Dict:
        """Extract video configuration for reconstruction."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        config = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        
        print("Video configuration:")
        print(f"  Resolution: {config['width']}x{config['height']}")
        print(f"  FPS: {config['fps']:.2f}")
        print(f"  Frame count: {config['frame_count']}")
        print(f"  Stride: {self.stride} (processing every {self.stride} frame{'s' if self.stride > 1 else ''})")
        
        return config
    
    def detect_crop_bounds_once(self, video_path: Path) -> Optional[Tuple]:
        """Detect crop bounds from first few frames to use for all processing."""
        print("\n[STEP 1] Detecting optimal crop bounds...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Sample frames from different parts of the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]
        
        all_bounds = []
        
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                try:
                    bounds = find_content_bounds(frame)
                    all_bounds.append(bounds)
                    print(f"  Frame {frame_idx}: bounds {bounds}")
                except Exception as e:
                    print(f"  Frame {frame_idx}: detection failed - {e}")
        
        cap.release()
        
        if not all_bounds:
            print("  [WARNING] No valid crop bounds detected, using full frame")
            return None
        
        # Use the most restrictive bounds (smallest crop area)
        min_top = max(b[0] for b in all_bounds)
        min_left = max(b[1] for b in all_bounds) 
        max_bottom = min(b[2] for b in all_bounds)
        max_right = min(b[3] for b in all_bounds)
        
        final_bounds = (min_top, min_left, max_bottom, max_right)
        print(f"  [OK] Final crop bounds: {final_bounds}")
        
        return final_bounds
    
    def extract_and_process_frames(self, video_path: Path, video_name: str) -> bool:
        """Extract all frames and apply all blur effects."""
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
        original_frames_dir = self.frames_base_path / "original" / video_name
        original_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Blurred frames directories
        blurred_base_dir = self.frames_base_path / "blurred" / video_name
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
                
                # Save original frame first
                frame_filename = f"{video_name}_{frame_num:06d}.png"
                original_frame_path = original_frames_dir / frame_filename
                cv2.imwrite(str(original_frame_path), frame)
                
                # Apply crop bounds if available
                cropped_frame = frame
                if self.crop_bounds:
                    top, left, bottom, right = self.crop_bounds
                    cropped_frame = frame[top:bottom, left:right]
                
                # Apply all blur effects and intensities
                for blur_type in self.blur_types:
                    for intensity in self.intensities:
                        # Apply blur
                        blurred_frame = self.apply_blur_effect(cropped_frame, blur_type, intensity)
                        
                        # Save frame with proper numbering (use original frame number)
                        frame_path = frame_dirs[f"{blur_type}_{intensity}"] / frame_filename
                        cv2.imwrite(str(frame_path), blurred_frame)
                
                processed_count += 1
                
                # Progress update (every 50 processed frames or at end)
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
    
    def apply_blur_effect(self, image: np.ndarray, blur_type: str, intensity: str) -> np.ndarray:
        """Apply specific blur effect to an image."""
        if blur_type == "gaussian":
            return self.apply_gaussian_blur(image, intensity)
        elif blur_type == "motion_blur":
            return self.apply_motion_blur(image, intensity)
        elif blur_type == "outoffocus":
            return self.apply_out_of_focus_blur(image, intensity)
        elif blur_type == "average":
            return self.apply_average_blur(image, intensity)
        elif blur_type == "median":
            return self.apply_median_blur(image, intensity)
        elif blur_type == "combined":
            return self.apply_combined_blur(image, intensity)
        else:
            return image
    
    def apply_gaussian_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
        """Apply Gaussian blur."""
        if intensity == "low":
            kernel_size = (5, 5)
            sigma = 1.5
        else:  # high
            kernel_size = (15, 15)
            sigma = 5.0
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def apply_motion_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
        """Apply motion blur with random angle."""        
        if intensity == "low":
            kernel_size = 10
            # Random angle between 0-180 degrees for low intensity
            angle = random.randint(0, 180)
        else:  # high
            kernel_size = 25
            # Different random angle range for high intensity
            angle = random.randint(0, 180)
        
        # Create motion blur kernel at specified angle
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # Create motion blur line
        center = kernel_size // 2
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * dx)
            y = int(center + offset * dy)
            
            # Check bounds
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # Normalize kernel
        if np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)
        else:
            # Fallback to simple horizontal blur if kernel creation fails
            kernel = np.zeros((kernel_size, kernel_size))
            middle_row = kernel_size // 2
            kernel[middle_row, :] = 1 / kernel_size
        
        return cv2.filter2D(image, -1, kernel)
    
    def apply_out_of_focus_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
        """Apply out-of-focus blur."""
        if intensity == "low":
            radius = 3
        else:  # high
            radius = 8
        
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        center = radius
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance <= radius:
                    kernel[i, j] = 1
        
        kernel = kernel / np.sum(kernel)
        
        if len(image.shape) == 3:
            blurred = np.zeros_like(image)
            for channel in range(image.shape[2]):
                blurred[:, :, channel] = signal.convolve2d(
                    image[:, :, channel], kernel, mode='same', boundary='symm'
                )
            return blurred.astype(np.uint8)
        else:
            return signal.convolve2d(image, kernel, mode='same', boundary='symm').astype(np.uint8)
    
    def apply_average_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
        """Apply average blur."""
        if intensity == "low":
            kernel_size = (5, 5)
        else:  # high
            kernel_size = (15, 15)
        return cv2.blur(image, kernel_size)
    
    def apply_median_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
        """Apply median blur."""
        if intensity == "low":
            kernel_size = 5
        else:  # high
            kernel_size = 15
        return cv2.medianBlur(image, kernel_size)
    
    def apply_combined_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
        """Apply combined blur: motion blur + out-of-focus blur + median blur."""
        # Apply motion blur first
        blurred = self.apply_motion_blur(image, intensity)
        
        # Apply out-of-focus blur
        blurred = self.apply_out_of_focus_blur(blurred, intensity)
        
        # Apply median blur last (helps reduce noise)
        blurred = self.apply_median_blur(blurred, intensity)
        
        return blurred
    
    def pad_frame_to_original_size(self, frame: np.ndarray) -> np.ndarray:
        """Pad frame with black pixels to match original video dimensions."""
        target_height = self.video_config['height']
        target_width = self.video_config['width']
        
        if frame.shape[:2] == (target_height, target_width):
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        
        # Create black canvas
        if len(frame.shape) == 3:
            padded_frame = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
        else:
            padded_frame = np.zeros((target_height, target_width), dtype=frame.dtype)
        
        # Center the frame
        start_y = (target_height - frame_height) // 2
        start_x = (target_width - frame_width) // 2
        
        # Handle oversized frames
        if start_y < 0 or start_x < 0 or frame_height > target_height or frame_width > target_width:
            scale = min(target_width / frame_width, target_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            start_y = (target_height - new_height) // 2
            start_x = (target_width - new_width) // 2
            frame_height, frame_width = new_height, new_width
        
        # Place frame in center
        end_y = start_y + frame_height
        end_x = start_x + frame_width
        
        if len(frame.shape) == 3:
            padded_frame[start_y:end_y, start_x:end_x, :] = frame
        else:
            padded_frame[start_y:end_y, start_x:end_x] = frame
            
        return padded_frame
    
    def create_videos_from_frames(self, video_name: str) -> int:
        """Create videos from all processed frame sets."""
        print("\n[STEP 3] Creating videos...")
        
        success_count = 0
        total_count = len(self.blur_types) * len(self.intensities)
        
        # Ensure output directory exists
        self.output_videos_path.mkdir(parents=True, exist_ok=True)
        
        for blur_type in self.blur_types:
            for intensity in self.intensities:
                dir_name = f"{blur_type}_{intensity}"
                frames_dir = self.frames_base_path / "blurred" / video_name / dir_name
                
                if not frames_dir.exists():
                    print(f"    [SKIP] {dir_name}: No frames directory")
                    continue
                
                # Get frame files
                frame_files = sorted(frames_dir.glob("*.png"))
                if not frame_files:
                    print(f"    [SKIP] {dir_name}: No frames found")
                    continue
                
                # Create video
                output_filename = f"{video_name}_{blur_type}_{intensity}.mp4"
                output_path = self.output_videos_path / output_filename
                
                if self.create_single_video(frame_files, output_path):
                    print(f"    [OK] {output_filename}")
                    success_count += 1
                else:
                    print(f"    [FAILED] {output_filename}")
        
        print(f"  [OK] Video creation complete: {success_count}/{total_count} videos created")
        return success_count
    
    def create_single_video(self, frame_files: List[Path], output_path: Path) -> bool:
        """Create a single video from frame files."""
        try:
            fps = self.video_config['fps']
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video_writer = None
            
            for frame_file in frame_files:
                # Load and pad frame
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue
                
                frame = self.pad_frame_to_original_size(frame)
                
                # Initialize video writer with first frame
                if video_writer is None:
                    height, width = frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(output_path), 
                        fourcc, 
                        fps, 
                        (width, height)
                    )
                    
                    if not video_writer.isOpened():
                        return False
                
                video_writer.write(frame)
            
            if video_writer is not None:
                video_writer.release()
                return True
            
            return False
            
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
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
            success_count = self.create_videos_from_frames(video_name)
            
            # Summary
            print("\n" + "=" * 60)
            print("[SUCCESS] PIPELINE COMPLETED!")
            print("=" * 60)
            print(f"Processed video: {video_name}")
            print(f"Videos created: {success_count}")
            print(f"Output videos: {self.output_videos_path}")
            print(f"Extracted frames: {self.frames_base_path}")
            print(f"Stride used: {self.stride}")
            print("\nGenerated videos:")
            
            # List generated videos
            for blur_type in self.blur_types:
                for intensity in self.intensities:
                    video_file = self.output_videos_path / f"{video_name}_{blur_type}_{intensity}.mp4"
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
