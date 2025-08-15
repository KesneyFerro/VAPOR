"""
Video Reconstructor Script
Creates a new video from a sequence of frames using the same configuration as a reference video.

Usage:
    python video_reconstructor.py <reference_video> <frames_folder> <output_video>
"""

import cv2
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional


class VideoReconstructor:
    """
    Reconstructs a video from frame sequences using reference video configuration.
    """
    
    def __init__(self, reference_video_path: str):
        """
        Initialize with reference video to extract configuration.
        
        Args:
            reference_video_path: Path to the reference video file
        """
        self.reference_video_path = Path(reference_video_path)
        if not self.reference_video_path.exists():
            raise FileNotFoundError(f"Reference video not found: {reference_video_path}")
        
        self.video_config = self._extract_video_config()
    
    def _extract_video_config(self) -> dict:
        """
        Extract video configuration from reference video.
        
        Returns:
            Dictionary containing video configuration parameters
        """
        cap = cv2.VideoCapture(str(self.reference_video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open reference video: {self.reference_video_path}")
        
        config = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        
        print("Reference video configuration:")
        print(f"  Resolution: {config['width']}x{config['height']}")
        print(f"  FPS: {config['fps']}")
        print(f"  Frame count: {config['frame_count']}")
        print(f"  FourCC: {config['fourcc']}")
        
        return config
    
    def _get_frame_files(self, frames_folder: str) -> List[Path]:
        """
        Get sorted list of frame files from the folder.
        
        Args:
            frames_folder: Path to folder containing frame images
            
        Returns:
            Sorted list of frame file paths
        """
        frames_path = Path(frames_folder)
        if not frames_path.exists():
            raise FileNotFoundError(f"Frames folder not found: {frames_folder}")
        
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        frame_files = []
        for file_path in frames_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                frame_files.append(file_path)
        
        if not frame_files:
            raise ValueError(f"No image files found in: {frames_folder}")
        
        # Sort files naturally (handle numeric sequences correctly)
        frame_files.sort(key=lambda x: self._natural_sort_key(x.name))
        
        print(f"Found {len(frame_files)} frame files")
        return frame_files
    
    def _natural_sort_key(self, filename: str) -> List:
        """
        Create a sort key for natural sorting of filenames with numbers.
        
        Args:
            filename: The filename to create sort key for
            
        Returns:
            Sort key for natural sorting
        """
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
    
    def _pad_frame_to_original_size(self, frame: np.ndarray) -> np.ndarray:
        """
        Pad frame with black pixels to match reference video dimensions while preserving aspect ratio.
        Centers the cropped content and fills surrounding area with black.
        
        Args:
            frame: Input frame array
            
        Returns:
            Padded frame array matching original video dimensions
        """
        target_height = self.video_config['height']
        target_width = self.video_config['width']
        
        # If frame already matches target dimensions, return as-is
        if frame.shape[:2] == (target_height, target_width):
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        
        # Create black canvas with target dimensions
        if len(frame.shape) == 3:  # Color image
            padded_frame = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
        else:  # Grayscale image
            padded_frame = np.zeros((target_height, target_width), dtype=frame.dtype)
        
        # Calculate position to center the frame
        start_y = (target_height - frame_height) // 2
        start_x = (target_width - frame_width) // 2
        
        # Handle cases where frame is larger than target (should not happen with cropped frames, but safety check)
        if start_y < 0 or start_x < 0 or frame_height > target_height or frame_width > target_width:
            # If frame is larger, resize it to fit while maintaining aspect ratio
            scale = min(target_width / frame_width, target_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Recalculate center position
            start_y = (target_height - new_height) // 2
            start_x = (target_width - new_width) // 2
            frame_height, frame_width = new_height, new_width
        
        # Place the frame in the center of the black canvas
        end_y = start_y + frame_height
        end_x = start_x + frame_width
        
        if len(frame.shape) == 3:
            padded_frame[start_y:end_y, start_x:end_x, :] = frame
        else:
            padded_frame[start_y:end_y, start_x:end_x] = frame
            
        return padded_frame
    
    def create_video(self, frames_folder: str, output_path: str, 
                    custom_fps: Optional[float] = None) -> bool:
        """
        Create video from frame sequence using reference video configuration.
        
        Args:
            frames_folder: Path to folder containing frame images
            output_path: Path for output video file
            custom_fps: Optional custom FPS (uses reference FPS if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get frame files
            frame_files = self._get_frame_files(frames_folder)
            
            # Setup output video writer
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fps = custom_fps if custom_fps is not None else self.video_config['fps']
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for compatibility
            
            # Initialize video writer (will be set up with first frame)
            video_writer = None
            
            print(f"Creating video: {output_path}")
            print(f"Using FPS: {fps}")
            
            # Process each frame
            for i, frame_file in enumerate(frame_files):
                # Load frame
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    print(f"Warning: Could not load frame {frame_file}")
                    continue
                
                # Pad frame to match reference video dimensions with black borders
                frame = self._pad_frame_to_original_size(frame)
                
                # Initialize video writer with first valid frame
                if video_writer is None:
                    height, width = frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(output_path), 
                        fourcc, 
                        fps, 
                        (width, height)
                    )
                    
                    if not video_writer.isOpened():
                        raise ValueError("Failed to initialize video writer")
                
                # Write frame to video
                video_writer.write(frame)
                
                # Progress indicator
                if (i + 1) % 10 == 0 or i == len(frame_files) - 1:
                    print(f"Processed {i + 1}/{len(frame_files)} frames")
            
            # Release video writer
            if video_writer is not None:
                video_writer.release()
                print(f"Video created successfully: {output_path}")
                return True
            else:
                print("Error: No valid frames found")
                return False
                
        except Exception as e:
            print(f"Error creating video: {e}")
            return False


def main():
    """Main function to handle command line arguments and execute video reconstruction."""
    parser = argparse.ArgumentParser(
        description="Create a video from frame sequences using reference video configuration"
    )
    parser.add_argument(
        "reference_video", 
        help="Path to reference video file"
    )
    parser.add_argument(
        "frames_folder", 
        help="Path to folder containing frame images"
    )
    parser.add_argument(
        "output_video", 
        help="Path for output video file"
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        help="Custom FPS (uses reference video FPS if not specified)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create video reconstructor
        reconstructor = VideoReconstructor(args.reference_video)
        
        # Create video from frames
        success = reconstructor.create_video(
            args.frames_folder, 
            args.output_video, 
            args.fps
        )
        
        if success:
            print("\n[SUCCESS] Video reconstruction completed successfully!")
            return 0
        else:
            print("\n[FAILED] Video reconstruction failed!")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
