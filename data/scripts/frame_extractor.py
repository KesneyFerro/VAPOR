"""
Frame extraction script for VAPOR project.
Extracts frames from video files and saves them with content cropping.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directories to path to import specularity modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from specularity.utils.content_detection import find_content_bounds, crop_to_content


class FrameExtractor:
    """Class for extracting frames from video files."""
    
    def __init__(self, video_path, output_dir=None):
        """
        Initialize frame extractor.
        
        Args:
            video_path: Path to the video file
            output_dir: Output directory for extracted frames (optional)
        """
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        
        if output_dir is None:
            # Use default extracted_frames/original directory
            script_dir = Path(__file__).parent
            self.output_dir = script_dir.parent / "extracted_frames" / "original" / self.video_name
        else:
            self.output_dir = Path(output_dir) / self.video_name
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video: {self.video_name}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Output directory: {self.output_dir}")
    
    def extract_frame(self, frame_number):
        """
        Extract a specific frame and apply content detection.
        
        Args:
            frame_number: Frame number to extract
            
        Returns:
            tuple: (original_frame, cropped_frame) or (None, None) if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return None, None
        
        # Apply content detection and cropping
        try:
            cropped_frame = crop_to_content(frame)
            return frame, cropped_frame
        except Exception as e:
            print(f"Warning: Content detection failed for frame {frame_number}: {e}")
            return frame, frame
    
    def extract_all_frames(self, crop_content=True, save_original=False):
        """
        Extract all frames from the video.
        
        Args:
            crop_content: Whether to apply content detection and cropping
            save_original: Whether to save original frames alongside cropped ones
        """
        print(f"\nExtracting {self.total_frames} frames...")
        
        extracted_count = 0
        failed_count = 0
        
        for frame_num in range(self.total_frames):
            try:
                original_frame, processed_frame = self.extract_frame(frame_num)
                
                if original_frame is None:
                    failed_count += 1
                    continue
                
                # Generate filename
                frame_filename = f"{self.video_name}_extracted_{frame_num:06d}.png"
                frame_path = self.output_dir / frame_filename
                
                # Save the processed frame (cropped or original)
                if crop_content:
                    success = cv2.imwrite(str(frame_path), processed_frame)
                else:
                    success = cv2.imwrite(str(frame_path), original_frame)
                
                if success:
                    extracted_count += 1
                else:
                    failed_count += 1
                    print(f"Failed to save frame {frame_num}")
                
                # Optionally save original frame
                if save_original and crop_content:
                    original_filename = f"{self.video_name}_original_{frame_num:06d}.png"
                    original_path = self.output_dir / original_filename
                    cv2.imwrite(str(original_path), original_frame)
                
                # Progress update
                if (frame_num + 1) % 100 == 0 or frame_num == self.total_frames - 1:
                    progress = (frame_num + 1) / self.total_frames * 100
                    print(f"Progress: {progress:.1f}% ({frame_num + 1}/{self.total_frames})")
                    
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                failed_count += 1
        
        print(f"\nExtraction complete!")
        print(f"Successfully extracted: {extracted_count} frames")
        print(f"Failed extractions: {failed_count} frames")
        print(f"Output directory: {self.output_dir}")
    
    def extract_sample_frames(self, num_samples=10, crop_content=True):
        """
        Extract a sample of frames evenly distributed throughout the video.
        
        Args:
            num_samples: Number of sample frames to extract
            crop_content: Whether to apply content detection and cropping
        """
        if num_samples >= self.total_frames:
            print("Number of samples exceeds total frames. Extracting all frames.")
            return self.extract_all_frames(crop_content)
        
        # Calculate frame indices for sampling
        frame_indices = []
        step = self.total_frames / num_samples
        for i in range(num_samples):
            frame_idx = int(i * step)
            frame_indices.append(frame_idx)
        
        print(f"\nExtracting {num_samples} sample frames from {self.total_frames} total frames...")
        print(f"Frame indices: {frame_indices}")
        
        extracted_count = 0
        failed_count = 0
        
        for i, frame_num in enumerate(frame_indices):
            try:
                original_frame, processed_frame = self.extract_frame(frame_num)
                
                if original_frame is None:
                    failed_count += 1
                    continue
                
                # Generate filename with sample index
                frame_filename = f"{self.video_name}_sample_{i:03d}_frame_{frame_num:06d}.png"
                frame_path = self.output_dir / frame_filename
                
                # Save the processed frame
                if crop_content:
                    success = cv2.imwrite(str(frame_path), processed_frame)
                else:
                    success = cv2.imwrite(str(frame_path), original_frame)
                
                if success:
                    extracted_count += 1
                    print(f"Extracted sample {i+1}/{num_samples}: frame {frame_num}")
                else:
                    failed_count += 1
                    print(f"Failed to save frame {frame_num}")
                    
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                failed_count += 1
        
        print(f"\nSample extraction complete!")
        print(f"Successfully extracted: {extracted_count} frames")
        print(f"Failed extractions: {failed_count} frames")
        print(f"Output directory: {self.output_dir}")
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from video files with content detection")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("-o", "--output", help="Output directory for extracted frames")
    parser.add_argument("-s", "--samples", type=int, help="Extract only N sample frames instead of all frames")
    parser.add_argument("--no-crop", action="store_true", help="Don't apply content detection/cropping")
    parser.add_argument("--save-original", action="store_true", help="Save original frames alongside cropped ones")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found!")
        return 1
    
    try:
        # Create frame extractor
        extractor = FrameExtractor(args.video_path, args.output)
        
        # Extract frames
        if args.samples:
            extractor.extract_sample_frames(
                num_samples=args.samples,
                crop_content=not args.no_crop
            )
        else:
            extractor.extract_all_frames(
                crop_content=not args.no_crop,
                save_original=args.save_original
            )
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    # Interactive mode if no command line arguments
    if len(sys.argv) == 1:
        print("Frame Extractor for VAPOR Project")
        print("=" * 40)
        
        # Get video path from user
        video_path = input("Enter the path to your video file: ").strip().strip('"')
        
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found!")
            sys.exit(1)
        
        # Get extraction options
        print("\nExtraction options:")
        print("1. Extract all frames")
        print("2. Extract sample frames")
        
        choice = input("Choose option (1 or 2): ").strip()
        
        try:
            extractor = FrameExtractor(video_path)
            
            if choice == "2":
                num_samples = int(input("Number of sample frames to extract (default 10): ") or "10")
                extractor.extract_sample_frames(num_samples)
            else:
                # Ask about content cropping
                crop_choice = input("Apply content detection and cropping? (y/n, default y): ").strip().lower()
                crop_content = crop_choice != 'n'
                
                # Ask about saving originals
                if crop_content:
                    save_orig = input("Save original frames alongside cropped ones? (y/n, default n): ").strip().lower()
                    save_original = save_orig == 'y'
                else:
                    save_original = False
                
                extractor.extract_all_frames(crop_content, save_original)
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Command line mode
        sys.exit(main())
