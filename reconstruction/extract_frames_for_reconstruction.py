#!/usr/bin/env python
"""
Extract frames from pat3.mp4 with stride 3 to get sufficient frames for reconstruction.
Target: 56 frames (minimum 51 for netvlad + 5 extra as requested)
"""

import cv2
import sys
from pathlib import Path

def extract_frames_with_stride(video_path, output_dir, stride=3, max_frames=56):
    """Extract frames from video with specified stride."""
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path.name}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Target frames: {max_frames}")
    print(f"Stride: {stride}")
    print(f"Output: {output_dir}")
    
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract every stride-th frame
        if frame_count % stride == 0:
            # Generate filename using 6-digit zero-padded numbering
            filename = f"{video_path.stem}_{extracted_count:06d}.png"
            filepath = output_dir / filename
            
            # Save frame
            if cv2.imwrite(str(filepath), frame):
                extracted_count += 1
                if extracted_count % 10 == 0:
                    print(f"Extracted {extracted_count}/{max_frames} frames...")
            else:
                print(f"Error saving frame {extracted_count}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"Extracted {extracted_count} frames")
    print(f"Saved to: {output_dir}")
    
    return extracted_count >= max_frames

def main():
    """Main function."""
    
    # Paths
    video_path = r"S:\Kesney\VAPOR\data\videos\original\pat3.mp4"
    output_dir = r"S:\Kesney\VAPOR\data\frames\original\pat3"
    
    # Parameters
    stride = 3
    target_frames = 56  # Minimum 51 for netvlad + 5 extra
    
    print("VAPOR Frame Extraction for Reconstruction")
    print("=" * 50)
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    # Clear existing frames (optional - ask user)
    output_path = Path(output_dir)
    if output_path.exists():
        existing_files = list(output_path.glob("*.png"))
        if existing_files:
            print(f"Found {len(existing_files)} existing PNG files in output directory.")
            response = input("Delete existing files? (y/n): ").strip().lower()
            if response == 'y':
                for f in existing_files:
                    f.unlink()
                print("Existing files deleted.")
    
    # Extract frames
    success = extract_frames_with_stride(video_path, output_dir, stride, target_frames)
    
    if success:
        print(f"\nSuccess! Extracted sufficient frames for reconstruction.")
        print(f"You can now run: python reconstruction_pipeline.py --folder \"{output_dir}\"")
        return 0
    else:
        print(f"\nWarning: Could not extract the target number of frames.")
        return 1

if __name__ == "__main__":
    sys.exit(main())