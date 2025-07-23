"""
Example usage of the video frame processor.
This script demonstrates how to use the VideoFrameProcessor class.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path to import specularity modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from specularity.core.video_processor import VideoFrameProcessor


def example_usage():
    """Basic usage example of the VideoFrameProcessor."""
    # Example video path - replace with your actual video path
    video_path = input("Enter the path to your video file: ").strip().strip('"')
    
    # Check if video exists
    if not os.path.exists(video_path):
        print("Video file not found. Please check the path.")
        print("Common video formats supported: .mp4, .avi, .mov, .mkv, .wmv")
        return
    
    # Create processor and run
    try:
        processor = VideoFrameProcessor(video_path)
        processor.process_video()
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    example_usage()
