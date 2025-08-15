"""
Main entry point for VAPOR (Video Analysis Processing for Object Recognition) project.
Specularity detection and analysis tool for video frames.
"""

import os
import sys
from pathlib import Path

# Add project modules to path
sys.path.append(str(Path(__file__).parent))

from specularity.core.video_processor import VideoFrameProcessor
from data.utils.filename_normalizer import normalize_videos_in_directory


def get_video_files(videos_dir):
    """Get all video files from the videos directory."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    videos = []
    
    if videos_dir.exists():
        for file_path in videos_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                videos.append(file_path)
    
    return sorted(videos)


def select_video():
    """Select a video from the videos/original directory."""
    videos_dir = Path(__file__).parent / "data" / "videos" / "original"
    
    # Normalize video filenames first
    print("Normalizing video filenames...")
    normalize_videos_in_directory(str(videos_dir))
    
    # Get available videos
    videos = get_video_files(videos_dir)
    
    if not videos:
        print("\nNo videos found in data/videos/original directory.")
        print("Please add video files to data/videos/original/ and run the program again.")
        return None
    
    # Display available videos
    print(f"\nAvailable videos in {videos_dir}:")
    print("-" * 40)
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video.name}")
    
    # Get user selection
    while True:
        try:
            choice = input(f"\nSelect a video (1-{len(videos)}): ").strip()
            index = int(choice) - 1
            
            if 0 <= index < len(videos):
                return str(videos[index])
            else:
                print(f"Please enter a number between 1 and {len(videos)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return None


def main():
    """Main entry point for the application."""
    print("VAPOR - Video Analysis Processing for Object Recognition")
    print("=" * 60)
    
    # Select video from available options
    video_path = select_video()
    
    if not video_path:
        print("No video selected. Exiting.")
        return 1
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return 1
    
    try:
        processor = VideoFrameProcessor(video_path)
        processor.process_video()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
