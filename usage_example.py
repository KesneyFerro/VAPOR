"""
Example usage of the video frame processor.
This script demonstrates how to use the VideoFrameProcessor class.
"""

from test import VideoFrameProcessor
import os

def example_usage():
    # Example video path - replace with your actual video path
    video_path = r"C:\Users\kesne\Downloads\pat3_raw.mp4"
    
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
