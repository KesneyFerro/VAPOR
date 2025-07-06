"""
Create a test video with RGB patterns to verify channel filtering in the video processor
"""

import cv2
import numpy as np
import os

def create_rgb_test_video():
    """Create a test video with RGB color patterns"""
    output_path = "rgb_test_video.mp4"
    
    # Video properties
    width, height = 900, 400
    fps = 30
    duration = 5  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video: {output_path}")
    print(f"Frames: {total_frames}, Size: {width}x{height}")
    
    for i in range(total_frames):
        # Create frame with RGB stripes
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Divide into 3 sections
        section_width = width // 3
        
        # Section 1: Red (BGR: 0, 0, 255)
        frame[:, 0:section_width] = [0, 0, 255]
        
        # Section 2: Green (BGR: 0, 255, 0)  
        frame[:, section_width:section_width*2] = [0, 255, 0]
        
        # Section 3: Blue (BGR: 255, 0, 0)
        frame[:, section_width*2:width] = [255, 0, 0]
        
        # Add frame number
        cv2.putText(frame, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add color labels
        cv2.putText(frame, 'RED', (section_width//2 - 50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(frame, 'GREEN', (section_width + section_width//2 - 70, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(frame, 'BLUE', (section_width*2 + section_width//2 - 60, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Add changing pattern to make it more interesting
        if i % 30 < 15:  # Change every half second
            # Add some noise to make it more realistic
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            noise = rng.integers(0, 50, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")
    return output_path

def test_video_with_processor():
    """Test the created video with the VideoFrameProcessor"""
    video_path = create_rgb_test_video()
    
    print("\nNow testing with VideoFrameProcessor...")
    print(f"Video path: {video_path}")
    print("\nInstructions:")
    print("1. At the initial selection, type: 1 3 4 5 (Original, R, G, B channels)")
    print("2. Look for RGB channel separation patterns")
    print("3. Press 'c' to advance frames, 'q' to quit")
    
    try:
        from test import VideoFrameProcessor
        processor = VideoFrameProcessor(video_path)
        processor.process_video()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if os.path.exists(video_path):
            print(f"\nCleaning up test video: {video_path}")
            os.remove(video_path)

if __name__ == "__main__":
    test_video_with_processor()
