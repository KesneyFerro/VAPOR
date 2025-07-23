#!/usr/bin/env python3
"""
Test script to demonstrate the new functionality
"""

import cv2
import numpy as np
from test import VideoFrameProcessor

def create_test_image():
    """Create a test image with black borders to test cropping"""
    # Create a 400x400 black image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Add a colored rectangle in the center with some content
    # This simulates a video frame with black borders
    cv2.rectangle(img, (50, 50), (350, 350), (100, 150, 200), -1)  # Orange rectangle
    cv2.rectangle(img, (100, 100), (300, 300), (50, 200, 50), -1)   # Green rectangle
    cv2.circle(img, (200, 200), 50, (255, 255, 255), -1)            # White circle
    
    return img

def test_edge_detection():
    """Test the edge detection and cropping functionality"""
    print("Testing edge detection and cropping...")
    
    # Create a mock processor (we'll just test the methods)
    class MockProcessor:
        def find_content_bounds(self, frame):
            # Convert to grayscale for edge detection
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            h, w = gray.shape
            
            # Find first non-black pixel from each direction (from center edges)
            # Top edge going down
            top_bound = 0
            for y in range(h):
                if gray[y, w//2] > 0:  # Non-black pixel
                    top_bound = y
                    break
            
            # Bottom edge going up
            bottom_bound = h - 1
            for y in range(h-1, -1, -1):
                if gray[y, w//2] > 0:  # Non-black pixel
                    bottom_bound = y
                    break
            
            # Left edge going right
            left_bound = 0
            for x in range(w):
                if gray[h//2, x] > 0:  # Non-black pixel
                    left_bound = x
                    break
            
            # Right edge going left
            right_bound = w - 1
            for x in range(w-1, -1, -1):
                if gray[h//2, x] > 0:  # Non-black pixel
                    right_bound = x
                    break
            
            # Add small padding to ensure we don't crop too aggressively
            padding = 5
            top_bound = max(0, top_bound - padding)
            bottom_bound = min(h - 1, bottom_bound + padding)
            left_bound = max(0, left_bound - padding)
            right_bound = min(w - 1, right_bound + padding)
            
            return top_bound, bottom_bound, left_bound, right_bound
        
        def crop_to_content(self, frame):
            """Crop frame to content bounds"""
            top, bottom, left, right = self.find_content_bounds(frame)
            
            # Crop the frame
            if len(frame.shape) == 3:
                cropped = frame[top:bottom+1, left:right+1, :]
            else:
                cropped = frame[top:bottom+1, left:right+1]
            
            return cropped
    
    processor = MockProcessor()
    test_img = create_test_image()
    
    print(f"Original image shape: {test_img.shape}")
    
    # Test edge detection
    bounds = processor.find_content_bounds(test_img)
    print(f"Content bounds (top, bottom, left, right): {bounds}")
    
    # Test cropping
    cropped = processor.crop_to_content(test_img)
    print(f"Cropped image shape: {cropped.shape}")
    
    # Save test images to see the results
    cv2.imwrite("test_original.png", test_img)
    cv2.imwrite("test_cropped.png", cropped)
    print("Saved test_original.png and test_cropped.png")

def test_pipeline_parsing():
    """Test the pipeline parsing functionality"""
    print("\nTesting pipeline parsing...")
    
    class MockProcessor:
        def parse_pipeline(self, pipeline_str):
            """Parse pipeline string like '57' into ['5', '7'] or '5 7' into ['5', '7']"""
            pipeline_str = pipeline_str.strip()
            
            # If contains spaces, split by spaces
            if ' ' in pipeline_str:
                return pipeline_str.split()
            
            # Otherwise, treat each character as a separate operation
            return list(pipeline_str)
    
    processor = MockProcessor()
    
    test_cases = [
        "57",      # Should become ['5', '7']
        "5 7",     # Should become ['5', '7']
        "123",     # Should become ['1', '2', '3']
        "1 2 3",   # Should become ['1', '2', '3']
        "a0",      # Should become ['a', '0']
    ]
    
    for test_case in test_cases:
        result = processor.parse_pipeline(test_case)
        print(f"Pipeline '{test_case}' -> {result}")

if __name__ == "__main__":
    test_edge_detection()
    test_pipeline_parsing()
    print("\nAll tests completed!")
    print("\nTo use the new functionality:")
    print("1. Run the main script with: python test.py")
    print("2. When prompted for selection, try:")
    print("   - '5 7 57' to get B Channel, CLAHE, and B+CLAHE pipeline")
    print("   - '57 68' to get B+CLAHE and G+Gamma pipelines")
    print("3. During runtime, press 'p' to manage pipelines interactively")
    print("   - Type '+57' to add B Channel + CLAHE pipeline")
    print("   - Type '-57' to remove that pipeline")
    print("   - Type 'clear' to remove all pipelines")
