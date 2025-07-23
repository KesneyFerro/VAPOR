#!/usr/bin/env python3
"""
Test script for core functionality of specularity detection.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path to import specularity modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from specularity.utils.content_detection import find_content_bounds, crop_to_content
from specularity.utils.pipeline_manager import PipelineManager


def create_test_image():
    """Create a test image with black borders to test cropping."""
    # Create a 400x400 black image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Add a colored rectangle in the center with some content
    # This simulates a video frame with black borders
    cv2.rectangle(img, (50, 50), (350, 350), (100, 150, 200), -1)  # Orange rectangle
    cv2.rectangle(img, (100, 100), (300, 300), (50, 200, 50), -1)   # Green rectangle
    cv2.circle(img, (200, 200), 50, (255, 255, 255), -1)            # White circle
    
    return img


def test_content_detection():
    """Test the content detection and cropping functionality."""
    print("Testing content detection and cropping...")
    
    test_img = create_test_image()
    
    print(f"Original image shape: {test_img.shape}")
    
    # Test edge detection
    bounds = find_content_bounds(test_img)
    print(f"Content bounds (top, bottom, left, right): {bounds}")
    
    # Test cropping
    cropped = crop_to_content(test_img)
    print(f"Cropped image shape: {cropped.shape}")
    
    # Save test images to see the results
    cv2.imwrite("test_original.png", test_img)
    cv2.imwrite("test_cropped.png", cropped)
    print("Saved test_original.png and test_cropped.png")
    
    return True


def test_pipeline_management():
    """Test the pipeline management functionality."""
    print("\nTesting pipeline management...")
    
    # Create available manipulations (same as in frame operations)
    available_manipulations = {
        '1': 'Original',
        '2': 'Grayscale', 
        '3': 'R Channel',
        '4': 'G Channel',
        '5': 'B Channel',
        '6': 'Histogram Normalization',
        '7': 'Grayscale + CLAHE',
        '8': 'Gamma Correction (γ=1.5)',
        '9': 'Bilateral Filtering',
        '0': 'Unsharp Masking',
        'a': 'Logarithmic Transformation'
    }
    
    pipeline_manager = PipelineManager(available_manipulations)
    
    # Test pipeline parsing
    test_cases = [
        "57",      # Should become ['5', '7']
        "5 7",     # Should become ['5', '7']
        "123",     # Should become ['1', '2', '3']
        "1 2 3",   # Should become ['1', '2', '3']
        "a0",      # Should become ['a', '0']
    ]
    
    print("Testing pipeline parsing:")
    for test_case in test_cases:
        result = pipeline_manager.parse_pipeline(test_case)
        print(f"Pipeline '{test_case}' -> {result}")
    
    # Test pipeline management
    print("\nTesting pipeline addition/removal:")
    
    # Add some pipelines
    pipeline_manager.add_pipeline("57")  # B Channel + CLAHE
    pipeline_manager.add_pipeline("68")  # G Channel + Gamma
    print(f"Added pipelines: {pipeline_manager.pipelines}")
    
    # Test command processing
    print("\nTesting command processing:")
    commands = [
        "+90",      # Add Bilateral + Unsharp
        "-57",      # Remove B + CLAHE
        "+123 -68", # Add Original+Gray+R, Remove G+Gamma
        "clear"     # Clear all
    ]
    
    for command in commands:
        print(f"Command: '{command}'")
        changed = pipeline_manager.process_commands(command)
        print(f"Changed: {changed}, Pipelines: {pipeline_manager.pipelines}")
    
    return True


def run_all_tests():
    """Run all functionality tests."""
    print("Running Core Functionality Tests")
    print("=" * 40)
    
    tests_passed = 0
    tests_total = 2
    
    try:
        if test_content_detection():
            tests_passed += 1
            print("✓ Content detection test passed")
        else:
            print("✗ Content detection test failed")
    except Exception as e:
        print(f"✗ Content detection test failed with error: {e}")
    
    try:
        if test_pipeline_management():
            tests_passed += 1
            print("✓ Pipeline management test passed")
        else:
            print("✗ Pipeline management test failed")
    except Exception as e:
        print(f"✗ Pipeline management test failed with error: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("All tests passed! ✓")
        return True
    else:
        print("Some tests failed! ✗")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    print("\nTo use the new functionality:")
    print("1. Run the main script with: python main.py")
    print("2. When prompted for selection, try:")
    print("   - '5 7 57' to get B Channel, CLAHE, and B+CLAHE pipeline")
    print("   - '57 68' to get B+CLAHE and G+Gamma pipelines")
    print("3. During runtime, press 'p' to manage pipelines interactively")
    print("   - Type '+57' to add B Channel + CLAHE pipeline")
    print("   - Type '-57' to remove that pipeline")
    print("   - Type 'clear' to remove all pipelines")
    
    sys.exit(0 if success else 1)
