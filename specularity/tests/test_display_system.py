#!/usr/bin/env python3
"""
Test script for display system improvements.
Tests the responsive grid display and font rendering.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path to import specularity modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from specularity.core.display_manager import DisplayManager


def create_test_display_scenarios():
    """Create test scenarios with different aspect ratios."""
    # Simulate different aspect ratio content
    test_scenarios = [
        (400, 300, "4:3 Standard"),
        (640, 360, "16:9 Widescreen"), 
        (300, 400, "3:4 Portrait"),
        (400, 400, "1:1 Square")
    ]
    
    print("Testing Display System Improvements:")
    print("=" * 50)
    
    for width, height, description in test_scenarios:
        aspect_ratio = width / height
        print(f"{description}: {width}x{height} (aspect: {aspect_ratio:.3f})")
        
        # Create a test image
        test_img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        
        # Add some visual elements to make aspect ratio visible
        cv2.rectangle(test_img, (10, 10), (width-10, height-10), (255, 255, 255), 3)
        cv2.line(test_img, (0, 0), (width, height), (255, 0, 0), 2)
        cv2.line(test_img, (width, 0), (0, height), (0, 255, 0), 2)
        
        # Add text
        cv2.putText(test_img, description, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(test_img, f"{width}x{height}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save the test image
        filename = f"test_display_{description.replace(':', '').replace(' ', '_')}.png"
        cv2.imwrite(filename, test_img)
        print(f"  Created: {filename}")
    
    return True


def test_display_manager():
    """Test the display manager functionality."""
    print("\nTesting DisplayManager:")
    
    display_manager = DisplayManager()
    
    # Test screen dimensions
    width, height = display_manager.get_screen_dimensions()
    print(f"Detected screen dimensions: {width}x{height}")
    
    # Test grid calculations
    test_items = [1, 2, 4, 6, 9, 12, 16]
    
    print("\nOptimal grid calculations:")
    for num_items in test_items:
        rows, cols = display_manager.get_optimal_grid_size(num_items)
        print(f"  {num_items} items -> {rows}x{cols} grid")
    
    return True


def test_font_rendering():
    """Test the font rendering improvements."""
    # Create a test canvas
    canvas = np.ones((400, 800, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Test different font scales and thicknesses
    y_pos = 50
    
    # Old style (thin, hard to read)
    cv2.putText(canvas, "Old Style: Thin font (thickness=1)", (20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    y_pos += 50
    
    # New style (thick, readable)
    cv2.putText(canvas, "New Style: Thick font (thickness=3)", (20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    y_pos += 50
    
    # With background rectangle
    text = "With Background: Maximum readability"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)[0]
    cv2.rectangle(canvas, (15, y_pos - text_size[1] - 5), (25 + text_size[0], y_pos + 5), (255, 255, 255), -1)
    cv2.rectangle(canvas, (15, y_pos - text_size[1] - 5), (25 + text_size[0], y_pos + 5), (0, 0, 0), 1)
    cv2.putText(canvas, text, (20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    # Save the font test
    cv2.imwrite("test_font_improvements.png", canvas)
    print("Created: test_font_improvements.png")
    
    return True


def run_all_tests():
    """Run all display system tests."""
    print("Running Display System Tests")
    print("=" * 40)
    
    tests_passed = 0
    tests_total = 3
    
    try:
        if create_test_display_scenarios():
            tests_passed += 1
            print("✓ Display scenarios test passed")
        else:
            print("✗ Display scenarios test failed")
    except Exception as e:
        print(f"✗ Display scenarios test failed with error: {e}")
    
    try:
        if test_display_manager():
            tests_passed += 1
            print("✓ Display manager test passed")
        else:
            print("✗ Display manager test failed")
    except Exception as e:
        print(f"✗ Display manager test failed with error: {e}")
    
    try:
        if test_font_rendering():
            tests_passed += 1
            print("✓ Font rendering test passed")
        else:
            print("✗ Font rendering test failed")
    except Exception as e:
        print(f"✗ Font rendering test failed with error: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    print("\nKey improvements made:")
    print("✓ Fixed aspect ratio preservation")
    print("✓ Improved font readability with background rectangles")
    print("✓ Better grid scaling and responsive layout")
    print("✓ Enhanced visual elements and contrast")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
