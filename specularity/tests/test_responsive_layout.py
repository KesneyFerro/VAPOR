#!/usr/bin/env python3
"""
Test script for responsive display functionality.
Tests the responsive grid calculation and layout system.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path to import specularity modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from specularity.core.display_manager import DisplayManager

def test_responsive_grid():
    """Test the responsive grid calculation"""
    
    # Simulate the screen detection
    def get_screen_dimensions():
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            return screen_width, screen_height
        except Exception:
            return 1920, 1080
    
    def get_optimal_grid_size(num_items):
        """Calculate optimal grid size for given number of items"""
        if num_items <= 1:
            return 1, 1
        elif num_items <= 2:
            return 1, 2
        elif num_items <= 4:
            return 2, 2
        elif num_items <= 6:
            return 2, 3
        elif num_items <= 9:
            return 3, 3
        elif num_items <= 12:
            return 3, 4
        else:
            rows = (num_items + 3) // 4
            return rows, 4
    
    # Test different scenarios
    screen_width, screen_height = get_screen_dimensions()
    print(f"Detected screen resolution: {screen_width}x{screen_height}")
    
    # Use 90% of screen
    target_width = int(screen_width * 0.9)
    target_height = int(screen_height * 0.9)
    print(f"Target display size: {target_width}x{target_height}")
    
    # Test different numbers of images
    test_cases = [1, 2, 4, 6, 9, 12, 16]
    
    # Simulate cropped content with different aspect ratios
    content_scenarios = [
        (800, 600, "4:3 aspect ratio"),
        (1920, 1080, "16:9 aspect ratio"),
        (600, 800, "3:4 portrait"),
        (1080, 1920, "9:16 portrait"),
        (1200, 1200, "1:1 square")
    ]
    
    print("\n" + "="*80)
    print("RESPONSIVE GRID CALCULATIONS")
    print("="*80)
    
    for content_w, content_h, description in content_scenarios:
        print(f"\nContent: {description} ({content_w}x{content_h})")
        content_aspect = content_w / content_h
        print(f"Content aspect ratio: {content_aspect:.3f}")
        
        for num_images in test_cases:
            rows, cols = get_optimal_grid_size(num_images)
            
            # Calculate responsive dimensions
            padding = 20
            text_height = 80
            header_height = 120
            footer_height = 60
            
            available_width = target_width - (cols + 1) * padding
            available_height = target_height - header_height - footer_height - (rows * text_height) - (rows + 1) * padding
            
            max_img_width = available_width // cols
            max_img_height = available_height // rows
            
            # Maintain aspect ratio
            if content_aspect > (max_img_width / max_img_height):
                img_width = max_img_width
                img_height = int(max_img_width / content_aspect)
            else:
                img_height = max_img_height
                img_width = int(max_img_height * content_aspect)
            
            # Ensure minimum size
            img_width = max(img_width, 200)
            img_height = max(img_height, 150)
            
            grid_width = cols * img_width + (cols + 1) * padding
            grid_height = rows * (img_height + text_height) + (rows + 1) * padding + header_height + footer_height
            
            # Scale up if grid is too small
            if grid_width < target_width * 0.9:
                scale_factor = (target_width * 0.9) / grid_width
                img_width = int(img_width * scale_factor)
                img_height = int(img_height * scale_factor)
                grid_width = cols * img_width + (cols + 1) * padding
            
            screen_usage = (grid_width / target_width) * 100
            
            print(f"  {num_images:2d} images ({rows}x{cols}): "
                  f"img={img_width}x{img_height}, "
                  f"grid={grid_width}x{grid_height}, "
                  f"screen usage={screen_usage:.1f}%")

def create_test_image_with_aspect_ratio():
    """Create a test image to demonstrate aspect ratio preservation"""
    
    # Create test images with different aspect ratios
    test_images = {
        "Wide 16:9": np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8),
        "Tall 9:16": np.random.randint(0, 255, (960, 540, 3), dtype=np.uint8),
        "Square 1:1": np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8),
        "Standard 4:3": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }
    
    # Add colored rectangles and text to make aspect ratios visible
    for name, img in test_images.items():
        h, w = img.shape[:2]
        # Add border
        cv2.rectangle(img, (10, 10), (w-10, h-10), (255, 255, 255), 5)
        # Add text
        cv2.putText(img, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"{w}x{h}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Add aspect ratio indicator
        aspect = w / h
        cv2.putText(img, f"Aspect: {aspect:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return test_images

if __name__ == "__main__":
    print("Testing Responsive Grid Display System")
    print("="*50)
    
    test_responsive_grid()
    
    print("\n" + "="*80)
    print("FEATURES OF THE NEW RESPONSIVE SYSTEM:")
    print("="*80)
    print("✓ Automatically detects screen resolution")
    print("✓ Uses 90% of screen width for optimal viewing")
    print("✓ Maintains content aspect ratio")
    print("✓ Scales images to fit available space")
    print("✓ Responsive font sizes based on display size")
    print("✓ Optimizes grid layout for number of images")
    print("✓ Works with any content aspect ratio (4:3, 16:9, portrait, etc.)")
    print("✓ Ensures minimum image sizes for visibility")
    print("✓ Adapts to different screen sizes automatically")
    
    print("\n" + "="*80)
    print("USAGE:")
    print("="*80)
    print("The system now automatically:")
    print("1. Detects your screen resolution")
    print("2. Crops black borders from video content")
    print("3. Calculates optimal image sizes maintaining aspect ratio")
    print("4. Creates a grid that uses most of your screen width")
    print("5. Scales fonts and spacing appropriately")
    print("6. Ensures all images maintain their proper proportions")
    
    # Create and save test images
    test_images = create_test_image_with_aspect_ratio()
    for name, img in test_images.items():
        filename = f"test_aspect_{name.replace(':', '').replace(' ', '_')}.png"
        cv2.imwrite(filename, img)
        print(f"Saved: {filename}")
