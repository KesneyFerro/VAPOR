"""
Blur Effects Script
Applies different types of blur effects to image sequences from extracted frames.

Supports five blur techniques:
- Gaussian blur (low and high intensity)
- Motion blur (horizontal and diagonal)
- Out-of-focus blur (circular disk kernel)
- Average blur (box filter)
- Median blur (noise reduction)

Usage:
    python blur_effects.py <folder_name> [--blur_types <types>] [--intensity <low|high>]
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from scipy import signal
import math


class BlurEffects:
    """
    Applies various blur effects to image sequences.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize with base path for data folders.
        
        Args:
            base_path: Base path to the data folder (defaults to detecting from script location)
        """
        if base_path is None:
            # Auto-detect base path relative to script location
            script_dir = Path(__file__).parent
            self.base_path = script_dir.parent.parent / "data"
        else:
            self.base_path = Path(base_path)
        
        self.original_frames_path = self.base_path / "extracted_frames" / "original"
        self.blurred_frames_path = self.base_path / "extracted_frames" / "blurred"
        
        # Ensure output directory exists
        self.blurred_frames_path.mkdir(parents=True, exist_ok=True)
    
    def apply_gaussian_blur(self, image: np.ndarray, intensity: str = "low") -> np.ndarray:
        """
        Apply Gaussian blur to an image.
        
        Args:
            image: Input image array
            intensity: "low" or "high" blur intensity
            
        Returns:
            Blurred image array
        """
        if intensity == "low":
            kernel_size = (5, 5)
            sigma = 1.5
        else:  # high
            kernel_size = (15, 15)
            sigma = 5.0
        
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def apply_motion_blur(self, image: np.ndarray, intensity: str = "low", direction: str = "horizontal") -> np.ndarray:
        """
        Apply motion blur to an image.
        
        Args:
            image: Input image array
            intensity: "low" or "high" blur intensity
            direction: "horizontal" or "diagonal" motion direction
            
        Returns:
            Motion blurred image array
        """
        if intensity == "low":
            kernel_size = 10
        else:  # high
            kernel_size = 25
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        
        if direction == "horizontal":
            # Horizontal motion blur
            middle_row = kernel_size // 2
            kernel[middle_row, :] = 1
        else:  # diagonal
            # Diagonal motion blur
            for i in range(kernel_size):
                kernel[i, i] = 1
        
        # Normalize the kernel
        kernel = kernel / np.sum(kernel)
        
        return cv2.filter2D(image, -1, kernel)
    
    def apply_out_of_focus_blur(self, image: np.ndarray, intensity: str = "low") -> np.ndarray:
        """
        Apply out-of-focus blur using a circular disk kernel.
        
        Args:
            image: Input image array
            intensity: "low" or "high" blur intensity
            
        Returns:
            Out-of-focus blurred image array
        """
        if intensity == "low":
            radius = 3
        else:  # high
            radius = 8
        
        # Create circular disk kernel
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        center = radius
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance <= radius:
                    kernel[i, j] = 1
        
        # Normalize the kernel
        kernel = kernel / np.sum(kernel)
        
        # Apply convolution to each channel separately
        if len(image.shape) == 3:
            blurred = np.zeros_like(image)
            for channel in range(image.shape[2]):
                blurred[:, :, channel] = signal.convolve2d(
                    image[:, :, channel], kernel, mode='same', boundary='symm'
                )
            return blurred.astype(np.uint8)
        else:
            return signal.convolve2d(image, kernel, mode='same', boundary='symm').astype(np.uint8)
    
    def apply_average_blur(self, image: np.ndarray, intensity: str = "low") -> np.ndarray:
        """
        Apply average blur (box filter) to an image.
        
        Args:
            image: Input image array
            intensity: "low" or "high" blur intensity
            
        Returns:
            Average blurred image array
        """
        if intensity == "low":
            kernel_size = (5, 5)
        else:  # high
            kernel_size = (15, 15)
        
        return cv2.blur(image, kernel_size)
    
    def apply_median_blur(self, image: np.ndarray, intensity: str = "low") -> np.ndarray:
        """
        Apply median blur to an image.
        
        Args:
            image: Input image array
            intensity: "low" or "high" blur intensity
            
        Returns:
            Median blurred image array
        """
        if intensity == "low":
            kernel_size = 5
        else:  # high
            kernel_size = 15
        
        return cv2.medianBlur(image, kernel_size)
    
    def get_image_files(self, folder_path: Path) -> List[Path]:
        """
        Get sorted list of image files from a folder.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Sorted list of image file paths
        """
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        image_files = []
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        if not image_files:
            raise ValueError(f"No image files found in: {folder_path}")
        
        # Sort files naturally
        image_files.sort(key=lambda x: self._natural_sort_key(x.name))
        
        return image_files
    
    def _natural_sort_key(self, filename: str) -> List:
        """
        Create a sort key for natural sorting of filenames with numbers.
        
        Args:
            filename: The filename to create sort key for
            
        Returns:
            Sort key for natural sorting
        """
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]
    
    def process_folder(self, folder_name: str, blur_types: List[str] = None, 
                      intensity: str = "low") -> bool:
        """
        Process all images in a folder with specified blur effects.
        
        Args:
            folder_name: Name of the folder in original frames directory
            blur_types: List of blur types to apply (default: all types)
            intensity: Blur intensity level ("low" or "high")
            
        Returns:
            True if successful, False otherwise
        """
        if blur_types is None:
            blur_types = ["gaussian", "motion_horizontal", "motion_diagonal", 
                         "outoffocus", "average", "median"]
        
        try:
            # Input folder path
            input_folder = self.original_frames_path / folder_name
            if not input_folder.exists():
                raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
            # Get image files
            image_files = self.get_image_files(input_folder)
            print(f"Found {len(image_files)} images in {folder_name}")
            
            # Process each blur type
            for blur_type in blur_types:
                print(f"\nApplying {blur_type} blur ({intensity} intensity)...")
                
                # Create output folder
                output_folder = self.blurred_frames_path / folder_name / blur_type
                output_folder.mkdir(parents=True, exist_ok=True)
                
                # Process each image
                for i, image_file in enumerate(image_files):
                    # Load image
                    image = cv2.imread(str(image_file))
                    if image is None:
                        print(f"Warning: Could not load image {image_file}")
                        continue
                    
                    # Apply blur effect
                    blurred_image = self._apply_blur_by_type(image, blur_type, intensity)
                    
                    # Save blurred image
                    output_path = output_folder / image_file.name
                    cv2.imwrite(str(output_path), blurred_image)
                    
                    # Progress indicator
                    if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                        print(f"  Processed {i + 1}/{len(image_files)} images")
                
                print(f"  [OK] {blur_type} blur completed: {output_folder}")
            
            print(f"\n[OK] All blur effects applied to folder: {folder_name}")
            return True
            
        except Exception as e:
            print(f"Error processing folder {folder_name}: {e}")
            return False
    
    def _apply_blur_by_type(self, image: np.ndarray, blur_type: str, intensity: str) -> np.ndarray:
        """
        Apply blur effect based on type string.
        
        Args:
            image: Input image array
            blur_type: Type of blur to apply
            intensity: Blur intensity level
            
        Returns:
            Blurred image array
        """
        if blur_type == "gaussian":
            return self.apply_gaussian_blur(image, intensity)
        elif blur_type == "motion_horizontal":
            return self.apply_motion_blur(image, intensity, "horizontal")
        elif blur_type == "motion_diagonal":
            return self.apply_motion_blur(image, intensity, "diagonal")
        elif blur_type == "outoffocus":
            return self.apply_out_of_focus_blur(image, intensity)
        elif blur_type == "average":
            return self.apply_average_blur(image, intensity)
        elif blur_type == "median":
            return self.apply_median_blur(image, intensity)
        else:
            raise ValueError(f"Unknown blur type: {blur_type}")
    
    def list_available_folders(self) -> List[str]:
        """
        List available folders in the original frames directory.
        
        Returns:
            List of folder names
        """
        if not self.original_frames_path.exists():
            return []
        
        folders = []
        for item in self.original_frames_path.iterdir():
            if item.is_dir():
                folders.append(item.name)
        
        return sorted(folders)


def main():
    """Main function to handle command line arguments and execute blur effects."""
    parser = argparse.ArgumentParser(
        description="Apply blur effects to image sequences from extracted frames"
    )
    parser.add_argument(
        "folder_name", 
        help="Name of the folder in data/extracted_frames/original/"
    )
    parser.add_argument(
        "--blur_types", 
        nargs='+',
        choices=["gaussian", "motion_horizontal", "motion_diagonal", 
                "outoffocus", "average", "median"],
        help="Types of blur to apply (default: all types)"
    )
    parser.add_argument(
        "--intensity", 
        choices=["low", "high"],
        default="low",
        help="Blur intensity level (default: low)"
    )
    parser.add_argument(
        "--list_folders", 
        action="store_true",
        help="List available folders and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Create blur effects processor
        blur_processor = BlurEffects()
        
        # List folders if requested
        if args.list_folders:
            folders = blur_processor.list_available_folders()
            if folders:
                print("Available folders:")
                for folder in folders:
                    print(f"  - {folder}")
            else:
                print("No folders found in data/extracted_frames/original/")
            return 0
        
        # Process the specified folder
        success = blur_processor.process_folder(
            args.folder_name, 
            args.blur_types, 
            args.intensity
        )
        
        if success:
            print("\n[SUCCESS] Blur effects processing completed successfully!")
            return 0
        else:
            print("\n[FAILED] Blur effects processing failed!")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
