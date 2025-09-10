"""
Shared Blur Effects Module for VAPOR Project

This module provides unified blur effect implementations that can be used across
different parts of the VAPOR pipeline. Extracted from duplicated code in various modules.

Supports the following blur techniques:
- Gaussian blur (low and high intensity)
- Motion blur (with random angles)
- Out-of-focus blur (circular disk kernel)
- Average blur (box filter)
- Median blur (noise reduction)
- Combined blur (sequential application of multiple effects)
"""

import cv2
import numpy as np
import random
import math
from scipy import signal
from typing import Dict, Callable


class BlurEffectsEngine:
    """
    Unified blur effects engine with configurable intensity levels.
    
    This class provides a consistent interface for applying various blur effects
    with both low and high intensity settings. Designed to be reusable across
    different VAPOR modules.
    """
    
    def __init__(self):
        """Initialize the blur effects engine."""
        self.blur_methods: Dict[str, Callable] = {
            "gaussian": self.apply_gaussian_blur,
            "motion_blur": self.apply_motion_blur,
            "outoffocus": self.apply_out_of_focus_blur,
            "average": self.apply_average_blur,
            "median": self.apply_median_blur,
            "combined": self.apply_combined_blur,
            # Aliases for compatibility
            "motion_horizontal": lambda img, intensity: self.apply_motion_blur_directional(img, intensity, "horizontal"),
            "motion_diagonal": lambda img, intensity: self.apply_motion_blur_directional(img, intensity, "diagonal"),
        }
    
    def apply_blur_effect(self, image: np.ndarray, blur_type: str, intensity: str) -> np.ndarray:
        """
        Apply a specific blur effect to an image.
        
        Args:
            image: Input image array
            blur_type: Type of blur to apply
            intensity: Blur intensity level ("low" or "high")
            
        Returns:
            Blurred image array
            
        Raises:
            ValueError: If blur_type is not supported
        """
        if blur_type not in self.blur_methods:
            raise ValueError(f"Unsupported blur type: {blur_type}. Supported types: {list(self.blur_methods.keys())}")
        
        return self.blur_methods[blur_type](image, intensity)
    
    def get_supported_blur_types(self) -> list:
        """Get list of supported blur types."""
        return list(self.blur_methods.keys())
    
    def apply_gaussian_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
        """
        Apply Gaussian blur to an image.
        
        Args:
            image: Input image array
            intensity: "low" or "high" blur intensity
            
        Returns:
            Gaussian blurred image array
        """
        if intensity == "low":
            kernel_size = (5, 5)
            sigma = 1.5
        else:  # high
            kernel_size = (15, 15)
            sigma = 5.0
        
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def apply_motion_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
        """
        Apply motion blur with random angle to an image.
        
        Args:
            image: Input image array
            intensity: "low" or "high" blur intensity
            
        Returns:
            Motion blurred image array
        """
        if intensity == "low":
            kernel_size = 10
            # Random angle between 0-180 degrees for low intensity
            angle = random.randint(0, 180)
        else:  # high
            kernel_size = 25
            # Different random angle range for high intensity
            angle = random.randint(0, 180)
        
        # Create motion blur kernel at specified angle
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # Create motion blur line
        center = kernel_size // 2
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * dx)
            y = int(center + offset * dy)
            
            # Check bounds
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # Normalize kernel
        if np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)
        else:
            # Fallback to simple horizontal blur if kernel creation fails
            kernel = np.zeros((kernel_size, kernel_size))
            middle_row = kernel_size // 2
            kernel[middle_row, :] = 1 / kernel_size
        
        return cv2.filter2D(image, -1, kernel)
    
    def apply_motion_blur_directional(self, image: np.ndarray, intensity: str, direction: str) -> np.ndarray:
        """
        Apply motion blur in a specific direction (for compatibility with existing modules).
        
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
    
    def apply_out_of_focus_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
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
        
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        center = radius
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance <= radius:
                    kernel[i, j] = 1
        
        kernel = kernel / np.sum(kernel)
        
        if len(image.shape) == 3:
            blurred = np.zeros_like(image)
            for channel in range(image.shape[2]):
                blurred[:, :, channel] = signal.convolve2d(
                    image[:, :, channel], kernel, mode='same', boundary='symm'
                )
            return blurred.astype(np.uint8)
        else:
            return signal.convolve2d(image, kernel, mode='same', boundary='symm').astype(np.uint8)
    
    def apply_average_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
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
    
    def apply_median_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
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
    
    def apply_combined_blur(self, image: np.ndarray, intensity: str) -> np.ndarray:
        """
        Apply combined blur: motion blur + out-of-focus blur + median blur.
        
        Args:
            image: Input image array
            intensity: "low" or "high" blur intensity
            
        Returns:
            Combined blurred image array
        """
        # Apply motion blur first
        blurred = self.apply_motion_blur(image, intensity)
        
        # Apply out-of-focus blur
        blurred = self.apply_out_of_focus_blur(blurred, intensity)
        
        # Apply median blur last (helps reduce noise)
        blurred = self.apply_median_blur(blurred, intensity)
        
        return blurred


# Global instance for easy access
blur_engine = BlurEffectsEngine()

# Convenience functions for direct access
def apply_blur_effect(image: np.ndarray, blur_type: str, intensity: str) -> np.ndarray:
    """Convenience function for applying blur effects."""
    return blur_engine.apply_blur_effect(image, blur_type, intensity)

def get_supported_blur_types() -> list:
    """Get list of supported blur types."""
    return blur_engine.get_supported_blur_types()
