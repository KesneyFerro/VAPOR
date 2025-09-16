"""
Enhanced Blur Effects Implementation for VAPOR fx_01_blur

This module implements deterministic blur effects using pre-generated kernels
to ensure reproducible results. Supports:

- Motion blur (low and high intensity)
- Defocus blur (low and high intensity) 
- Haze blur (low and high intensity)
- Gaussian blur (low and high intensity)
- Combined blur (motion + defocus + gaussian, all low intensity)

All blur effects use deterministic kernels for reproducibility.

Author: Kesney de Oliveira
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Union
from pathlib import Path
import logging

from ..kernels.generator import DeterministicKernelGenerator, KernelConfig


class EnhancedBlurEffects:
    """
    Enhanced blur effects with deterministic kernel generation.
    
    This class provides blur effects that use pre-generated deterministic kernels
    to ensure reproducible results across experiments.
    """
    
    def __init__(self, kernel_config: KernelConfig = None):
        """
        Initialize the blur effects engine.
        
        Args:
            kernel_config: Configuration for kernel generation
        """
        self.kernel_generator = DeterministicKernelGenerator(kernel_config)
        self.kernels_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def _get_or_generate_kernel(self, blur_type: str, intensity: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get kernel from cache or generate it.
        
        Args:
            blur_type: Type of blur effect
            intensity: 'low' or 'high'
            
        Returns:
            Tuple of (kernel, metadata)
        """
        cache_key = f"{blur_type}_{intensity}"
        
        if cache_key not in self.kernels_cache:
            if blur_type == 'motion_blur':
                self.kernels_cache[cache_key] = self.kernel_generator.get_motion_blur_kernel(intensity)
            elif blur_type == 'gaussian_blur':
                self.kernels_cache[cache_key] = self.kernel_generator.get_gaussian_blur_kernel(intensity)
            elif blur_type == 'defocus_blur':
                self.kernels_cache[cache_key] = self.kernel_generator.get_defocus_blur_kernel(intensity)
            elif blur_type == 'haze_blur':
                self.kernels_cache[cache_key] = self.kernel_generator.get_haze_blur_kernel(intensity)
            else:
                raise ValueError(f"Unknown blur type: {blur_type}")
                
        return self.kernels_cache[cache_key]
    
    def apply_motion_blur(self, image: np.ndarray, intensity: str = 'low') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply motion blur effect.
        
        Args:
            image: Input image (BGR or grayscale)
            intensity: 'low' or 'high'
            
        Returns:
            Tuple of (blurred_image, metadata)
        """
        kernel, metadata = self._get_or_generate_kernel('motion_blur', intensity)
        
        # Apply convolution
        if len(image.shape) == 3:
            # Color image - apply to each channel
            blurred = np.zeros_like(image)
            for i in range(image.shape[2]):
                blurred[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
        else:
            # Grayscale image
            blurred = cv2.filter2D(image, -1, kernel)
            
        return blurred, metadata
    
    def apply_gaussian_blur(self, image: np.ndarray, intensity: str = 'low') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Gaussian blur effect.
        
        Args:
            image: Input image (BGR or grayscale)
            intensity: 'low' or 'high'
            
        Returns:
            Tuple of (blurred_image, metadata)
        """
        kernel, metadata = self._get_or_generate_kernel('gaussian_blur', intensity)
        
        # Apply convolution
        if len(image.shape) == 3:
            # Color image - apply to each channel
            blurred = np.zeros_like(image)
            for i in range(image.shape[2]):
                blurred[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
        else:
            # Grayscale image
            blurred = cv2.filter2D(image, -1, kernel)
            
        return blurred, metadata
    
    def apply_defocus_blur(self, image: np.ndarray, intensity: str = 'low') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply defocus (out-of-focus) blur effect.
        
        Args:
            image: Input image (BGR or grayscale)
            intensity: 'low' or 'high'
            
        Returns:
            Tuple of (blurred_image, metadata)
        """
        kernel, metadata = self._get_or_generate_kernel('defocus_blur', intensity)
        
        # Apply convolution
        if len(image.shape) == 3:
            # Color image - apply to each channel
            blurred = np.zeros_like(image)
            for i in range(image.shape[2]):
                blurred[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
        else:
            # Grayscale image
            blurred = cv2.filter2D(image, -1, kernel)
            
        return blurred, metadata
    
    def apply_haze_blur(self, image: np.ndarray, intensity: str = 'low') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply haze blur effect (atmospheric scattering simulation).
        
        Args:
            image: Input image (BGR or grayscale)
            intensity: 'low' or 'high'
            
        Returns:
            Tuple of (blurred_image, metadata)
        """
        kernel, metadata = self._get_or_generate_kernel('haze_blur', intensity)
        
        # Apply convolution
        if len(image.shape) == 3:
            # Color image - apply to each channel
            blurred = np.zeros_like(image)
            for i in range(image.shape[2]):
                blurred[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
        else:
            # Grayscale image
            blurred = cv2.filter2D(image, -1, kernel)
            
        # Add slight brightness reduction to simulate atmospheric effects
        alpha = metadata['alpha']
        brightness_factor = 1.0 - (alpha * 0.1)  # Slight darkening
        blurred = (blurred * brightness_factor).astype(image.dtype)
            
        return blurred, metadata
    
    def apply_combined_blur(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply combined blur effect (motion + defocus + gaussian, all low intensity).
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (blurred_image, metadata)
        """
        # Get combined kernels
        kernels, metadata = self.kernel_generator.get_combined_blur_kernels()
        
        # Apply effects sequentially
        result = image.copy().astype(np.float32)
        
        # Apply motion blur
        motion_kernel = kernels['motion']
        if len(result.shape) == 3:
            for i in range(result.shape[2]):
                result[:, :, i] = cv2.filter2D(result[:, :, i], -1, motion_kernel)
        else:
            result = cv2.filter2D(result, -1, motion_kernel)
        
        # Apply defocus blur
        defocus_kernel = kernels['defocus']
        if len(result.shape) == 3:
            for i in range(result.shape[2]):
                result[:, :, i] = cv2.filter2D(result[:, :, i], -1, defocus_kernel)
        else:
            result = cv2.filter2D(result, -1, defocus_kernel)
        
        # Apply Gaussian blur
        gaussian_kernel = kernels['gaussian']
        if len(result.shape) == 3:
            for i in range(result.shape[2]):
                result[:, :, i] = cv2.filter2D(result[:, :, i], -1, gaussian_kernel)
        else:
            result = cv2.filter2D(result, -1, gaussian_kernel)
        
        # Convert back to original dtype
        result = result.astype(image.dtype)
        
        return result, metadata
    
    def apply_blur_effect(self, image: np.ndarray, blur_type: str, intensity: str = 'low') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply any blur effect by name.
        
        Args:
            image: Input image (BGR or grayscale)
            blur_type: Type of blur ('motion', 'gaussian', 'defocus', 'haze', 'combined')
            intensity: 'low' or 'high' (ignored for combined blur)
            
        Returns:
            Tuple of (blurred_image, metadata)
        """
        blur_methods = {
            'motion': self.apply_motion_blur,
            'motion_blur': self.apply_motion_blur,
            'gaussian': self.apply_gaussian_blur,
            'gaussian_blur': self.apply_gaussian_blur,
            'defocus': self.apply_defocus_blur,
            'defocus_blur': self.apply_defocus_blur,
            'haze': self.apply_haze_blur,
            'haze_blur': self.apply_haze_blur,
            'combined': self.apply_combined_blur,
            'combined_blur': self.apply_combined_blur
        }
        
        if blur_type not in blur_methods:
            raise ValueError(f"Unknown blur type: {blur_type}. Available: {list(blur_methods.keys())}")
        
        method = blur_methods[blur_type]
        
        if blur_type in ['combined', 'combined_blur']:
            return method(image)
        else:
            return method(image, intensity)
    
    def get_all_kernels(self) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get all generated kernels with metadata.
        
        Returns:
            Dictionary mapping kernel names to (kernel, metadata) tuples
        """
        return self.kernel_generator.get_all_kernels()
    
    def save_kernels(self, filepath: str):
        """
        Save all kernels to file for reuse.
        
        Args:
            filepath: Path to save the kernels
        """
        from ..kernels.generator import save_kernels_to_file
        kernels = self.get_all_kernels()
        save_kernels_to_file(kernels, filepath)
        self.logger.info(f"Kernels saved to {filepath}")


def create_blur_effects_engine(seed: int = 42) -> EnhancedBlurEffects:
    """
    Create a blur effects engine with custom seed.
    
    Args:
        seed: Random seed for deterministic generation
        
    Returns:
        Configured EnhancedBlurEffects instance
    """
    config = KernelConfig(seed=seed)
    return EnhancedBlurEffects(config)
