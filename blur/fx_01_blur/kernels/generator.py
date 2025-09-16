"""
Deterministic Kernel Generation for Blur Effects

This module provides deterministic kernel generation for all blur effects to ensure
reproducible results across experiments. All kernels are generated with fixed seeds
to maintain consistency.

Author: Kesney de Oliveira
"""

import numpy as np
import cv2
import math
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class KernelConfig:
    """Configuration for kernel generation."""
    seed: int = 42
    motion_angle_low: float = 45.0  # degrees
    motion_angle_high: float = 135.0  # degrees
    motion_length_low: int = 15
    motion_length_high: int = 30
    gaussian_sigma_low: float = 2.0
    gaussian_sigma_high: float = 5.0
    defocus_radius_low: int = 8
    defocus_radius_high: int = 15
    haze_alpha_low: float = 0.3
    haze_alpha_high: float = 0.7


class DeterministicKernelGenerator:
    """
    Generates deterministic kernels for blur effects.
    
    All kernels are generated with fixed parameters to ensure reproducibility
    across different runs and experiments.
    """
    
    def __init__(self, config: KernelConfig = None):
        """
        Initialize the kernel generator.
        
        Args:
            config: Kernel configuration. Uses default if None.
        """
        self.config = config or KernelConfig()
        np.random.seed(self.config.seed)
        
    def get_motion_blur_kernel(self, intensity: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate motion blur kernel.
        
        Args:
            intensity: 'low' or 'high'
            
        Returns:
            Tuple of (kernel, metadata)
        """
        if intensity == 'low':
            length = self.config.motion_length_low
            angle = self.config.motion_angle_low
        else:  # high
            length = self.config.motion_length_high
            angle = self.config.motion_angle_high
            
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Create kernel
        kernel = np.zeros((length, length), dtype=np.float32)
        
        # Calculate line endpoints
        center = length // 2
        dx = int(center * math.cos(angle_rad))
        dy = int(center * math.sin(angle_rad))
        
        # Draw line in kernel
        cv2.line(kernel, 
                (center - dx, center - dy), 
                (center + dx, center + dy), 
                1.0, 1)
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        metadata = {
            'type': 'motion_blur',
            'intensity': intensity,
            'length': length,
            'angle': angle,
            'seed': self.config.seed
        }
        
        return kernel, metadata
    
    def get_gaussian_blur_kernel(self, intensity: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate Gaussian blur kernel.
        
        Args:
            intensity: 'low' or 'high'
            
        Returns:
            Tuple of (kernel, metadata)
        """
        if intensity == 'low':
            sigma = self.config.gaussian_sigma_low
        else:  # high
            sigma = self.config.gaussian_sigma_high
            
        # Calculate kernel size (6*sigma + 1, must be odd)
        ksize = int(6 * sigma) + 1
        if ksize % 2 == 0:
            ksize += 1
            
        # Generate 1D Gaussian kernel
        kernel_1d = cv2.getGaussianKernel(ksize, sigma)
        
        # Create 2D kernel
        kernel = np.outer(kernel_1d, kernel_1d)
        
        metadata = {
            'type': 'gaussian_blur',
            'intensity': intensity,
            'sigma': sigma,
            'kernel_size': ksize,
            'seed': self.config.seed
        }
        
        return kernel, metadata
    
    def get_defocus_blur_kernel(self, intensity: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate defocus (out-of-focus) blur kernel.
        
        Args:
            intensity: 'low' or 'high'
            
        Returns:
            Tuple of (kernel, metadata)
        """
        if intensity == 'low':
            radius = self.config.defocus_radius_low
        else:  # high
            radius = self.config.defocus_radius_high
            
        # Create circular kernel
        ksize = 2 * radius + 1
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        
        center = radius
        y, x = np.ogrid[:ksize, :ksize]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        kernel[mask] = 1.0
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        metadata = {
            'type': 'defocus_blur',
            'intensity': intensity,
            'radius': radius,
            'kernel_size': ksize,
            'seed': self.config.seed
        }
        
        return kernel, metadata
    
    def get_haze_blur_kernel(self, intensity: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate haze blur kernel (atmospheric scattering simulation).
        
        Args:
            intensity: 'low' or 'high'
            
        Returns:
            Tuple of (kernel, metadata)
        """
        if intensity == 'low':
            alpha = self.config.haze_alpha_low
            sigma = 1.5
        else:  # high
            alpha = self.config.haze_alpha_high
            sigma = 3.0
            
        # Create haze kernel (combination of Gaussian and uniform distribution)
        ksize = int(6 * sigma) + 1
        if ksize % 2 == 0:
            ksize += 1
            
        # Base Gaussian
        gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
        gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
        
        # Uniform component for atmospheric scattering
        uniform = np.ones((ksize, ksize), dtype=np.float32)
        uniform = uniform / np.sum(uniform)
        
        # Combine with alpha blending
        kernel = alpha * gaussian_2d + (1 - alpha) * uniform
        kernel = kernel / np.sum(kernel)
        
        metadata = {
            'type': 'haze_blur',
            'intensity': intensity,
            'alpha': alpha,
            'sigma': sigma,
            'kernel_size': ksize,
            'seed': self.config.seed
        }
        
        return kernel, metadata
    
    def get_combined_blur_kernels(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Generate kernels for combined blur effect.
        
        Combined blur applies: low motion + low defocus + low gaussian
        
        Returns:
            Tuple of (kernels_dict, metadata)
        """
        motion_kernel, motion_meta = self.get_motion_blur_kernel('low')
        defocus_kernel, defocus_meta = self.get_defocus_blur_kernel('low')
        gaussian_kernel, gaussian_meta = self.get_gaussian_blur_kernel('low')
        
        kernels = {
            'motion': motion_kernel,
            'defocus': defocus_kernel,
            'gaussian': gaussian_kernel
        }
        
        metadata = {
            'type': 'combined_blur',
            'components': {
                'motion': motion_meta,
                'defocus': defocus_meta,
                'gaussian': gaussian_meta
            },
            'seed': self.config.seed
        }
        
        return kernels, metadata
    
    def get_all_kernels(self) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Generate all kernels at once.
        
        Returns:
            Dictionary mapping kernel names to (kernel, metadata) tuples
        """
        kernels = {}
        
        # Individual blur kernels
        blur_types = ['motion_blur', 'gaussian_blur', 'defocus_blur', 'haze_blur']
        intensities = ['low', 'high']
        
        for blur_type in blur_types:
            for intensity in intensities:
                key = f"{blur_type}_{intensity}"
                if blur_type == 'motion_blur':
                    kernels[key] = self.get_motion_blur_kernel(intensity)
                elif blur_type == 'gaussian_blur':
                    kernels[key] = self.get_gaussian_blur_kernel(intensity)
                elif blur_type == 'defocus_blur':
                    kernels[key] = self.get_defocus_blur_kernel(intensity)
                elif blur_type == 'haze_blur':
                    kernels[key] = self.get_haze_blur_kernel(intensity)
        
        # Combined blur
        combined_kernels, combined_meta = self.get_combined_blur_kernels()
        kernels['combined_blur'] = (combined_kernels, combined_meta)
        
        return kernels


def save_kernels_to_file(kernels: Dict, filepath: str):
    """
    Save kernels to numpy file for reuse.
    
    Args:
        kernels: Dictionary of kernels from get_all_kernels()
        filepath: Path to save the kernels
    """
    np.savez_compressed(filepath, **{k: v[0] for k, v in kernels.items()})
    

def load_kernels_from_file(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load kernels from numpy file.
    
    Args:
        filepath: Path to the saved kernels file
        
    Returns:
        Dictionary of loaded kernels
    """
    return dict(np.load(filepath))
