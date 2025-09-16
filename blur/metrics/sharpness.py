"""
Sharpness Metrics

These metrics directly measure sharpness or blur levels using image processing 
techniques. They analyze edge content, gradient strength, and frequency characteristics.

Supported Metrics:
- Acutance (Edge Sharpness)
- Mean Gradient (Edge Strength)
- FFT High-Frequency Energy (Frequency Domain Analysis)
- Laplacian Variance (Focus/Blur Measurement)

Author: Kesney de Oliveira
Date: September 2025
"""

import cv2
import numpy as np
import logging
from typing import Dict, Union
from scipy.fft import fft2, fftshift
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class SharpnessMetrics:
    """
    Sharpness and focus metrics calculator.
    
    These metrics measure image sharpness, edge content, and focus quality
    without requiring a reference image.
    """
    
    def __init__(self):
        """Initialize the sharpness metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_all(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate all sharpness metrics.
        
        Args:
            image: Input image to evaluate
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        try:
            # Prepare image
            img = self._prepare_image(image)
            
            # Calculate metrics
            metrics['acutance'] = self.calculate_acutance(img)
            metrics['mean_gradient'] = self.calculate_mean_gradient(img)
            metrics['fft_high_freq'] = self.calculate_fft_high_freq(img)
            metrics['laplacian_variance'] = self.calculate_laplacian_variance(img)
            
        except Exception as e:
            self.logger.error(f"Error calculating sharpness metrics: {e}")
            
        return metrics
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for sharpness metrics calculation.
        
        Args:
            image: Input image
            
        Returns:
            Processed image ready for metrics calculation
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Convert to float32 for processing
        if gray.dtype == np.uint8:
            gray = gray.astype(np.float32)
        else:
            gray = gray.astype(np.float32)
            
        return gray
    
    def calculate_acutance(self, image: np.ndarray) -> float:
        """
        Calculate Acutance (edge sharpness measure).
        
        Acutance measures perceived sharpness by calculating the average magnitude 
        of local gradients relative to image intensity. Higher values indicate sharper images.
        
        Args:
            image: Input image (grayscale, float32)
            
        Returns:
            Acutance score (higher values indicate sharper images)
        """
        try:
            img = self._prepare_image(image)
            
            # Calculate Sobel gradients
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            
            # Calculate acutance (normalized by mean intensity)
            mean_intensity = np.mean(img)
            if mean_intensity > 0:
                acutance = np.mean(gradient_magnitude) / mean_intensity
            else:
                acutance = np.mean(gradient_magnitude)
            
            return float(acutance)
            
        except Exception as e:
            self.logger.warning(f"Acutance calculation failed: {e}")
            return 0.0
    
    def calculate_mean_gradient(self, image: np.ndarray) -> float:
        """
        Calculate Mean Gradient (edge strength measure).
        
        Mean Gradient computes the mean gradient magnitude to measure average edge strength.
        Higher values indicate more edges and sharper content.
        
        Args:
            image: Input image (grayscale, float32)
            
        Returns:
            Mean gradient value (higher values indicate more edge content)
        """
        try:
            img = self._prepare_image(image)
            
            # Calculate Sobel gradients
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            
            # Return mean gradient
            return float(np.mean(gradient_magnitude))
            
        except Exception as e:
            self.logger.warning(f"Mean gradient calculation failed: {e}")
            return 0.0
    
    def calculate_fft_high_freq(self, image: np.ndarray) -> float:
        """
        Calculate FFT High-Frequency Energy.
        
        Uses Fourier transform to calculate the proportion of high-frequency components,
        which correspond to image detail and sharpness.
        
        Args:
            image: Input image (grayscale, float32)
            
        Returns:
            High-frequency energy ratio (higher values indicate sharper images)
        """
        try:
            img = self._prepare_image(image)
            
            # Apply 2D FFT
            f = fft2(img)
            fshift = fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            
            # Create high-frequency mask (ignore low-frequency center)
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            
            # Define radius for low-frequency region to mask out
            radius = min(h, w) // 8  # Mask central 25% of spectrum
            
            # Create mask - 1 for high frequencies, 0 for low frequencies
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x)**2 + (y - center_y)**2) > radius**2
            
            # Calculate energy ratios
            total_energy = np.sum(magnitude_spectrum**2)
            high_freq_energy = np.sum(magnitude_spectrum[mask]**2)
            
            if total_energy > 0:
                ratio = high_freq_energy / total_energy
            else:
                ratio = 0.0
            
            return float(ratio)
            
        except Exception as e:
            self.logger.warning(f"FFT high-frequency calculation failed: {e}")
            return 0.0
    
    def calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """
        Calculate Laplacian Variance (focus/blur measure).
        
        Calculates the variance of the Laplacian filter response. Higher variance 
        indicates more focus and sharpness, while lower variance indicates blur.
        
        Args:
            image: Input image (grayscale, float32)
            
        Returns:
            Laplacian variance (higher values indicate sharper/more focused images)
        """
        try:
            img = self._prepare_image(image)
            
            # Convert to uint8 for OpenCV Laplacian (more stable)
            if img.max() <= 1.0:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            
            # Apply Laplacian filter
            laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
            
            # Calculate variance
            variance = np.var(laplacian)
            
            return float(variance)
            
        except Exception as e:
            self.logger.warning(f"Laplacian variance calculation failed: {e}")
            return 0.0


def main():
    """Example usage of sharpness metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate image sharpness metrics")
    parser.add_argument("image", help="Path to image")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load image
    img = cv2.imread(args.image)
    
    if img is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    # Calculate metrics
    metrics_calc = SharpnessMetrics()
    metrics = metrics_calc.calculate_all(img)
    
    # Display results
    print("\nSharpness Metrics:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric.upper():20s}: {value:.4f}")
    print("\nNote: Higher values generally indicate sharper images")


if __name__ == "__main__":
    main()
