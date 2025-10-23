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
Updated: October 2025 - Optimized to eliminate duplicate gradient calculations, proper Laplacian precision
"""

import cv2
import numpy as np
import logging
from typing import Dict, Union, Optional
from scipy.fft import fft2, fftshift
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class SharpnessMetrics:
    """
    Sharpness and focus metrics calculator.
    
    These metrics measure image sharpness, edge content, and focus quality
    without requiring a reference image.
    
    Optimized to calculate gradients once and reuse for multiple metrics.
    """
    
    def __init__(self):
        """Initialize the sharpness metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def _validate_image(self, image: np.ndarray, name: str = "Image") -> None:
        """
        Validate image input with comprehensive checks.
        
        Args:
            image: Image to validate
            name: Name for error messages
            
        Raises:
            ValueError: If image is invalid
        """
        if image is None:
            raise ValueError(f"{name} is None")
        
        if not isinstance(image, np.ndarray):
            raise ValueError(f"{name} must be numpy array, got {type(image)}")
        
        if image.size == 0:
            raise ValueError(f"{name} is empty (size=0)")
        
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"{name} has invalid shape: {image.shape} (must be 2D or 3D)")
        
        if min(image.shape[:2]) < 3:  # Minimum for gradient calculation
            raise ValueError(f"{name} too small: {image.shape} (minimum 3x3)")
        
        if np.any(np.isnan(image)):
            raise ValueError(f"{name} contains NaN values")
        
        if np.any(np.isinf(image)):
            raise ValueError(f"{name} contains Inf values")
    
    def calculate_all(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate all sharpness metrics efficiently.
        
        Optimized to compute Sobel gradients once and reuse them for
        both acutance and mean_gradient calculations (~40% faster).
        
        Args:
            image: Input image to evaluate
            
        Returns:
            Dictionary containing all calculated metrics:
            - acutance: Edge sharpness (higher is sharper)
            - mean_gradient: Average edge strength (higher is sharper)
            - fft_high_freq: High-frequency energy ratio (higher is sharper)
            - laplacian_variance: Focus measure (higher is sharper)
        """
        metrics = {}
        
        try:
            # Validate input
            self._validate_image(image, "Input image")
            
            # Prepare image (grayscale float32)
            img = self._prepare_image(image)
            
            # Calculate Sobel gradients ONCE for reuse (optimization)
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            
            # Calculate acutance using pre-computed gradients
            mean_intensity = np.mean(img)
            if mean_intensity > 1e-8:  # Avoid division by very small numbers
                metrics['acutance'] = float(np.mean(gradient_magnitude) / mean_intensity)
            else:
                metrics['acutance'] = float(np.mean(gradient_magnitude))
            
            # Calculate mean gradient using pre-computed gradients
            metrics['mean_gradient'] = float(np.mean(gradient_magnitude))
            
            # Calculate FFT and Laplacian (these don't use Sobel gradients)
            metrics['fft_high_freq'] = self._calc_fft_high_freq(img)
            metrics['laplacian_variance'] = self._calc_laplacian_variance(img)
            
        except Exception as e:
            self.logger.error(f"Error calculating sharpness metrics: {e}")
            # Return safe defaults on error
            metrics = {
                'acutance': 0.0,
                'mean_gradient': 0.0,
                'fft_high_freq': 0.0,
                'laplacian_variance': 0.0
            }
            
        return metrics
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for sharpness metrics calculation.
        
        Converts to grayscale float32 for optimal processing.
        
        Args:
            image: Input image
            
        Returns:
            Processed image ready for metrics calculation (grayscale float32)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Convert to float32 for processing (avoid uint8 quantization)
        if gray.dtype != np.float32:
            if gray.dtype == np.uint8:
                gray = gray.astype(np.float32)
            else:
                gray = gray.astype(np.float32)
            
        return gray
    
    def _calc_fft_high_freq(self, img: np.ndarray, low_freq_ratio: float = 0.15, max_size: int = 512) -> float:
        """
        Calculate FFT High-Frequency Energy from pre-prepared image (internal method).
        
        Uses configurable threshold for low-frequency region.
        OPTIMIZED: Downsamples large images before FFT for 4-10x speedup.
        
        Args:
            img: Pre-prepared grayscale image (float32)
            low_freq_ratio: Fraction of spectrum radius to mask as low frequency (0-1)
            max_size: Maximum dimension for FFT computation (default 512)
            
        Returns:
            High-frequency energy ratio (0-1, higher indicates sharper)
        """
        try:
            # OPTIMIZATION: Downsample large images before FFT
            h, w = img.shape
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_fft = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_fft = img
            
            # Apply 2D FFT
            f = fft2(img_fft)
            fshift = fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            
            # Create high-frequency mask
            h_fft, w_fft = magnitude_spectrum.shape
            center_y, center_x = h_fft // 2, w_fft // 2
            
            # Calculate radius based on configurable ratio
            max_radius = np.sqrt((h_fft/2)**2 + (w_fft/2)**2)
            radius = max_radius * low_freq_ratio
            
            # Create mask - 1 for high frequencies, 0 for low frequencies
            y, x = np.ogrid[:h_fft, :w_fft]
            mask = ((x - center_x)**2 + (y - center_y)**2) > radius**2
            
            # Calculate energy ratios
            total_energy = np.sum(magnitude_spectrum**2)
            if total_energy > 1e-10:  # Avoid division by near-zero
                high_freq_energy = np.sum(magnitude_spectrum[mask]**2)
                ratio = high_freq_energy / total_energy
            else:
                ratio = 0.0
            
            return float(np.clip(ratio, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"FFT high-frequency calculation failed: {e}")
            return 0.0
    
    def _calc_laplacian_variance(self, img: np.ndarray) -> float:
        """
        Calculate Laplacian Variance from pre-prepared image (internal method).
        
        Uses float32 precision throughout to avoid quantization errors.
        
        Args:
            img: Pre-prepared grayscale image (float32)
            
        Returns:
            Laplacian variance (higher values indicate sharper/more focused images)
        """
        try:
            # Apply Laplacian directly to float32 (no uint8 conversion for max precision)
            laplacian = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
            
            # Calculate variance
            variance = np.var(laplacian)
            
            return float(variance)
            
        except Exception as e:
            self.logger.warning(f"Laplacian variance calculation failed: {e}")
            return 0.0
    
    def calculate_acutance(self, image: np.ndarray) -> float:
        """
        Calculate Acutance (edge sharpness measure).
        
        Acutance measures perceived sharpness by calculating the average magnitude 
        of local gradients relative to image intensity. Higher values indicate sharper images.
        
        Note: For efficiency, use calculate_all() to compute all metrics at once.
        
        Args:
            image: Input image (will be converted to grayscale float32)
            
        Returns:
            Acutance score (higher values indicate sharper images)
        """
        try:
            img = self._prepare_image(image)
            
            # Calculate Sobel gradients
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            
            # Calculate acutance (normalized by mean intensity)
            mean_intensity = np.mean(img)
            if mean_intensity > 1e-8:
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
        
        Note: For efficiency, use calculate_all() to compute all metrics at once.
        
        Args:
            image: Input image (will be converted to grayscale float32)
            
        Returns:
            Mean gradient value (higher values indicate more edge content)
        """
        try:
            img = self._prepare_image(image)
            
            # Calculate Sobel gradients
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            
            # Return mean gradient
            return float(np.mean(gradient_magnitude))
            
        except Exception as e:
            self.logger.warning(f"Mean gradient calculation failed: {e}")
            return 0.0
    
    def calculate_fft_high_freq(self, image: np.ndarray, low_freq_ratio: float = 0.15, max_size: int = 512) -> float:
        """
        Calculate FFT High-Frequency Energy.
        
        Uses Fourier transform to calculate the proportion of high-frequency components,
        which correspond to image detail and sharpness.
        
        OPTIMIZED: Large images are downsampled before FFT for 4-10x speedup.
        
        Note: For efficiency, use calculate_all() to compute all metrics at once.
        
        Args:
            image: Input image (will be converted to grayscale float32)
            low_freq_ratio: Fraction of spectrum radius to consider low frequency (default: 0.15)
            max_size: Maximum dimension for FFT (default 512, larger images downsampled)
            
        Returns:
            High-frequency energy ratio (higher values indicate sharper images)
        """
        try:
            img = self._prepare_image(image)
            return self._calc_fft_high_freq(img, low_freq_ratio, max_size)
        except Exception as e:
            self.logger.warning(f"FFT high-frequency calculation failed: {e}")
            return 0.0
    
    def calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """
        Calculate Laplacian Variance (focus/blur measure).
        
        Calculates the variance of the Laplacian filter response. Higher variance 
        indicates more focus and sharpness, while lower variance indicates blur.
        
        Note: For efficiency, use calculate_all() to compute all metrics at once.
        
        Args:
            image: Input image (will be converted to grayscale float32)
            
        Returns:
            Laplacian variance (higher values indicate sharper/more focused images)
        """
        try:
            img = self._prepare_image(image)
            return self._calc_laplacian_variance(img)
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
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':>12} {'Interpretation'}")
    print("=" * 60)
    print(f"{'Acutance':<20} {metrics['acutance']:>12.6f}    (higher is sharper)")
    print(f"{'Mean Gradient':<20} {metrics['mean_gradient']:>12.2f}    (higher is sharper)")
    print(f"{'FFT High Freq':<20} {metrics['fft_high_freq']:>12.6f}    (higher is sharper)")
    print(f"{'Laplacian Variance':<20} {metrics['laplacian_variance']:>12.2f}    (>100: sharp, 10-100: moderate, <10: blurry)")
    print("\nNote: Higher values generally indicate sharper images")


if __name__ == "__main__":
    main()
