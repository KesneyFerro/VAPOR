"""
Full-Reference Image Quality Metrics

These metrics require a clean ground truth (reference) image to compare 
against the test image. They measure how close the test image is to the reference.

Supported Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index) 
- MS-SSIM (Multi-Scale Structural Similarity)

Author: Kesney de Oliveira
Date: September 2025
"""

import cv2
import numpy as np
import logging
from typing import Dict, Union
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class FullReferenceMetrics:
    """
    Full-reference image quality metrics calculator.
    
    These metrics require both a test image and a reference (ground truth) image.
    """
    
    def __init__(self):
        """Initialize the full-reference metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_all(self, test_image: np.ndarray, reference_image: np.ndarray) -> Dict[str, float]:
        """
        Calculate all full-reference metrics.
        
        Args:
            test_image: Test image (deblurred, processed, etc.)
            reference_image: Reference/ground truth image
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        try:
            # Ensure images are same size
            test_img, ref_img = self._prepare_images(test_image, reference_image)
            
            # Calculate metrics
            metrics['psnr'] = self.calculate_psnr(test_img, ref_img)
            metrics['ssim'] = self.calculate_ssim(test_img, ref_img)
            metrics['ms_ssim'] = self.calculate_ms_ssim(test_img, ref_img)
            
        except Exception as e:
            self.logger.error(f"Error calculating full-reference metrics: {e}")
            
        return metrics
    
    def _prepare_images(self, test_image: np.ndarray, reference_image: np.ndarray):
        """
        Prepare images for comparison (ensure same size, format).
        
        Args:
            test_image: Test image
            reference_image: Reference image
            
        Returns:
            Tuple of (test_image, reference_image) prepared for comparison
        """
        # Convert to grayscale if needed
        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()
            
        if len(reference_image.shape) == 3:
            ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference_image.copy()
        
        # Ensure same size
        if test_gray.shape != ref_gray.shape:
            min_h = min(test_gray.shape[0], ref_gray.shape[0])
            min_w = min(test_gray.shape[1], ref_gray.shape[1])
            test_gray = test_gray[:min_h, :min_w]
            ref_gray = ref_gray[:min_h, :min_w]
        
        # Normalize to [0, 1] range
        if test_gray.dtype == np.uint8:
            test_gray = test_gray.astype(np.float64) / 255.0
        if ref_gray.dtype == np.uint8:
            ref_gray = ref_gray.astype(np.float64) / 255.0
            
        return test_gray, ref_gray
    
    def calculate_psnr(self, test_image: np.ndarray, reference_image: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        PSNR measures how close the test image is to the reference using 
        logarithmic ratio of signal to noise. Higher values indicate better quality.
        
        Args:
            test_image: Test image (normalized to [0,1])
            reference_image: Reference image (normalized to [0,1])
            
        Returns:
            PSNR value in dB (higher is better)
        """
        try:
            test_img, ref_img = self._prepare_images(test_image, reference_image)
            return float(psnr(ref_img, test_img, data_range=1.0))
        except Exception as e:
            self.logger.warning(f"PSNR calculation failed: {e}")
            return 0.0
    
    def calculate_ssim(self, test_image: np.ndarray, reference_image: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        SSIM evaluates similarity based on luminance, contrast, and structure.
        Values range from -1 to 1, with 1 indicating perfect similarity.
        
        Args:
            test_image: Test image
            reference_image: Reference image
            
        Returns:
            SSIM value between -1 and 1 (higher is better)
        """
        try:
            test_img, ref_img = self._prepare_images(test_image, reference_image)
            return float(ssim(ref_img, test_img, data_range=1.0))
        except Exception as e:
            self.logger.warning(f"SSIM calculation failed: {e}")
            return 0.0
    
    def calculate_ms_ssim(self, test_image: np.ndarray, reference_image: np.ndarray) -> float:
        """
        Calculate Multi-Scale Structural Similarity Index (MS-SSIM).
        
        MS-SSIM is a multi-scale version of SSIM that evaluates structural 
        similarity at multiple image resolutions. More stable for larger images.
        
        Args:
            test_image: Test image
            reference_image: Reference image
            
        Returns:
            MS-SSIM value between 0 and 1 (higher is better)
        """
        try:
            test_img, ref_img = self._prepare_images(test_image, reference_image)
            
            # Calculate SSIM at multiple scales
            scales = [1.0, 0.5, 0.25, 0.125]
            ms_ssim_values = []
            
            for scale in scales:
                if scale < 1.0:
                    # Resize images for this scale
                    h, w = int(test_img.shape[0] * scale), int(test_img.shape[1] * scale)
                    if h < 7 or w < 7:  # Skip if too small for SSIM
                        continue
                    test_scaled = cv2.resize(test_img, (w, h))
                    ref_scaled = cv2.resize(ref_img, (w, h))
                else:
                    test_scaled = test_img
                    ref_scaled = ref_img
                
                # Calculate SSIM for this scale
                if min(test_scaled.shape) >= 7:  # Minimum size for SSIM
                    scale_ssim = ssim(ref_scaled, test_scaled, data_range=1.0)
                    ms_ssim_values.append(scale_ssim)
            
            # Return mean SSIM across scales
            if ms_ssim_values:
                return float(np.mean(ms_ssim_values))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"MS-SSIM calculation failed: {e}")
            return 0.0


def main():
    """Example usage of full-reference metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate full-reference image quality metrics")
    parser.add_argument("test_image", help="Path to test image")
    parser.add_argument("reference_image", help="Path to reference image")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load images
    test_img = cv2.imread(args.test_image)
    ref_img = cv2.imread(args.reference_image)
    
    if test_img is None:
        print(f"Error: Could not load test image {args.test_image}")
        return
    if ref_img is None:
        print(f"Error: Could not load reference image {args.reference_image}")
        return
    
    # Calculate metrics
    metrics_calc = FullReferenceMetrics()
    metrics = metrics_calc.calculate_all(test_img, ref_img)
    
    # Display results
    print("\nFull-Reference Quality Metrics:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric.upper():10s}: {value:.4f}")


if __name__ == "__main__":
    main()
