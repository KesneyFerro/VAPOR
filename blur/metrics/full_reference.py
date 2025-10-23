"""
Full-Reference Image Quality Metrics

These metrics require a clean ground truth (reference) image to compare 
against the test image. They measure how close the test image is to the reference.

Supported Metrics:
- PSNR (Peak Signal-to-Noise Ratio) - Uses piq library
- SSIM (Structural Similarity Index) - Uses piq library
- MS-SSIM (Multi-Scale Structural Similarity) - Uses piq library
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

Author: Kesney de Oliveira
Date: September 2025
Updated: October 2025 - Using piq for PSNR/SSIM/MS-SSIM, proper implementations
"""

import cv2
import numpy as np
import logging
from typing import Dict, Union, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Try to import piq for metrics
try:
    import torch
    import piq
    PIQ_AVAILABLE = True
except ImportError:
    PIQ_AVAILABLE = False


class FullReferenceMetrics:
    """
    Full-reference image quality metrics calculator.
    
    These metrics require both a test image and a reference (ground truth) image.
    Uses piq library for PSNR, SSIM, MS-SSIM with proper implementations.
    """
    
    def __init__(self):
        """Initialize the full-reference metrics calculator."""
        self.logger = logging.getLogger(__name__)
        
        if not PIQ_AVAILABLE:
            self.logger.warning("piq library not available. Install: pip install piq torch torchvision")
    
    def _prepare_tensor(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Prepare image as PyTorch tensor for piq metrics.
        
        Args:
            image: Input numpy array (HxW or HxWxC)
            
        Returns:
            Tensor of shape (1, C, H, W) normalized to [0, 1], or None on error
        """
        try:
            # Convert to RGB if grayscale
            if len(image.shape) == 2:
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 1:
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                img = image.copy()
            
            # Convert BGR to RGB if needed (OpenCV loads as BGR)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
            
            # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"Failed to prepare tensor: {e}")
            return None
    
    def _prepare_matched_tensors(self, test_image: np.ndarray, reference_image: np.ndarray) -> tuple:
        """
        Prepare two images as matched PyTorch tensors (same size).
        
        Args:
            test_image: Test image numpy array
            reference_image: Reference image numpy array
            
        Returns:
            Tuple of (test_tensor, ref_tensor) with matching sizes, or (None, None) on error
        """
        try:
            # First, ensure both images are the same size (crop to smaller dimensions)
            test_h, test_w = test_image.shape[:2]
            ref_h, ref_w = reference_image.shape[:2]
            
            if (test_h, test_w) != (ref_h, ref_w):
                # Crop to minimum dimensions
                min_h = min(test_h, ref_h)
                min_w = min(test_w, ref_w)
                
                test_image = test_image[:min_h, :min_w]
                reference_image = reference_image[:min_h, :min_w]
                
                self.logger.debug(f"Resized images to match: ({min_h}, {min_w})")
            
            # Now prepare tensors
            test_tensor = self._prepare_tensor(test_image)
            ref_tensor = self._prepare_tensor(reference_image)
            
            if test_tensor is None or ref_tensor is None:
                return None, None
            
            # Double-check sizes match
            if test_tensor.shape != ref_tensor.shape:
                self.logger.error(f"Tensor size mismatch after preparation: {test_tensor.shape} vs {ref_tensor.shape}")
                return None, None
            
            return test_tensor, ref_tensor
            
        except Exception as e:
            self.logger.error(f"Failed to prepare matched tensors: {e}")
            return None, None
    
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
        
        if min(image.shape[:2]) < 7:  # Minimum for SSIM
            raise ValueError(f"{name} too small: {image.shape} (minimum 7x7 for SSIM)")
        
        if np.any(np.isnan(image)):
            raise ValueError(f"{name} contains NaN values")
        
        if np.any(np.isinf(image)):
            raise ValueError(f"{name} contains Inf values")
    
    def calculate_all(self, test_image: np.ndarray, reference_image: np.ndarray) -> Dict[str, float]:
        """
        Calculate all full-reference metrics efficiently.
        
        Optimized to prepare images once and pass to all metrics,
        avoiding redundant conversions.
        
        Args:
            test_image: Test image (deblurred, processed, etc.)
            reference_image: Reference/ground truth image
            
        Returns:
            Dictionary containing all calculated metrics:
            - psnr: Peak Signal-to-Noise Ratio (dB, higher is better)
            - ssim: Structural Similarity Index (0-1, higher is better)
            - ms_ssim: Multi-Scale SSIM (0-1, higher is better)
            - mse: Mean Squared Error (lower is better, 0 is perfect)
            - mae: Mean Absolute Error (lower is better, 0 is perfect)
        """
        metrics = {}
        
        try:
            # Validate inputs
            self._validate_image(test_image, "Test image")
            self._validate_image(reference_image, "Reference image")
            
            # Prepare matched tensors - ensures same size, converts to PyTorch tensors (1, C, H, W) [0,1]
            test_tensor, ref_tensor = self._prepare_matched_tensors(test_image, reference_image)
            
            if test_tensor is None or ref_tensor is None:
                raise ValueError("Failed to prepare matched image tensors")
            
            # Calculate all metrics using prepared tensors (no redundant conversions)
            metrics['psnr'] = self._calc_psnr(test_tensor, ref_tensor)
            metrics['ssim'] = self._calc_ssim(test_tensor, ref_tensor)
            metrics['ms_ssim'] = self._calc_ms_ssim(test_tensor, ref_tensor)
            metrics['mse'] = self._calc_mse(test_tensor, ref_tensor)
            metrics['mae'] = self._calc_mae(test_tensor, ref_tensor)
            
        except Exception as e:
            self.logger.error(f"Error calculating full-reference metrics: {e}")
            # Return None for all metrics on error (no fallbacks policy)
            metrics = {
                'psnr': None,
                'ssim': None,
                'ms_ssim': None,
                'mse': None,
                'mae': None
            }
            
        return metrics
    
    def _prepare_images(self, test_image: np.ndarray, reference_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def _calc_psnr(self, test_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> Optional[float]:
        """
        Calculate PSNR using piq library.
        
        Args:
            test_tensor: Test image tensor (1, C, H, W)
            ref_tensor: Reference image tensor (1, C, H, W)
            
        Returns:
            PSNR value in dB (higher is better) or None if unavailable
        """
        if not PIQ_AVAILABLE:
            self.logger.error("piq not available for PSNR")
            return None
        
        try:
            score = piq.psnr(test_tensor, ref_tensor, data_range=1.0, reduction='none')
            return float(score.item())
        except Exception as e:
            self.logger.error(f"PSNR calculation failed: {e}")
            return None
    
    def _calc_ssim(self, test_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> Optional[float]:
        """
        Calculate SSIM using piq library.
        
        Args:
            test_tensor: Test image tensor (1, C, H, W)
            ref_tensor: Reference image tensor (1, C, H, W)
            
        Returns:
            SSIM value (0-1, higher is better) or None if unavailable
        """
        if not PIQ_AVAILABLE:
            self.logger.error("piq not available for SSIM")
            return None
        
        try:
            score = piq.ssim(test_tensor, ref_tensor, data_range=1.0, reduction='none')
            return float(score.item())
        except Exception as e:
            self.logger.error(f"SSIM calculation failed: {e}")
            return None
    
    def _calc_ms_ssim(self, test_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> Optional[float]:
        """
        Calculate Multi-Scale SSIM using piq library.
        
        Args:
            test_tensor: Test image tensor (1, C, H, W)
            ref_tensor: Reference image tensor (1, C, H, W)
            
        Returns:
            MS-SSIM value (0-1, higher is better) or None if unavailable
        """
        if not PIQ_AVAILABLE:
            self.logger.error("piq not available for MS-SSIM")
            return None
        
        try:
            score = piq.multi_scale_ssim(test_tensor, ref_tensor, data_range=1.0, reduction='none')
            return float(score.item())
        except Exception as e:
            self.logger.error(f"MS-SSIM calculation failed: {e}")
            return None
    
    def _calc_mse(self, test_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> Optional[float]:
        """
        Calculate Mean Squared Error.
        
        Args:
            test_tensor: Test image tensor (1, C, H, W)
            ref_tensor: Reference image tensor (1, C, H, W)
            
        Returns:
            MSE value (lower is better, 0 is perfect) or None on error
        """
        try:
            mse = torch.mean((test_tensor - ref_tensor) ** 2)
            return float(mse.item())
        except Exception as e:
            self.logger.error(f"MSE calculation failed: {e}")
            return None
    
    def _calc_mae(self, test_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> Optional[float]:
        """
        Calculate Mean Absolute Error.
        
        Args:
            test_tensor: Test image tensor (1, C, H, W)
            ref_tensor: Reference image tensor (1, C, H, W)
            
        Returns:
            MAE value (lower is better, 0 is perfect) or None on error
        """
        try:
            mae = torch.mean(torch.abs(test_tensor - ref_tensor))
            return float(mae.item())
        except Exception as e:
            self.logger.error(f"MAE calculation failed: {e}")
            return None
    
    def calculate_psnr(self, test_image: np.ndarray, reference_image: np.ndarray) -> Optional[float]:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Note: For efficiency, use calculate_all() to compute all metrics at once.
        
        Args:
            test_image: Test image
            reference_image: Reference image
            
        Returns:
            PSNR value in dB (higher is better) or None if unavailable
        """
        try:
            test_tensor = self._prepare_tensor(test_image)
            ref_tensor = self._prepare_tensor(reference_image)
            if test_tensor is None or ref_tensor is None:
                return None
            return self._calc_psnr(test_tensor, ref_tensor)
        except Exception as e:
            self.logger.error(f"PSNR calculation failed: {e}")
            return None
    
    def calculate_ssim(self, test_image: np.ndarray, reference_image: np.ndarray) -> Optional[float]:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Note: For efficiency, use calculate_all() to compute all metrics at once.
        
        Args:
            test_image: Test image
            reference_image: Reference image
            
        Returns:
            SSIM value (0-1, higher is better) or None if unavailable
        """
        try:
            test_tensor = self._prepare_tensor(test_image)
            ref_tensor = self._prepare_tensor(reference_image)
            if test_tensor is None or ref_tensor is None:
                return None
            return self._calc_ssim(test_tensor, ref_tensor)
        except Exception as e:
            self.logger.error(f"SSIM calculation failed: {e}")
            return None
    
    def calculate_ms_ssim(self, test_image: np.ndarray, reference_image: np.ndarray) -> Optional[float]:
        """
        Calculate Multi-Scale Structural Similarity Index (MS-SSIM).
        
        Note: For efficiency, use calculate_all() to compute all metrics at once.
        
        Args:
            test_image: Test image
            reference_image: Reference image
            
        Returns:
            MS-SSIM value (0-1, higher is better) or None if unavailable
        """
        try:
            test_tensor = self._prepare_tensor(test_image)
            ref_tensor = self._prepare_tensor(reference_image)
            if test_tensor is None or ref_tensor is None:
                return None
            return self._calc_ms_ssim(test_tensor, ref_tensor)
        except Exception as e:
            self.logger.error(f"MS-SSIM calculation failed: {e}")
            return None
    
    def calculate_mse(self, test_image: np.ndarray, reference_image: np.ndarray) -> Optional[float]:
        """
        Calculate Mean Squared Error (MSE).
        
        MSE measures the average squared difference between pixels.
        Lower values indicate better quality (0 = perfect match).
        
        Note: For efficiency, use calculate_all() to compute all metrics at once.
        
        Args:
            test_image: Test image
            reference_image: Reference image
            
        Returns:
            MSE value (lower is better, 0 is perfect)
        """
        try:
            test_tensor = self._prepare_tensor(test_image)
            ref_tensor = self._prepare_tensor(reference_image)
            if test_tensor is None or ref_tensor is None:
                return None
            return self._calc_mse(test_tensor, ref_tensor)
        except Exception as e:
            self.logger.error(f"MSE calculation failed: {e}")
            return None
    
    def calculate_mae(self, test_image: np.ndarray, reference_image: np.ndarray) -> Optional[float]:
        """
        Calculate Mean Absolute Error (MAE).
        
        Note: For efficiency, use calculate_all() to compute all metrics at once.
        
        Args:
            test_image: Test image
            reference_image: Reference image
            
        Returns:
            MAE value (lower is better, 0 is perfect) or None if unavailable
        """
        try:
            test_tensor = self._prepare_tensor(test_image)
            ref_tensor = self._prepare_tensor(reference_image)
            if test_tensor is None or ref_tensor is None:
                return None
            return self._calc_mae(test_tensor, ref_tensor)
        except Exception as e:
            self.logger.error(f"MAE calculation failed: {e}")
            return None


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
    print("=" * 50)
    print(f"{'Metric':<12} {'Value':>10} {'Interpretation'}")
    print("=" * 50)
    print(f"{'PSNR':<12} {metrics['psnr']:>10.2f} dB (>40: excellent, 30-40: good, 20-30: fair)")
    print(f"{'SSIM':<12} {metrics['ssim']:>10.4f}    (>0.95: excellent, 0.90-0.95: good)")
    print(f"{'MS-SSIM':<12} {metrics['ms_ssim']:>10.4f}    (>0.95: excellent, 0.90-0.95: good)")
    print(f"{'MSE':<12} {metrics['mse']:>10.6f}    (<0.001: excellent, lower is better)")
    print(f"{'MAE':<12} {metrics['mae']:>10.6f}    (<0.01: excellent, lower is better)")


if __name__ == "__main__":
    main()
