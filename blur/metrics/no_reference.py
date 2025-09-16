"""
No-Reference Image Quality Metrics

These metrics do not require a ground truth image. They evaluate the image 
based on natural statistics, learned models, or image characteristics.

Supported Metrics:
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
- NIQE (Natural Image Quality Evaluator)

Author: Kesney de Oliveira
Date: September 2025
"""

import cv2
import numpy as np
import logging
from typing import Dict, Union
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class NoReferenceMetrics:
    """
    No-reference (blind) image quality metrics calculator.
    
    These metrics evaluate image quality without requiring a reference image.
    """
    
    def __init__(self):
        """Initialize the no-reference metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_all(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate all no-reference metrics.
        
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
            metrics['brisque'] = self.calculate_brisque(img)
            metrics['niqe'] = self.calculate_niqe(img)
            
        except Exception as e:
            self.logger.error(f"Error calculating no-reference metrics: {e}")
            
        return metrics
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for no-reference metrics calculation.
        
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
        
        # Ensure uint8 format
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
                
        return gray
    
    def calculate_brisque(self, image: np.ndarray) -> float:
        """
        Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).
        
        BRISQUE predicts quality by analyzing natural scene statistics and distortions.
        This is a simplified implementation. For full accuracy, use pre-trained models.
        
        Args:
            image: Input image (grayscale, uint8)
            
        Returns:
            BRISQUE score (lower values indicate better quality)
        """
        try:
            img = self._prepare_image(image)
            
            # Calculate local mean and standard deviation using Gaussian filtering
            mu = cv2.GaussianBlur(img.astype(np.float64), (7, 7), 1.166)
            mu_sq = mu * mu
            sigma = cv2.GaussianBlur((img.astype(np.float64) * img.astype(np.float64)), (7, 7), 1.166)
            sigma = np.sqrt(np.abs(sigma - mu_sq))
            
            # Normalize (MSCN coefficients)
            structdis = (img.astype(np.float64) - mu) / (sigma + 1)
            
            # Calculate features only where sigma > 0
            valid_mask = sigma > 0
            if not np.any(valid_mask):
                return 100.0  # Return poor quality score if no valid pixels
            
            structdis_valid = structdis[valid_mask]
            
            # Calculate statistical features
            mean_mscn = np.mean(structdis_valid)
            std_mscn = np.std(structdis_valid)
            
            # Calculate skewness and kurtosis
            if std_mscn > 0:
                skewness = np.mean(((structdis_valid - mean_mscn) / std_mscn) ** 3)
                kurtosis = np.mean(((structdis_valid - mean_mscn) / std_mscn) ** 4) - 3
            else:
                skewness = 0
                kurtosis = 0
            
            # Calculate pairwise product features (simplified)
            # Horizontal pairs
            h_pairs = structdis_valid[:-1] * structdis_valid[1:]
            mean_h = np.mean(h_pairs) if len(h_pairs) > 0 else 0
            std_h = np.std(h_pairs) if len(h_pairs) > 0 else 0
            
            # Combine features into quality score
            # This is a simplified scoring - real BRISQUE uses trained SVR model
            features = np.array([
                abs(mean_mscn),
                std_mscn,
                abs(skewness),
                abs(kurtosis),
                abs(mean_h),
                std_h
            ])
            
            # Simple linear combination (in practice, this would be learned)
            weights = np.array([1.0, 2.0, 1.5, 1.5, 1.0, 1.0])
            brisque_score = np.sum(features * weights)
            
            return float(brisque_score)
            
        except Exception as e:
            self.logger.warning(f"BRISQUE calculation failed: {e}")
            return 100.0  # Return poor quality score on error
    
    def calculate_niqe(self, image: np.ndarray) -> float:
        """
        Calculate NIQE (Natural Image Quality Evaluator).
        
        NIQE uses natural scene statistical models to predict how "natural" 
        an image looks based on spatial domain NSS features.
        
        Args:
            image: Input image (grayscale, uint8)
            
        Returns:
            NIQE score (lower values indicate better quality)
        """
        try:
            img = self._prepare_image(image).astype(np.float64)
            
            # Calculate local patches
            patch_size = 6
            h, w = img.shape
            patches = []
            
            # Extract non-overlapping patches
            for i in range(0, h - patch_size + 1, patch_size):
                for j in range(0, w - patch_size + 1, patch_size):
                    patch = img[i:i+patch_size, j:j+patch_size]
                    if patch.shape == (patch_size, patch_size):
                        patches.append(patch.flatten())
            
            if len(patches) == 0:
                return 100.0  # Return poor quality if no patches
            
            patches = np.array(patches)
            
            # Calculate patch statistics
            patch_means = np.mean(patches, axis=1)
            patch_stds = np.std(patches, axis=1)
            
            # Calculate quality features based on natural scene statistics
            # In real NIQE, these would be compared to learned natural scene model
            
            # Feature 1: Mean luminance distribution
            mean_deviation = np.std(patch_means)
            
            # Feature 2: Contrast distribution  
            contrast_deviation = np.std(patch_stds)
            
            # Feature 3: Spatial correlation (simplified)
            spatial_corr = 0
            if len(patches) > 1:
                # Calculate correlation between adjacent patches
                correlations = []
                for i in range(len(patches) - 1):
                    corr = np.corrcoef(patches[i], patches[i+1])[0,1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                if correlations:
                    spatial_corr = np.mean(correlations)
            
            # Feature 4: Local variance uniformity
            variance_uniformity = np.std(patch_stds) / (np.mean(patch_stds) + 1e-8)
            
            # Combine features into NIQE score
            # Real NIQE uses multivariate Gaussian model fitted to natural images
            features = np.array([
                mean_deviation,
                contrast_deviation, 
                spatial_corr,
                variance_uniformity
            ])
            
            # Simple distance-based scoring (simplified version)
            # In practice, this uses Mahalanobis distance to natural image model
            niqe_score = np.sqrt(np.sum(features ** 2))
            
            return float(niqe_score)
            
        except Exception as e:
            self.logger.warning(f"NIQE calculation failed: {e}")
            return 100.0  # Return poor quality score on error


def main():
    """Example usage of no-reference metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate no-reference image quality metrics")
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
    metrics_calc = NoReferenceMetrics()
    metrics = metrics_calc.calculate_all(img)
    
    # Display results
    print("\nNo-Reference Quality Metrics:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric.upper():10s}: {value:.4f}")
    print("\nNote: Lower values indicate better quality for these metrics")


if __name__ == "__main__":
    main()
