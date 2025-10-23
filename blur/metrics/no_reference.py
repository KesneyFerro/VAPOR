"""
No-Reference Image Quality Metrics

These metrics do not require a ground truth image. They evaluate the image 
based on natural statistics, learned models, or image characteristics.

Supported Metrics:
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) - Uses pyiqa library with trained model
- NIQE (Natural Image Quality Evaluator) - Uses pyiqa library with trained model

Author: Kesney de Oliveira
Date: September 2025
Updated: October 2025 - Proper BRISQUE and NIQE using pyiqa with trained models
"""

import cv2
import numpy as np
import logging
from typing import Dict, Union, Optional
from pathlib import Path
import warnings

# Suppress specific numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')

# Try to import pyiqa for BRISQUE and NIQE
try:
    import torch
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False


class NoReferenceMetrics:
    """
    No-reference (blind) image quality metrics calculator.
    
    These metrics evaluate image quality without requiring a reference image.
    Uses proper trained models from pyiqa for BRISQUE and NIQE.
    """
    
    def __init__(self):
        """Initialize the no-reference metrics calculator."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize pyiqa metrics if available
        self.brisque_metric = None
        self.niqe_metric = None
        
        if PYIQA_AVAILABLE:
            try:
                # Create metric instances (models will be downloaded on first use)
                self.brisque_metric = pyiqa.create_metric('brisque')
                self.niqe_metric = pyiqa.create_metric('niqe')
            except Exception as e:
                self.logger.error(f"Failed to initialize pyiqa metrics: {e}")
        else:
            self.logger.warning("pyiqa library not available. Install: pip install pyiqa torch torchvision")
    
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
    
    def calculate_all(self, image: np.ndarray) -> Dict[str, Optional[float]]:
        """
        Calculate all no-reference metrics.
        
        Args:
            image: Input image to evaluate
            
        Returns:
            Dictionary containing all calculated metrics (None if metric unavailable)
        """
        metrics = {}
        
        try:
            # Prepare image
            img = self._prepare_image(image)
            
            # Calculate metrics - returns None if unavailable
            metrics['brisque'] = self.calculate_brisque(img)
            metrics['niqe'] = self.calculate_niqe(img)
            
        except Exception as e:
            self.logger.error(f"Error calculating no-reference metrics: {e}")
            metrics['brisque'] = None
            metrics['niqe'] = None
            
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
    
    def calculate_brisque(self, image: np.ndarray) -> Optional[float]:
        """
        Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).
        
        Uses pyiqa library's trained BRISQUE model with proper SVR prediction.
        BRISQUE scores range from 0-100, where lower is better quality.
        
        Typical ranges:
        - Excellent: 0-20
        - Good: 20-40
        - Fair: 40-60
        - Poor: 60-100
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            BRISQUE score (0-100, lower is better quality) or None if unavailable
        """
        if self.brisque_metric is None:
            return None
        
        try:
            # Prepare image as tensor
            tensor = self._prepare_tensor(image)
            if tensor is None:
                return None
            
            # Calculate BRISQUE using pyiqa (returns tensor)
            with torch.no_grad():
                score = self.brisque_metric(tensor)
            
            # Convert to float
            return float(score.item())
            
        except Exception as e:
            self.logger.error(f"BRISQUE calculation failed: {e}")
            return None
    
    def calculate_niqe(self, image: np.ndarray) -> Optional[float]:
        """
        Calculate NIQE (Natural Image Quality Evaluator).
        
        Uses pyiqa's implementation with trained natural scene statistics.
        NIQE scores typically range 0-10+, where lower is better quality.
        
        Typical ranges:
        - Excellent: 0-3
        - Good: 3-5
        - Fair: 5-7
        - Poor: 7+
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            NIQE score (lower is better quality) or None if unavailable
        """
        if self.niqe_metric is None:
            return None
        
        try:
            # Prepare image as tensor
            tensor = self._prepare_tensor(image)
            if tensor is None:
                return None
            
            # Calculate NIQE using pyiqa (returns tensor)
            with torch.no_grad():
                score = self.niqe_metric(tensor)
            
            return float(score.item())
            
        except Exception as e:
            self.logger.error(f"NIQE calculation failed: {e}")
            return None


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
