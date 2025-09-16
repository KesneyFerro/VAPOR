"""
Image Quality Metrics Calculator

Main interface for calculating all types of image quality metrics.
Combines full-reference, no-reference, and sharpness metrics into a unified system.

Author: Kesney de Oliveira
Date: September 2025
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Union, Optional, List

from .full_reference import FullReferenceMetrics
from .no_reference import NoReferenceMetrics
from .sharpness import SharpnessMetrics


class MetricsCalculator:
    """
    Unified interface for calculating image quality metrics.
    
    Provides convenient methods for batch processing, file I/O, and result organization.
    Combines full-reference, no-reference, and sharpness metrics.
    """
    
    def __init__(self):
        """Initialize the metrics calculator with all metric types."""
        self.full_reference = FullReferenceMetrics()
        self.no_reference = NoReferenceMetrics()
        self.sharpness = SharpnessMetrics()
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_metrics(self, 
                            test_image: np.ndarray, 
                            reference_image: Optional[np.ndarray] = None,
                            include_full_reference: bool = True,
                            include_no_reference: bool = True,
                            include_sharpness: bool = True) -> Dict[str, float]:
        """
        Calculate all available quality metrics for an image.
        
        Args:
            test_image: Test image to evaluate
            reference_image: Reference image for full-reference metrics (optional)
            include_full_reference: Include full-reference metrics (requires reference_image)
            include_no_reference: Include no-reference metrics
            include_sharpness: Include sharpness metrics
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Full-reference metrics (require reference image)
        if include_full_reference and reference_image is not None:
            try:
                fr_metrics = self.full_reference.calculate_all(test_image, reference_image)
                metrics.update(fr_metrics)
            except Exception as e:
                self.logger.warning(f"Error calculating full-reference metrics: {e}")
        
        # No-reference metrics
        if include_no_reference:
            try:
                nr_metrics = self.no_reference.calculate_all(test_image)
                metrics.update(nr_metrics)
            except Exception as e:
                self.logger.warning(f"Error calculating no-reference metrics: {e}")
        
        # Sharpness metrics
        if include_sharpness:
            try:
                sharpness_metrics = self.sharpness.calculate_all(test_image)
                metrics.update(sharpness_metrics)
            except Exception as e:
                self.logger.warning(f"Error calculating sharpness metrics: {e}")
        
        return metrics
    
    def calculate_metrics_for_image_pair(self, 
                                       test_path: Union[str, Path], 
                                       reference_path: Optional[Union[str, Path]] = None) -> Dict[str, float]:
        """
        Calculate metrics for a test image (with optional reference).
        
        Args:
            test_path: Path to test image
            reference_path: Path to reference image (optional)
            
        Returns:
            Dictionary of calculated metrics
        """
        try:
            # Load test image
            test_image = cv2.imread(str(test_path))
            if test_image is None:
                raise ValueError(f"Could not load test image: {test_path}")
            
            # Load reference image if provided
            reference_image = None
            if reference_path:
                reference_image = cv2.imread(str(reference_path))
                if reference_image is None:
                    self.logger.warning(f"Could not load reference image: {reference_path}")
                    reference_image = None
            
            # Calculate metrics
            metrics = self.calculate_all_metrics(test_image, reference_image)
            
            # Add metadata
            metrics['test_image'] = str(test_path)
            if reference_path:
                metrics['reference_image'] = str(reference_path)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {test_path}: {e}")
            return {}
    
    def calculate_metrics_for_directory(self, 
                                      test_dir: Union[str, Path],
                                      reference_dir: Optional[Union[str, Path]] = None,
                                      output_file: Optional[Union[str, Path]] = None,
                                      image_extensions: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for all images in a directory.
        
        Args:
            test_dir: Directory containing test images
            reference_dir: Directory containing reference images (optional)
            output_file: Path to save results JSON file (optional)
            image_extensions: List of image extensions to process
            
        Returns:
            Dictionary mapping image names to their metrics
        """
        test_dir = Path(test_dir)
        reference_dir = Path(reference_dir) if reference_dir else None
        
        # Default image extensions
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Find all image files
        test_images = []
        for ext in image_extensions:
            test_images.extend(test_dir.glob(f'*{ext}'))
            test_images.extend(test_dir.glob(f'*{ext.upper()}'))
        
        if not test_images:
            self.logger.warning(f"No images found in {test_dir}")
            return {}
        
        results = {}
        total_images = len(test_images)
        
        for i, test_path in enumerate(test_images, 1):
            self.logger.info(f"Processing image {i}/{total_images}: {test_path.name}")
            
            # Find corresponding reference image
            reference_path = None
            if reference_dir:
                reference_path = self._find_reference_image(test_path, reference_dir, image_extensions)
            
            # Calculate metrics
            metrics = self.calculate_metrics_for_image_pair(test_path, reference_path)
            if metrics:
                results[test_path.name] = metrics
        
        # Save results if requested
        if output_file and results:
            self._save_results(results, output_file)
        
        return results
    
    def _find_reference_image(self, test_path: Path, reference_dir: Path, 
                             image_extensions: List[str]) -> Optional[Path]:
        """
        Find corresponding reference image for a test image.
        
        Args:
            test_path: Path to test image
            reference_dir: Directory containing reference images
            image_extensions: List of valid image extensions
            
        Returns:
            Path to reference image or None if not found
        """
        # Try exact name match first
        reference_path = reference_dir / test_path.name
        if reference_path.exists():
            return reference_path
        
        # Try with different extensions
        stem = test_path.stem
        for ext in image_extensions:
            potential_ref = reference_dir / f"{stem}{ext}"
            if potential_ref.exists():
                return potential_ref
            potential_ref = reference_dir / f"{stem}{ext.upper()}"
            if potential_ref.exists():
                return potential_ref
        
        return None
    
    def _save_results(self, results: Dict, output_file: Union[str, Path]):
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary to save
            output_file: Output file path
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def compare_blur_effects(self, 
                           original_image: Union[str, Path],
                           blurred_images_dir: Union[str, Path],
                           output_file: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple blur effects against an original image.
        
        Args:
            original_image: Path to original (reference) image
            blurred_images_dir: Directory containing blurred versions
            output_file: Path to save comparison results
            
        Returns:
            Dictionary mapping blur effect names to their metrics
        """
        original_path = Path(original_image)
        if not original_path.exists():
            raise FileNotFoundError(f"Original image not found: {original_path}")
        
        # Create temporary reference directory with just the original image
        reference_dir = original_path.parent
        
        # Calculate metrics for all blurred images
        results = self.calculate_metrics_for_directory(
            blurred_images_dir, 
            reference_dir,
            output_file
        )
        
        # Filter to only include comparisons with the original
        original_name = original_path.name
        filtered_results = {}
        
        for blur_name, metrics in results.items():
            if 'reference_image' in metrics:
                ref_name = Path(metrics['reference_image']).name
                if ref_name == original_name:
                    filtered_results[blur_name] = metrics
        
        return filtered_results
    
    def generate_summary_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate a summary report of metrics results.
        
        Args:
            results: Results dictionary from metrics calculation
            
        Returns:
            Formatted summary report string
        """
        if not results:
            return "No results to summarize."
        
        report = ["Image Quality Metrics Summary", "=" * 50, ""]
        
        # Count metrics types
        metric_types = {
            'full_reference': ['psnr', 'ssim', 'ms_ssim'],
            'no_reference': ['brisque', 'niqe'],
            'sharpness': ['acutance', 'mean_gradient', 'fft_high_freq', 'laplacian_variance']
        }
        
        # Calculate averages
        all_metrics = {}
        for image_name, metrics in results.items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Generate summary by category
        for category, metric_names in metric_types.items():
            category_metrics = {m: all_metrics.get(m, []) for m in metric_names if m in all_metrics}
            
            if category_metrics:
                report.append(f"{category.replace('_', ' ').title()} Metrics:")
                report.append("-" * 30)
                
                for metric, values in category_metrics.items():
                    if values:
                        avg_val = np.mean(values)
                        std_val = np.std(values)
                        report.append(f"{metric.upper():20s}: {avg_val:.4f} ± {std_val:.4f}")
                
                report.append("")
        
        # Overall statistics
        report.append(f"Total Images Processed: {len(results)}")
        report.append(f"Total Metrics Calculated: {len(all_metrics)}")
        
        return "\n".join(report)


def main():
    """Example usage of the metrics calculator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate comprehensive image quality metrics")
    parser.add_argument("test_image", help="Path to test image or directory")
    parser.add_argument("--reference", help="Path to reference image or directory")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--summary", action="store_true", help="Generate summary report")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Initialize calculator
    calculator = MetricsCalculator()
    
    test_path = Path(args.test_image)
    
    if test_path.is_file():
        # Single image
        metrics = calculator.calculate_metrics_for_image_pair(args.test_image, args.reference)
        
        print("\nImage Quality Metrics:")
        print("=" * 50)
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric:20s}: {value:.4f}")
            else:
                print(f"{metric:20s}: {value}")
        
        if args.output:
            calculator._save_results({test_path.name: metrics}, args.output)
            
    elif test_path.is_dir():
        # Directory processing
        reference_dir = args.reference if args.reference else None
        results = calculator.calculate_metrics_for_directory(
            args.test_image, reference_dir, args.output
        )
        
        print(f"\nProcessed {len(results)} images")
        
        if args.summary:
            summary = calculator.generate_summary_report(results)
            print("\n" + summary)
    
    else:
        print(f"Error: {args.test_image} is not a valid file or directory")


if __name__ == "__main__":
    main()
