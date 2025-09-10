"""
VAPOR Quality Metrics Module
Calculates various image quality metrics for blur analysis.

Metrics included:
- Laplacian Score: Variance of Laplacian (measures edge content/sharpness)
- FFT Score: High-frequency content analysis using Fast Fourier Transform
- Acutance Score: Perceived sharpness based on edge contrast
- mGradient Score: Mean gradient magnitude (edge strength)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
from scipy import ndimage
from scipy.fft import fft2, fftshift


def sharpness_metrics(image):
    """
    Compute different sharpness/blur metrics for a grayscale image.
    
    Parameters:
        image (numpy.ndarray): Input image (grayscale preferred).
    
    Returns:
        dict: Dictionary with Laplacian variance, FFT energy, acutance, mean gradient.
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Laplacian Variance
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    lap_var = laplacian.var()

    # 2. FFT High-Frequency Energy
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = image.shape
    center = (h // 2, w // 2)
    radius = min(h, w) // 8  # low frequency cutoff radius
    
    mask = np.ones_like(magnitude)
    cv2.circle(mask, center, radius, 0, -1)  # zero out low freq
    fft_energy = (magnitude * mask).mean()

    # 3. Acutance
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    acutance = np.mean(grad_mag)

    # 4. Mean Gradient
    mean_grad = (np.abs(sobelx).mean() + np.abs(sobely).mean()) / 2

    return {
        "laplacian_variance": float(lap_var),
        "fft_energy": float(fft_energy),
        "acutance": float(acutance),
        "mean_gradient": float(mean_grad)
    }


class QualityMetricsCalculator:
    """Calculate various image quality metrics for blur analysis."""
    
    def __init__(self):
        """Initialize the quality metrics calculator."""
        pass
    
    def calculate_all_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate all quality metrics for an image using the unified sharpness_metrics function.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Dictionary with all metric scores
        """
        metrics = sharpness_metrics(image)
        return {
            'laplacian_score': metrics['laplacian_variance'],
            'fft_score': metrics['fft_energy'], 
            'acutance_score': metrics['acutance'],
            'mgradient_score': metrics['mean_gradient']
        }


class QualityMetricsLogger:
    """Log quality metrics to CSV files."""
    
    def __init__(self, base_output_path: Path):
        """
        Initialize the metrics logger.
        
        Args:
            base_output_path: Base path for output CSV files
        """
        self.base_output_path = Path(base_output_path)
        self.calculator = QualityMetricsCalculator()
        self.metrics_data = []  # Store all metrics for summary
    
    def process_image_file(self, image_path: Path, method: str, frame_number: int, 
                          relative_path: str) -> Dict[str, float]:
        """
        Process a single image file and calculate metrics.
        
        Args:
            image_path: Path to the image file
            method: Method used (e.g., 'original', 'gaussian_low', etc.)
            frame_number: Frame number in the video
            relative_path: Relative path to the image file
            
        Returns:
            Dictionary with all metrics
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Calculate all metrics
        metrics = self.calculator.calculate_all_metrics(image)
        
        # Add metadata
        metrics.update({
            'method': method,
            'frame_number': frame_number,
            'relative_path': relative_path
        })
        
        # Store for summary
        self.metrics_data.append(metrics.copy())
        
        return metrics
    
    def process_frames_directory(self, frames_dir: Path, method: str) -> Optional[Path]:
        """
        Process all frames in a directory and create CSV.
        
        Args:
            frames_dir: Directory containing frame images
            method: Method used (e.g., 'original', 'gaussian_low', etc.)
            
        Returns:
            Path to the created CSV file, or None if no frames processed
        """
        if not frames_dir.exists():
            print(f"    [SKIP] Directory not found: {frames_dir}")
            return None
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(frames_dir.glob(f"*{ext}"))
        
        if not image_files:
            print(f"    [SKIP] No image files found in: {frames_dir}")
            return None
        
        # Sort by frame number (extracted from filename)
        image_files.sort(key=lambda x: self._extract_frame_number(x.name))
        
        print(f"    [METRICS] Processing {len(image_files)} frames for {method}...")
        
        # Process each image
        metrics_list = []
        for image_file in image_files:
            try:
                frame_number = self._extract_frame_number(image_file.name)
                relative_path = str(image_file.relative_to(self.base_output_path))
                
                metrics = self.process_image_file(
                    image_file, method, frame_number, relative_path
                )
                metrics_list.append(metrics)
                
            except Exception as e:
                print(f"      [WARNING] Failed to process {image_file.name}: {e}")
                continue
        
        if not metrics_list:
            print(f"    [SKIP] No metrics calculated for {method}")
            return None
        
        # Create DataFrame and save CSV
        df = pd.DataFrame(metrics_list)
        
        # Reorder columns
        column_order = ['method', 'frame_number', 'laplacian_score', 'fft_score', 
                       'acutance_score', 'mgradient_score', 'relative_path']
        df = df[column_order]
        
        # Create output directory and save CSV
        csv_path = frames_dir / 'quality_metrics.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        print(f"    [OK] Saved metrics CSV: {csv_path}")
        return csv_path
    
    def _extract_frame_number(self, filename: str) -> int:
        """
        Extract frame number from filename.
        Expected format: {video_name}_{frame_number:06d}.png
        
        Args:
            filename: Name of the frame file
            
        Returns:
            Frame number as integer
        """
        try:
            # Remove extension and split by underscore
            name_parts = Path(filename).stem.split('_')
            # Last part should be the frame number
            return int(name_parts[-1])
        except (ValueError, IndexError):
            # Fallback: try to find any number in the filename
            import re
            numbers = re.findall(r'\d+', filename)
            return int(numbers[-1]) if numbers else 0
    
    def create_summary_csv(self, video_name: str, original_frames_path: Path) -> Optional[Path]:
        """
        Create summary CSV with mean and std deviation for all methods.
        
        Args:
            video_name: Name of the video
            original_frames_path: Path to original frames directory for output
            
        Returns:
            Path to the created summary CSV file
        """
        if not self.metrics_data:
            print("    [SKIP] No metrics data available for summary")
            return None
        
        print(f"    [SUMMARY] Creating summary statistics for {len(self.metrics_data)} measurements...")
        
        # Create DataFrame from all metrics
        df = pd.DataFrame(self.metrics_data)
        
        # Calculate summary statistics by method
        summary_stats = []
        metric_columns = ['laplacian_score', 'fft_score', 'acutance_score', 'mgradient_score']
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            summary_row = {'method': method}
            
            for metric in metric_columns:
                if metric in method_data.columns:
                    values = method_data[metric]
                    summary_row[f'{metric}_mean'] = values.mean()
                    summary_row[f'{metric}_std'] = values.std()
                    summary_row['frame_count'] = len(values)
            
            summary_stats.append(summary_row)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_stats)
        
        # Reorder columns
        column_order = ['method', 'frame_count']
        for metric in metric_columns:
            column_order.extend([f'{metric}_mean', f'{metric}_std'])
        
        summary_df = summary_df[column_order]
        
        # Save summary CSV
        summary_csv_path = original_frames_path / f'{video_name}_quality_metrics_summary.csv'
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False)
        
        print(f"    [OK] Saved summary CSV: {summary_csv_path}")
        print(f"    [INFO] Summary includes {len(summary_stats)} methods with statistics")
        
        return summary_csv_path
    
    def clear_metrics_data(self):
        """Clear stored metrics data."""
        self.metrics_data.clear()
