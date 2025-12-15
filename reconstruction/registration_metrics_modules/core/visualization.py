"""
Visualization Generation Module (Step 10)
========================================

Generate colored point clouds, distance heatmaps, and plots.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class VisualizationGenerator:
    """Generate all visualizations for QA."""
    
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
    def generate_distance_heatmap(self,
                                   point_cloud: o3d.geometry.PointCloud,
                                   distances: np.ndarray,
                                   output_name: str,
                                   max_distance: float = 0.01):
        """
        Color point cloud by distance error.
        
        Args:
            point_cloud: Point cloud to color
            distances: Distance values
            output_name: Output filename (without extension)
            max_distance: Max distance for color scale
        """
        self.logger.info(f"Generating distance heatmap: {output_name}...")
        
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Normalize distances
        distances_norm = np.clip(distances / max_distance, 0, 1)
        
        # Create colormap (blue=good, red=bad)
        colors = np.zeros((len(distances_norm), 3))
        for i, d in enumerate(distances_norm):
            if d < 0.33:
                colors[i] = [0, d * 3, 1 - d * 3]
            elif d < 0.67:
                colors[i] = [(d - 0.33) * 3, 1, 0]
            else:
                colors[i] = [1, 1 - (d - 0.67) * 3, 0]
        
        # Apply colors
        colored_pcd = point_cloud.__copy__()
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save
        ply_path = plots_dir / f'{output_name}_heatmap.ply'
        o3d.io.write_point_cloud(str(ply_path), colored_pcd)
        
        self.logger.info(f"Heatmap saved to {ply_path}")
        
    def generate_histogram(self,
                            distances: np.ndarray,
                            output_name: str,
                            title: str = "Distance Distribution",
                            xlabel: str = "Distance (mm)"):
        """Generate distance histogram."""
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        distances_mm = distances * 1000
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(distances_mm, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        png_path = plots_dir / f'{output_name}_histogram.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Histogram saved to {png_path}")
