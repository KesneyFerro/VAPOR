"""
Noise, Smoothness, and Feature Preservation Module (Step 6)
==========================================================

Computes local roughness, eigenvalue shape descriptors, and curvature retention.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import numpy as np
import open3d as o3d


class NoiseSmoothnessAnalyzer:
    """Analyze noise, smoothness, and feature preservation."""
    
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
    def compute_local_roughness(self,
                                 point_cloud: o3d.geometry.PointCloud,
                                 knn: int = 30) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute local roughness via PCA residuals.
        
        Args:
            point_cloud: Input point cloud
            knn: Number of nearest neighbors
        
        Returns:
            Tuple of (roughness_array, statistics)
        """
        self.logger.info(f"Computing local roughness (knn={knn})...")
        
        points = np.asarray(point_cloud.points)
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        
        roughness_values = np.zeros(len(points))
        
        for i, point in enumerate(points):
            # Find neighbors
            [k, idx, _] = kdtree.search_knn_vector_3d(point, knn)
            neighbors = points[idx]
            
            # Compute covariance
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov = centered.T @ centered / len(neighbors)
            
            # Eigenvalues
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]  # Descending
            
            # Roughness = RMS distance to best-fit plane
            # Plane normal = eigenvector of smallest eigenvalue
            # Roughness ≈ sqrt(smallest eigenvalue)
            roughness_values[i] = np.sqrt(eigvals[2]) if eigvals[2] > 0 else 0
        
        stats = {
            'median': float(np.median(roughness_values)),
            'mean': float(np.mean(roughness_values)),
            'std': float(np.std(roughness_values)),
            'p95': float(np.percentile(roughness_values, 95)),
            'units': 'meters',
        }
        
        self.logger.info(f"Roughness: median={stats['median']*1000:.4f}mm, p95={stats['p95']*1000:.4f}mm")
        
        return roughness_values, stats
    
    def compute_shape_descriptors(self,
                                    point_cloud: o3d.geometry.PointCloud,
                                    knn: int = 30) -> Dict[str, Any]:
        """
        Compute eigenvalue-based shape descriptors (linearity, planarity, scattering).
        
        Args:
            point_cloud: Input point cloud
            knn: Number of nearest neighbors
        """
        self.logger.info(f"Computing shape descriptors (knn={knn})...")
        
        points = np.asarray(point_cloud.points)
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        
        linearity_vals = []
        planarity_vals = []
        scattering_vals = []
        
        for i, point in enumerate(points):
            [k, idx, _] = kdtree.search_knn_vector_3d(point, knn)
            neighbors = points[idx]
            
            # Covariance
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov = centered.T @ centered / len(neighbors)
            
            # Eigenvalues (descending)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            l1, l2, l3 = eigvals
            
            if l1 > 0:
                linearity = (l1 - l2) / l1
                planarity = (l2 - l3) / l1
                scattering = l3 / l1
            else:
                linearity = planarity = scattering = 0
            
            linearity_vals.append(linearity)
            planarity_vals.append(planarity)
            scattering_vals.append(scattering)
        
        linearity_vals = np.array(linearity_vals)
        planarity_vals = np.array(planarity_vals)
        scattering_vals = np.array(scattering_vals)
        
        return {
            'linearity': {
                'median': float(np.median(linearity_vals)),
                'mean': float(np.mean(linearity_vals)),
                'p95': float(np.percentile(linearity_vals, 95)),
            },
            'planarity': {
                'median': float(np.median(planarity_vals)),
                'mean': float(np.mean(planarity_vals)),
                'p95': float(np.percentile(planarity_vals, 95)),
            },
            'scattering': {
                'median': float(np.median(scattering_vals)),
                'mean': float(np.mean(scattering_vals)),
                'p95': float(np.percentile(scattering_vals, 95)),
            },
        }
    
    def save_results(self, model_name: str, roughness: np.ndarray, 
                     roughness_stats: Dict, shape_descriptors: Dict):
        """Save noise/smoothness analysis results."""
        smoothness_dir = self.output_dir / 'smoothness'
        smoothness_dir.mkdir(exist_ok=True)
        
        # Save roughness array
        np.save(smoothness_dir / f'roughness_{model_name}.npy', roughness)
        
        # Save stats
        results = {
            'roughness_statistics': roughness_stats,
            'shape_descriptors': shape_descriptors,
        }
        
        json_path = smoothness_dir / f'smoothness_{model_name}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Smoothness analysis saved to {smoothness_dir}")
