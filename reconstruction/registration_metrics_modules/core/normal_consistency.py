"""
Normal Consistency Module (Step 4)
==================================

Computes angle differences between reconstruction point cloud normals
and CT mesh normals at corresponding surface points.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np
import open3d as o3d


class NormalConsistencyCalculator:
    """Calculate normal angle consistency between reconstruction and CT mesh."""
    
    def __init__(self, output_dir: Path, h: float, logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.h = h
        self.logger = logger or logging.getLogger(__name__)
        self.angle_errors: Optional[np.ndarray] = None
    
    def compute_normal_consistency(self,
                                    point_cloud: o3d.geometry.PointCloud,
                                    ct_mesh: o3d.geometry.TriangleMesh,
                                    raycasting_scene: o3d.t.geometry.RaycastingScene,
                                    max_distance: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute normal angle errors.
        
        Args:
            point_cloud: Reconstruction point cloud
            ct_mesh: CT mesh with normals
            raycasting_scene: Raycasting scene for closest point queries
            max_distance: Max distance for valid pairs (default: 2h)
        """
        if max_distance is None:
            max_distance = 2 * self.h
        
        self.logger.info(f"Computing normal consistency (max_distance={max_distance*1000:.2f}mm)...")
        
        # Estimate normals on point cloud
        if not point_cloud.has_normals():
            point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
            )
            point_cloud.orient_normals_consistent_tangent_plane(30)
        
        pc_points = np.asarray(point_cloud.points, dtype=np.float32)
        pc_normals = np.asarray(point_cloud.normals)
        
        # Find closest points on CT mesh
        query_points = o3d.core.Tensor(pc_points, dtype=o3d.core.Dtype.Float32)
        result = raycasting_scene.compute_closest_points(query_points)
        
        closest_points = result['points'].numpy()
        closest_tri_ids = result['primitive_ids'].numpy()
        distances = np.linalg.norm(pc_points - closest_points, axis=1)
        
        # Get CT mesh normals at closest triangles
        ct_normals_array = np.asarray(ct_mesh.triangle_normals)
        ct_normals_at_points = ct_normals_array[closest_tri_ids]
        
        # Filter by distance
        valid_mask = distances <= max_distance
        
        pc_normals_valid = pc_normals[valid_mask]
        ct_normals_valid = ct_normals_at_points[valid_mask]
        
        # Compute angle errors (in degrees)
        dot_products = np.abs(np.sum(pc_normals_valid * ct_normals_valid, axis=1))
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angle_errors_rad = np.arccos(dot_products)
        angle_errors_deg = np.degrees(angle_errors_rad)
        
        self.angle_errors = angle_errors_deg
        
        stats = {
            'n_valid_pairs': int(np.sum(valid_mask)),
            'n_total_points': len(pc_points),
            'median_angle_deg': float(np.median(angle_errors_deg)),
            'mean_angle_deg': float(np.mean(angle_errors_deg)),
            'p95_angle_deg': float(np.percentile(angle_errors_deg, 95)),
            'max_angle_deg': float(np.max(angle_errors_deg)),
            'max_distance_threshold_m': max_distance,
        }
        
        self.logger.info(f"Normal consistency: median={stats['median_angle_deg']:.2f}°, p95={stats['p95_angle_deg']:.2f}°")
        
        return stats
    
    def save_results(self, model_name: str):
        """Save normal consistency results."""
        normals_dir = self.output_dir / 'normals'
        normals_dir.mkdir(exist_ok=True)
        
        # Save angle errors
        np.save(normals_dir / f'normal_error_{model_name}.npy', self.angle_errors)
        
        # Save stats
        stats_path = normals_dir / f'normal_stats_{model_name}.json'
        with open(stats_path, 'w') as f:
            json.dump({
                'median_angle_deg': float(np.median(self.angle_errors)),
                'mean_angle_deg': float(np.mean(self.angle_errors)),
                'p95_angle_deg': float(np.percentile(self.angle_errors, 95)),
            }, f, indent=2)
        
        self.logger.info(f"Normal consistency saved to {normals_dir}")
