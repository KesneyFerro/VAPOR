"""
Registration Sanity Check Module (Step 7)
=========================================

Validates registration quality by running ICP and analyzing residual transforms.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np
import open3d as o3d


class RegistrationSanityChecker:
    """Validate registration quality with ICP sanity checks."""
    
    def __init__(self, output_dir: Path, h: float, logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.h = h
        self.logger = logger or logging.getLogger(__name__)
    
    def run_icp_sanity_check(self,
                              point_cloud: o3d.geometry.PointCloud,
                              ct_roi_mesh: o3d.geometry.TriangleMesh,
                              threshold: Optional[float] = None,
                              max_iterations: int = 50) -> Dict[str, Any]:
        """
        Run point-to-plane ICP from identity to check registration quality.
        
        Args:
            point_cloud: Registered reconstruction
            ct_roi_mesh: CT ROI mesh
            threshold: Distance threshold (default: 2h)
            max_iterations: Max ICP iterations
        """
        if threshold is None:
            threshold = 2 * self.h
        
        self.logger.info(f"Running ICP sanity check (threshold={threshold*1000:.2f}mm)...")
        
        # Ensure normals
        if not ct_roi_mesh.has_vertex_normals():
            ct_roi_mesh.compute_vertex_normals()
        
        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            source=point_cloud,
            target=ct_roi_mesh,
            max_correspondence_distance=threshold,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            )
        )
        
        # Extract transform
        T = result.transformation
        R = T[:3, :3]
        t = T[:3, 3]
        
        # Rotation angle
        rotation_angle_rad = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        rotation_angle_deg = np.degrees(rotation_angle_rad)
        
        # Translation magnitude
        translation_magnitude = np.linalg.norm(t)
        
        # Inlier ratio
        inlier_rmse = result.inlier_rmse
        fitness = result.fitness
        
        stats = {
            'rmse_m': float(inlier_rmse),
            'rmse_mm': float(inlier_rmse * 1000),
            'fitness': float(fitness),
            'rotation_deg': float(rotation_angle_deg),
            'translation_m': float(translation_magnitude),
            'translation_mm': float(translation_magnitude * 1000),
            'threshold_m': threshold,
        }
        
        self.logger.info(f"ICP sanity: RMSE={stats['rmse_mm']:.3f}mm, "
                        f"fitness={fitness:.3f}, rot={rotation_angle_deg:.2f}°")
        
        return stats
    
    def save_results(self, model_name: str, stats: Dict[str, Any]):
        """Save registration sanity check results."""
        tables_dir = self.output_dir / 'tables'
        tables_dir.mkdir(exist_ok=True)
        
        json_path = tables_dir / f'registration_qc_{model_name}.json'
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Registration QC saved to {json_path}")
