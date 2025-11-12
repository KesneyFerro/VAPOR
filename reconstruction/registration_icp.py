"""
ICP-based Point Cloud Registration for VAPOR
Implements Iterative Closest Point registration requiring initial alignment.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import json


class ICPRegistration:
    """ICP registration with configurable parameters."""
    
    def __init__(self, 
                 max_correspondence_distance: float = 0.2,  # Increased from 0.05 to 0.2 (200mm)
                 relative_fitness: float = 1e-4,  # Less strict (was 1e-6)
                 relative_rmse: float = 1e-4,     # Less strict (was 1e-6)
                 max_iterations: int = 300,       # Increased from 100 to 300
                 init_transformation: Optional[np.ndarray] = None):
        """
        Initialize ICP registration.
        
        Args:
            max_correspondence_distance: Maximum distance for point correspondences (meters)
            relative_fitness: Convergence criteria for fitness
            relative_rmse: Convergence criteria for RMSE
            max_iterations: Maximum number of ICP iterations
            init_transformation: Initial 4x4 transformation matrix (required for ICP)
        """
        self.max_correspondence_distance = max_correspondence_distance
        self.relative_fitness = relative_fitness
        self.relative_rmse = relative_rmse
        self.max_iterations = max_iterations
        self.init_transformation = init_transformation if init_transformation is not None else np.eye(4)
        
        self.logger = logging.getLogger(__name__)
    
    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud, voxel_size: float = 0.005) -> o3d.geometry.PointCloud:
        """
        Preprocess point cloud with downsampling and normal estimation.
        
        Args:
            pcd: Input point cloud
            voxel_size: Voxel size for downsampling (meters)
            
        Returns:
            Preprocessed point cloud
        """
        # Downsample
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # Estimate normals
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        
        return pcd_down
    
    def register_point_to_point(self, 
                                  source: o3d.geometry.PointCloud,
                                  target: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """
        Point-to-point ICP registration.
        
        Args:
            source: Source point cloud (to be transformed)
            target: Target point cloud (reference)
            
        Returns:
            Registration result dictionary
        """
        self.logger.info("Running point-to-point ICP...")
        
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            self.max_correspondence_distance,
            self.init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.relative_fitness,
                relative_rmse=self.relative_rmse,
                max_iteration=self.max_iterations
            )
        )
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set,
            'num_correspondences': len(result.correspondence_set)
        }
    
    def register_point_to_plane(self,
                                  source: o3d.geometry.PointCloud,
                                  target: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """
        Point-to-plane ICP registration (more accurate, requires normals).
        
        Args:
            source: Source point cloud (to be transformed)
            target: Target point cloud (reference, must have normals)
            
        Returns:
            Registration result dictionary
        """
        self.logger.info("Running point-to-plane ICP...")
        
        # Ensure target has normals
        if not target.has_normals():
            self.logger.warning("Target lacks normals, estimating...")
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            self.max_correspondence_distance,
            self.init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.relative_fitness,
                relative_rmse=self.relative_rmse,
                max_iteration=self.max_iterations
            )
        )
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set,
            'num_correspondences': len(result.correspondence_set)
        }
    
    def multi_scale_icp(self,
                        source: o3d.geometry.PointCloud,
                        target: o3d.geometry.PointCloud,
                        voxel_sizes: list = [0.05, 0.025, 0.01],
                        max_iters: list = [50, 30, 20]) -> Dict[str, Any]:
        """
        Multi-scale ICP registration (coarse to fine).
        
        Args:
            source: Source point cloud
            target: Target point cloud
            voxel_sizes: List of voxel sizes from coarse to fine
            max_iters: List of max iterations for each scale
            
        Returns:
            Final registration result
        """
        self.logger.info(f"Running multi-scale ICP with {len(voxel_sizes)} scales...")
        
        current_transformation = self.init_transformation.copy()
        
        for i, (voxel_size, max_iter) in enumerate(zip(voxel_sizes, max_iters)):
            self.logger.info(f"Scale {i+1}/{len(voxel_sizes)}: voxel_size={voxel_size}, max_iter={max_iter}")
            
            # Downsample both clouds
            source_down = self.preprocess_point_cloud(source, voxel_size)
            target_down = self.preprocess_point_cloud(target, voxel_size)
            
            # Adjust correspondence distance based on voxel size
            correspondence_dist = voxel_size * 1.5
            
            # Run ICP at this scale
            result = o3d.pipelines.registration.registration_icp(
                source_down, target_down,
                correspondence_dist,
                current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=self.relative_fitness,
                    relative_rmse=self.relative_rmse,
                    max_iteration=max_iter
                )
            )
            
            current_transformation = result.transformation
            self.logger.info(f"  Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.6f}")
        
        return {
            'transformation': current_transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set,
            'num_correspondences': len(result.correspondence_set)
        }
    
    def register(self,
                 source_path: Path,
                 target_path: Path,
                 output_dir: Path,
                 method: str = "point_to_plane",
                 preprocess: bool = True,
                 voxel_size: float = 0.005) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
        """
        Main registration interface.
        
        Args:
            source_path: Path to source PLY file (reconstruction)
            target_path: Path to target PLY file (ground truth)
            output_dir: Directory to save outputs
            method: 'point_to_point', 'point_to_plane', or 'multi_scale'
            preprocess: Whether to preprocess clouds
            voxel_size: Voxel size for preprocessing
            
        Returns:
            Tuple of (transformed_source, registration_info)
        """
        self.logger.info(f"Loading point clouds...")
        self.logger.info(f"  Source: {source_path}")
        self.logger.info(f"  Target: {target_path}")
        
        # Load point clouds
        source = o3d.io.read_point_cloud(str(source_path))
        target = o3d.io.read_point_cloud(str(target_path))
        
        self.logger.info(f"Source points: {len(source.points)}")
        self.logger.info(f"Target points: {len(target.points)}")
        
        # Preprocess if requested
        if preprocess:
            self.logger.info(f"Preprocessing with voxel_size={voxel_size}...")
            source_processed = self.preprocess_point_cloud(source, voxel_size)
            target_processed = self.preprocess_point_cloud(target, voxel_size)
            self.logger.info(f"After preprocessing - Source: {len(source_processed.points)}, Target: {len(target_processed.points)}")
        else:
            source_processed = source
            target_processed = target
        
        # Run registration based on method
        if method == "point_to_point":
            result = self.register_point_to_point(source_processed, target_processed)
        elif method == "point_to_plane":
            result = self.register_point_to_plane(source_processed, target_processed)
        elif method == "multi_scale":
            result = self.multi_scale_icp(source, target)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Transform the original (non-downsampled) source
        source_transformed = source.transform(result['transformation'])
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save transformed point cloud
        transformed_path = output_dir / "registered_cloud.ply"
        o3d.io.write_point_cloud(str(transformed_path), source_transformed)
        self.logger.info(f"Saved transformed cloud to: {transformed_path}")
        
        # Save transformation matrix
        transform_path = output_dir / "transformation.txt"
        np.savetxt(transform_path, result['transformation'], fmt='%.8f')
        self.logger.info(f"Saved transformation matrix to: {transform_path}")
        
        # Save registration metadata
        metadata = {
            'method': 'ICP',
            'icp_variant': method,
            'source_file': str(source_path),
            'target_file': str(target_path),
            'source_points': len(source.points),
            'target_points': len(target.points),
            'preprocessed': preprocess,
            'voxel_size': voxel_size if preprocess else None,
            'parameters': {
                'max_correspondence_distance': self.max_correspondence_distance,
                'max_iterations': self.max_iterations,
                'relative_fitness': self.relative_fitness,
                'relative_rmse': self.relative_rmse
            },
            'result': {
                'fitness': float(result['fitness']),
                'inlier_rmse': float(result['inlier_rmse']),
                'num_correspondences': int(result['num_correspondences']),
                'transformation': result['transformation'].tolist()
            }
        }
        
        metadata_path = output_dir / "registration_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata to: {metadata_path}")
        
        self.logger.info(f"Registration complete - Fitness: {result['fitness']:.4f}, RMSE: {result['inlier_rmse']:.6f}")
        
        return source_transformed, metadata
