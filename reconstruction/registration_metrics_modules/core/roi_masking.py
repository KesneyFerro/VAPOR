"""
ROI (Region of Interest) Masking Module (Step 2)
===============================================

Computes regions of CT mesh that were plausibly observed by the reconstruction.

Two strategies:
1. Camera frusta-based: geometric visibility from known camera poses
2. Proximity-based: triangles within 5h of reconstruction point cloud

Production-grade using Open3D for geometry operations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class ROIMaskGenerator:
    """Generate Region of Interest masks for partial reconstructions."""
    
    def __init__(self, output_dir: Path, h: float, logger: Optional[logging.Logger] = None):
        """
        Initialize ROI mask generator.
        
        Args:
            output_dir: Output directory for ROI masks and visualizations
            h: Characteristic length (meters)
            logger: Optional logger
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.h = h
        self.logger = logger or logging.getLogger(__name__)
        
        self.roi_mask: Optional[np.ndarray] = None
        self.roi_info: Dict[str, Any] = {}
        
    def compute_triangle_centroids(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        """
        Compute centroids of all triangles.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Array of triangle centroids (N, 3)
        """
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        if len(triangles) == 0:
            return np.array([]).reshape(0, 3)
        
        # Get vertices for each triangle
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        # Centroid = mean of vertices
        centroids = (v0 + v1 + v2) / 3.0
        
        return centroids
    
    def generate_proximity_based_roi(self,
                                      ct_mesh: o3d.geometry.TriangleMesh,
                                      point_cloud: o3d.geometry.PointCloud,
                                      distance_threshold: Optional[float] = None) -> np.ndarray:
        """
        Generate ROI based on proximity to point cloud.
        
        Keeps CT triangles whose centroid is within threshold of any reconstruction point.
        
        Args:
            ct_mesh: CT reference mesh
            point_cloud: Reconstruction point cloud
            distance_threshold: Distance threshold (default: 5h)
            
        Returns:
            Boolean mask array for triangle indices
        """
        if distance_threshold is None:
            distance_threshold = 5 * self.h
        
        self.logger.info(f"Generating proximity-based ROI (threshold={distance_threshold:.4f}m = {distance_threshold*1000:.2f}mm)")
        
        # Compute triangle centroids
        centroids = self.compute_triangle_centroids(ct_mesh)
        n_triangles = len(centroids)
        
        if n_triangles == 0:
            self.logger.warning("CT mesh has no triangles")
            return np.array([], dtype=bool)
        
        # Build KDTree on point cloud
        pc_points = np.asarray(point_cloud.points)
        kdtree = cKDTree(pc_points)
        
        # Query nearest neighbor for each centroid
        distances, _ = kdtree.query(centroids, k=1)
        
        # Create mask
        roi_mask = distances <= distance_threshold
        
        n_roi = np.sum(roi_mask)
        roi_percentage = 100 * n_roi / n_triangles
        
        self.logger.info(f"ROI mask: {n_roi}/{n_triangles} triangles ({roi_percentage:.1f}%)")
        
        self.roi_mask = roi_mask
        self.roi_info = {
            'method': 'proximity',
            'distance_threshold_m': distance_threshold,
            'distance_threshold_mm': distance_threshold * 1000,
            'n_total_triangles': n_triangles,
            'n_roi_triangles': int(n_roi),
            'roi_percentage': roi_percentage,
        }
        
        return roi_mask
    
    def extract_roi_mesh(self,
                          ct_mesh: o3d.geometry.TriangleMesh,
                          roi_mask: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Extract mesh subset using ROI mask.
        
        Args:
            ct_mesh: Full CT mesh
            roi_mask: Boolean mask for triangles
            
        Returns:
            ROI mesh subset
        """
        triangles = np.asarray(ct_mesh.triangles)
        
        # Select ROI triangles
        roi_triangles = triangles[roi_mask]
        
        # Create new mesh
        roi_mesh = o3d.geometry.TriangleMesh()
        roi_mesh.vertices = ct_mesh.vertices
        roi_mesh.triangles = o3d.utility.Vector3iVector(roi_triangles)
        
        # Copy normals if available
        if ct_mesh.has_vertex_normals():
            roi_mesh.vertex_normals = ct_mesh.vertex_normals
        
        # Remove unreferenced vertices
        roi_mesh = roi_mesh.remove_unreferenced_vertices()
        
        self.logger.debug(f"ROI mesh extracted: {len(roi_mesh.vertices)} vertices, {len(roi_mesh.triangles)} triangles")
        
        return roi_mesh
    
    def sample_roi_poisson_disk(self,
                                 roi_mesh: o3d.geometry.TriangleMesh,
                                 radius: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        Poisson-disk sample the ROI mesh.
        
        Args:
            roi_mesh: ROI mesh subset
            radius: Sampling radius (default: h)
            
        Returns:
            Sampled point cloud
        """
        if radius is None:
            radius = self.h
        
        self.logger.info(f"Poisson-disk sampling ROI mesh (radius={radius:.6f}m)")
        
        # Sample mesh
        sampled_pcd = roi_mesh.sample_points_poisson_disk(
            number_of_points=100000,  # Upper limit
            init_factor=5
        )
        
        n_samples = len(sampled_pcd.points)
        self.logger.info(f"Sampled {n_samples} points from ROI")
        
        return sampled_pcd
    
    def save_roi_mask(self, model_name: str) -> Path:
        """
        Save ROI mask to file.
        
        Args:
            model_name: Model name for filename
            
        Returns:
            Path to saved mask
        """
        if self.roi_mask is None:
            raise ValueError("No ROI mask to save")
        
        roi_dir = self.output_dir / 'roi'
        roi_dir.mkdir(exist_ok=True)
        
        mask_path = roi_dir / f'roi_{model_name}_mask.npy'
        np.save(mask_path, self.roi_mask)
        
        # Save info
        info_path = roi_dir / f'roi_{model_name}_info.json'
        with open(info_path, 'w') as f:
            json.dump(self.roi_info, f, indent=2)
        
        self.logger.info(f"ROI mask saved: {mask_path}")
        
        return mask_path
    
    def save_roi_visualization(self,
                                ct_mesh: o3d.geometry.TriangleMesh,
                                roi_mask: np.ndarray,
                                model_name: str) -> Path:
        """
        Save visualization of ROI (colored mesh).
        
        Args:
            ct_mesh: Full CT mesh
            roi_mask: ROI mask
            model_name: Model name
            
        Returns:
            Path to saved PLY
        """
        roi_dir = self.output_dir / 'roi'
        roi_dir.mkdir(exist_ok=True)
        
        # Create colored mesh (ROI = green, non-ROI = gray)
        vertices = np.asarray(ct_mesh.vertices)
        triangles = np.asarray(ct_mesh.triangles)
        
        # Assign colors to vertices based on triangle membership
        vertex_colors = np.ones((len(vertices), 3)) * 0.5  # Gray default
        
        for tri_idx, tri in enumerate(triangles):
            if roi_mask[tri_idx]:
                # ROI: green
                vertex_colors[tri] = [0.0, 0.8, 0.0]
        
        colored_mesh = o3d.geometry.TriangleMesh()
        colored_mesh.vertices = ct_mesh.vertices
        colored_mesh.triangles = ct_mesh.triangles
        colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        # Save
        viz_path = roi_dir / f'roi_{model_name}_preview.ply'
        o3d.io.write_triangle_mesh(str(viz_path), colored_mesh)
        
        self.logger.info(f"ROI visualization saved: {viz_path}")
        
        return viz_path


def generate_roi_mask(
    ct_mesh: o3d.geometry.TriangleMesh,
    point_cloud: o3d.geometry.PointCloud,
    h: float,
    output_dir: Path,
    model_name: str,
    distance_threshold: Optional[float] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """
    Convenience function to generate ROI mask and samples.
    
    Args:
        ct_mesh: CT reference mesh
        point_cloud: Reconstruction point cloud
        h: Characteristic length
        output_dir: Output directory
        model_name: Model name
        distance_threshold: Distance threshold (default: 5h)
        logger: Optional logger
        
    Returns:
        Tuple of (roi_mask, roi_mesh, roi_samples)
    """
    generator = ROIMaskGenerator(output_dir, h, logger)
    
    # Generate mask
    roi_mask = generator.generate_proximity_based_roi(ct_mesh, point_cloud, distance_threshold)
    
    # Extract ROI mesh
    roi_mesh = generator.extract_roi_mesh(ct_mesh, roi_mask)
    
    # Sample ROI
    roi_samples = generator.sample_roi_poisson_disk(roi_mesh)
    
    # Save
    generator.save_roi_mask(model_name)
    generator.save_roi_visualization(ct_mesh, roi_mask, model_name)
    
    return roi_mask, roi_mesh, roi_samples
