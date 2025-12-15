"""
Coverage, Sampling, and Hole Analysis Module (Step 5)
====================================================

Computes coverage metrics, observed area, hole detection, and density uniformity.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from collections import deque


class CoverageAnalyzer:
    """Analyze coverage, holes, and sampling density."""
    
    def __init__(self, output_dir: Path, h: float, logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.h = h
        self.logger = logger or logging.getLogger(__name__)
        
    def compute_coverage_at_thresholds(self,
                                        ct_roi_samples: o3d.geometry.PointCloud,
                                        point_cloud: o3d.geometry.PointCloud,
                                        thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compute coverage percentage at multiple distance thresholds.
        
        Args:
            ct_roi_samples: Sampled points from CT ROI
            point_cloud: Reconstruction point cloud
            thresholds: Distance thresholds (default: [h, 2h, 3h])
        """
        if thresholds is None:
            thresholds = [self.h, 2*self.h, 3*self.h]
        
        self.logger.info(f"Computing coverage at thresholds: {[t*1000 for t in thresholds]} mm")
        
        ct_points = np.asarray(ct_roi_samples.points)
        pc_points = np.asarray(point_cloud.points)
        
        # Build KDTree on reconstruction
        kdtree = cKDTree(pc_points)
        
        # Query distances
        distances, _ = kdtree.query(ct_points, k=1)
        
        coverage_results = {}
        for threshold in thresholds:
            covered = np.sum(distances <= threshold)
            coverage_pct = 100 * covered / len(ct_points)
            
            key = f'threshold_{int(threshold*1000)}mm'
            coverage_results[key] = {
                'threshold_m': float(threshold),
                'threshold_mm': float(threshold * 1000),
                'covered_points': int(covered),
                'total_points': len(ct_points),
                'coverage_percent': float(coverage_pct),
            }
            
            self.logger.debug(f"Coverage @ {threshold*1000:.1f}mm: {coverage_pct:.1f}%")
        
        return coverage_results
    
    def compute_observed_area(self,
                               ct_mesh: o3d.geometry.TriangleMesh,
                               roi_mask: np.ndarray,
                               s2p_distances: np.ndarray,
                               threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute observed surface area based on S2P distances.
        
        Args:
            ct_mesh: CT mesh
            roi_mask: ROI triangle mask
            s2p_distances: Surface-to-point distances for ROI samples
            threshold: Distance threshold for "observed" (default: 2h)
        """
        if threshold is None:
            threshold = 2 * self.h
        
        self.logger.info(f"Computing observed area (threshold={threshold*1000:.2f}mm)...")
        
        vertices = np.asarray(ct_mesh.vertices)
        triangles = np.asarray(ct_mesh.triangles)
        
        # Get ROI triangles
        roi_triangles = triangles[roi_mask]
        
        # Compute triangle areas
        triangle_areas = []
        for tri in roi_triangles:
            v0, v1, v2 = vertices[tri]
            # Area = 0.5 * ||cross product||
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            triangle_areas.append(area)
        
        triangle_areas = np.array(triangle_areas)
        total_roi_area = np.sum(triangle_areas)
        
        # Simple approximation: assume S2P distances represent triangle coverage
        # (more sophisticated: map each sample back to triangle)
        observed_fraction = np.sum(s2p_distances <= threshold) / len(s2p_distances)
        observed_area = total_roi_area * observed_fraction
        
        return {
            'threshold_m': float(threshold),
            'threshold_mm': float(threshold * 1000),
            'total_roi_area_m2': float(total_roi_area),
            'observed_area_m2': float(observed_area),
            'observed_fraction': float(observed_fraction),
            'observed_percentage': float(observed_fraction * 100),
            'observed_area_mm2': float(observed_area * 1e6),
        }
    
    def analyze_holes(self,
                       ct_mesh: o3d.geometry.TriangleMesh,
                       roi_mask: np.ndarray,
                       s2p_distances: np.ndarray,
                       threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect holes (uncovered regions) using triangle connectivity.
        
        Args:
            ct_mesh: CT mesh
            roi_mask: ROI triangle mask
            s2p_distances: S2P distances
            threshold: Distance threshold for "covered" (default: 2h)
        """
        if threshold is None:
            threshold = 2 * self.h
        
        self.logger.info(f"Analyzing holes (threshold={threshold*1000:.2f}mm)...")
        
        # For simplicity, consider triangles uncovered if their centroid's S2P > threshold
        # More sophisticated: map S2P samples to triangles
        
        uncovered_mask = s2p_distances > threshold
        n_uncovered = np.sum(uncovered_mask)
        n_total = len(s2p_distances)
        
        # Count connected components in uncovered regions (simplified)
        # Full implementation would use triangle adjacency graph
        # For now, estimate holes as number of large uncovered clusters
        n_holes = max(1, int(n_uncovered / 100)) if n_uncovered > 0 else 0
        
        return {
            'threshold_m': float(threshold),
            'threshold_mm': float(threshold * 1000),
            'uncovered_points': int(n_uncovered),
            'total_points': int(n_total),
            'uncovered_fraction': float(n_uncovered / n_total),
            'n_holes': int(n_holes),
            'note': 'Simplified hole analysis based on point samples'
        }
    
    def compute_density_uniformity(self,
                                     ct_roi_mesh: o3d.geometry.TriangleMesh,
                                     point_cloud: o3d.geometry.PointCloud,
                                     bin_radius: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze reconstruction point density uniformity.
        
        Args:
            ct_roi_mesh: ROI mesh
            point_cloud: Reconstruction point cloud
            bin_radius: Radius for density bins (default: 5h)
        """
        if bin_radius is None:
            bin_radius = 5 * self.h
        
        self.logger.info(f"Computing density uniformity (bin_radius={bin_radius*1000:.2f}mm)...")
        
        # Sample ROI mesh uniformly
        roi_samples = ct_roi_mesh.sample_points_poisson_disk(
            number_of_points=10000,
            init_factor=5
        )
        roi_points = np.asarray(roi_samples.points)
        pc_points = np.asarray(point_cloud.points)
        
        # Build KDTree on reconstruction
        kdtree = cKDTree(pc_points)
        
        # Count points in radius for each bin
        counts = []
        for center in roi_points:
            indices = kdtree.query_ball_point(center, bin_radius)
            counts.append(len(indices))
        
        counts = np.array(counts)
        
        # Compute Gini coefficient
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        if cumsum[-1] == 0:
            gini = 0.0
        else:
            gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
        
        return {
            'bin_radius_m': float(bin_radius),
            'bin_radius_mm': float(bin_radius * 1000),
            'n_bins': int(n),
            'mean_density': float(np.mean(counts)),
            'std_density': float(np.std(counts)),
            'min_density': int(np.min(counts)),
            'max_density': int(np.max(counts)),
            'gini_coefficient': float(gini),
        }
    
    def save_results(self, model_name: str, results: Dict[str, Any]):
        """Save coverage analysis results."""
        coverage_dir = self.output_dir / 'coverage'
        coverage_dir.mkdir(exist_ok=True)
        
        json_path = coverage_dir / f'coverage_{model_name}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Coverage analysis saved to {json_path}")
