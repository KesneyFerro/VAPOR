"""
Point Cloud Comparison Metrics Calculator for VAPOR
Computes distance metrics between registered and ground truth point clouds.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from scipy.spatial import cKDTree


class PointCloudMetrics:
    """Calculate comparison metrics between two point clouds."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_chamfer_distance(self,
                                   source: o3d.geometry.PointCloud,
                                   target: o3d.geometry.PointCloud) -> Dict[str, float]:
        """
        Compute bidirectional Chamfer distance.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            
        Returns:
            Dictionary with chamfer distance metrics
        """
        self.logger.info("Computing Chamfer distance...")
        
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        # Build KD-trees for fast nearest neighbor search
        source_tree = cKDTree(source_points)
        target_tree = cKDTree(target_points)
        
        # Source to target distances
        distances_s2t, _ = target_tree.query(source_points, k=1)
        
        # Target to source distances
        distances_t2s, _ = source_tree.query(target_points, k=1)
        
        # Chamfer distance (mean of both directions)
        chamfer_distance = np.mean(distances_s2t) + np.mean(distances_t2s)
        
        return {
            'chamfer_distance': float(chamfer_distance),
            'chamfer_distance_s2t': float(np.mean(distances_s2t)),
            'chamfer_distance_t2s': float(np.mean(distances_t2s)),
            'chamfer_distance_units': 'meters'
        }
    
    def compute_hausdorff_distance(self,
                                     source: o3d.geometry.PointCloud,
                                     target: o3d.geometry.PointCloud) -> Dict[str, float]:
        """
        Compute Hausdorff distance (maximum distance).
        
        Args:
            source: Source point cloud
            target: Target point cloud
            
        Returns:
            Dictionary with Hausdorff distance metrics
        """
        self.logger.info("Computing Hausdorff distance...")
        
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        # Build KD-tree
        target_tree = cKDTree(target_points)
        source_tree = cKDTree(source_points)
        
        # Distances
        distances_s2t, _ = target_tree.query(source_points, k=1)
        distances_t2s, _ = source_tree.query(target_points, k=1)
        
        # Hausdorff distance (max of both directions)
        hausdorff_s2t = np.max(distances_s2t)
        hausdorff_t2s = np.max(distances_t2s)
        hausdorff_distance = max(hausdorff_s2t, hausdorff_t2s)
        
        return {
            'hausdorff_distance': float(hausdorff_distance),
            'hausdorff_distance_s2t': float(hausdorff_s2t),
            'hausdorff_distance_t2s': float(hausdorff_t2s),
            'hausdorff_distance_units': 'meters'
        }
    
    def compute_rms_error(self,
                          source: o3d.geometry.PointCloud,
                          target: o3d.geometry.PointCloud) -> Dict[str, float]:
        """
        Compute RMS (Root Mean Square) error.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            
        Returns:
            Dictionary with RMS error metrics
        """
        self.logger.info("Computing RMS error...")
        
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        # Build KD-tree
        target_tree = cKDTree(target_points)
        
        # Find nearest neighbors
        distances, _ = target_tree.query(source_points, k=1)
        
        # RMS error
        rms_error = np.sqrt(np.mean(distances ** 2))
        
        return {
            'rms_error': float(rms_error),
            'rms_error_units': 'meters'
        }
    
    def compute_mae(self,
                    source: o3d.geometry.PointCloud,
                    target: o3d.geometry.PointCloud) -> Dict[str, float]:
        """
        Compute Mean Absolute Error.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            
        Returns:
            Dictionary with MAE metrics
        """
        self.logger.info("Computing Mean Absolute Error...")
        
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        # Build KD-tree
        target_tree = cKDTree(target_points)
        
        # Find nearest neighbors
        distances, _ = target_tree.query(source_points, k=1)
        
        # MAE
        mae = np.mean(distances)
        median_ae = np.median(distances)
        
        return {
            'mean_absolute_error': float(mae),
            'median_absolute_error': float(median_ae),
            'mae_units': 'meters'
        }
    
    def compute_completeness(self,
                              source: o3d.geometry.PointCloud,
                              target: o3d.geometry.PointCloud,
                              threshold: float = 0.01) -> Dict[str, float]:
        """
        Compute completeness and accuracy percentages.
        
        Args:
            source: Source point cloud (reconstruction)
            target: Target point cloud (ground truth)
            threshold: Distance threshold for considering a point "aligned" (meters)
            
        Returns:
            Dictionary with completeness metrics
        """
        self.logger.info(f"Computing completeness (threshold={threshold}m)...")
        
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        # Build KD-trees
        source_tree = cKDTree(source_points)
        target_tree = cKDTree(target_points)
        
        # Completeness: % of source points within threshold of target
        distances_s2t, _ = target_tree.query(source_points, k=1)
        completeness = np.sum(distances_s2t <= threshold) / len(source_points)
        
        # Accuracy: % of target points within threshold of source
        distances_t2s, _ = source_tree.query(target_points, k=1)
        accuracy = np.sum(distances_t2s <= threshold) / len(target_points)
        
        # F1 score
        if completeness + accuracy > 0:
            f1_score = 2 * (completeness * accuracy) / (completeness + accuracy)
        else:
            f1_score = 0.0
        
        return {
            'completeness_percent': float(completeness * 100),
            'accuracy_percent': float(accuracy * 100),
            'f1_score': float(f1_score),
            'threshold_meters': threshold,
            'description': f'Percentage of points within {threshold}m of corresponding surface'
        }
    
    def compute_distance_statistics(self,
                                      source: o3d.geometry.PointCloud,
                                      target: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """
        Compute detailed distance statistics.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            
        Returns:
            Dictionary with distance statistics
        """
        self.logger.info("Computing distance statistics...")
        
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        # Build KD-tree
        target_tree = cKDTree(target_points)
        
        # Find distances
        distances, _ = target_tree.query(source_points, k=1)
        
        # Convert to millimeters for easier interpretation
        distances_mm = distances * 1000
        
        # Compute percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(distances_mm, percentiles)
        
        stats = {
            'mean_distance_mm': float(np.mean(distances_mm)),
            'std_distance_mm': float(np.std(distances_mm)),
            'min_distance_mm': float(np.min(distances_mm)),
            'max_distance_mm': float(np.max(distances_mm)),
            'median_distance_mm': float(np.median(distances_mm)),
            'percentiles': {
                f'p{p}': float(v) for p, v in zip(percentiles, percentile_values)
            },
            'distance_ranges': {
                'excellent_lt_1mm': int(np.sum(distances_mm < 1)),
                'good_1_to_3mm': int(np.sum((distances_mm >= 1) & (distances_mm < 3))),
                'fair_3_to_5mm': int(np.sum((distances_mm >= 3) & (distances_mm < 5))),
                'poor_5_to_10mm': int(np.sum((distances_mm >= 5) & (distances_mm < 10))),
                'very_poor_gt_10mm': int(np.sum(distances_mm >= 10))
            },
            'distance_range_percentages': {
                'excellent_lt_1mm': float(np.sum(distances_mm < 1) / len(distances_mm) * 100),
                'good_1_to_3mm': float(np.sum((distances_mm >= 1) & (distances_mm < 3)) / len(distances_mm) * 100),
                'fair_3_to_5mm': float(np.sum((distances_mm >= 3) & (distances_mm < 5)) / len(distances_mm) * 100),
                'poor_5_to_10mm': float(np.sum((distances_mm >= 5) & (distances_mm < 10)) / len(distances_mm) * 100),
                'very_poor_gt_10mm': float(np.sum(distances_mm >= 10) / len(distances_mm) * 100)
            }
        }
        
        return stats
    
    def compute_all_metrics(self,
                             source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             thresholds: list = [0.001, 0.005, 0.01]) -> Dict[str, Any]:
        """
        Compute all available metrics.
        
        Args:
            source: Source point cloud (registered reconstruction)
            target: Target point cloud (ground truth)
            thresholds: List of distance thresholds for completeness (meters)
            
        Returns:
            Dictionary with all metrics
        """
        self.logger.info("Computing all metrics...")
        
        metrics = {
            'point_counts': {
                'source_points': len(source.points),
                'target_points': len(target.points)
            }
        }
        
        # Primary metrics
        metrics['chamfer'] = self.compute_chamfer_distance(source, target)
        metrics['hausdorff'] = self.compute_hausdorff_distance(source, target)
        metrics['rms'] = self.compute_rms_error(source, target)
        metrics['mae'] = self.compute_mae(source, target)
        
        # Completeness at multiple thresholds
        metrics['completeness'] = {}
        for threshold in thresholds:
            threshold_key = f'threshold_{int(threshold * 1000)}mm'
            metrics['completeness'][threshold_key] = self.compute_completeness(source, target, threshold)
        
        # Detailed statistics
        metrics['distance_statistics'] = self.compute_distance_statistics(source, target)
        
        return metrics
    
    def generate_distance_heatmap(self,
                                   source: o3d.geometry.PointCloud,
                                   target: o3d.geometry.PointCloud,
                                   output_path: Path,
                                   max_distance: float = 0.01):
        """
        Generate colored point cloud showing distance errors.
        
        Args:
            source: Source point cloud (registered)
            target: Target point cloud (ground truth)
            output_path: Path to save colored PLY
            max_distance: Maximum distance for color scale (meters)
        """
        self.logger.info("Generating distance heatmap...")
        
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        # Build KD-tree
        target_tree = cKDTree(target_points)
        
        # Find distances
        distances, _ = target_tree.query(source_points, k=1)
        
        # Normalize distances to [0, 1] for colormap
        distances_normalized = np.clip(distances / max_distance, 0, 1)
        
        # Create colormap (blue=good, green=medium, red=bad)
        colors = np.zeros((len(distances_normalized), 3))
        for i, d in enumerate(distances_normalized):
            if d < 0.33:  # Good (blue to cyan)
                colors[i] = [0, d * 3, 1 - d * 3]
            elif d < 0.67:  # Medium (cyan to yellow)
                colors[i] = [(d - 0.33) * 3, 1, 0]
            else:  # Bad (yellow to red)
                colors[i] = [1, 1 - (d - 0.67) * 3, 0]
        
        # Create colored point cloud
        source_colored = source.__copy__()
        source_colored.colors = o3d.utility.Vector3dVector(colors)
        
        # Save
        o3d.io.write_point_cloud(str(output_path), source_colored)
        self.logger.info(f"Saved distance heatmap to: {output_path}")
        
        return {
            'max_distance_meters': max_distance,
            'color_scheme': 'blue (good) -> cyan -> yellow -> red (bad)',
            'file': str(output_path)
        }
