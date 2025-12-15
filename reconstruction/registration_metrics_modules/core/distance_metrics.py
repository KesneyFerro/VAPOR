"""
Core Distance Metrics Module (Step 3)
=====================================

Computes comprehensive distance metrics between point clouds and CT mesh:
- Point-to-Surface (P2S): point cloud → mesh distances
- Surface-to-Point (S2P): mesh → point cloud distances (on ROI)
- Symmetric Chamfer Distance (squared)
- Hausdorff Distance (HD) and HD95
- F-score @ multiple thresholds (Precision/Recall)
- ASSD (Average Symmetric Surface Distance)

Production-grade using:
- Open3D RaycastingScene for exact point-to-triangle distances
- Open3D KDTree for fast nearest neighbor queries
- NumPy for statistical analysis
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class DistanceMetricsCalculator:
    """
    Calculate comprehensive distance metrics between reconstruction and CT mesh.
    All computations use production-grade algorithms from Open3D.
    """
    
    def __init__(self, 
                 output_dir: Path,
                 h: float,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize distance metrics calculator.
        
        Args:
            output_dir: Output directory for distance arrays and metrics
            h: Characteristic length (meters) for threshold calculations
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.h = h
        self.logger = logger or logging.getLogger(__name__)
        
        # Store results
        self.p2s_distances: Optional[np.ndarray] = None
        self.s2p_distances: Optional[np.ndarray] = None
        self.metrics: Dict[str, Any] = {}
        
    def compute_p2s_distances(self,
                               point_cloud: o3d.geometry.PointCloud,
                               raycasting_scene: o3d.t.geometry.RaycastingScene) -> np.ndarray:
        """
        Compute Point-to-Surface (P2S) unsigned distances.
        
        Uses Open3D RaycastingScene for exact point-to-triangle distances.
        
        Args:
            point_cloud: Reconstruction point cloud
            raycasting_scene: Raycasting scene built from CT mesh
            
        Returns:
            Array of distances (meters) for each point
        """
        self.logger.info("Computing Point-to-Surface (P2S) distances...")
        
        points = np.asarray(point_cloud.points, dtype=np.float32)
        n_points = len(points)
        
        self.logger.debug(f"Computing distances for {n_points} points")
        
        # Convert to Open3D tensor
        query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        
        # Compute unsigned distances using RaycastingScene
        result = raycasting_scene.compute_distance(query_points)
        
        # Extract distances
        distances = result.numpy()
        
        self.logger.info(f"P2S distances computed: median={np.median(distances):.6f}m, "
                        f"mean={np.mean(distances):.6f}m, max={np.max(distances):.6f}m")
        
        self.p2s_distances = distances
        
        return distances
    
    def compute_s2p_distances(self,
                               ct_mesh_samples: o3d.geometry.PointCloud,
                               point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Compute Surface-to-Point (S2P) distances.
        
        For each CT mesh sample point, finds nearest neighbor in reconstruction.
        
        Args:
            ct_mesh_samples: Poisson-disk sampled points from CT mesh ROI
            point_cloud: Reconstruction point cloud
            
        Returns:
            Array of distances (meters) for each CT sample
        """
        self.logger.info("Computing Surface-to-Point (S2P) distances...")
        
        ct_points = np.asarray(ct_mesh_samples.points)
        pc_points = np.asarray(point_cloud.points)
        
        n_ct = len(ct_points)
        n_pc = len(pc_points)
        
        self.logger.debug(f"Computing distances for {n_ct} CT samples to {n_pc} PC points")
        
        # Build KDTree on reconstruction point cloud
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        
        distances = np.zeros(n_ct)
        
        for i, point in enumerate(ct_points):
            # Query nearest neighbor
            [k, idx, dist_sq] = kdtree.search_knn_vector_3d(point, 1)
            distances[i] = np.sqrt(dist_sq[0])
        
        self.logger.info(f"S2P distances computed: median={np.median(distances):.6f}m, "
                        f"mean={np.mean(distances):.6f}m, max={np.max(distances):.6f}m")
        
        self.s2p_distances = distances
        
        return distances
    
    def compute_summary_statistics(self, distances: np.ndarray, name: str) -> Dict[str, float]:
        """
        Compute comprehensive summary statistics for distance array.
        
        Args:
            distances: Array of distances
            name: Name for logging
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'median': float(np.median(distances)),
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'p90': float(np.percentile(distances, 90)),
            'p95': float(np.percentile(distances, 95)),
            'p99': float(np.percentile(distances, 99)),
            'units': 'meters',
        }
        
        # Convert to mm for easier interpretation
        stats_mm = {
            'median_mm': stats['median'] * 1000,
            'mean_mm': stats['mean'] * 1000,
            'std_mm': stats['std'] * 1000,
            'p95_mm': stats['p95'] * 1000,
        }
        
        self.logger.debug(f"{name} statistics: median={stats_mm['median_mm']:.3f}mm, "
                         f"mean={stats_mm['mean_mm']:.3f}mm, p95={stats_mm['p95_mm']:.3f}mm")
        
        return stats
    
    def compute_chamfer_distance(self,
                                  p2s_distances: np.ndarray,
                                  s2p_distances: np.ndarray) -> Dict[str, float]:
        """
        Compute Symmetric Chamfer Distance (squared).
        
        CD = mean(d_p2s^2) + mean(d_s2p^2)
        
        Args:
            p2s_distances: Point-to-surface distances
            s2p_distances: Surface-to-point distances
            
        Returns:
            Dictionary with Chamfer metrics
        """
        self.logger.info("Computing Chamfer Distance...")
        
        # Squared distances
        p2s_sq = p2s_distances ** 2
        s2p_sq = s2p_distances ** 2
        
        # Chamfer components
        cd_p2s = float(np.mean(p2s_sq))
        cd_s2p = float(np.mean(s2p_sq))
        
        # Total chamfer
        cd = cd_p2s + cd_s2p
        
        self.logger.info(f"Chamfer Distance: {cd:.8f} (P2S: {cd_p2s:.8f}, S2P: {cd_s2p:.8f})")
        
        return {
            'chamfer_distance': cd,
            'chamfer_p2s_component': cd_p2s,
            'chamfer_s2p_component': cd_s2p,
            'units': 'meters^2',
        }
    
    def compute_hausdorff_distance(self,
                                     p2s_distances: np.ndarray,
                                     s2p_distances: np.ndarray) -> Dict[str, float]:
        """
        Compute Hausdorff Distance (HD) and HD95.
        
        HD = max(max(d_p2s), max(d_s2p))
        HD95 = percentile_95(both directions)
        
        Args:
            p2s_distances: Point-to-surface distances
            s2p_distances: Surface-to-point distances
            
        Returns:
            Dictionary with Hausdorff metrics
        """
        self.logger.info("Computing Hausdorff Distance...")
        
        # Directed Hausdorff
        hd_p2s = float(np.max(p2s_distances))
        hd_s2p = float(np.max(s2p_distances))
        
        # Symmetric Hausdorff
        hd = max(hd_p2s, hd_s2p)
        
        # HD95 (more robust to outliers)
        hd95_p2s = float(np.percentile(p2s_distances, 95))
        hd95_s2p = float(np.percentile(s2p_distances, 95))
        hd95 = max(hd95_p2s, hd95_s2p)
        
        self.logger.info(f"Hausdorff Distance: HD={hd:.6f}m, HD95={hd95:.6f}m")
        
        return {
            'hausdorff_distance': hd,
            'hausdorff_p2s': hd_p2s,
            'hausdorff_s2p': hd_s2p,
            'hd95': hd95,
            'hd95_p2s': hd95_p2s,
            'hd95_s2p': hd95_s2p,
            'units': 'meters',
        }
    
    def compute_assd(self,
                     p2s_distances: np.ndarray,
                     s2p_distances: np.ndarray) -> Dict[str, float]:
        """
        Compute Average Symmetric Surface Distance (ASSD).
        
        ASSD = (mean(d_p2s) + mean(d_s2p)) / 2
        
        Args:
            p2s_distances: Point-to-surface distances
            s2p_distances: Surface-to-point distances
            
        Returns:
            Dictionary with ASSD metric
        """
        mean_p2s = float(np.mean(p2s_distances))
        mean_s2p = float(np.mean(s2p_distances))
        
        assd = (mean_p2s + mean_s2p) / 2
        
        self.logger.info(f"ASSD: {assd:.6f}m")
        
        return {
            'assd': assd,
            'mean_p2s': mean_p2s,
            'mean_s2p': mean_s2p,
            'units': 'meters',
        }
    
    def compute_fscore(self,
                       p2s_distances: np.ndarray,
                       s2p_distances: np.ndarray,
                       thresholds: Optional[List[float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute F-score (Precision, Recall, F1) at multiple thresholds.
        
        Precision = fraction of P2S distances <= threshold
        Recall = fraction of S2P distances <= threshold
        F1 = 2 * P * R / (P + R)
        
        Args:
            p2s_distances: Point-to-surface distances
            s2p_distances: Surface-to-point distances
            thresholds: List of thresholds (meters). Default: [h, 2h, 3h]
            
        Returns:
            Dictionary mapping threshold -> {precision, recall, f1}
        """
        if thresholds is None:
            thresholds = [self.h, 2 * self.h, 3 * self.h]
        
        self.logger.info(f"Computing F-scores at thresholds: {thresholds}")
        
        results = {}
        
        for threshold in thresholds:
            # Precision: fraction of reconstruction points within threshold of CT
            precision = float(np.mean(p2s_distances <= threshold))
            
            # Recall: fraction of CT points within threshold of reconstruction
            recall = float(np.mean(s2p_distances <= threshold))
            
            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            key = f'threshold_{threshold:.6f}m'
            results[key] = {
                'threshold_m': threshold,
                'threshold_mm': threshold * 1000,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            }
            
            self.logger.debug(f"F-score @ {threshold*1000:.3f}mm: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        return results
    
    def compute_all_metrics(self,
                            point_cloud: o3d.geometry.PointCloud,
                            raycasting_scene: o3d.t.geometry.RaycastingScene,
                            ct_roi_samples: o3d.geometry.PointCloud,
                            thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compute all distance metrics.
        
        Args:
            point_cloud: Reconstruction point cloud
            raycasting_scene: RaycastingScene from CT mesh
            ct_roi_samples: Poisson-disk sampled points from CT ROI
            thresholds: F-score thresholds (default: [h, 2h, 3h])
            
        Returns:
            Comprehensive metrics dictionary
        """
        self.logger.info("Computing all distance metrics...")
        
        metrics = {
            'n_reconstruction_points': len(point_cloud.points),
            'n_ct_samples': len(ct_roi_samples.points),
            'characteristic_length_h': self.h,
        }
        
        # P2S distances
        p2s = self.compute_p2s_distances(point_cloud, raycasting_scene)
        metrics['p2s_statistics'] = self.compute_summary_statistics(p2s, "P2S")
        
        # S2P distances
        s2p = self.compute_s2p_distances(ct_roi_samples, point_cloud)
        metrics['s2p_statistics'] = self.compute_summary_statistics(s2p, "S2P")
        
        # Chamfer distance
        metrics['chamfer'] = self.compute_chamfer_distance(p2s, s2p)
        
        # Hausdorff distance
        metrics['hausdorff'] = self.compute_hausdorff_distance(p2s, s2p)
        
        # ASSD
        metrics['assd'] = self.compute_assd(p2s, s2p)
        
        # F-scores
        metrics['fscores'] = self.compute_fscore(p2s, s2p, thresholds)
        
        self.metrics = metrics
        self.logger.info("All distance metrics computed")
        
        return metrics
    
    def save_distance_arrays(self, model_name: str) -> Tuple[Path, Path]:
        """
        Save P2S and S2P distance arrays to disk.
        
        Args:
            model_name: Name of the model (for filename)
            
        Returns:
            Tuple of (p2s_path, s2p_path)
        """
        if self.p2s_distances is None or self.s2p_distances is None:
            raise ValueError("No distances to save. Run compute_all_metrics() first.")
        
        # Create distances subdirectory
        distances_dir = self.output_dir / 'distances'
        distances_dir.mkdir(exist_ok=True)
        
        # Save arrays
        p2s_path = distances_dir / f'p2s_{model_name}.npy'
        s2p_path = distances_dir / f's2p_{model_name}.npy'
        
        np.save(p2s_path, self.p2s_distances)
        np.save(s2p_path, self.s2p_distances)
        
        self.logger.info(f"Distance arrays saved: {p2s_path}, {s2p_path}")
        
        return p2s_path, s2p_path
    
    def save_metrics_table(self, model_name: str) -> Path:
        """
        Save metrics to CSV table.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to saved CSV
        """
        if not self.metrics:
            raise ValueError("No metrics to save. Run compute_all_metrics() first.")
        
        # Create tables subdirectory
        tables_dir = self.output_dir / 'tables'
        tables_dir.mkdir(exist_ok=True)
        
        csv_path = tables_dir / f'metrics_{model_name}.csv'
        
        # Flatten metrics for CSV
        rows = []
        
        # P2S stats
        for key, val in self.metrics['p2s_statistics'].items():
            if key != 'units':
                rows.append(f"p2s_{key},{val}\n")
        
        # S2P stats
        for key, val in self.metrics['s2p_statistics'].items():
            if key != 'units':
                rows.append(f"s2p_{key},{val}\n")
        
        # Chamfer
        rows.append(f"chamfer_distance,{self.metrics['chamfer']['chamfer_distance']}\n")
        
        # Hausdorff
        rows.append(f"hausdorff_distance,{self.metrics['hausdorff']['hausdorff_distance']}\n")
        rows.append(f"hd95,{self.metrics['hausdorff']['hd95']}\n")
        
        # ASSD
        rows.append(f"assd,{self.metrics['assd']['assd']}\n")
        
        # F-scores
        for threshold_key, fmetrics in self.metrics['fscores'].items():
            rows.append(f"fscore_{threshold_key}_precision,{fmetrics['precision']}\n")
            rows.append(f"fscore_{threshold_key}_recall,{fmetrics['recall']}\n")
            rows.append(f"fscore_{threshold_key}_f1,{fmetrics['f1_score']}\n")
        
        with open(csv_path, 'w') as f:
            f.write("metric,value\n")
            f.writelines(rows)
        
        self.logger.info(f"Metrics table saved to: {csv_path}")
        
        # Also save as JSON
        json_path = tables_dir / f'metrics_{model_name}.json'
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics JSON saved to: {json_path}")
        
        return csv_path


def compute_distance_metrics(
    point_cloud: o3d.geometry.PointCloud,
    raycasting_scene: o3d.t.geometry.RaycastingScene,
    ct_roi_samples: o3d.geometry.PointCloud,
    h: float,
    output_dir: Path,
    model_name: str,
    thresholds: Optional[List[float]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Convenience function to compute all distance metrics.
    
    Args:
        point_cloud: Reconstruction point cloud
        raycasting_scene: CT mesh raycasting scene
        ct_roi_samples: Poisson-disk samples from CT ROI
        h: Characteristic length
        output_dir: Output directory
        model_name: Model name
        thresholds: F-score thresholds
        logger: Optional logger
        
    Returns:
        Metrics dictionary
    """
    calculator = DistanceMetricsCalculator(output_dir, h, logger)
    metrics = calculator.compute_all_metrics(
        point_cloud, raycasting_scene, ct_roi_samples, thresholds
    )
    calculator.save_distance_arrays(model_name)
    calculator.save_metrics_table(model_name)
    
    return metrics
