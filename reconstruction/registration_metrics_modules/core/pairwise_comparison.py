"""
Cross-Model Pairwise Comparison Module (Step 8)
==============================================

Computes pairwise metrics between different reconstruction models.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

import numpy as np
import open3d as o3d
import pandas as pd


class PairwiseComparator:
    """Compare multiple reconstructions pairwise."""
    
    def __init__(self, output_dir: Path, h: float, logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.h = h
        self.logger = logger or logging.getLogger(__name__)
        
    def compute_pairwise_chamfer(self,
                                  models: Dict[str, o3d.geometry.PointCloud]) -> np.ndarray:
        """
        Compute pairwise Chamfer distances between all models.
        
        Args:
            models: Dictionary mapping model_name -> point_cloud
            
        Returns:
            NxN matrix of Chamfer distances
        """
        self.logger.info(f"Computing pairwise Chamfer distances for {len(models)} models...")
        
        model_names = list(models.keys())
        n = len(model_names)
        chamfer_matrix = np.zeros((n, n))
        
        for i, name_i in enumerate(model_names):
            for j, name_j in enumerate(model_names):
                if i == j:
                    chamfer_matrix[i, j] = 0.0
                    continue
                
                pc_i = models[name_i]
                pc_j = models[name_j]
                
                # Compute symmetric Chamfer
                points_i = np.asarray(pc_i.points)
                points_j = np.asarray(pc_j.points)
                
                # Build KDTrees
                kdtree_i = o3d.geometry.KDTreeFlann(pc_i)
                kdtree_j = o3d.geometry.KDTreeFlann(pc_j)
                
                # i -> j distances
                dists_i2j = []
                for pt in points_i:
                    [k, idx, dist_sq] = kdtree_j.search_knn_vector_3d(pt, 1)
                    dists_i2j.append(np.sqrt(dist_sq[0]))
                
                # j -> i distances
                dists_j2i = []
                for pt in points_j:
                    [k, idx, dist_sq] = kdtree_i.search_knn_vector_3d(pt, 1)
                    dists_j2i.append(np.sqrt(dist_sq[0]))
                
                # Symmetric Chamfer (squared)
                chamfer = np.mean(np.array(dists_i2j)**2) + np.mean(np.array(dists_j2i)**2)
                chamfer_matrix[i, j] = chamfer
        
        return chamfer_matrix, model_names
    
    def save_results(self, chamfer_matrix: np.ndarray, model_names: List[str]):
        """Save pairwise comparison results."""
        pairwise_dir = self.output_dir / 'pairwise'
        pairwise_dir.mkdir(exist_ok=True)
        
        # Save as CSV
        df = pd.DataFrame(chamfer_matrix, index=model_names, columns=model_names)
        csv_path = pairwise_dir / 'chamfer_matrix.csv'
        df.to_csv(csv_path)
        
        self.logger.info(f"Pairwise Chamfer matrix saved to {csv_path}")
