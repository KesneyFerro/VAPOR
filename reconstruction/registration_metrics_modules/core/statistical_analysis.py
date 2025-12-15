"""
Statistical Uncertainty Analysis Module (Step 9)
===============================================

Bootstrap confidence intervals and sensitivity analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np


class StatisticalAnalyzer:
    """Compute bootstrap confidence intervals and sensitivity."""
    
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
    def bootstrap_confidence_intervals(self,
                                        distances: np.ndarray,
                                        n_bootstrap: int = 1000,
                                        confidence: float = 0.95,
                                        seed: int = 42) -> Dict[str, Any]:
        """
        Compute bootstrap confidence intervals for median.
        
        Args:
            distances: Distance array
            n_bootstrap: Number of bootstrap resamples
            confidence: Confidence level
            seed: Random seed for reproducibility
        """
        self.logger.info(f"Computing bootstrap CIs (n={n_bootstrap}, confidence={confidence})...")
        
        np.random.seed(seed)
        
        medians = []
        n = len(distances)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(distances, size=n, replace=True)
            medians.append(np.median(sample))
        
        medians = np.array(medians)
        
        alpha = 1 - confidence
        lower = np.percentile(medians, 100 * alpha / 2)
        upper = np.percentile(medians, 100 * (1 - alpha / 2))
        
        return {
            'median': float(np.median(distances)),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'confidence_level': confidence,
            'n_bootstrap': n_bootstrap,
            'seed': seed,
        }
    
    def save_results(self, model_name: str, results: Dict[str, Any]):
        """Save statistical analysis results."""
        stats_dir = self.output_dir / 'statistics'
        stats_dir.mkdir(exist_ok=True)
        
        json_path = stats_dir / f'bootstrap_ci_{model_name}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Statistical analysis saved to {json_path}")
