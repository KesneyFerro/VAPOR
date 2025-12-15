"""
Configuration module for 3D Reconstruction QA
============================================

Centralized configuration management for all QA parameters.
"""

from typing import Dict, Any, List
from pathlib import Path
import yaml


class QAConfig:
    """Configuration for reconstruction QA pipeline."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        'paths': {
            'data_root': 'data',
            'point_clouds_subdir': 'point_clouds',
            'metrics_subdir': 'metrics',
            'logs_subdir': 'logs',
        },
        
        'ct_mesh': {
            'clean_mesh': True,
            'remove_duplicates': True,
            'remove_degenerates': True,
        },
        
        'roi': {
            'method': 'proximity',  # 'proximity' or 'frusta'
            'distance_multiplier': 5.0,  # ROI = points within distance_multiplier * h
        },
        
        'distance_metrics': {
            'threshold_multipliers': [1.0, 2.0, 3.0],  # F-score thresholds as multiples of h
            'poisson_disk_sampling': True,
            'poisson_init_factor': 5,
        },
        
        'normal_consistency': {
            'enabled': True,
            'max_distance_multiplier': 2.0,  # Only compare normals within 2h
            'knn_for_estimation': 30,
        },
        
        'registration_sanity': {
            'enabled': True,
            'icp_threshold_multiplier': 2.0,  # ICP distance threshold = 2h
            'icp_max_iterations': 50,
        },
        
        'output': {
            'save_distance_arrays': True,
            'save_visualizations': True,
            'save_json': True,
            'save_csv': True,
            'generate_report': True,
        },
        
        'logging': {
            'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
            'save_to_file': True,
        }
    }
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Optional configuration dictionary (overrides defaults)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_dict:
            self._update_nested(self.config, config_dict)
    
    def _update_nested(self, base: Dict, update: Dict):
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._update_nested(base[key], value)
            else:
                base[key] = value
    
    def get(self, *keys):
        """Get nested configuration value."""
        value = self.config
        for key in keys:
            value = value[key]
        return value
    
    def set(self, *keys, value):
        """Set nested configuration value."""
        config = self.config
        for key in keys[:-1]:
            config = config[key]
        config[keys[-1]] = value
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'QAConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def to_yaml(self, yaml_path: Path):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.copy()


# Convenience function
def load_config(yaml_path: Path = None) -> QAConfig:
    """
    Load configuration from YAML or use defaults.
    
    Args:
        yaml_path: Optional path to YAML config file
        
    Returns:
        QAConfig instance
    """
    if yaml_path and yaml_path.exists():
        return QAConfig.from_yaml(yaml_path)
    else:
        return QAConfig()
