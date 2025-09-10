"""
VAPOR Core Module
Shared utilities and components for the VAPOR project.

This module provides reusable components across different VAPOR pipelines:
- effects: Blur and other image effects
- video: Video processing utilities  
- image: Image manipulation and cropping utilities
- utils: Common pipeline utilities
"""

from . import effects
from . import video
from . import image
from . import quality_metrics
from .utils import (
    VideoSelector,
    ProcessingModeSelector,
    setup_project_paths,
    ensure_directories_exist,
    add_project_to_path
)
from .quality_metrics import QualityMetricsCalculator, QualityMetricsLogger

__version__ = "1.0.0"
__all__ = [
    "effects", 
    "video", 
    "image",
    "quality_metrics",
    "VideoSelector",
    "ProcessingModeSelector", 
    "setup_project_paths",
    "ensure_directories_exist",
    "add_project_to_path",
    "QualityMetricsCalculator",
    "QualityMetricsLogger"
]
