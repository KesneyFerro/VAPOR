"""
Image Processing Module
Image manipulation, cropping, and processing utilities.
"""

from .processing import (
    find_content_bounds,
    crop_to_content,
    pad_frame_to_size,
    detect_optimal_crop_bounds,
    get_image_files,
    resize_image_keeping_aspect_ratio
)

__all__ = [
    'find_content_bounds',
    'crop_to_content', 
    'pad_frame_to_size',
    'detect_optimal_crop_bounds',
    'get_image_files',
    'resize_image_keeping_aspect_ratio'
]
