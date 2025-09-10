"""
Content detection utilities for VAPOR project.
LEGACY MODULE - Use blur.core.image.processing for new implementations.

This module is kept for backward compatibility. New code should use
the improved content detection methods in blur.core.image.processing which
include diagonal corner detection and better accuracy.
"""

import warnings
from blur.core.image.processing import find_content_bounds as _new_find_content_bounds
from blur.core.image.processing import crop_to_content as _new_crop_to_content

# Issue deprecation warning
warnings.warn(
    "specularity.utils.content_detection is deprecated. "
    "Use blur.core.image.processing instead for improved content detection.",
    DeprecationWarning,
    stacklevel=2
)


def find_content_bounds(frame):
    """
    LEGACY: Find the bounds of non-black content.
    
    This function is deprecated. Use blur.core.image.processing.find_content_bounds
    for improved diagonal corner detection.
    
    Args:
        frame: Input frame (BGR or grayscale)
        
    Returns:
        tuple: (top_bound, left_bound, bottom_bound, right_bound)
    """
    warnings.warn(
        "find_content_bounds is deprecated. Use blur.core.image.processing.find_content_bounds",
        DeprecationWarning,
        stacklevel=2
    )
    return _new_find_content_bounds(frame)


def crop_to_content(frame, bounds=None):
    """
    LEGACY: Crop frame to content bounds.
    
    This function is deprecated. Use blur.core.image.processing.crop_to_content
    for consistent behavior across VAPOR modules.
    
    Args:
        frame: Input frame to crop
        bounds: Optional pre-computed bounds
    
    Returns:
        Cropped frame
    """
    warnings.warn(
        "crop_to_content is deprecated. Use blur.core.image.processing.crop_to_content",
        DeprecationWarning,
        stacklevel=2
    )
    return _new_crop_to_content(frame, bounds)
