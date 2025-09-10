"""
Blurring Module
Provides various blur effects for image sequences using shared blur core utilities.

This module imports from the blur.core module for consistent blur
implementations within the blur processing pipeline.
"""

# Import from shared blur core module
from ..core.effects import BlurEffectsEngine as BlurEffects, apply_blur_effect, get_supported_blur_types

__all__ = ['BlurEffects', 'apply_blur_effect', 'get_supported_blur_types']
__version__ = '3.0.0'
