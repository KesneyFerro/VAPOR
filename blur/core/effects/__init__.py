"""
Effects Module
Blur and other image effects for VAPOR processing.
"""

from .blur_effects import BlurEffectsEngine, blur_engine, apply_blur_effect, get_supported_blur_types

__all__ = [
    'BlurEffectsEngine',
    'blur_engine', 
    'apply_blur_effect',
    'get_supported_blur_types'
]
