"""
Content detection utilities for VAPOR project.
Contains functions for finding and cropping content bounds in video frames.
"""

import cv2
import numpy as np


def find_content_bounds(frame):
    """
    Find the bounds of non-black content from center edges.
    
    Args:
        frame: Input frame (BGR or grayscale)
        
    Returns:
        tuple: (top_bound, bottom_bound, left_bound, right_bound)
    """
    # Convert to grayscale for edge detection
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    h, w = gray.shape
    
    # Find first non-black pixel from each direction (from center edges)
    # Top edge going down
    top_bound = 0
    for y in range(h):
        if gray[y, w//2] > 0:  # Non-black pixel
            top_bound = y
            break
    
    # Bottom edge going up
    bottom_bound = h - 1
    for y in range(h-1, -1, -1):
        if gray[y, w//2] > 0:  # Non-black pixel
            bottom_bound = y
            break
    
    # Left edge going right
    left_bound = 0
    for x in range(w):
        if gray[h//2, x] > 0:  # Non-black pixel
            left_bound = x
            break
    
    # Right edge going left
    right_bound = w - 1
    for x in range(w-1, -1, -1):
        if gray[h//2, x] > 0:  # Non-black pixel
            right_bound = x
            break
    
    # Add small padding to ensure we don't crop too aggressively
    padding = 5
    top_bound = max(0, top_bound - padding)
    bottom_bound = min(h - 1, bottom_bound + padding)
    left_bound = max(0, left_bound - padding)
    right_bound = min(w - 1, right_bound + padding)
    
    return top_bound, bottom_bound, left_bound, right_bound


def crop_to_content(frame):
    """
    Crop frame to content bounds.
    
    Args:
        frame: Input frame (BGR or grayscale)
        
    Returns:
        numpy.ndarray: Cropped frame
    """
    top, bottom, left, right = find_content_bounds(frame)
    
    # Crop the frame
    if len(frame.shape) == 3:
        cropped = frame[top:bottom+1, left:right+1, :]
    else:
        cropped = frame[top:bottom+1, left:right+1]
    
    return cropped
