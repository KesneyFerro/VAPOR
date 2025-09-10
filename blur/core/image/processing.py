"""
Image Processing Utilities for VAPOR Project

This module provides unified image processing functions including:
- Content detection and cropping
- Frame padding and resizing
- Image format handling

Consolidates functionality from specularity.utils.content_detection and other modules.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def find_content_bounds(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the bounds of non-black content using advanced diagonal corner detection.
    
    This method first uses edge-based detection from center points, then refines
    the bounds using diagonal corner analysis for more accurate content detection.
    
    Args:
        frame: Input frame (BGR or grayscale)
        
    Returns:
        tuple: (top_bound, left_bound, bottom_bound, right_bound)
    """
    # Convert to grayscale for edge detection
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    h, w = gray.shape
    
    # Step 1: Find initial bounds using edge-based detection from center
    # Top edge going down from center
    top_bound = 0
    for y in range(h):
        if gray[y, w//2] > 0:  # Non-black pixel
            top_bound = y
            break
    
    # Bottom edge going up from center
    bottom_bound = h - 1
    for y in range(h-1, -1, -1):
        if gray[y, w//2] > 0:  # Non-black pixel
            bottom_bound = y
            break
    
    # Left edge going right from center
    left_bound = 0
    for x in range(w):
        if gray[h//2, x] > 0:  # Non-black pixel
            left_bound = x
            break
    
    # Right edge going left from center
    right_bound = w - 1
    for x in range(w-1, -1, -1):
        if gray[h//2, x] > 0:  # Non-black pixel
            right_bound = x
            break
    
    # Step 2: Refine bounds using diagonal corner detection
    refined_bounds = _refine_bounds_with_diagonal_detection(gray, top_bound, left_bound, bottom_bound, right_bound)
    
    return refined_bounds


def _refine_bounds_with_diagonal_detection(gray: np.ndarray, initial_top: int, initial_left: int, 
                                         initial_bottom: int, initial_right: int) -> Tuple[int, int, int, int]:
    """
    Refine content bounds using diagonal corner detection.
    
    This method analyzes diagonal paths from each corner toward the center of the image
    to find more accurate content boundaries.
    
    Args:
        gray: Grayscale image
        initial_top, initial_left, initial_bottom, initial_right: Initial bounds from edge detection
        
    Returns:
        tuple: Refined (top_bound, left_bound, bottom_bound, right_bound)
    """
    h, w = gray.shape
    
    # Calculate center point
    center_y, center_x = h // 2, w // 2
    
    # Define corner points based on initial bounds
    corners = {
        'top_left': (initial_top, initial_left),
        'top_right': (initial_top, initial_right),
        'bottom_left': (initial_bottom, initial_left),
        'bottom_right': (initial_bottom, initial_right)
    }
    
    # Find refined bounds by analyzing diagonals from corners to center
    
    # Top-left corner: diagonal down-right toward center
    corner_y, corner_x = corners['top_left']
    refined_top_left = _find_diagonal_content_bound(gray, corner_y, corner_x, center_y, center_x)
    
    # Top-right corner: diagonal down-left toward center  
    corner_y, corner_x = corners['top_right']
    refined_top_right = _find_diagonal_content_bound(gray, corner_y, corner_x, center_y, center_x)
    
    # Bottom-left corner: diagonal up-right toward center
    corner_y, corner_x = corners['bottom_left']
    refined_bottom_left = _find_diagonal_content_bound(gray, corner_y, corner_x, center_y, center_x)
    
    # Bottom-right corner: diagonal up-left toward center
    corner_y, corner_x = corners['bottom_right']
    refined_bottom_right = _find_diagonal_content_bound(gray, corner_y, corner_x, center_y, center_x)
    
    # Calculate refined bounds from the diagonal detections
    # Top bound: average of y-coordinates from top corners
    refined_top = int((refined_top_left[0] + refined_top_right[0]) / 2)
    
    # Left bound: average of x-coordinates from left corners
    refined_left = int((refined_top_left[1] + refined_bottom_left[1]) / 2)
    
    # Bottom bound: average of y-coordinates from bottom corners
    refined_bottom = int((refined_bottom_left[0] + refined_bottom_right[0]) / 2)
    
    # Right bound: average of x-coordinates from right corners
    refined_right = int((refined_top_right[1] + refined_bottom_right[1]) / 2)
    
    # Ensure bounds are valid and within image dimensions
    refined_top = max(0, min(refined_top, h-1))
    refined_left = max(0, min(refined_left, w-1))
    refined_bottom = max(refined_top, min(refined_bottom, h-1))
    refined_right = max(refined_left, min(refined_right, w-1))
    
    return (refined_top, refined_left, refined_bottom, refined_right)


def _find_diagonal_content_bound(gray: np.ndarray, start_y: int, start_x: int, 
                               end_y: int, end_x: int) -> Tuple[int, int]:
    """
    Find the first non-black pixel along a diagonal path from corner toward center.
    
    Args:
        gray: Grayscale image
        start_y, start_x: Starting corner coordinates
        end_y, end_x: Target center coordinates
        
    Returns:
        tuple: (y, x) coordinates of first non-black pixel found
    """
    h, w = gray.shape
    
    # Calculate direction vector from start to end
    dy = end_y - start_y
    dx = end_x - start_x
    
    # Calculate the number of steps needed for the diagonal
    steps = max(abs(dy), abs(dx))
    
    if steps == 0:
        return (start_y, start_x)
    
    # Calculate step increments
    step_y = dy / steps
    step_x = dx / steps
    
    # Walk along the diagonal from corner toward center
    for i in range(steps + 1):
        y = int(start_y + i * step_y)
        x = int(start_x + i * step_x)
        
        # Ensure coordinates are within bounds
        if 0 <= y < h and 0 <= x < w:
            # Check if pixel is non-black
            if gray[y, x] > 0:
                return (y, x)
    
    # If no content found, return the end point (center)
    return (end_y, end_x)


def crop_to_content(frame: np.ndarray, bounds: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Crop frame to content bounds.
    
    Args:
        frame: Input frame to crop
        bounds: Optional pre-computed bounds (top, left, bottom, right).
                If None, bounds will be computed automatically.
    
    Returns:
        Cropped frame
    """
    if bounds is None:
        bounds = find_content_bounds(frame)
    
    top, left, bottom, right = bounds
    
    # Ensure bounds are valid
    h, w = frame.shape[:2]
    top = max(0, min(top, h-1))
    left = max(0, min(left, w-1))
    bottom = max(top, min(bottom, h-1))
    right = max(left, min(right, w-1))
    
    return frame[top:bottom+1, left:right+1]


def pad_frame_to_size(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Pad frame with black pixels to match target dimensions.
    
    Args:
        frame: Input frame to pad
        target_width: Target width
        target_height: Target height
        
    Returns:
        Padded frame with target dimensions
    """
    if frame.shape[:2] == (target_height, target_width):
        return frame
    
    frame_height, frame_width = frame.shape[:2]
    
    # Create black canvas
    if len(frame.shape) == 3:
        padded_frame = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
    else:
        padded_frame = np.zeros((target_height, target_width), dtype=frame.dtype)
    
    # Handle oversized frames by scaling down
    if frame_height > target_height or frame_width > target_width:
        scale = min(target_width / frame_width, target_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
        frame_height, frame_width = new_height, new_width
    
    # Center the frame
    start_y = (target_height - frame_height) // 2
    start_x = (target_width - frame_width) // 2
    
    # Place frame in center
    end_y = start_y + frame_height
    end_x = start_x + frame_width
    
    if len(frame.shape) == 3:
        padded_frame[start_y:end_y, start_x:end_x, :] = frame
    else:
        padded_frame[start_y:end_y, start_x:end_x] = frame
        
    return padded_frame


def detect_optimal_crop_bounds(video_path: Path, sample_count: int = 5) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect optimal crop bounds by sampling frames from different parts of the video.
    
    This method samples exactly 5 random frames (as specified) and uses the new
    diagonal corner detection method for each frame, then averages the results.
    
    Args:
        video_path: Path to video file
        sample_count: Number of frames to sample (kept for API compatibility, always uses 5)
        
    Returns:
        Optimal crop bounds (top, left, bottom, right) or None if detection fails
    """
    import random
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Always sample exactly 5 random frames as specified (ignore sample_count parameter)
    frames_to_sample = 5
    if total_frames <= frames_to_sample:
        sample_indices = list(range(total_frames))
    else:
        # Select 5 random frame indices
        sample_indices = random.sample(range(total_frames), frames_to_sample)
        sample_indices.sort()  # Sort for consistent processing
    
    print(f"  Sampling {len(sample_indices)} frames: {sample_indices}")
    
    all_bounds = []
    
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            try:
                bounds = find_content_bounds(frame)
                all_bounds.append(bounds)
                print(f"  Frame {frame_idx}: bounds {bounds} (diagonal method)")
            except Exception as e:
                print(f"  Frame {frame_idx}: detection failed - {e}")
                continue  # Skip frames that cause issues
    
    cap.release()
    
    if not all_bounds:
        print("  [WARNING] No valid crop bounds detected")
        return None
    
    # Calculate average bounds from all successful detections
    avg_top = sum(b[0] for b in all_bounds) / len(all_bounds)
    avg_left = sum(b[1] for b in all_bounds) / len(all_bounds)
    avg_bottom = sum(b[2] for b in all_bounds) / len(all_bounds)
    avg_right = sum(b[3] for b in all_bounds) / len(all_bounds)
    
    # Convert to integers
    final_top = int(avg_top)
    final_left = int(avg_left)
    final_bottom = int(avg_bottom)
    final_right = int(avg_right)
    
    # Validate bounds
    if final_top >= final_bottom or final_left >= final_right:
        print("  [WARNING] Invalid averaged bounds detected")
        return None
    
    print(f"  [OK] Averaged bounds from {len(all_bounds)} frames: ({final_top}, {final_left}, {final_bottom}, {final_right})")
    
    return (final_top, final_left, final_bottom, final_right)


def get_image_files(folder_path: Path, extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) -> list:
    """
    Get all image files from a folder sorted naturally.
    
    Args:
        folder_path: Path to folder containing images
        extensions: Tuple of valid image extensions
        
    Returns:
        List of image file paths sorted naturally
    """
    if not folder_path.exists():
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    # Natural sort by filename
    def natural_sort_key(path):
        import re
        filename = path.name
        # Split into text and number parts for natural sorting
        parts = re.split(r'(\d+)', filename)
        # Convert numeric parts to integers for proper sorting
        return [int(part) if part.isdigit() else part.lower() for part in parts]
    
    return sorted(image_files, key=natural_sort_key)


def resize_image_keeping_aspect_ratio(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize image while keeping aspect ratio, fitting within target dimensions.
    
    Args:
        image: Input image
        target_width: Maximum width
        target_height: Maximum height
        
    Returns:
        Resized image that fits within target dimensions
    """
    h, w = image.shape[:2]
    
    # Calculate scale to fit within target dimensions
    scale = min(target_width / w, target_height / h)
    
    if scale >= 1.0:
        return image  # No need to resize
    
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
