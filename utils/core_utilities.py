"""
Video Processing Core Utilities

This module provides core utility functions used across the video processing project including:
- Video file discovery and handling
- Image processing and cropping with diagonal ROI detection
- Frame extraction and processing
- File management utilities

Consolidated from various modules to eliminate code duplication.

Author: Kesney de Oliveira
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Generator
import random
import re
import json


# =============================================================================
# Video File Discovery and Management
# =============================================================================

def get_all_videos(videos_dir: Path, include_subdirs: bool = True) -> Dict[str, List[Path]]:
    """
    Get all video files organized by category from the data/videos directory.
    
    Args:
        videos_dir: Path to videos directory (e.g., data/videos)
        include_subdirs: Whether to search in subdirectories
        
    Returns:
        Dictionary mapping category names to lists of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    videos_dir = Path(videos_dir)
    
    if not videos_dir.exists():
        return {}
    
    if include_subdirs:
        return _get_videos_from_subdirs(videos_dir, video_extensions)
    else:
        return _get_videos_from_root(videos_dir, video_extensions)


def _get_videos_from_subdirs(videos_dir: Path, video_extensions: List[str]) -> Dict[str, List[Path]]:
    """Get videos organized by subdirectories."""
    video_categories = {}
    
    for subdir in videos_dir.iterdir():
        if subdir.is_dir():
            category_videos = _find_videos_in_dir(subdir, video_extensions)
            if category_videos:
                video_categories[subdir.name] = sorted(category_videos)
    
    return video_categories


def _get_videos_from_root(videos_dir: Path, video_extensions: List[str]) -> Dict[str, List[Path]]:
    """Get videos from root directory only."""
    all_videos = _find_videos_in_dir(videos_dir, video_extensions)
    
    if all_videos:
        return {videos_dir.name: sorted(all_videos)}
    else:
        return {}


def _find_videos_in_dir(directory: Path, video_extensions: List[str]) -> List[Path]:
    """Find all videos in a specific directory."""
    videos = []
    for ext in video_extensions:
        videos.extend(directory.glob(f"*{ext}"))
        videos.extend(directory.glob(f"*{ext.upper()}"))
    return videos


def find_matching_videos(videos_dir: Path, pattern: str) -> List[Path]:
    """
    Find videos matching a specific pattern.
    
    Args:
        videos_dir: Path to videos directory
        pattern: Pattern to match (supports wildcards)
        
    Returns:
        List of matching video paths
    """
    videos_dir = Path(videos_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    matching_videos = []
    
    # Search recursively
    for ext in video_extensions:
        matching_videos.extend(videos_dir.rglob(f"{pattern}{ext}"))
        matching_videos.extend(videos_dir.rglob(f"{pattern}{ext.upper()}"))
    
    return sorted(matching_videos)


# =============================================================================
# ROI Detection with Diagonal Method
# =============================================================================

def find_content_bounds_diagonal(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the bounds of non-black content using advanced diagonal corner detection.
    
    This method uses diagonal corner analysis for more accurate content detection
    compared to simple edge-based methods.
    
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
    refined_bounds = _refine_bounds_with_diagonal_detection(
        gray, top_bound, left_bound, bottom_bound, right_bound
    )
    
    return refined_bounds


def _refine_bounds_with_diagonal_detection(gray: np.ndarray, initial_top: int, initial_left: int, 
                                         initial_bottom: int, initial_right: int) -> Tuple[int, int, int, int]:
    """
    Refine content bounds using diagonal corner detection.
    
    This method analyzes diagonal paths from each corner toward the center of the image
    to find more accurate content boundaries.
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


def crop_video_roi_diagonal(video_path: Path, sample_count: int = 5) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect optimal crop bounds for video using diagonal ROI method.
    
    Samples multiple frames and uses diagonal corner detection to find
    consistent content boundaries.
    
    Args:
        video_path: Path to video file
        sample_count: Number of frames to sample for analysis
        
    Returns:
        Optimal crop bounds (top, left, bottom, right) or None if detection fails
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly distributed through the video
    if total_frames <= sample_count:
        sample_indices = list(range(total_frames))
    else:
        # Select frames evenly distributed throughout video
        step = total_frames // sample_count
        sample_indices = [i * step for i in range(sample_count)]
        # Ensure we don't exceed total frames
        sample_indices = [min(idx, total_frames - 1) for idx in sample_indices]
    
    all_bounds = []
    
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            try:
                bounds = find_content_bounds_diagonal(frame)
                all_bounds.append(bounds)
            except Exception:
                continue  # Skip frames that cause issues
    
    cap.release()
    
    if not all_bounds:
        return None
    
    # Calculate average bounds from all successful detections
    avg_top = sum(b[0] for b in all_bounds) / len(all_bounds)
    avg_left = sum(b[1] for b in all_bounds) / len(all_bounds)
    avg_bottom = sum(b[2] for b in all_bounds) / len(all_bounds)
    avg_right = sum(b[3] for b in all_bounds) / len(all_bounds)
    
    # Convert to integers
    final_bounds = (int(avg_top), int(avg_left), int(avg_bottom), int(avg_right))
    
    # Validate bounds
    if final_bounds[0] >= final_bounds[2] or final_bounds[1] >= final_bounds[3]:
        return None
    
    return final_bounds


def crop_to_content(frame: np.ndarray, bounds: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Crop frame to content bounds.
    
    Args:
        frame: Input frame to crop
        bounds: Optional pre-computed bounds (top, left, bottom, right).
                If None, bounds will be computed automatically using diagonal method.
    
    Returns:
        Cropped frame
    """
    if bounds is None:
        bounds = find_content_bounds_diagonal(frame)
    
    top, left, bottom, right = bounds
    
    # Ensure bounds are valid
    h, w = frame.shape[:2]
    top = max(0, min(top, h-1))
    left = max(0, min(left, w-1))
    bottom = max(top, min(bottom, h-1))
    right = max(left, min(right, w-1))
    
    return frame[top:bottom+1, left:right+1]


# =============================================================================
# Frame Processing Utilities
# =============================================================================

def get_image_files(folder_path: Path, extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) -> List[Path]:
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
        filename = path.name
        # Split into text and number parts for natural sorting
        parts = re.split(r'(\d+)', filename)
        # Convert numeric parts to integers for proper sorting
        return [int(part) if part.isdigit() else part.lower() for part in parts]
    
    return sorted(image_files, key=natural_sort_key)


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


# =============================================================================
# Video Configuration and Metadata
# =============================================================================

class VideoConfig:
    """Video configuration container."""
    
    def __init__(self, video_path: Path):
        """
        Initialize video configuration from a video file.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        self._extract_config()
    
    def _extract_config(self):
        """Extract configuration from video file."""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        cap.release()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_count': self.frame_count,
            'fourcc': self.fourcc,
            'duration': self.duration
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"VideoConfig(resolution={self.width}x{self.height}, "
                f"fps={self.fps:.2f}, frames={self.frame_count}, "
                f"duration={self.duration:.2f}s)")


# =============================================================================
# File Management Utilities
# =============================================================================

def create_output_structure(base_output_dir: Path, categories: List[str], 
                          subdirs: List[str] = None) -> Dict[str, Path]:
    """
    Create organized output directory structure.
    
    Args:
        base_output_dir: Base output directory
        categories: List of category names (e.g., ['original', 'blurred', 'deblurred'])
        subdirs: Optional subdirectories within each category
        
    Returns:
        Dictionary mapping category names to their directory paths
    """
    base_output_dir = Path(base_output_dir)
    structure = {}
    
    for category in categories:
        category_dir = base_output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        structure[category] = category_dir
        
        if subdirs:
            for subdir in subdirs:
                subdir_path = category_dir / subdir
                subdir_path.mkdir(parents=True, exist_ok=True)
                structure[f"{category}_{subdir}"] = subdir_path
    
    return structure


def copy_file_with_new_name(source_path: Path, target_dir: Path, new_name: str) -> Path:
    """
    Copy file to target directory with a new name.
    
    Args:
        source_path: Source file path
        target_dir: Target directory
        new_name: New filename (without extension)
        
    Returns:
        Path to the copied file
    """
    import shutil
    
    source_path = Path(source_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Preserve original extension
    extension = source_path.suffix
    target_path = target_dir / f"{new_name}{extension}"
    
    shutil.copy2(source_path, target_path)
    return target_path


def save_processing_metadata(output_dir: Path, metadata: Dict[str, Any], filename: str = "metadata.json"):
    """
    Save processing metadata to JSON file.
    
    Args:
        output_dir: Output directory
        metadata: Metadata dictionary
        filename: Metadata filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = output_dir / filename
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_processing_metadata(output_dir: Path, filename: str = "metadata.json") -> Optional[Dict[str, Any]]:
    """
    Load processing metadata from JSON file.
    
    Args:
        output_dir: Output directory
        filename: Metadata filename
        
    Returns:
        Metadata dictionary or None if file doesn't exist
    """
    metadata_file = Path(output_dir) / filename
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None
