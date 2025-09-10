"""
Video Processing Utilities for VAPOR Project

This module provides unified video processing functions including:
- Video configuration extraction
- Frame extraction from videos
- Video reconstruction from frames
- Video file management

Consolidates functionality from various VAPOR modules.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import os


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


class VideoFrameExtractor:
    """Extracts frames from video files with various options."""
    
    def __init__(self, video_path: Path):
        """
        Initialize frame extractor.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        self.config = VideoConfig(video_path)
        
    def extract_frames(self, output_dir: Path, stride: int = 1, 
                      start_frame: int = 0, end_frame: Optional[int] = None) -> List[Path]:
        """
        Extract frames from video.
        
        Args:
            output_dir: Directory to save extracted frames
            stride: Extract every Nth frame (1 = all frames)
            start_frame: Starting frame number
            end_frame: Ending frame number (None = until end)
            
        Returns:
            List of extracted frame file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        video_name = self.video_path.stem
        extracted_files = []
        
        if end_frame is None:
            end_frame = self.config.frame_count
        
        frame_num = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame if it matches stride
            if (frame_num - start_frame) % stride == 0:
                frame_filename = f"{video_name}_{frame_num:06d}.png"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                extracted_files.append(frame_path)
            
            frame_num += 1
        
        cap.release()
        return extracted_files
    
    def extract_frame_generator(self, stride: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames from video.
        
        Args:
            stride: Extract every Nth frame
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % stride == 0:
                yield frame_num, frame
            
            frame_num += 1
        
        cap.release()


class VideoReconstructor:
    """Reconstructs videos from frame sequences."""
    
    def __init__(self, reference_config: VideoConfig):
        """
        Initialize with reference video configuration.
        
        Args:
            reference_config: VideoConfig object with target video settings
        """
        self.config = reference_config
    
    def create_video_from_frames(self, frame_files: List[Path], output_path: Path, 
                                fps: Optional[float] = None) -> bool:
        """
        Create video from a list of frame files.
        
        Args:
            frame_files: List of frame file paths (sorted)
            output_path: Output video file path
            fps: Frame rate (uses reference config if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not frame_files:
            return False
        
        fps = fps or self.config.fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        video_writer = None
        
        try:
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue
                
                # Initialize video writer with first frame
                if video_writer is None:
                    height, width = frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(output_path), 
                        fourcc, 
                        fps, 
                        (width, height)
                    )
                    
                    if not video_writer.isOpened():
                        return False
                
                video_writer.write(frame)
            
            return True
            
        except Exception:
            return False
        finally:
            if video_writer is not None:
                video_writer.release()
    
    def create_video_from_folder(self, frames_folder: Path, output_path: Path, 
                                fps: Optional[float] = None) -> bool:
        """
        Create video from all frames in a folder.
        
        Args:
            frames_folder: Folder containing frame images
            output_path: Output video file path
            fps: Frame rate (uses reference config if None)
            
        Returns:
            True if successful, False otherwise
        """
        from ..image import get_image_files
        
        frame_files = get_image_files(frames_folder)
        return self.create_video_from_frames(frame_files, output_path, fps)


def list_video_files(directory: Path, extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')) -> List[Path]:
    """
    List all video files in a directory.
    
    Args:
        directory: Directory to search
        extensions: Tuple of valid video extensions
        
    Returns:
        List of video file paths sorted by name
    """
    if not directory.exists():
        return []
    
    video_files = []
    seen_files = set()  # Track files we've already added to avoid duplicates
    
    for ext in extensions:
        # Find files with both lowercase and uppercase extensions
        for pattern in [f"*{ext}", f"*{ext.upper()}"]:
            for file_path in directory.glob(pattern):
                # Use the resolved path to avoid duplicates on case-insensitive filesystems
                resolved_path = file_path.resolve()
                if resolved_path not in seen_files:
                    seen_files.add(resolved_path)
                    video_files.append(file_path)
    
    return sorted(video_files)


def get_video_info(video_path: Path) -> Dict:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    try:
        config = VideoConfig(video_path)
        file_size = video_path.stat().st_size / (1024 * 1024)  # MB
        
        return {
            'name': video_path.name,
            'size_mb': round(file_size, 1),
            'resolution': f"{config.width}x{config.height}",
            'fps': round(config.fps, 2),
            'duration': round(config.duration, 2),
            'frame_count': config.frame_count
        }
    except Exception:
        return {
            'name': video_path.name,
            'error': 'Could not read video information'
        }
