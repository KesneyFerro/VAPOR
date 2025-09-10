"""
Pipeline Utilities for VAPOR Project

Common pipeline operations and utilities that can be shared across different
VAPOR processing pipelines.
"""

import sys
from pathlib import Path
from typing import List, Optional


class VideoSelector:
    """Interactive video selection utility."""
    
    def __init__(self, videos_directory: Path):
        """
        Initialize video selector.
        
        Args:
            videos_directory: Directory containing video files
        """
        self.videos_directory = Path(videos_directory)
    
    def list_available_videos(self) -> List[Path]:
        """List all available videos in the directory."""
        from .video import list_video_files
        return list_video_files(self.videos_directory)
    
    def select_video_interactive(self) -> Optional[Path]:
        """
        Interactive video selection with user prompts.
        
        Returns:
            Selected video path or None if cancelled
        """
        from .video import get_video_info
        
        videos = self.list_available_videos()
        
        if not videos:
            print(f"[ERROR] No videos found in {self.videos_directory}")
            return None
        
        print("\nAvailable videos:")
        print("=" * 50)
        
        for i, video in enumerate(videos, 1):
            info = get_video_info(video)
            if 'error' in info:
                print(f"{i:2d}. {info['name']} (Error reading file)")
            else:
                print(f"{i:2d}. {info['name']} ({info['size_mb']} MB, "
                      f"{info['resolution']}, {info['duration']}s)")
        
        while True:
            try:
                choice = input(f"\nSelect video (1-{len(videos)}): ").strip()
                if not choice:
                    continue
                
                index = int(choice) - 1
                if 0 <= index < len(videos):
                    return videos[index]
                else:
                    print(f"[ERROR] Please enter a number between 1 and {len(videos)}")
            except ValueError:
                print("[ERROR] Please enter a valid number")
            except KeyboardInterrupt:
                print("\n[CANCELLED] Operation cancelled by user")
                return None


class ProcessingModeSelector:
    """Utility for selecting processing modes and options."""
    
    @staticmethod
    def select_stride_interactive(current_stride: int = 1) -> int:
        """
        Interactive stride selection for frame processing.
        
        Args:
            current_stride: Current stride value
            
        Returns:
            Selected stride value
        """
        if current_stride != 1:
            return current_stride  # Already set via command line
        
        print("\nProcessing options:")
        print("1. Full video (all frames) - SLOW but complete")
        print("2. Sample frames (every 10th frame) - FAST for testing")
        print("3. Sample frames (every 5th frame) - MEDIUM")
        print("4. Custom stride - specify your own")
        
        while True:
            try:
                mode = input("\nChoose processing mode (1-4): ").strip()
                if mode == '1':
                    return 1  # Process all frames
                elif mode == '2':
                    return 10  # Every 10th frame
                elif mode == '3':
                    return 5   # Every 5th frame
                elif mode == '4':
                    while True:
                        try:
                            custom_stride = int(input("Enter custom stride (1-50): "))
                            if 1 <= custom_stride <= 50:
                                return custom_stride
                            else:
                                print("Please enter a number between 1 and 50")
                        except ValueError:
                            print("Please enter a valid number")
                else:
                    print("Please enter 1, 2, 3, or 4")
            except KeyboardInterrupt:
                print("\n[CANCELLED] Using default stride")
                return 1


def setup_project_paths(base_path: Optional[Path] = None) -> dict:
    """
    Setup standard VAPOR project paths.
    
    Args:
        base_path: Base project path (auto-detected if None)
        
    Returns:
        Dictionary with standard project paths
    """
    if base_path is None:
        # Try to auto-detect from current file location
        current_file = Path(__file__)
        base_path = current_file.parent.parent.parent  # Go up from core/
    
    base_path = Path(base_path)
    
    return {
        'base': base_path,
        'data': base_path / "data",
        'videos_original': base_path / "data" / "videos" / "original",
        'videos_blurred': base_path / "data" / "videos" / "blurred", 
        'videos_deblurred': base_path / "data" / "videos" / "deblurred",
        'frames_base': base_path / "data" / "extracted_frames",
        'frames_original': base_path / "data" / "extracted_frames" / "original",
        'frames_blurred': base_path / "data" / "extracted_frames" / "blurred",
        'frames_deblurred': base_path / "data" / "extracted_frames" / "deblurred",
    }


def ensure_directories_exist(paths: dict) -> None:
    """
    Ensure all directories in the paths dictionary exist.
    
    Args:
        paths: Dictionary of paths (from setup_project_paths)
    """
    for path_name, path_obj in paths.items():
        if isinstance(path_obj, Path):
            path_obj.mkdir(parents=True, exist_ok=True)


def add_project_to_path(base_path: Optional[Path] = None) -> None:
    """
    Add project modules to Python path for imports.
    
    Args:
        base_path: Base project path (auto-detected if None)
    """
    if base_path is None:
        current_file = Path(__file__)
        base_path = current_file.parent.parent.parent
    
    base_path = Path(base_path)
    
    # Add base path to sys.path if not already there
    base_str = str(base_path)
    if base_str not in sys.path:
        sys.path.insert(0, base_str)
