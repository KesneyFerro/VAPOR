"""
Video filename normalization utility.
Normalizes video filenames by replacing spaces with underscores and removing illegal characters.
"""

import os
import re
import string
from pathlib import Path


def normalize_filename(filename):
    """
    Normalize a filename by:
    - Replacing spaces with underscores
    - Removing illegal filesystem characters
    - Ensuring proper extension handling
    - Converting to lowercase for consistency
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Normalized filename
    """
    # Get the file extension
    name, ext = os.path.splitext(filename)
    
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    
    # Remove illegal characters for filesystem
    # Keep only alphanumeric, underscores, hyphens, and dots
    illegal_chars = r'[<>:"/\\|?*]'
    name = re.sub(illegal_chars, '', name)
    
    # Remove any remaining problematic characters
    valid_chars = f"-_.{string.ascii_letters}{string.digits}"
    name = ''.join(c for c in name if c in valid_chars)
    
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Convert to lowercase for consistency
    name = name.lower()
    
    # Ensure extension is lowercase and valid
    ext = ext.lower()
    
    # If no extension, don't add one
    if not ext:
        return name
    
    return f"{name}{ext}"


def normalize_videos_in_directory(directory_path):
    """
    Normalize all video filenames in a directory.
    
    Args:
        directory_path (str): Path to the directory containing videos
        
    Returns:
        dict: Mapping of old filenames to new filenames
    """
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory {directory_path} does not exist")
        return {}
    
    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    renamed_files = {}
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            original_name = file_path.name
            normalized_name = normalize_filename(original_name)
            
            if original_name != normalized_name:
                new_path = file_path.parent / normalized_name
                
                # Avoid overwriting existing files
                counter = 1
                while new_path.exists():
                    name, ext = os.path.splitext(normalized_name)
                    new_path = file_path.parent / f"{name}_{counter}{ext}"
                    counter += 1
                    normalized_name = new_path.name
                
                try:
                    file_path.rename(new_path)
                    renamed_files[original_name] = normalized_name
                    print(f"Renamed: '{original_name}' -> '{normalized_name}'")
                except OSError as e:
                    print(f"Error renaming '{original_name}': {e}")
    
    if renamed_files:
        print(f"\nNormalized {len(renamed_files)} video filename(s)")
    
    return renamed_files


if __name__ == "__main__":
    # Test the normalization
    test_names = [
        "My Video File.mp4",
        "Test Video (1080p).avi",
        "Video With Spaces & Special Chars!.mkv",
        "normal_video.mp4"
    ]
    
    print("Testing filename normalization:")
    for name in test_names:
        normalized = normalize_filename(name)
        print(f"'{name}' -> '{normalized}'")
