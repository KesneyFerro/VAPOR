#!/usr/bin/env python
"""
Example script showing how to use the VAPOR reconstruction pipeline for PNG folder processing.

This script demonstrates how to run 3D reconstruction on a folder containing PNG images.
"""

import sys
from pathlib import Path

# Add the reconstruction directory to path
sys.path.append(str(Path(__file__).parent))

def run_reconstruction_example():
    """Example of running reconstruction on pat3 folder."""
    
    # Example folder path (adjust as needed)
    folder_path = r"S:\Kesney\VAPOR\data\frames\original\pat3"
    
    print("VAPOR Reconstruction Pipeline Example")
    print("=" * 50)
    print(f"Target folder: {folder_path}")
    
    # Check if folder exists
    if not Path(folder_path).exists():
        print(f"ERROR: Folder does not exist: {folder_path}")
        return False
        
    # Count PNG files
    png_files = list(Path(folder_path).glob("*.png"))
    print(f"Found {len(png_files)} PNG files")
    
    if len(png_files) == 0:
        print("ERROR: No PNG files found in the folder")
        return False
    
    print("\nTo run the reconstruction pipeline:")
    print(f'python reconstruction_pipeline.py --folder "{folder_path}"')
    print("\nOptional parameters:")
    print("  --feature disk          # Feature detector (disk, superpoint, aliked-n16, sift)")
    print("  --matcher disk+lightglue # Feature matcher (disk+lightglue, superglue, etc.)")
    
    print("\nExample with custom settings:")
    print(f'python reconstruction_pipeline.py --folder "{folder_path}" --feature superpoint --matcher superglue')
    
    print("\nNote: Make sure to install required dependencies first:")
    print("  pip install h5py numpy opencv-python")
    print("  # And follow the hloc/maploc setup instructions")
    
    return True

if __name__ == "__main__":
    run_reconstruction_example()