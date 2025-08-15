"""
Batch Blur Processor
Applies blur effects to all available folders or specific folders in batch.

Usage:
    python batch_blur.py [--folders <folder1> <folder2> ...] [--intensity <low|high>] [--blur_types <types>]
"""

import sys
import argparse
from pathlib import Path
from blur_effects import BlurEffects


def main():
    """Main function for batch blur processing."""
    parser = argparse.ArgumentParser(
        description="Apply blur effects to multiple folders in batch"
    )
    parser.add_argument(
        "--folders", 
        nargs='+',
        help="Specific folders to process (default: all available folders)"
    )
    parser.add_argument(
        "--blur_types", 
        nargs='+',
        choices=["gaussian", "motion_horizontal", "motion_diagonal", 
                "outoffocus", "average", "median"],
        help="Types of blur to apply (default: all types)"
    )
    parser.add_argument(
        "--intensity", 
        choices=["low", "high"],
        default="low",
        help="Blur intensity level (default: low)"
    )
    parser.add_argument(
        "--list_folders", 
        action="store_true",
        help="List available folders and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Create blur effects processor
        blur_processor = BlurEffects()
        
        # List folders if requested
        if args.list_folders:
            folders = blur_processor.list_available_folders()
            if folders:
                print("Available folders:")
                for folder in folders:
                    print(f"  - {folder}")
            else:
                print("No folders found in data/extracted_frames/original/")
            return 0
        
        # Determine which folders to process
        if args.folders:
            folders_to_process = args.folders
        else:
            folders_to_process = blur_processor.list_available_folders()
            if not folders_to_process:
                print("No folders found in data/extracted_frames/original/")
                return 1
        
        print(f"Processing {len(folders_to_process)} folders with {args.intensity} intensity blur...")
        if args.blur_types:
            print(f"Blur types: {', '.join(args.blur_types)}")
        else:
            print("Blur types: all types")
        print()
        
        # Process each folder
        success_count = 0
        for i, folder in enumerate(folders_to_process, 1):
            print(f"[{i}/{len(folders_to_process)}] Processing folder: {folder}")
            
            success = blur_processor.process_folder(
                folder, 
                args.blur_types, 
                args.intensity
            )
            
            if success:
                success_count += 1
                print(f"Completed: {folder}")
            else:
                print(f"Failed: {folder}")

            print("-" * 50)
        
        # Summary
        print(f"\nBatch processing completed!")
        print(f"Successfully processed: {success_count}/{len(folders_to_process)} folders")
        
        if success_count == len(folders_to_process):
            print("✓ All folders processed successfully!")
            return 0
        else:
            print(f"✗ {len(folders_to_process) - success_count} folders failed")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
