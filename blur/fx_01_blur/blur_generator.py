#!/usr/bin/env python3
"""
fx_01_blur - Deterministic Blur Effects Generator

This script provides a command-line interface for generating deterministic blur effects
on images and videos. All effects use pre-generated kernels with fixed seeds to ensure
reproducible results across experiments.

Usage:
    python blur_generator.py --input <path> --output <path> --effect <type> --intensity <level>
    python blur_generator.py --input data/videos/original/pat3.mp4 --output data/videos/blurred/ --effect motion --intensity high
    python blur_generator.py --input data/extracted_frames/original/pat3/ --output data/extracted_frames/blurred/pat3/ --effect combined

Author: Kesney de Oliveira
"""

import argparse
import logging
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Any
import json
from datetime import datetime
import os

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from blur.fx_01_blur.effects.blur_engine import EnhancedBlurEffects, create_blur_effects_engine
from blur.fx_01_blur.kernels.generator import KernelConfig


def setup_logging(log_dir: Path, run_id: str) -> logging.Logger:
    """
    Setup logging for the blur generation run.
    
    Args:
        log_dir: Directory for log files
        run_id: Unique run identifier
        
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('blur_generator')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    log_file = log_dir / f"blur_generation_{run_id}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_run_metadata(log_dir: Path, run_id: str, args: argparse.Namespace, 
                     blur_engine: EnhancedBlurEffects, processing_results: List[Dict]):
    """
    Save comprehensive metadata for the run.
    
    Args:
        log_dir: Directory for log files
        run_id: Unique run identifier
        args: Command line arguments
        blur_engine: Blur effects engine
        processing_results: List of processing results for each file
    """
    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'command_args': {
            'input': str(args.input),
            'output': str(args.output),
            'effect': args.effect,
            'intensity': args.intensity,
            'seed': args.seed,
            'format': args.format
        },
        'kernel_config': {
            'seed': blur_engine.kernel_generator.config.seed,
            'motion_angle_low': blur_engine.kernel_generator.config.motion_angle_low,
            'motion_angle_high': blur_engine.kernel_generator.config.motion_angle_high,
            'motion_length_low': blur_engine.kernel_generator.config.motion_length_low,
            'motion_length_high': blur_engine.kernel_generator.config.motion_length_high,
            'gaussian_sigma_low': blur_engine.kernel_generator.config.gaussian_sigma_low,
            'gaussian_sigma_high': blur_engine.kernel_generator.config.gaussian_sigma_high,
            'defocus_radius_low': blur_engine.kernel_generator.config.defocus_radius_low,
            'defocus_radius_high': blur_engine.kernel_generator.config.defocus_radius_high,
            'haze_alpha_low': blur_engine.kernel_generator.config.haze_alpha_low,
            'haze_alpha_high': blur_engine.kernel_generator.config.haze_alpha_high
        },
        'processing_results': processing_results,
        'total_files_processed': len(processing_results),
        'successful_files': len([r for r in processing_results if r['success']]),
        'failed_files': len([r for r in processing_results if not r['success']])
    }
    
    metadata_file = log_dir / f"run_metadata_{run_id}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def process_image(image_path: Path, output_path: Path, blur_engine: EnhancedBlurEffects,
                 effect: str, intensity: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Process a single image file.
    
    Args:
        image_path: Path to input image
        output_path: Path to output image
        blur_engine: Blur effects engine
        effect: Blur effect type
        intensity: Blur intensity
        logger: Logger instance
        
    Returns:
        Processing result dictionary
    """
    result = {
        'input_file': str(image_path),
        'output_file': str(output_path),
        'effect': effect,
        'intensity': intensity,
        'success': False,
        'error': None,
        'metadata': None
    }
    
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Apply blur effect
        blurred_image, metadata = blur_engine.apply_blur_effect(image, effect, intensity)
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), blurred_image)
        
        if not success:
            raise ValueError(f"Could not save image: {output_path}")
        
        result['success'] = True
        result['metadata'] = metadata
        logger.info(f"Processed image: {image_path.name} -> {output_path.name}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Failed to process image {image_path}: {e}")
        
    return result


def process_video(video_path: Path, output_path: Path, blur_engine: EnhancedBlurEffects,
                 effect: str, intensity: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Process a single video file.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        blur_engine: Blur effects engine
        effect: Blur effect type
        intensity: Blur intensity
        logger: Logger instance
        
    Returns:
        Processing result dictionary
    """
    result = {
        'input_file': str(video_path),
        'output_file': str(output_path),
        'effect': effect,
        'intensity': intensity,
        'success': False,
        'error': None,
        'metadata': None,
        'frame_count': 0,
        'processed_frames': 0
    }
    
    try:
        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        result['frame_count'] = total_frames
        
        # Setup output video
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        # Process frames
        frame_metadata = None
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply blur effect
            blurred_frame, metadata = blur_engine.apply_blur_effect(frame, effect, intensity)
            
            # Save metadata from first frame
            if frame_metadata is None:
                frame_metadata = metadata
            
            # Write frame
            out.write(blurred_frame)
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        out.release()
        
        result['success'] = True
        result['metadata'] = frame_metadata
        result['processed_frames'] = processed_count
        logger.info(f"Processed video: {video_path.name} -> {output_path.name} ({processed_count} frames)")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Failed to process video {video_path}: {e}")
        
        # Cleanup on error
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
            
    return result


def get_files_to_process(input_path: Path, supported_formats: List[str]) -> List[Path]:
    """
    Get list of files to process from input path.
    
    Args:
        input_path: Input path (file or directory)
        supported_formats: List of supported file extensions
        
    Returns:
        List of files to process
    """
    if input_path.is_file():
        if input_path.suffix.lower() in supported_formats:
            return [input_path]
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    elif input_path.is_dir():
        files = []
        for ext in supported_formats:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(files)
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="fx_01_blur - Deterministic Blur Effects Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply motion blur (high intensity) to a video
  python blur_generator.py --input data/videos/original/pat3.mp4 --output data/videos/blurred/ --effect motion --intensity high
  
  # Apply combined blur to all images in a directory
  python blur_generator.py --input data/extracted_frames/original/pat3/ --output data/extracted_frames/blurred/pat3/ --effect combined
  
  # Apply defocus blur (low intensity) to a single image
  python blur_generator.py --input image.jpg --output blurred_image.jpg --effect defocus --intensity low

Supported effects: motion, gaussian, defocus, haze, combined
Supported intensities: low, high (combined blur uses low intensity for all components)
        """
    )
    
    parser.add_argument('--input', '-i', type=Path, required=True,
                       help='Input file or directory path')
    parser.add_argument('--output', '-o', type=Path, required=True,
                       help='Output file or directory path')
    parser.add_argument('--effect', '-e', type=str, required=True,
                       choices=['motion', 'gaussian', 'defocus', 'haze', 'combined'],
                       help='Blur effect type')
    parser.add_argument('--intensity', '-int', type=str, default='low',
                       choices=['low', 'high'],
                       help='Blur intensity (ignored for combined effect)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for deterministic generation')
    parser.add_argument('--format', '-f', type=str, default='auto',
                       choices=['auto', 'image', 'video'],
                       help='Force processing format (auto-detects by default)')
    
    return parser.parse_args()


def determine_process_mode(input_path: Path, format_arg: str, logger):
    """Determine the processing mode and supported formats."""
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    if format_arg != 'auto':
        process_mode = format_arg
        supported_formats = image_formats if process_mode == 'image' else video_formats
        return process_mode, supported_formats
    
    # Auto-detect based on file extensions
    if input_path.is_file():
        if input_path.suffix.lower() in image_formats:
            return 'image', image_formats
        elif input_path.suffix.lower() in video_formats:
            return 'video', video_formats
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Directory - check what files are present
    all_files = list(input_path.iterdir())
    image_files = [f for f in all_files if f.suffix.lower() in image_formats]
    video_files = [f for f in all_files if f.suffix.lower() in video_formats]
    
    if image_files and not video_files:
        return 'image', image_formats
    elif video_files and not image_files:
        return 'video', video_formats
    elif image_files and video_files:
        logger.warning("Mixed file types detected, defaulting to image processing")
        return 'image', image_formats
    else:
        raise ValueError("No supported files found in directory")


def generate_output_path(input_file: Path, output_base: Path, effect: str, intensity: str, 
                        is_multiple_files: bool) -> Path:
    """Generate output path for a file."""
    if output_base.is_dir() or (not output_base.exists() and is_multiple_files):
        # Output directory
        output_base.mkdir(parents=True, exist_ok=True)
        
        stem = input_file.stem
        suffix = input_file.suffix
        
        if effect == 'combined':
            output_name = f"{stem}_{effect}{suffix}"
        else:
            output_name = f"{stem}_{effect}_{intensity}{suffix}"
        
        return output_base / output_name
    else:
        return output_base


def process_files_batch(files_to_process: List[Path], args, blur_engine, 
                       process_mode: str, logger) -> List[Dict[str, Any]]:
    """Process a batch of files."""
    processing_results = []
    
    for i, input_file in enumerate(files_to_process, 1):
        logger.info(f"Processing file {i}/{len(files_to_process)}: {input_file.name}")
        
        output_file = generate_output_path(
            input_file, args.output, args.effect, args.intensity,
            len(files_to_process) > 1
        )
        
        if process_mode == 'image':
            result = process_image(input_file, output_file, blur_engine, 
                                 args.effect, args.intensity, logger)
        else:
            result = process_video(input_file, output_file, blur_engine,
                                 args.effect, args.intensity, logger)
        
        processing_results.append(result)
    
    return processing_results


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate paths
    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return 1
    
    # Generate run ID and setup logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "runs" / run_id
    logger = setup_logging(log_dir, run_id)
    
    logger.info(f"Starting blur generation run: {run_id}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Effect: {args.effect}, Intensity: {args.intensity}")
    logger.info(f"Seed: {args.seed}")
    
    try:
        # Create blur engine
        config = KernelConfig(seed=args.seed)
        blur_engine = EnhancedBlurEffects(config)
        
        # Determine processing mode
        process_mode, supported_formats = determine_process_mode(
            args.input, args.format, logger
        )
        logger.info(f"Processing mode: {process_mode}")
        
        # Get files to process
        files_to_process = get_files_to_process(args.input, supported_formats)
        
        if not files_to_process:
            logger.error("No files to process found")
            return 1
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files
        processing_results = process_files_batch(
            files_to_process, args, blur_engine, process_mode, logger
        )
        
        # Save run metadata
        save_run_metadata(log_dir, run_id, args, blur_engine, processing_results)
        
        # Summary
        successful = len([r for r in processing_results if r['success']])
        failed = len([r for r in processing_results if not r['success']])
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        logger.info(f"Run metadata saved to: {log_dir}")
        
        if failed > 0:
            logger.warning("Some files failed to process. Check the log for details.")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
