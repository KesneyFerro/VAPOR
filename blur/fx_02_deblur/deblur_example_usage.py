#!/usr/bin/env python3
"""
VAPOR fx_02_deblur - Deblurring Example Usage

Example script showing how to use different deblurring methods with automatic environment switching.
NOTE: The main pipeline (vapor_complete_pipeline.py) uses direct module imports instead of this CLI.

Supported methods:
- DeblurGAN-v2
- Restormer
- Uformer  
- DPIR
- MPRNet

Usage:
    python deblur_example_usage.py --input <path> --output <path> --method <method>
    
Examples:
    # Deblur with specific method
    python deblur_example_usage.py --input blurred_image.jpg --output deblurred/ --method Restormer
    
    # Deblur with all methods
    python deblur_example_usage.py --input blurred_folder/ --output results/ --all-methods
    
    # Deblur single frame
    python deblur_example_usage.py --input frame_001.jpg --output deblurred_frame.jpg --method DeblurGANv2

Author: Kesney de Oliveira
"""

import argparse
import subprocess
import sys
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import cv2
import numpy as np

def monitor_gpu_usage(stop_event, interval=2):
    """Monitor GPU usage in a separate thread."""
    while not stop_event.is_set():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                if len(gpu_data) == 4:
                    gpu_util, mem_used, mem_total, temp = gpu_data
                    print(f"GPU: {gpu_util}% utilization, {mem_used}/{mem_total}MB memory, {temp}°C", end='\r')
        except Exception:
            pass
        time.sleep(interval)

# Check for CUDA/GPU availability
def check_gpu_availability() -> Dict[str, any]:
    """Check if GPU is available for deep learning frameworks."""
    gpu_info = {
        'cuda_available': False,
        'torch_cuda': False,
        'tensorflow_gpu': False,
        'gpu_count': 0,
        'gpu_names': []
    }
    
    try:
        # Check PyTorch CUDA
        import torch
        gpu_info['torch_cuda'] = torch.cuda.is_available()
        if gpu_info['torch_cuda']:
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(gpu_info['gpu_count'])]
            gpu_info['cuda_available'] = True
    except ImportError:
        pass
    
    try:
        # Check TensorFlow GPU
        import tensorflow as tf
        gpu_devices = tf.config.list_physical_devices('GPU')
        gpu_info['tensorflow_gpu'] = len(gpu_devices) > 0
        if gpu_info['tensorflow_gpu']:
            gpu_info['cuda_available'] = True
            if not gpu_info['gpu_count']:
                gpu_info['gpu_count'] = len(gpu_devices)
    except ImportError:
        pass
    
    return gpu_info


# Method configurations
DEBLUR_METHODS = {
    "DeblurGANv2": {
        "conda_env": "deblurgan_v2",
        "folder": "DeblurGANv2", 
        "script": "predict.py",
        "model_path": "fpn_inception.h5",  # Correct model filename
        "input_arg": "", # Uses positional arguments, not flags
        "output_arg": "",
        "extra_args": [],
        "framework": "tensorflow",
        "supports_gpu": True,
        "custom_command_builder": True  # Uses custom command building logic
    },
    "Restormer": {
        "conda_env": "restormer",
        "folder": "Restormer",
        "script": "demo.py", 
        "model_path": "Motion_Deblurring/pretrained_models/", 
        "input_arg": "--input_dir",
        "output_arg": "--result_dir", 
        "extra_args": ["--task", "Motion_Deblurring", "--tile", "512", "--tile_overlap", "32"],
        "framework": "pytorch",
        "supports_gpu": True
    },
    "Uformer": {
        "conda_env": "uformer",
        "folder": "Uformer",
        "script": "test/test_gopro_hide.py",
        "model_path": "saved_models/",
        "input_arg": "--input_dir", 
        "output_arg": "--result_dir",
        "extra_args": ["--gpus", "0", "--weights", "saved_models/Uformer_B.pth"],  # Add weights path
        "framework": "pytorch",
        "supports_gpu": True,
        "isolated_pythonpath": True  # Run with isolated PYTHONPATH
    },
    "DPIR": {
        "conda_env": "dpir",
        "folder": "DPIR",
        "script": "main_dpir_deblur.py", 
        "model_path": "model_zoo/",
        "input_arg": "--input", 
        "output_arg": "--output",
        "extra_args": ["--gpu_ids", "0"],
        "framework": "pytorch",
        "supports_gpu": True,
        "isolated_pythonpath": True  # Run with isolated PYTHONPATH
    },
    "MPRNet": {
        "conda_env": "mprnet", 
        "folder": "MPRNet",
        "script": "demo.py",
        "model_path": "Deblurring/pretrained_models/",
        "input_arg": "--input_dir",
        "output_arg": "--result_dir", 
        "extra_args": ["--task", "Deblurring"],
        "framework": "pytorch",
        "supports_gpu": True,
        "custom_model_setup": True
    }
}


def setup_logging(log_dir: Path, run_id: str) -> logging.Logger:
    """Setup logging for the deblur run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('deblur_cli')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = log_dir / f"deblur_run_{run_id}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def check_environment_setup(base_dir: Path) -> Dict:
    """Check if environments are properly set up."""
    env_info_file = base_dir / "environment_info.json"
    
    if not env_info_file.exists():
        raise FileNotFoundError(
            "Environment info file not found. Please run setup_repositories.py first."
        )
    
    with open(env_info_file, 'r') as f:
        return json.load(f)


def setup_gpu_environment(method_config: Dict, logger: logging.Logger, force_gpu: bool = True) -> Dict[str, str]:
    """Setup environment variables for GPU processing with optimization."""
    env_vars = {}
    
    # Handle isolated PYTHONPATH for methods that need it
    if method_config.get('isolated_pythonpath', False):
        # Remove VAPOR from PYTHONPATH to avoid import conflicts
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        if current_pythonpath:
            # Filter out VAPOR paths
            paths = current_pythonpath.split(os.pathsep)
            filtered_paths = [p for p in paths if 'VAPOR' not in p]
            env_vars['PYTHONPATH'] = os.pathsep.join(filtered_paths)
            logger.info("Using isolated PYTHONPATH to avoid import conflicts")
        else:
            env_vars['PYTHONPATH'] = ''
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    
    if gpu_info['cuda_available'] and method_config.get('supports_gpu', False):
        logger.info(f"GPU detected: {gpu_info['gpu_count']} device(s)")
        for i, name in enumerate(gpu_info['gpu_names']):
            logger.info(f"  GPU {i}: {name}")
        
        # Force GPU usage with optimized settings
        env_vars['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        env_vars['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async execution
        env_vars['CUDA_CACHE_DISABLE'] = '0'    # Enable CUDA cache
        
        if method_config['framework'] == 'pytorch':
            env_vars['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,expandable_segments:True'
            env_vars['TORCH_CUDA_ARCH_LIST'] = 'Auto'  # Auto-detect architecture
            # Force PyTorch to use GPU with optimizations
            env_vars['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            env_vars['TORCH_CUDNN_BENCHMARK'] = '1'  # Optimize for consistent input sizes
            env_vars['TORCH_BACKENDS_CUDNN_DETERMINISTIC'] = '0'  # Allow non-deterministic for speed
            env_vars['OMP_NUM_THREADS'] = '4'  # Limit CPU threads for GPU focus
            env_vars['MKL_NUM_THREADS'] = '4'
            # Suppress PyTorch warnings
            env_vars['PYTORCH_NO_WARN_EXPANDABLE_SEGMENTS'] = '1'  # Suppress expandable_segments warning
            env_vars['PYTHONWARNINGS'] = 'ignore::UserWarning'    # Suppress torch.load warnings
            
        elif method_config['framework'] == 'tensorflow':
            env_vars['TF_GPU_MEMORY_GROWTH'] = '1'
            env_vars['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            env_vars['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging
            # Force TensorFlow to use GPU
            env_vars['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        logger.info("GPU processing enabled with optimized settings")
        
    elif force_gpu:
        logger.error("GPU acceleration requested but not available!")
        if not gpu_info['cuda_available']:
            logger.error("CUDA is not available on this system")
        if not method_config.get('supports_gpu', False):
            logger.error(f"Method {method_config.get('folder', 'unknown')} does not support GPU")
        # Still try to use GPU settings in case detection failed
        env_vars['CUDA_VISIBLE_DEVICES'] = '0'
        
    else:
        logger.warning("GPU not available or disabled - using CPU")
        env_vars['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
    
    return env_vars


def process_single_frame(input_file: Path, output_file: Path, method_name: str, 
                        method_config: Dict, method_dir: Path, temp_dir: Path, 
                        logger: logging.Logger) -> bool:
    """
    Process a single frame through a deblur method with immediate saving.
    
    Args:
        input_file: Path to input image file
        output_file: Path where output should be saved
        method_name: Name of the deblur method
        method_config: Method configuration dictionary
        method_dir: Directory containing the method
        temp_dir: Temporary directory for processing
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create temporary input/output directories for this frame
        frame_temp_dir = temp_dir / f"frame_{input_file.stem}"
        frame_input_dir = frame_temp_dir / "input"
        frame_output_dir = frame_temp_dir / "output"
        
        frame_input_dir.mkdir(parents=True, exist_ok=True)
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy input file to temp input directory
        temp_input_file = frame_input_dir / input_file.name
        import shutil
        shutil.copy2(input_file, temp_input_file)
        
        # Handle special directory structures for certain methods
        if method_name == "Uformer":
            # Uformer expects input/groundtruth subdirectories
            uformer_input_dir = frame_input_dir / "input"
            uformer_input_dir.mkdir(parents=True, exist_ok=True)
            # Move the file to the input subdirectory
            uformer_input_file = uformer_input_dir / input_file.name
            shutil.move(temp_input_file, uformer_input_file)
            
            # Create empty groundtruth directory (required by Uformer dataset loader)
            uformer_gt_dir = frame_input_dir / "groundtruth"
            uformer_gt_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command for single frame processing
        script_path = method_dir / method_config["script"]
        if not script_path.exists():
            possible_scripts = list(method_dir.rglob(method_config["script"]))
            if possible_scripts:
                script_path = possible_scripts[0]
            else:
                raise FileNotFoundError(f"Script not found: {method_config['script']}")
        
        # Build command arguments
        if method_config.get("custom_command_builder"):
            # Custom command building for special methods
            if method_name == "DeblurGANv2":
                # DeblurGANv2 uses Fire CLI with positional arguments
                input_pattern = str(frame_input_dir / "*.png")
                cmd_parts = [
                    "python", str(script_path),
                    f'"{input_pattern}"',  # img_pattern
                    "--mask_pattern", "None",
                    "--weights_path", "fpn_inception.h5",
                    "--out_dir", f'"{frame_output_dir}"',
                    "--side_by_side", "False",
                    "--video", "False"  # Explicitly disable video processing
                ]
            else:
                # Fallback for other custom methods
                cmd_parts = [
                    "python", str(script_path),
                    method_config["input_arg"], str(frame_input_dir),
                    method_config["output_arg"], str(frame_output_dir)
                ]
        else:
            # Standard command building
            cmd_parts = [
                "python", str(script_path),
                method_config["input_arg"], str(frame_input_dir),
                method_config["output_arg"], str(frame_output_dir)
            ]
        
        # Add extra arguments
        cmd_parts.extend(method_config["extra_args"])
        
        command = " ".join(cmd_parts)
        
        # Setup GPU environment with force flag
        gpu_env = setup_gpu_environment(method_config, logger, force_gpu=True)
        
        # Run command with GPU environment
        success, stdout, stderr = run_conda_command(
            command, method_config["conda_env"], 
            cwd=method_dir, timeout=300,  # Shorter timeout for single frame
            extra_env=gpu_env
        )
        
        if success:
            # Find and copy the output file immediately
            output_files = list(frame_output_dir.rglob("*.png")) + list(frame_output_dir.rglob("*.jpg"))
            if output_files:
                # Copy the first output file to the final location
                output_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_files[0], output_file)
                logger.info(f"[OK] Saved: {output_file.name}")
                
                # Clean up temp directory for this frame
                shutil.rmtree(frame_temp_dir)
                return True
            else:
                logger.warning(f"No output generated for {input_file.name}")
                return False
        else:
            logger.error(f"Processing failed for {input_file.name}: {stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing single frame {input_file.name}: {e}")
        return False


def run_conda_command(command: str, conda_env: str, cwd: Path = None, 
                     timeout: int = 1800, extra_env: Dict[str, str] = None) -> Tuple[bool, str, str]:
    """
    Run a command in a specific conda environment.
    
    Args:
        command: Command to run
        conda_env: Conda environment name
        cwd: Working directory
        timeout: Command timeout in seconds
        extra_env: Additional environment variables
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    if sys.platform == "win32":
        full_command = f"conda activate {conda_env} && {command}"
        shell_cmd = ["cmd", "/c", full_command]
    else:
        full_command = f"conda activate {conda_env} && {command}"
        shell_cmd = ["bash", "-c", full_command]
    
    # Prepare environment with extra variables
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    
    try:
        # Start GPU monitoring if GPU is being used
        stop_monitoring = threading.Event()
        gpu_monitor_thread = None
        
        if extra_env and any('CUDA' in key for key in extra_env.keys()):
            print(f"\n[GPU MONITORING] Starting GPU monitoring for {conda_env}...")
            gpu_monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(stop_monitoring,))
            gpu_monitor_thread.daemon = True
            gpu_monitor_thread.start()
        
        # Use Popen for real-time output streaming
        print(f"\n[COMMAND] {full_command}")
        process = subprocess.Popen(
            shell_cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        stdout_lines = []
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())  # Print to console immediately
            stdout_lines.append(line)
        
        process.wait(timeout=timeout)
        stdout = ''.join(stdout_lines)
        
        # Stop GPU monitoring
        if gpu_monitor_thread:
            stop_monitoring.set()
            gpu_monitor_thread.join(timeout=1)
            print()  # New line after GPU monitoring
        
        return process.returncode == 0, stdout, ""
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def prepare_input_for_method(input_path: Path, method_config: Dict, temp_dir: Path) -> Path:
    """
    Prepare input for a specific method (handle different input format requirements).
    
    Args:
        input_path: Original input path
        method_config: Method configuration
        temp_dir: Temporary directory for processing
        
    Returns:
        Prepared input path
    """
    method_folder = method_config.get('folder', '')
    
    # Most methods expect directory input, some handle single files
    if input_path.is_file():
        # Create temp directory and copy file
        method_input_dir = temp_dir / f"input_{method_folder}"
        method_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file to temp directory
        import shutil
        temp_file = method_input_dir / input_path.name
        shutil.copy2(input_path, temp_file)
        
        # Handle special directory structures
        if method_folder == "Uformer":
            # Uformer expects input/groundtruth subdirectories
            uformer_input_dir = method_input_dir / "input"
            uformer_input_dir.mkdir(parents=True, exist_ok=True)
            # Move the file to the input subdirectory
            uformer_input_file = uformer_input_dir / input_path.name
            shutil.move(temp_file, uformer_input_file)
            
            # Create empty groundtruth directory (required by Uformer dataset loader)
            uformer_gt_dir = method_input_dir / "groundtruth"
            uformer_gt_dir.mkdir(parents=True, exist_ok=True)
        
        return method_input_dir
    else:
        # Directory input
        if method_folder == "Uformer":
            # For Uformer, need to restructure the directory
            method_input_dir = temp_dir / f"input_{method_folder}"
            method_input_dir.mkdir(parents=True, exist_ok=True)
            
            # Create input subdirectory and copy all files
            uformer_input_dir = method_input_dir / "input"
            uformer_input_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            # Copy all image files to input subdirectory
            for file in input_path.iterdir():
                if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    shutil.copy2(file, uformer_input_dir / file.name)
            
            # Create empty groundtruth directory
            uformer_gt_dir = method_input_dir / "groundtruth"
            uformer_gt_dir.mkdir(parents=True, exist_ok=True)
            
            return method_input_dir
        else:
            # Use directory as is for other methods
            return input_path


def ensure_model_files(base_dir: Path, logger: logging.Logger) -> bool:
    """
    Ensure that pretrained model files are in the correct locations for current methods.
    Only checks for available methods: Restormer, MPRNet, Uformer
    
    Args:
        base_dir: Base directory containing method folders
        logger: Logger instance
        
    Returns:
        True if all models are available, False otherwise
    """
    try:
        import shutil  # Import shutil at the beginning of the function
        
        # VAPOR pretrained models directory
        vapor_models_dir = base_dir.parent / "pretrained_models"
        
        # Only check for current available methods
        success = True
        
        # Restormer model setup
        restormer_source = vapor_models_dir / "restormer" / "motion_deblurring.pth"
        restormer_dest = base_dir / "Restormer" / "Motion_Deblurring" / "pretrained_models" / "motion_deblurring.pth"
        
        if restormer_source.exists() and not restormer_dest.exists():
            logger.info(f"Copying Restormer model: {restormer_source} -> {restormer_dest}")
            restormer_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(restormer_source, restormer_dest)
            logger.info("Restormer model copied successfully")
        elif restormer_dest.exists():
            logger.info("Restormer model already in place")
        elif not restormer_source.exists():
            logger.warning(f"Restormer model not found: {restormer_source}")
            success = False
            
        # MPRNet model setup (corrected filename)
        mprnet_source = vapor_models_dir / "mprnet" / "mprnet_deblur.pth"  # Source filename
        mprnet_dest = base_dir / "MPRNet" / "Deblurring" / "pretrained_models" / "model_deblurring.pth"  # Expected filename by demo.py
        
        if mprnet_source.exists() and not mprnet_dest.exists():
            logger.info(f"Copying MPRNet model: {mprnet_source} -> {mprnet_dest}")
            mprnet_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mprnet_source, mprnet_dest)
            logger.info("MPRNet model copied successfully")
        elif mprnet_dest.exists():
            logger.info("MPRNet model already in place")
        elif not mprnet_source.exists():
            logger.warning(f"MPRNet model not found: {mprnet_source}")
            success = False
        
        # Uformer model setup
        uformer_source = vapor_models_dir / "uformer" / "Uformer_B.pth"
        uformer_dest = base_dir / "Uformer" / "saved_models" / "Uformer_B.pth"
        
        if uformer_source.exists() and not uformer_dest.exists():
            logger.info(f"Copying Uformer model: {uformer_source} -> {uformer_dest}")
            uformer_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(uformer_source, uformer_dest)
            logger.info("Uformer model copied successfully")
        elif uformer_dest.exists():
            logger.info("Uformer model already in place")
        elif not uformer_source.exists():
            logger.warning(f"Uformer model not found: {uformer_source}")
            success = False
        
        # Log final status
        if success:
            logger.info("All required model files are available")
        else:
            logger.warning("Some model files are missing, but continuing with available methods")
            
        # Return True even if some models are missing - let individual methods handle their requirements
        return True
        
    except Exception as e:
        logger.error(f"Error setting up model files: {e}")
        return False


def run_deblur_method(method_name: str, input_path: Path, output_path: Path, 
                     base_dir: Path, temp_dir: Path, logger: logging.Logger,
                     blur_method: str = None, save_as_you_go: bool = False) -> Dict:
    """
    Run a specific deblurring method.
    
    Args:
        method_name: Name of the deblurring method
        input_path: Input path (file or directory)
        output_path: Output path 
        base_dir: Base directory containing method folders
        temp_dir: Temporary directory for processing
        logger: Logger instance
        
    Returns:
        Result dictionary with success status and metadata
    """
    result = {
        'method': method_name,
        'success': False,
        'error': None,
        'input_path': str(input_path),
        'output_path': str(output_path),
        'processing_time': 0
    }
    
    try:
        if method_name not in DEBLUR_METHODS:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Auto-detect blur method from input path if not provided
        if blur_method is None:
            # Try to extract blur method from the input path structure
            # Expected structure: .../blurred/{video_name}/{blur_method}/
            path_parts = input_path.parts
            if 'blurred' in path_parts:
                blur_idx = path_parts.index('blurred')
                if blur_idx + 2 < len(path_parts):
                    blur_method = path_parts[blur_idx + 2]
                    logger.info(f"Auto-detected blur method: {blur_method}")
                else:
                    logger.warning("Could not auto-detect blur method from path structure")
            else:
                logger.info("No blur method specified and path doesn't contain 'blurred' - using fallback structure")
        
        config = DEBLUR_METHODS[method_name]
        method_dir = base_dir / config["folder"]
        
        if not method_dir.exists():
            raise FileNotFoundError(f"Method directory not found: {method_dir}")
        
        logger.info(f"Running {method_name} deblurring...")
        
        # Check if input is a directory with multiple frames and save_as_you_go is enabled
        if save_as_you_go and input_path.is_dir():
            # Frame-by-frame processing with immediate saving
            image_files = [f for f in input_path.iterdir() 
                          if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            
            if not image_files:
                raise ValueError("No image files found in input directory")
            
            # Prepare final output directory with new structure: {output_path}/{method_name}/{blur_method}/
            if blur_method:
                final_output = output_path / method_name / blur_method
            else:
                # Fallback for backward compatibility
                final_output = output_path / method_name
            final_output.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing {len(image_files)} frames with save-as-you-go...")
            
            successful_frames = 0
            failed_frames = []
            
            for i, image_file in enumerate(sorted(image_files)):
                logger.info(f"Processing frame {i+1}/{len(image_files)}: {image_file.name}")
                
                output_file = final_output / image_file.name
                
                frame_success = process_single_frame(
                    image_file, output_file, method_name, config, 
                    method_dir, temp_dir, logger
                )
                
                if frame_success:
                    successful_frames += 1
                else:
                    failed_frames.append(image_file.name)
            
            if successful_frames > 0:
                result['success'] = True
                result['output_path'] = str(final_output)
                result['frames_processed'] = successful_frames
                result['frames_failed'] = len(failed_frames)
                
                if failed_frames:
                    logger.warning(f"Failed to process {len(failed_frames)} frames: {failed_frames[:5]}...")
                    result['failed_frames'] = failed_frames
                
                logger.info(f"Successfully processed {successful_frames}/{len(image_files)} frames")
            else:
                raise RuntimeError("No frames processed successfully")
                
        else:
            # Original batch processing
            # Prepare input
            prepared_input = prepare_input_for_method(input_path, config, temp_dir)
            
            # Prepare output directory
            method_output = temp_dir / f"output_{config['folder']}"
            method_output.mkdir(parents=True, exist_ok=True)
            
            # Build command
            script_path = method_dir / config["script"]
            if not script_path.exists():
                # Try alternative script locations
                possible_scripts = list(method_dir.rglob(config["script"]))
                if possible_scripts:
                    script_path = possible_scripts[0]
                else:
                    raise FileNotFoundError(f"Script not found: {config['script']}")
            
            # Build command arguments
            if config.get("custom_command_builder"):
                # Custom command building for special methods
                if method_name == "DeblurGANv2":
                    # DeblurGANv2 uses Fire CLI with positional arguments
                    # Function signature: main(img_pattern, mask_pattern=None, weights_path='fpn_inception.h5', out_dir='submit/', side_by_side=False, video=False)
                    input_pattern = str(prepared_input / "*.png") if prepared_input.is_dir() else str(prepared_input)
                    cmd_parts = [
                        "python", str(script_path),
                        f'"{input_pattern}"',  # img_pattern
                        "--mask_pattern", "None",
                        "--weights_path", "fpn_inception.h5", 
                        "--out_dir", f'"{method_output}"',
                        "--side_by_side", "False",
                        "--video", "False"  # Explicitly disable video processing
                    ]
                else:
                    # Fallback for other custom methods
                    cmd_parts = [
                        "python", str(script_path),
                        config["input_arg"], str(prepared_input),
                        config["output_arg"], str(method_output)
                    ]
            else:
                # Standard command building
                cmd_parts = [
                    "python", str(script_path),
                    config["input_arg"], str(prepared_input),
                    config["output_arg"], str(method_output)
                ]
            
            # Add extra arguments
            cmd_parts.extend(config["extra_args"])
            
            command = " ".join(cmd_parts)
            
            logger.info(f"Command: {command}")
            logger.info(f"Working directory: {method_dir}")
            logger.info(f"Conda environment: {config['conda_env']}")
            
            # Setup GPU environment with force flag
            gpu_env = setup_gpu_environment(config, logger, force_gpu=True)
            
            # Run command
            start_time = datetime.now()
            success, _, stderr = run_conda_command(
                command, config["conda_env"], cwd=method_dir, timeout=1800, extra_env=gpu_env
            )
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            result['processing_time'] = processing_time
            
            if success:
                # Copy results to final output location
                if method_output.exists() and any(method_output.iterdir()):
                    # Copy results with new structure
                    if blur_method:
                        final_output = output_path / method_name / blur_method
                    else:
                        # Fallback for backward compatibility
                        final_output = output_path / method_name
                    final_output.mkdir(parents=True, exist_ok=True)
                    
                    import shutil
                    for item in method_output.iterdir():
                        if item.is_file():
                            shutil.copy2(item, final_output / item.name)
                        else:
                            shutil.copytree(item, final_output / item.name, dirs_exist_ok=True)
                    
                    result['success'] = True
                    result['output_path'] = str(final_output)
                    logger.info(f"Successfully completed {method_name} in {processing_time:.1f}s")
                else:
                    raise ValueError("No output files generated")
            else:
                raise RuntimeError(f"Method failed: {stderr}")
            
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Failed to run {method_name}: {e}")
    
    return result


def save_run_metadata(log_dir: Path, run_id: str, args: argparse.Namespace, results: List[Dict]):
    """Save comprehensive metadata for the deblur run."""
    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'command_args': {
            'input': str(args.input),
            'output': str(args.output),
            'method': args.method,
            'all_methods': args.all_methods
        },
        'results': results,
        'total_methods': len(results),
        'successful_methods': len([r for r in results if r['success']]),
        'failed_methods': len([r for r in results if not r['success']]),
        'total_processing_time': sum(r.get('processing_time', 0) for r in results)
    }
    
    metadata_file = log_dir / f"deblur_metadata_{run_id}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_supported_image_files(path: Path) -> List[Path]:
    """Get list of supported image files from a path."""
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    if path.is_file():
        if path.suffix.lower() in supported_extensions:
            return [path]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    elif path.is_dir():
        files = []
        for ext in supported_extensions:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return sorted(files)
    else:
        raise ValueError(f"Input path does not exist: {path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="VAPOR fx_02_deblur - Unified Deblurring CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Deblur with specific method
    python deblur_cli.py --input blurred_image.jpg --output deblurred/ --method Restormer --blur-method motion_blur_high
    
    # Deblur with all methods  
    python deblur_cli.py --input blurred_folder/ --output results/ --all-methods --blur-method gaussian_blur_medium
    
    # List available methods
    python deblur_cli.py --list-methods
        
Available methods: DeblurGANv2, Restormer, Uformer, DPIR, MPRNet
        """
    )
    
    parser.add_argument('--input', '-i', type=Path,
                       help='Input file or directory path')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output directory path')
    parser.add_argument('--blur-method', type=str,
                       help='Blur method name for organizing output (e.g., motion_blur_high)')
    parser.add_argument('--method', '-m', type=str,
                       choices=list(DEBLUR_METHODS.keys()),
                       help='Deblurring method to use')
    parser.add_argument('--all-methods', action='store_true',
                       help='Run all available methods')
    parser.add_argument('--list-methods', action='store_true',
                       help='List available methods')
    parser.add_argument('--gpu', action='store_true', default=False,
                       help='Enable GPU acceleration (default: False, CPU-only)')
    parser.add_argument('--save-as-you-go', action='store_true', default=False,
                       help='Save frames immediately as processed for progress monitoring')
    
    args = parser.parse_args()
    
    if args.list_methods:
        print("Available deblurring methods:")
        for method in DEBLUR_METHODS.keys():
            print(f"  - {method}")
        return 0
    
    if not args.input or not args.output:
        parser.print_help()
        return 1
    
    if not args.method and not args.all_methods:
        print("Error: Must specify either --method or --all-methods")
        return 1
    
    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return 1
    
    # Generate run ID and setup logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "runs" / f"deblur_{run_id}"
    logger = setup_logging(log_dir, run_id)
    
    logger.info(f"Starting deblur run: {run_id}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Check environment setup
        base_dir = Path(__file__).parent
        check_environment_setup(base_dir)
        logger.info("Environment setup verified")
        
        # Ensure model files are in the correct locations
        if not ensure_model_files(base_dir, logger):
            logger.error("Failed to setup required model files")
            return 1
        
        # Validate input
        input_files = get_supported_image_files(args.input)
        logger.info(f"Found {len(input_files)} image files to process")
        
        # Determine methods to run
        if args.all_methods:
            methods_to_run = list(DEBLUR_METHODS.keys())
        else:
            methods_to_run = [args.method]
        
        logger.info(f"Methods to run: {', '.join(methods_to_run)}")
        
        # Create temporary directory
        temp_dir = log_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Run methods
        results = []
        
        for method_name in methods_to_run:
            logger.info(f"Processing with {method_name}...")
            
            result = run_deblur_method(
                method_name, args.input, args.output, base_dir, temp_dir, logger,
                blur_method=getattr(args, 'blur_method', None),
                save_as_you_go=getattr(args, 'save_as_you_go', False)
            )
            results.append(result)
            
            if result['success']:
                logger.info(f"{method_name} completed successfully")
            else:
                logger.error(f"{method_name} failed: {result['error']}")
        
        # Save metadata
        save_run_metadata(log_dir, run_id, args, results)
        
        # Summary
        successful = len([r for r in results if r['success']])
        total_time = sum(r.get('processing_time', 0) for r in results)
        
        logger.info(f"Processing complete: {successful}/{len(results)} methods successful")
        logger.info(f"Total processing time: {total_time:.1f} seconds")
        logger.info(f"Results saved to: {args.output}")
        logger.info(f"Logs saved to: {log_dir}")
        
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return 0 if successful > 0 else 1
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
