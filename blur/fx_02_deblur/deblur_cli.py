#!/usr/bin/env python3
"""
VAPOR fx_02_deblur - Unified Deblurring CLI

This script provides a unified command-line interface for running different deblurring
methods with automatic environment switching. Supports:

- DeblurGAN-v2
- Restormer
- Uformer  
- DeblurDiNAT
- DPIR
- MPRNet

Usage:
    python deblur_cli.py --input <path> --output <path> --method <method> [--all-methods]
    
Examples:
    # Deblur with specific method
    python deblur_cli.py --input blurred_image.jpg --output deblurred/ --method Restormer
    
    # Deblur with all methods
    python deblur_cli.py --input blurred_folder/ --output results/ --all-methods
    
    # Deblur single frame
    python deblur_cli.py --input frame_001.jpg --output deblurred_frame.jpg --method DeblurGANv2

Author: Kesney de Oliveira
"""

import argparse
import subprocess
import sys
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import cv2
import numpy as np


# Method configurations
DEBLUR_METHODS = {
    "DeblurGANv2": {
        "conda_env": "deblurgan_v2",
        "folder": "DeblurGANv2", 
        "script": "predict.py",
        "model_path": "fpn_mobilenet.h5",
        "input_arg": "--input",
        "output_arg": "--output",
        "extra_args": []
    },
    "Restormer": {
        "conda_env": "restormer",
        "folder": "Restormer",
        "script": "demo.py", 
        "model_path": "Deblurring/pretrained_models/", 
        "input_arg": "--input_dir",
        "output_arg": "--result_dir", 
        "extra_args": ["--task", "Deblurring"]
    },
    "Uformer": {
        "conda_env": "uformer",
        "folder": "Uformer",
        "script": "test.py",
        "model_path": "saved_models/",
        "input_arg": "--input_dir", 
        "output_arg": "--result_dir",
        "extra_args": []
    },
    "DeblurDiNAT": {
        "conda_env": "deblurdinat",
        "folder": "DeblurDiNAT", 
        "script": "predict.py",
        "model_path": "experiments/",
        "input_arg": "--input",
        "output_arg": "--output",
        "extra_args": []
    },
    "DPIR": {
        "conda_env": "dpir",
        "folder": "DPIR",
        "script": "main_dpir_deblur.py", 
        "model_path": "model_zoo/",
        "input_arg": "--input", 
        "output_arg": "--output",
        "extra_args": []
    },
    "MPRNet": {
        "conda_env": "mprnet", 
        "folder": "MPRNet",
        "script": "demo.py",
        "model_path": "Deblurring/pretrained_models/",
        "input_arg": "--input_dir",
        "output_arg": "--result_dir", 
        "extra_args": ["--task", "Deblurring"]
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


def run_conda_command(command: str, conda_env: str, cwd: Path = None, timeout: int = 1800) -> Tuple[bool, str, str]:
    """
    Run a command in a specific conda environment.
    
    Args:
        command: Command to run
        conda_env: Conda environment name
        cwd: Working directory
        timeout: Command timeout in seconds
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    if sys.platform == "win32":
        full_command = f"conda activate {conda_env} && {command}"
        shell_cmd = ["cmd", "/c", full_command]
    else:
        full_command = f"conda activate {conda_env} && {command}"
        shell_cmd = ["bash", "-c", full_command]
    
    try:
        result = subprocess.run(
            shell_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
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
    # Most methods expect directory input, some handle single files
    if input_path.is_file():
        # Create temp directory and copy file
        method_input_dir = temp_dir / f"input_{method_config['folder']}"
        method_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file to temp directory
        import shutil
        temp_file = method_input_dir / input_path.name
        shutil.copy2(input_path, temp_file)
        
        return method_input_dir
    else:
        # Directory input - use as is
        return input_path


def run_deblur_method(method_name: str, input_path: Path, output_path: Path, 
                     base_dir: Path, temp_dir: Path, logger: logging.Logger) -> Dict:
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
        
        config = DEBLUR_METHODS[method_name]
        method_dir = base_dir / config["folder"]
        
        if not method_dir.exists():
            raise FileNotFoundError(f"Method directory not found: {method_dir}")
        
        logger.info(f"Running {method_name} deblurring...")
        
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
        
        # Run command
        start_time = datetime.now()
        success, _, stderr = run_conda_command(
            command, config["conda_env"], cwd=method_dir, timeout=1800
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        result['processing_time'] = processing_time
        
        if success:
            # Copy results to final output location
            if method_output.exists() and any(method_output.iterdir()):
                # Copy results 
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
    python deblur_cli.py --input blurred_image.jpg --output deblurred/ --method Restormer
    
    # Deblur with all methods  
    python deblur_cli.py --input blurred_folder/ --output results/ --all-methods
    
    # List available methods
    python deblur_cli.py --list-methods
        
Available methods: DeblurGANv2, Restormer, Uformer, DeblurDiNAT, DPIR, MPRNet
        """
    )
    
    parser.add_argument('--input', '-i', type=Path,
                       help='Input file or directory path')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output directory path')
    parser.add_argument('--method', '-m', type=str,
                       choices=list(DEBLUR_METHODS.keys()),
                       help='Deblurring method to use')
    parser.add_argument('--all-methods', action='store_true',
                       help='Run all available methods')
    parser.add_argument('--list-methods', action='store_true',
                       help='List available methods')
    
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
                method_name, args.input, args.output, base_dir, temp_dir, logger
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
