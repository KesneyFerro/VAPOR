#!/usr/bin/env python3
"""
VAPOR fx_02_deblur Repository Setup Script

This script clones and sets up conda environments for various deblurring models:
- DeblurGAN-v2
- Restormer  
- Uformer
- DPIR
- MPRNet

Each model gets its own conda environment with appropriate Python version.
Dependencies must be installed manually after running this script.

Usage:
    python setup_repositories.py [--all] [--model MODEL_NAME]

Author: Kesney de Oliveira
"""

import argparse
import subprocess
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List
import yaml
import os


# Repository configurations
REPOS_CONFIG = {
    "DeblurGANv2": {
        "url": "https://github.com/VITA-Group/DeblurGANv2.git",
        "python_version": "3.7",
        "conda_env": "deblurgan_v2",
        "folder": "DeblurGANv2",
        "requirements": "requirements.txt",
        "setup_commands": [
            "pip install torch torchvision torchaudio",
            "pip install -r requirements.txt"
        ]
    },
    "Restormer": {
        "url": "https://github.com/swz30/Restormer.git", 
        "python_version": "3.8",
        "conda_env": "restormer",
        "folder": "Restormer",
        "requirements": "requirements.txt",
        "setup_commands": [
            "pip install torch torchvision torchaudio",
            "pip install -r requirements.txt"
        ]
    },
    "Uformer": {
        "url": "https://github.com/ZhendongWang6/Uformer.git",
        "python_version": "3.8", 
        "conda_env": "uformer",
        "folder": "Uformer",
        "requirements": "requirements.txt",
        "setup_commands": [
            "pip install torch torchvision torchaudio", 
            "pip install -r requirements.txt"
        ]
    },
    "DPIR": {
        "url": "https://github.com/cszn/DPIR.git",
        "python_version": "3.8",
        "conda_env": "dpir",
        "folder": "DPIR", 
        "requirements": "requirements.txt",
        "setup_commands": [
            "pip install torch torchvision torchaudio",
            "pip install -r requirements.txt"
        ]
    },
    "MPRNet": {
        "url": "https://github.com/swz30/MPRNet.git",
        "python_version": "3.8",
        "conda_env": "mprnet",
        "folder": "MPRNet",
        "requirements": "requirements.txt", 
        "setup_commands": [
            "pip install torch torchvision torchaudio",
            "pip install -r requirements.txt"
        ]
    }
}


def setup_logging() -> logging.Logger:
    """Setup logging for the setup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('setup_repositories.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_command(command: str, cwd: Path = None, conda_env: str = None) -> tuple:
    """
    Run a shell command and return success status and output.
    
    Args:
        command: Command to run
        cwd: Working directory
        conda_env: Conda environment to activate
        
    Returns:
        Tuple of (success, output, error)
    """
    if conda_env:
        # Activate conda environment and run command
        if sys.platform == "win32":
            full_command = f"conda activate {conda_env} && {command}"
            shell_cmd = ["cmd", "/c", full_command]
        else:
            full_command = f"source activate {conda_env} && {command}"
            shell_cmd = ["bash", "-c", full_command]
    else:
        if sys.platform == "win32":
            shell_cmd = ["cmd", "/c", command]
        else:
            shell_cmd = ["bash", "-c", command]
    
    try:
        result = subprocess.run(
            shell_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_conda_available() -> bool:
    """Check if conda is available."""
    success, _, _ = run_command("conda --version")
    return success


def check_git_available() -> bool:
    """Check if git is available."""
    success, _, _ = run_command("git --version")
    return success


def create_conda_environment(env_name: str, python_version: str, logger: logging.Logger) -> bool:
    """
    Create a conda environment.
    
    Args:
        env_name: Name of the environment
        python_version: Python version to install
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Creating conda environment: {env_name} with Python {python_version}")
    
    # Check if environment already exists
    success, output, _ = run_command("conda env list")
    if success and env_name in output:
        logger.info(f"Environment {env_name} already exists, skipping creation")
        return True
    
    # Create environment
    command = f"conda create -n {env_name} python={python_version} -y"
    success, output, error = run_command(command)
    
    if success:
        logger.info(f"Successfully created environment: {env_name}")
        return True
    else:
        logger.error(f"Failed to create environment {env_name}: {error}")
        return False


def clone_repository(repo_config: Dict, base_dir: Path, logger: logging.Logger) -> bool:
    """
    Clone a repository.
    
    Args:
        repo_config: Repository configuration
        base_dir: Base directory for cloning
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    repo_url = repo_config["url"]
    folder_name = repo_config["folder"]
    target_dir = base_dir / folder_name
    
    logger.info(f"Cloning repository: {repo_url}")
    
    # Check if directory already exists
    if target_dir.exists():
        logger.info(f"Repository {folder_name} already exists, skipping clone")
        return True
    
    # Clone repository
    command = f"git clone {repo_url} {folder_name}"
    success, output, error = run_command(command, cwd=base_dir)
    
    if success:
        logger.info(f"Successfully cloned: {folder_name}")
        return True
    else:
        logger.error(f"Failed to clone {folder_name}: {error}")
        return False


def setup_repository_dependencies(repo_config: Dict, base_dir: Path, logger: logging.Logger) -> bool:
    """
    Set up dependencies for a repository.
    
    Args:
        repo_config: Repository configuration
        base_dir: Base directory
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    folder_name = repo_config["folder"]
    conda_env = repo_config["conda_env"]
    setup_commands = repo_config["setup_commands"]
    
    repo_dir = base_dir / folder_name
    
    if not repo_dir.exists():
        logger.error(f"Repository directory not found: {repo_dir}")
        return False
    
    logger.info(f"Setting up dependencies for {folder_name}")
    
    # Run setup commands
    for command in setup_commands:
        logger.info(f"Running: {command}")
        success, output, error = run_command(command, cwd=repo_dir, conda_env=conda_env)
        
        if not success:
            logger.error(f"Failed to run command '{command}': {error}")
            return False
        
        logger.info(f"Command completed successfully")
    
    logger.info(f"Successfully set up dependencies for {folder_name}")
    return True


def create_environment_info_file(base_dir: Path, logger: logging.Logger):
    """Create a file with environment information."""
    env_info = {
        "setup_date": str(Path(__file__).stat().st_mtime),
        "environments": {}
    }
    
    for model_name, config in REPOS_CONFIG.items():
        env_info["environments"][model_name] = {
            "conda_env": config["conda_env"],
            "python_version": config["python_version"],
            "folder": config["folder"],
            "repository": config["url"]
        }
    
    info_file = base_dir / "environment_info.json"
    with open(info_file, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    logger.info(f"Environment info saved to: {info_file}")


def setup_model(model_name: str, base_dir: Path, logger: logging.Logger) -> bool:
    """
    Set up a specific model.
    
    Args:
        model_name: Name of the model to setup
        base_dir: Base directory for setup
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    if model_name not in REPOS_CONFIG:
        logger.error(f"Unknown model: {model_name}")
        return False
    
    config = REPOS_CONFIG[model_name]
    logger.info(f"Setting up model: {model_name}")
    
    # Create conda environment
    if not create_conda_environment(config["conda_env"], config["python_version"], logger):
        return False
    
    # Clone repository
    if not clone_repository(config, base_dir, logger):
        return False
    
    # Note: Dependencies should be installed manually using the setup guide
    logger.info(f"Repository cloned for {model_name}. Install dependencies manually using the setup guide.")
    
    logger.info(f"Successfully set up model: {model_name}")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup deblurring model repositories and conda environments"
    )
    parser.add_argument('--all', action='store_true',
                       help='Setup all models')
    parser.add_argument('--model', type=str,
                       choices=list(REPOS_CONFIG.keys()),
                       help='Setup specific model')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model_name in REPOS_CONFIG.keys():
            print(f"  - {model_name}")
        return 0
    
    if not args.all and not args.model:
        parser.print_help()
        return 1
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting repository setup")
    
    # Check prerequisites
    if not check_conda_available():
        logger.error("Conda is not available. Please install Anaconda or Miniconda.")
        return 1
    
    if not check_git_available():
        logger.error("Git is not available. Please install Git.")
        return 1
    
    # Get base directory
    base_dir = Path(__file__).parent
    logger.info(f"Setup directory: {base_dir}")
    
    # Setup models
    success_count = 0
    total_count = 0
    
    if args.all:
        models_to_setup = list(REPOS_CONFIG.keys())
    else:
        models_to_setup = [args.model]
    
    for model_name in models_to_setup:
        total_count += 1
        if setup_model(model_name, base_dir, logger):
            success_count += 1
        else:
            logger.error(f"Failed to setup model: {model_name}")
    
    # Create environment info file
    create_environment_info_file(base_dir, logger)
    
    # Summary
    logger.info(f"Setup complete: {success_count}/{total_count} models successful")
    
    if success_count == total_count:
        logger.info("All models set up successfully!")
        return 0
    else:
        logger.error("Some models failed to setup. Check the log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
