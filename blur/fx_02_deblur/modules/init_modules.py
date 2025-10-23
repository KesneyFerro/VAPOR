#!/usr/bin/env python3
"""
Module Initialization Script
Sets up the deblur modules environment and validates the setup.

This script:
1. Checks if all required model directories exist
2. Validates that Python environments are set up
3. Checks for pre-trained model weights
4. Creates output directories
5. Provides diagnostic information

Usage:
    python init_modules.py
"""

import os
import sys
from pathlib import Path
import json


def check_model_directories():
    """Check if all model directories exist."""
    print("Checking model directories...")
    
    current_dir = Path(__file__).parent.parent
    models = ["DeblurDiNAT", "DeblurGANv2", "DPIR", "MPRNet", "Restormer", "Uformer"]
    
    missing_models = []
    for model in models:
        model_dir = current_dir / model
        if model_dir.exists():
            print(f"   {model}: {model_dir}")
        else:
            print(f"  [ERROR]  {model}: {model_dir} (MISSING)")
            missing_models.append(model)
    
    return missing_models


def check_pretrained_weights():
    """Check for pre-trained model weights."""
    print("\nChecking for pre-trained weights...")
    
    current_dir = Path(__file__).parent.parent
    weights_base = current_dir.parent.parent / "pretrained_models"
    
    model_weights = {
        "DeblurGANv2": [
            weights_base / "deblurgan_v2" / "best_fpn.h5",
            weights_base / "deblurgan_v2" / "fpn_inception.h5"
        ],
        "DPIR": [
            current_dir / "DPIR" / "model_zoo" / "drunet_color.pth",
            current_dir / "DPIR" / "model_zoo" / "drunet_gray.pth"
        ],
        "MPRNet": [
            weights_base / "mprnet" / "mprnet_deblur.pth",
            current_dir / "MPRNet" / "Deblurring" / "pretrained_models" / "model_deblurring.pth"
        ],
        "Restormer": [
            weights_base / "restormer" / "motion_deblurring.pth",
            weights_base / "restormer" / "restormer_deblur.pth"
        ],
        "Uformer": [
            weights_base / "uformer" / "uformer_deblur.pth",
            current_dir / "Uformer" / "logs" / "GoPro" / "models" / "model_best.pth"
        ],
        "DeblurDiNAT": [
            weights_base / "deblurdinat" / "deblurdinat_gopro.pth",
            current_dir / "DeblurDiNAT" / "results" / "GoPro" / "models" / "model_best.pth"
        ]
    }
    
    for model, paths in model_weights.items():
        print(f"\n  {model}:")
        found_weight = False
        for path in paths:
            if path.exists():
                print(f"     {path}")
                found_weight = True
            else:
                print(f"    [ERROR]  {path} (not found)")
        
        if not found_weight:
            print(f"    ⚠️  No weights found for {model} - will run with random weights")


def check_environment_info():
    """Check environment configuration."""
    print("\nChecking environment information...")
    
    current_dir = Path(__file__).parent.parent
    env_file = current_dir / "environment_info.json"
    
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                env_info = json.load(f)
            
            print(f"   Environment info found: {env_file}")
            
            if "environments" in env_info:
                for model, info in env_info["environments"].items():
                    conda_env = info.get("conda_env", "N/A")
                    python_ver = info.get("python_version", "N/A")
                    print(f"    {model}: conda_env={conda_env}, python={python_ver}")
            
        except Exception as e:
            print(f"  [ERROR]  Error reading environment info: {e}")
    else:
        print(f"  ⚠️  Environment info not found: {env_file}")


def create_output_directories():
    """Create output directories for each model."""
    print("\nCreating output directories...")
    
    current_dir = Path(__file__).parent
    outputs_dir = current_dir / "outputs"
    
    models = ["deblurgan_v2", "dpir", "mprnet", "restormer", "uformer", "deblurdinat"]
    
    for model in models:
        model_output_dir = outputs_dir / model
        model_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   {model_output_dir}")


def check_python_dependencies():
    """Check if key Python dependencies are available."""
    print("\nChecking Python dependencies...")
    
    dependencies = [
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm")
    ]
    
    missing_deps = []
    for dep, name in dependencies:
        try:
            __import__(dep)
            print(f"   {name}")
        except ImportError:
            print(f"  [ERROR]  {name} (missing)")
            missing_deps.append(name)
    
    return missing_deps


def validate_input_directory():
    """Validate the default input directory."""
    print("\nValidating default input directory...")
    
    default_input = Path(r"S:\Kesney\VAPOR\data\frames\blurred\pat3\motion_blur_high")
    
    if default_input.exists():
        image_files = list(default_input.glob("*.png")) + list(default_input.glob("*.jpg"))
        print(f"   Default input directory exists: {default_input}")
        print(f"  📁 Found {len(image_files)} image files")
    else:
        print(f"  ⚠️  Default input directory not found: {default_input}")
        print("     You'll need to specify a valid input directory when running the models")


def main():
    """Main initialization function."""
    print("VAPOR Deblur Modules Initialization")
    print("=" * 60)
    
    # Check model directories
    missing_models = check_model_directories()
    
    # Check pre-trained weights
    check_pretrained_weights()
    
    # Check environment info
    check_environment_info()
    
    # Create output directories
    create_output_directories()
    
    # Check Python dependencies
    missing_deps = check_python_dependencies()
    
    # Validate input directory
    validate_input_directory()
    
    # Summary
    print("\n" + "=" * 60)
    print("INITIALIZATION SUMMARY")
    print("=" * 60)
    
    if missing_models:
        print(f"[ERROR]  Missing model directories: {missing_models}")
        print("   Please ensure all model repositories are cloned properly")
    else:
        print(" All model directories found")
    
    if missing_deps:
        print(f"[ERROR]  Missing dependencies: {missing_deps}")
        print("   Please install missing packages before running models")
    else:
        print(" All key dependencies available")
    
    print(f"\n📁 Output directories created in: {Path(__file__).parent / 'outputs'}")
    print(f"📝 Module scripts available in: {Path(__file__).parent / 'blur_modules'}")
    
    print("\nNext steps:")
    print("1. Ensure all conda environments are activated when running specific models")
    print("2. Download missing pre-trained weights if needed")
    print("3. Run individual model scripts or use run_all_models.py")
    
    print(f"\nExample usage:")
    print(f"  python run_all_models.py --input_dir \"path/to/blurred/images\"")
    print(f"  python blur_modules/mprnet_module.py --input_dir \"path/to/blurred/images\"")


if __name__ == "__main__":
    main()