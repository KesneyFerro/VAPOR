#!/usr/bin/env python3
"""
Install Missing Dependencies for Deblur Models
This script installs commonly missing dependencies for the deblur models.

Usage:
    python install_dependencies.py
"""

import subprocess
import sys
from pathlib import Path

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"[OK] Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR]  Failed to install {package}: {e}")
        return False

def main():
    """Install missing dependencies."""
    print("Installing missing dependencies for deblur models...")
    print("="*60)
    
    # Common missing dependencies
    dependencies = [
        'timm',                    # For Uformer
        'albumentations',          # For DeblurGANv2
        'efficientnet-pytorch',    # For DeblurGANv2
        'einops',                  # For various models
        'natten',                  # For DeblurDiNAT (may fail, that's ok)
        'basicsr',                 # For Restormer
    ]
    
    success_count = 0
    
    for package in dependencies:
        print(f"\nInstalling {package}...")
        if install_package(package):
            success_count += 1
        else:
            if package == 'natten':
                print("Note: natten installation often fails - this is normal")
                print("DeblurDiNAT will use fallback model if natten is not available")
    
    print("\n" + "="*60)
    print(f"Installation complete: {success_count}/{len(dependencies)} packages installed")
    print("\nNote: Some packages may require specific conda environments.")
    print("Refer to the original model repositories for detailed setup instructions.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())