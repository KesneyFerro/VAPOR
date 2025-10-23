#!/usr/bin/env python3
"""
VAPOR Pretrained Models Download Script
=======================================
Downloads all available pretrained models for the 5-model deblurring pipeline.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess

# Model configurations for 5-model pipeline
MODELS = {
    'restormer': {
        'status': 'completed',
        'description': 'Restormer motion deblurring model',
        'filename': 'motion_deblurring.pth',
        'target_dir': 'restormer',
        'size_mb': 100
    },
    
    'mprnet': {
        'status': 'completed', 
        'description': 'MPRNet motion deblurring model',
        'filename': 'mprnet_deblur.pth',
        'target_dir': 'mprnet',
        'size_mb': 77
    },
    
    'uformer': {
        'status': 'completed',
        'description': 'Uformer GoPro deblurring model',
        'filename': 'Uformer_B.pth',
        'target_dir': 'uformer',
        'size_mb': 584
    },
    
    'deblurdinat': {
        'status': 'download',
        'description': 'DeblurDiNAT pretrained model',
        'direct_url': 'https://drive.usercontent.google.com/download?id=1r2_clZ02-ai6xM7EOHW9APqY9IxkPYsS&export=download&confirm=t',
        'filename': 'vgg19-dcbb9e9d.pth',
        'target_dir': 'deblurdinat',
        'size_mb': 548
    },
    
    'deblurgan_v2': {
        'status': 'download',
        'description': 'DeblurGANv2 model',
        'direct_url': 'https://drive.usercontent.google.com/download?id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR&export=view&authuser=0&confirm=t',
        'filename': 'fpn_inception.h5',
        'target_dir': 'deblurgan_v2',
        'size_mb': 233
    }
}

def download_file_with_progress(url, filepath, description="file"):
    """Download a file with progress bar."""
    try:
        print(f"Downloading {description}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"Successfully downloaded: {filepath.name} ({size_mb:.1f}MB)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Download failed for {description}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading {description}: {e}")
        return False

def process_model(model_name, model_config, base_dir):
    """Process a single model download."""
    target_dir = base_dir / model_config['target_dir']
    target_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing {model_name.upper()}...")
    print(f"Description: {model_config['description']}")
    
    if model_config['status'] == 'completed':
        filepath = target_dir / model_config['filename']
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"Already downloaded: {filepath.name} ({size_mb:.1f}MB)")
            return True
        else:
            print(f"Expected file not found: {filepath}")
            return False
    
    elif model_config['status'] == 'download':
        filepath = target_dir / model_config['filename']
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"Already exists: {filepath.name} ({size_mb:.1f}MB)")
            return True
        
        if 'direct_url' in model_config:
            return download_file_with_progress(
                model_config['direct_url'], 
                filepath, 
                f"{model_name} model"
            )
        else:
            print("No download method configured")
            return False
    
    return False

def run_verification():
    """Run the verification script after downloads complete."""
    try:
        print("\n" + "="*50)
        print("Running model verification...")
        result = subprocess.run([sys.executable, 'verify_models.py'], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Verification warnings:")
            print(result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run verification: {e}")
        return False

def main():
    """Main download function."""
    print("VAPOR Deblurring Models Download Script")
    print("="*50)
    
    base_dir = Path.cwd()
    print(f"Base directory: {base_dir}")
    print("Processing models...\n")
    
    success_count = 0
    total_models = len(MODELS)
    
    for model_name, model_config in MODELS.items():
        if process_model(model_name, model_config, base_dir):
            success_count += 1
    
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print(f"Successfully processed: {success_count}/{total_models} models")
    
    if success_count == total_models:
        print("All models ready for use")
        # Run verification
        if run_verification():
            print("Model verification passed")
            return 0
        else:
            print("Model verification failed")
            return 1
    else:
        print("Some models require manual download")
        print("See README.md for manual download instructions")
        return 1

if __name__ == "__main__":
    sys.exit(main())