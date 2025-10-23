# VAPOR Blur Evaluation Setup Guide

This comprehensive guide covers the complete setup process for the VAPOR blur evaluation system, from initial conda environment creation to running deblur models.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Repository Cloning](#repository-cloning)
4. [Dependencies Installation](#dependencies-installation)
5. [Model Weights Download](#model-weights-download)
6. [Verification](#verification)
7. [Running the Code](#running-the-code)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

- **Conda**: Anaconda or Miniconda installed
- **Git**: For cloning repositories
- **GPU**: CUDA-compatible GPU recommended for optimal performance
- **Storage**: At least 10GB free space for models and dependencies

### Conda Setup (Windows)
If conda is not recognized in your terminal:

1. **PowerShell**: Add conda alias to your PowerShell profile:
```powershell
# Check if profile exists
Test-Path $PROFILE

# Create profile if it doesn't exist
if (!(Test-Path $PROFILE)) { New-Item -Type File -Path $PROFILE -Force }

# Add conda alias
Add-Content $PROFILE 'Set-Alias -Name conda -Value "C:\Users\$env:USERNAME\anaconda3\Scripts\conda.exe"'

# Reload profile
. $PROFILE
```

2. **Command Prompt**: Use full path to conda:
```cmd
C:\Users\%USERNAME%\anaconda3\Scripts\conda.exe
```

## Environment Setup

The blur evaluation system requires 6 different conda environments for the various deblurring models.

### 1. Create All Environments

Run the following commands to create all required environments:

```bash

# Other models (Python 3.8)
conda create -n restormer python=3.8 -y
conda create -n uformer python=3.8 -y
conda create -n mprnet python=3.8 -y
```

### 2. Verify Environment Creation

```bash
conda env list
```

You should see all 6 environments listed.

## Repository Cloning

### Automatic Setup (Recommended)

Use the provided setup script to clone all repositories:

```bash
cd 
python blur/fx_02_deblur/setup_repositories.py --all
```

### Manual Cloning

If you prefer manual setup:

```bash
cd \blur\fx_02_deblur

# Clone all deblur model repositories
git clone https://github.com/swz30/Restormer.git
git clone https://github.com/ZhendongWang6/Uformer.git
git clone https://github.com/swz30/MPRNet.git
```

## Dependencies Installation

Install dependencies for each environment using the setup script or manually:

### Automatic Installation

```bash
python blur/fx_02_deblur/setup_repositories.py --install-deps
```

### Manual Installation

#### Restormer Environment
```bash
conda run -n restormer pip install torch torchvision torchaudio
conda run -n restormer pip install opencv-python numpy pillow tqdm
conda run -n restormer pip install einops timm
```

#### Uformer Environment
```bash
conda run -n uformer pip install torch torchvision torchaudio
conda run -n uformer pip install opencv-python numpy pillow tqdm
conda run -n uformer pip install einops timm fvcore
```

#### MPRNet Environment
```bash
conda run -n mprnet pip install torch torchvision torchaudio
conda run -n mprnet pip install opencv-python numpy pillow tqdm
conda run -n mprnet pip install lpips
```

## Model Weights Download

Each model requires pre-trained weights. Download instructions:

### Restormer
```bash
cd \blur\fx_02_deblur\Restormer
# Download from: https://github.com/swz30/Restormer/releases
# Place in: Deblurring/pretrained_models/
```

### Uformer
```bash
cd \blur\fx_02_deblur\Uformer
# Download from: https://github.com/ZhendongWang6/Uformer/releases
# Place in: deblurring/pretrained_models/
```

### MPRNet
```bash
cd \blur\fx_02_deblur\MPRNet
# Download from: https://github.com/swz30/MPRNet/releases
# Place in: Deblurring/pretrained_models/
```

## Verification

### Test Environment Setup

Create and run a test script to verify all environments:

```python
# Create test_env.py in VAPOR root directory
import sys
import subprocess

def test_environment(env_name):
    print(f"\n=== Testing {env_name} environment ===")
    
    test_script = '''
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"PyTorch import error: {e}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV import error: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")
'''
    
    try:
        result = subprocess.run([
            'conda', 'run', '-n', env_name, 'python', '-c', test_script
        ], capture_output=True, text=True, shell=True)
        
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error testing {env_name}: {e}")
        return False

if __name__ == "__main__":
    environments = [
        'deblurgan_v2',
        'restormer', 
        'uformer',
        'mprnet',
        'dpir',
        'deblurdinat'
    ]
    
    print("Testing all blur evaluation environments...")
    results = {}
    
    for env in environments:
        results[env] = test_environment(env)
    
    print("\n=== SUMMARY ===")
    for env, success in results.items():
        status = "[OK] PASS" if success else "[ERROR] FAIL"
        print(f"{env}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall status: {'[OK] ALL ENVIRONMENTS READY' if all_passed else '[ERROR] SOME ENVIRONMENTS FAILED'}")
```

Run the test:
```bash
python test_env.py
```

## Running the Code

### Individual Model Testing

#### DeblurGANv2
```bash
cd \blur\fx_02_deblur\DeblurGANv2
conda run -n deblurgan_v2 python predict.py --input_path INPUT_IMAGE --output_path OUTPUT_IMAGE
```

#### Restormer
```bash
cd \blur\fx_02_deblur\Restormer\Deblurring
conda run -n restormer python test.py --input_dir INPUT_DIR --result_dir OUTPUT_DIR
```

#### Uformer
```bash
cd \blur\fx_02_deblur\Uformer\deblurring
conda run -n uformer python test.py --input_dir INPUT_DIR --result_dir OUTPUT_DIR
```

#### MPRNet
```bash
cd \blur\fx_02_deblur\MPRNet\Deblurring
conda run -n mprnet python test.py --input_dir INPUT_DIR --result_dir OUTPUT_DIR
```

#### DPIR
```bash
cd \blur\fx_02_deblur\DPIR
conda run -n dpir python main_dpir_deblur.py --input INPUT_IMAGE --output OUTPUT_IMAGE
```

#### DeblurDiNAT
```bash
cd \blur\fx_02_deblur\DeblurDiNAT
conda run -n deblurdinat python basicsr/test.py -opt options/test/test_deblurdinat.yml
```

### VAPOR Blur Pipeline

Run the complete blur evaluation pipeline:

```bash
python blur/blur_generator.py --input INPUT_VIDEO --output OUTPUT_DIR
```

### VAPOR Complete Pipeline

Run the full VAPOR pipeline including blur evaluation:

```bash
python vapor_complete_pipeline.py --input INPUT_VIDEO --output OUTPUT_DIR
```

## Troubleshooting

### Common Issues

#### 1. Conda Not Recognized
- **Solution**: Use full conda path or set up alias (see [Prerequisites](#prerequisites))

#### 2. CUDA Out of Memory
- **Solution**: Reduce batch size or use CPU mode
```bash
# For most models, add --device cpu
conda run -n model_env python script.py --device cpu
```

#### 3. Import Errors
- **Solution**: Reinstall dependencies
```bash
conda run -n env_name pip install --force-reinstall package_name
```

#### 4. Model Weights Not Found
- **Solution**: Verify weights are in correct directories (see [Model Weights Download](#model-weights-download))

#### 5. Path Issues on Windows
- **Solution**: Use forward slashes or raw strings in Python paths
```python
# Good
path = "S:/Kesney/VAPOR/input.jpg"
# Or
path = r"\input.jpg"
```

### Environment Reset

If you need to completely reset an environment:

```bash
# Remove environment
conda env remove -n environment_name

# Recreate environment
conda create -n environment_name python=3.8 -y

# Reinstall dependencies
# (follow installation steps above)
```

### Verification Commands

Quick verification commands:

```bash
# List all environments
conda env list


## Support

For additional help:
1. Check individual repository documentation
2. Verify CUDA compatibility for GPU acceleration
3. Ensure all model weights are downloaded
4. Test with small input images first

---

**Note**: This setup guide assumes Windows environment. For Linux/Mac, replace `conda run -n env_name` with `conda activate env_name` and adjust paths accordingly.