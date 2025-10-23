# Deblur Modules

This directory contains individual modules for each deblur model in the VAPOR project. Each module can be run independently to process blurred images and diagnose model performance.

## Structure

```
modules/
├── blur_modules/           # Individual model scripts
│   ├── mprnet_module.py
│   ├── restormer_module.py
│   └── uformer_module.py
├── outputs/               # Model outputs (auto-created)
│   ├── mprnet/
│   ├── restormer/
│   └── uformer/
├── init_modules.py        # Setup validation script
├── run_all_models.py      # Run all models at once
├── setup_environments.py  # Python environment setup script
├── setup_environments.bat # Windows batch setup script
├── dependency_guide.py    # Manual setup instructions
└── README.md             # This file
```

## Setup

### Option 1: Automatic Setup (Recommended)

**Windows users (easiest):**
```bash
setup_environments.bat
```

**Python script (cross-platform):**
```bash
python setup_environments.py
```

### Option 2: Manual Setup

Follow the dependency installation guide:
```bash
python dependency_guide.py
```

### Conda Environments

Each model runs in its dedicated conda environment:
- `restormer` - Restormer with lmdb, basicsr
- `uformer` - Uformer with timm, einops  
- `base` - MPRNet (works with base environment)

## Quick Start

1. **Set up environments (choose one option above)**

2. **Validate setup:**
   ```bash
   python init_modules.py
   ```

3. **Run all models on default directory:**
   ```bash
   python run_all_models.py
   ```

4. **Run specific models:**
   ```bash
   python run_all_models.py --models mprnet restormer
   ```

5. **Run individual model:**
   ```bash
   python blur_modules/mprnet_module.py --input_dir "path/to/blurred/images"
   ```

## Individual Model Usage

Each model script accepts the following arguments:

### Common Arguments
- `--input_dir`: Input directory containing blurred images (default: `S:\Kesney\VAPOR\data\frames\blurred\pat3\motion_blur_high`)
- `--weights_path`: Path to model weights file (optional, auto-detected if not specified)

### Model-Specific Arguments

#### DeblurGANv2
```bash
python blur_modules/deblurgan_v2_module.py --input_dir "path/to/images" --model_name "fpn_mobilenet"
```

#### DPIR
```bash
python blur_modules/dpir_module.py --input_dir "path/to/images" --model_name "drunet_color"
```
- `--model_name`: drunet_color, drunet_gray, ircnn_color, ircnn_gray

#### Restormer
```bash
python blur_modules/restormer_module.py --input_dir "path/to/images" --task "Motion_Deblurring"
```
- `--task`: Motion_Deblurring, Defocus_Deblurring, Deblurring

## Output Structure

Each model saves its results to:
```
outputs/{model_name}/
├── image1.png
├── image2.png
└── ...
```

The output images maintain the same filenames as the input images.

## Dependencies

Each model requires its own conda environment as specified in `../environment_info.json`:

- **DeblurGANv2**: `deblurgan_v2` environment
- **Restormer**: `restormer` environment  
- **Uformer**: `uformer` environment
- **DeblurDiNAT**: `deblurdinat` environment
- **MPRNet**: Can use any PyTorch environment

## Pre-trained Weights

The scripts automatically search for pre-trained weights in multiple locations:

1. `S:\Kesney\VAPOR\blur\pretrained_models\{model_name}/`
2. Model-specific directories within each repository
3. Standard checkpoint locations

If no weights are found, the models will run with random weights (poor results) but won't crash.

## Troubleshooting

1. **Import errors**: Ensure you're using the correct conda environment for each model and have installed all dependencies
2. **Missing dependencies**: Run `python dependency_guide.py` for exact installation commands
3. **Missing weights**: Download pre-trained weights or specify custom paths
4. **CUDA errors**: Models will automatically fallback to CPU if CUDA is unavailable
5. **Out of memory**: Process smaller batches or reduce image sizes

## Installing Dependencies

Run the dependency guide to see exactly what to install:
```bash
python dependency_guide.py
```

Each model requires its own conda environment with specific dependencies:
- **DeblurGANv2**: albumentations, efficientnet-pytorch
- **Restormer**: lmdb, basicsr
- **Uformer**: timm, einops  
- **DeblurDiNAT**: natten
- **MPRNet**: No additional dependencies

## Adding New Models

To add a new deblur model:

1. Create a new script in `blur_modules/` following the existing pattern
2. Add the model to `available_models` dict in `run_all_models.py`
3. Create output directory in `outputs/`
4. Update this README

## Environment Setup

If you need to set up the conda environments, refer to:
- `../setup_repositories.py` for automatic setup
- `../environment_info.json` for environment specifications
- Individual model repositories for manual setup

## Performance Comparison

After running multiple models, you can compare their outputs by examining the files in each `outputs/{model_name}/` directory. Consider using image quality metrics to quantitatively evaluate the results.