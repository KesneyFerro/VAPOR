# VAPOR Deblurring Models Setup Guide

This guide provides complete setup instructions for all 6 deblurring models in the VAPOR project.

## Prerequisites

Before starting, ensure you have:

- Anaconda or Miniconda installed
- Git installed
- Windows PowerShell (or equivalent terminal)

## Quick Setup

### Step 1: Clone Repositories and Create Environments

Run the setup script to clone all repositories and create conda environments:

```powershell
cd "C:\Users\kesne\Documents\MAPLE-25\VAPOR"
python blur/fx_02_deblur/setup_repositories.py --all
```

This will:

- Create 6 conda environments with appropriate Python versions
- Clone all 6 deblurring model repositories
- Generate an `environment_info.json` file with environment details

### Step 2: Install Dependencies for Each Environment

After the repositories are cloned, install dependencies for each model by activating their respective environments and running the installation commands below.

## Step 3: Download Pre-trained Models (CRITICAL!)

**⚠️ IMPORTANT: All models require pre-trained weights to function. Download them before attempting to use any model.**

### Download Links and Placement:

1. **DeblurGANv2 Models:**

   - [fpn_inception.h5](https://drive.google.com/uc?export=view&id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR) → `blur/fx_02_deblur/DeblurGANv2/`
   - [fpn_mobilenet.h5](https://drive.google.com/uc?export=view&id=1JhnT4BBeKBBSLqTo6UsJ13HeBXevarrU) → `blur/fx_02_deblur/DeblurGANv2/`

2. **Restormer Models:**

   - [Motion Deblurring Models](https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK) → `blur/fx_02_deblur/Restormer/Motion_Deblurring/pretrained_models/`

3. **Uformer Models:**

   - [GoPro Model](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhendongwang_mail_ustc_edu_cn/EfCPoTSEKJRAshoE6EAC_3YB7oNkbLUX6AUgWSCwoJe0oA) → `blur/fx_02_deblur/Uformer/`

4. **MPRNet Models:**

   - [Deblurring Model](https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb) → `blur/fx_02_deblur/MPRNet/Deblurring/pretrained_models/`

5. **DPIR Models:**

   - [All DPIR Models](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D) → `blur/fx_02_deblur/DPIR/model_zoo/`
   - Files: `drunet_gray.pth`, `drunet_color.pth`, `ircnn_gray.pth`, `ircnn_color.pth`

6. **DeblurDiNAT Models:**
   - [DeblurDiNATL.pth](https://drive.google.com/file/d/1VT7dpP550b83YZ0LjfmGA5t0nEA32EEs) → `blur/fx_02_deblur/DeblurDiNAT/`
   - [VGG19 Weights](https://drive.google.com/file/d/1r2_clZ02-ai6xM7EOHW9APqY9IxkPYsS) → `blur/fx_02_deblur/DeblurDiNAT/models/`

## Step 4: Additional Setup Commands

## Individual Model Setup Instructions

### 1. DeblurGANv2

**Environment:** `deblurgan_v2` (Python 3.7)

```powershell
conda activate deblurgan_v2
pip install torch torchvision torchaudio numpy opencv-python-headless joblib albumentations scikit-image==0.18.1 tqdm glog tensorboardx fire torchsummary pretrainedmodels
```

**Repository:** `blur/fx_02_deblur/DeblurGANv2/`

**Key Dependencies:**

- PyTorch with CUDA support
- OpenCV for image processing
- Albumentations for data augmentation
- TensorBoard for logging

### 2. Restormer

**Environment:** `restormer` (Python 3.8)

```powershell
conda activate restormer
pip install torch torchvision torchaudio matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

**Additional Setup:**

```powershell
cd blur/fx_02_deblur/Restormer
python setup.py develop --no_cuda_ext
```

**Additional Setup:**

```powershell
cd blur/fx_02_deblur/Restormer
python setup.py develop --no_cuda_ext
```

**Key Dependencies:**

- BasicSR framework
- LPIPS for perceptual loss
- Advanced image processing libraries

### 3. Uformer

**Environment:** `uformer` (Python 3.8)

```powershell
conda activate uformer
pip install torch==1.8.0 torchvision==0.9.0 matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm einops linformer timm ptflops dataclasses
```

**Repository:** `blur/fx_02_deblur/Uformer/`

**Key Dependencies:**

- Specific PyTorch versions (1.8.0)
- Einops for tensor operations
- Timm for vision transformer models
- Linformer for efficient attention

### 4. MPRNet

**Environment:** `mprnet` (Python 3.8)

```powershell
conda activate mprnet
pip install torch==1.10.2 torchvision matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

**Additional Setup:**

```powershell
cd blur/fx_02_deblur/MPRNet/pytorch-gradual-warmup-lr
python setup.py install
cd ../..
```

**Additional Setup:**

```powershell
cd blur/fx_02_deblur/MPRNet/pytorch-gradual-warmup-lr
python setup.py install
cd ../..
```

**Repository:** `blur/fx_02_deblur/MPRNet/`

**Key Dependencies:**

- PyTorch 1.10.2 for compatibility
- Gradual warmup scheduler
- Standard computer vision libraries

### 5. DPIR

**Environment:** `dpir` (Python 3.8)

```powershell
conda activate dpir
pip install torch torchvision scipy numpy opencv-python matplotlib hdf5storage scikit-image
```

**Alternative Download Method:**

```powershell
cd blur/fx_02_deblur/DPIR
python main_download_pretrained_models.py --models "DPIR IRCNN" --model_dir "model_zoo"
```

**Repository:** `blur/fx_02_deblur/DPIR/`

**Key Dependencies:**

- HDF5 storage for .mat files
- SciPy for scientific computing
- Plug-and-play denoising framework

### 6. DeblurDiNAT

**Environment:** `deblurdinat` (Python 3.8)

```powershell
conda activate deblurdinat
pip install einops glog opencv-python scikit-image timm tqdm tensorboardx
```

**Note:** NATTEN library requires CMake for compilation. If you need this library:

1. Install CMake: `conda install cmake`
2. Install NATTEN: `pip install natten==0.14.6+torch200cu118 -f https://shi-labs.com/natten/wheels`

**Additional Setup:**

```powershell
# Install CMake first if NATTEN is needed
conda activate deblurdinat
conda install cmake
pip install natten==0.14.6+torch200cu118 -f https://shi-labs.com/natten/wheels
```

**Repository:** `blur/fx_02_deblur/DeblurDiNAT/`

**Key Dependencies:**

- Vision transformer architecture
- NATTEN for neighborhood attention
- Advanced deep learning tools

## Environment Information

All created environments are documented in `blur/fx_02_deblur/environment_info.json`:

| Model       | Environment Name | Python Version | Repository             |
| ----------- | ---------------- | -------------- | ---------------------- |
| DeblurGANv2 | `deblurgan_v2`   | 3.7            | VITA-Group/DeblurGANv2 |
| Restormer   | `restormer`      | 3.8            | swz30/Restormer        |
| Uformer     | `uformer`        | 3.8            | ZhendongWang6/Uformer  |
| DeblurDiNAT | `deblurdinat`    | 3.8            | HanzhouLiu/DeblurDiNAT |
| DPIR        | `dpir`           | 3.8            | cszn/DPIR              |
| MPRNet      | `mprnet`         | 3.8            | swz30/MPRNet           |

## Usage Examples

### Running DeblurGANv2

```powershell
conda activate deblurgan_v2
cd blur/fx_02_deblur/DeblurGANv2
# Single image
python predict.py input_image.jpg
# Or specify model weights
python predict.py --weights_path fpn_mobilenet.h5 input_image.jpg
```

### Running Restormer

```powershell
conda activate restormer
cd blur/fx_02_deblur/Restormer
# Motion deblurring
python demo.py --task Motion_Deblurring --input_dir input/ --result_dir results/
```

### Running Uformer

```powershell
conda activate uformer
cd blur/fx_02_deblur/Uformer
# Test with pre-trained model
python test.py --input_dir input/ --result_dir results/
```

### Running MPRNet

```powershell
conda activate mprnet
cd blur/fx_02_deblur/MPRNet
python demo.py --task Deblurring --input_dir ./samples/input/ --result_dir ./samples/output/
```

### Running DPIR

```powershell
conda activate dpir
cd blur/fx_02_deblur/DPIR
python main_dpir_deblur.py
```

### Running DeblurDiNAT

```powershell
conda activate deblurdinat
cd blur/fx_02_deblur/DeblurDiNAT
# Test on single images
python predict_GoPro_test_results.py --job_name DeblurDiNATL --weight_name DeblurDiNATL.pth --blur_path input_folder/
```

## Troubleshooting

### Common Issues

1. **CUDA not available:** Ensure you have NVIDIA drivers and CUDA toolkit installed
2. **Import errors:** Verify all dependencies are installed in the correct environment
3. **Path issues:** Always use absolute paths when working with files
4. **Memory errors:** Reduce batch size or image resolution
5. **Missing pre-trained models:** Download all required .pth/.h5 model files before running
6. **CMake errors (DeblurDiNAT):** Install CMake via conda before installing NATTEN

### Missing Pre-trained Models

If you get errors about missing model files:

```powershell
# Check if model files exist in the right locations
ls blur/fx_02_deblur/*/pretrained_models/
ls blur/fx_02_deblur/*/*.pth
ls blur/fx_02_deblur/*/*.h5
```

### Environment Management

List all environments:

```powershell
conda env list
```

Remove an environment (if needed):

```powershell
conda env remove -n environment_name
```

Export environment:

```powershell
conda activate environment_name
conda env export > environment.yml
```

## Performance Tips

1. **Use GPU acceleration:** Ensure CUDA-enabled PyTorch is installed
2. **Optimize batch sizes:** Adjust based on your GPU memory
3. **Use mixed precision:** Enable for faster training/inference
4. **Monitor memory usage:** Use tools like `nvidia-smi` to track GPU usage

## Additional Resources

- [Original VAPOR repository](https://github.com/KesneyFerro/VAPOR)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- [Conda cheat sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)

## Support

For issues specific to individual models, refer to their original repositories:

- [DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2)
- [Restormer](https://github.com/swz30/Restormer)
- [Uformer](https://github.com/ZhendongWang6/Uformer)
- [DeblurDiNAT](https://github.com/HanzhouLiu/DeblurDiNAT)
- [DPIR](https://github.com/cszn/DPIR)
- [MPRNet](https://github.com/swz30/MPRNet)

---

_Last updated: September 11, 2025_
_Author: Kesney de Oliveira_
