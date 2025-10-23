# VAPOR Pretrained Models

This directory contains pretrained model weights for the VAPOR deblurring pipeline.

## Model Overview

The pipeline uses 5 specialized deblurring models:

- **Restormer**: Efficient Transformer for high-resolution image restoration (100MB)
- **MPRNet**: Multi-Path Network for image deblurring (77MB)  
- **Uformer**: U-shaped Vision Transformer for image restoration (584MB)
- **DeblurDiNAT**: Dilated Neighborhood Attention Transformer (548MB)
- **DeblurGANv2**: GAN-based deblurring model (233MB)

Total storage requirement: ~1.5GB

## Quick Start

1. Run the download script:
   ```bash
   python download_models.py
   ```

2. The script will automatically verify all downloads upon completion.

## Manual Downloads

Some models may require manual download due to cloud storage restrictions:

- **Uformer**: Download from OneDrive if automatic download fails
- **DeblurDiNAT**: Download from Google Drive if automatic download fails

Place downloaded files in their respective subdirectories:
- `uformer/Uformer_B.pth`
- `deblurdinat/vgg19-dcbb9e9d.pth`

## Verification

Run the verification script to check all models:
```bash
python verify_models.py
```

Expected output shows 5/5 models ready with correct file sizes.

## File Structure

```
pretrained_models/
├── download_models.py      # Main download script
├── verify_models.py        # Model verification
├── README.md              # This file
├── restormer/
│   └── motion_deblurring.pth
├── mprnet/
│   └── mprnet_deblur.pth
├── uformer/
│   └── Uformer_B.pth
├── deblurdinat/
│   └── vgg19-dcbb9e9d.pth
└── deblurgan_v2/
    └── fpn_inception.h5
```

## Troubleshooting

- Ensure internet connection for downloads
- Check available disk space (minimum 2GB)
- Verify Python packages: requests, tqdm
- For Google Drive links, manual download may be required due to virus scan warnings