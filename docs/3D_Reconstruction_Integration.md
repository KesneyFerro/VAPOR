# VAPOR 3D Reconstruction Integration

This document describes the new integrated VAPOR pipeline that combines blur processing with 3D reconstruction analysis using maploc.

## 📁 **New Directory Structure**

The updated VAPOR system now organizes data as follows:

```
data/
├── videos/
│   ├── original/           # Input videos (pat3.mp4, etc.)
│   └── blurred/           # Generated blurred videos
├── frames/
│   ├── original/{video}/   # Original extracted frames
│   ├── blurred/{video}/    # Blurred frame variations
│   └── deblurred/{video}/  # Deblurred frames (when available)
├── metrics/{video}/
│   ├── original/          # Metrics for original frames
│   ├── blurred/           # Metrics for blurred frames
│   └── deblurred/         # Metrics for deblurred frames
├── point_clouds/{video}/
│   ├── original/          # 3D reconstructions from original frames
│   ├── blurred/           # 3D reconstructions from blurred frames
│   └── deblurred/         # 3D reconstructions from deblurred frames
├── reports/               # Analysis reports
└── logs/                  # Pipeline execution logs
```

## 🚀 **Quick Start**

### **Option 1: Run Complete Pipeline (Recommended)**

```bash
# Run everything for pat3.mp4
python vapor_complete_pipeline.py --video pat3.mp4

# Run with selective steps
python vapor_complete_pipeline.py --video pat3.mp4 --skip-blur --skip-reconstruction
```

### **Option 2: Run Individual Components**

```bash
# 1. Generate blurred frames (if not done already)
python blur/simple_blur_pipeline.py --video pat3.mp4 --stride 60

# 2. Calculate metrics with new structure
python blur/metrics/updated_calculator.py --video pat3.mp4

# 3. Run 3D reconstruction analysis
python reconstruction/reconstruction_pipeline.py --video pat3.mp4
```

## 🔧 **Setup Requirements**

### **1. Install Maploc Dependencies**

```bash
# Setup maploc integration
python reconstruction/setup_vapor_maploc.py
```

### **2. Required Packages**

The system requires additional packages for 3D reconstruction:

```bash
pip install torch torchvision
pip install pycolmap==0.6.1
pip install kornia>=0.6.11
pip install h5py plotly
```

## 📊 **Output Analysis**

### **Metrics Structure**

- **Detailed Metrics**: Frame-by-frame analysis for each type
- **Summary Metrics**: Aggregated statistics by frame type
- **Comprehensive Comparison**: Cross-type analysis

### **3D Reconstruction Output**

- **Point Clouds**: `.ply` files for visualization
- **Reconstruction Stats**: Quality metrics (reprojection error, track length, etc.)
- **Comparison Data**: Quality degradation analysis across blur types

### **Generated Reports**

- **Complete Analysis Report**: Comprehensive pipeline summary
- **Reconstruction Summary**: 3D quality comparison across frame types
- **Execution Logs**: Detailed pipeline execution tracking

## 🎯 **Key Features**

### **1. Unified Processing**

- Single command processes original, blurred, and deblurred frames
- Automatic discovery of available frame sets
- Consistent output organization

### **2. Quality Assessment**

- **Image Quality Metrics**: BRISQUE, sharpness, SSIM, PSNR
- **3D Reconstruction Quality**: Point count, reprojection error, track length
- **Comparative Analysis**: Quantitative blur impact assessment

### **3. Flexible Execution**

- Skip individual pipeline steps as needed
- Process specific blur types or intensities
- Resume interrupted processing

## 📈 **Analysis Workflow**

### **Step 1: Frame Processing**

```bash
# Extract frames with stride and apply blur effects
python blur/simple_blur_pipeline.py --video pat3.mp4 --stride 60
```

### **Step 2: Quality Metrics**

```bash
# Calculate comprehensive quality metrics
python blur/metrics/updated_calculator.py --video pat3.mp4
```

### **Step 3: 3D Reconstruction**

```bash
# Run SfM reconstruction on all frame types
python reconstruction/reconstruction_pipeline.py --video pat3.mp4 --feature disk --matcher disk+lightglue
```

### **Step 4: Analysis**

Results are automatically organized and compared:

- Sharpness degradation across blur types
- 3D reconstruction quality impact
- Optimal deblurring method identification

## 🔍 **Understanding Results**

### **Metrics Interpretation**

- **Higher Laplacian Variance** = Sharper images
- **Lower BRISQUE Score** = Better perceptual quality
- **Higher SSIM/PSNR** = Better similarity to original

### **3D Reconstruction Quality**

- **More 3D Points** = Denser reconstruction
- **Lower Reprojection Error** = More accurate geometry
- **Longer Track Length** = More robust feature matching

### **Blur Impact Assessment**

Compare metrics across:

1. **Original frames** (baseline quality)
2. **Blurred variants** (degradation analysis)
3. **Deblurred frames** (restoration effectiveness)

## 🛠 **Troubleshooting**

### **Common Issues**

1. **Maploc Import Errors**

   ```bash
   python reconstruction/setup_vapor_maploc.py
   ```

2. **Missing COLMAP**

   - COLMAP is included via pycolmap, no separate installation needed

3. **GPU Memory Issues**

   - Reduce image resolution in preprocessing
   - Use smaller feature detector configurations

4. **Reconstruction Failures**
   - Check if sufficient image overlap exists
   - Verify feature matching quality
   - Consider reducing geometric verification thresholds

### **Performance Optimization**

- Use **stride > 1** for faster processing on long videos
- Choose **lighter feature detectors** (SIFT vs DISK) for speed
- Process **specific blur types** instead of all variants

## 📝 **Example Usage**

```bash
# Complete analysis for pat3.mp4
python vapor_complete_pipeline.py --video pat3.mp4

# Results will be saved to:
# - data/metrics/pat3/       (quality metrics)
# - data/point_clouds/pat3/  (3D reconstructions)
# - data/reports/            (analysis summary)
```

This integrated pipeline provides comprehensive analysis of how blur affects both 2D image quality and 3D reconstruction capabilities, enabling quantitative evaluation of deblurring methods for computer vision applications.
