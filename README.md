# VAPOR - Video Analysis Processing for Object Recognition

A comprehensive video analysis system that combines blur processing with 3D reconstruction analysis. VAPOR integrates deterministic blur generation, multiple deblurring models, comprehensive quality metrics calculation, and Structure from Motion (SfM) reconstruction to provide quantitative analysis of how blur affects both 2D image quality and 3D reconstruction capabilities.

## ✅ **Project Status: Complete and Organized**

The VAPOR system has been fully organized with comprehensive functionality:

- ✅ **Complete blur processing pipeline** with 5 blur types and 6 deblurring models
- ✅ **Comprehensive metrics system** with full-reference, no-reference, and sharpness metrics
- ✅ **3D reconstruction integration** with maploc SfM pipeline
- ✅ **Organized directory structure** with systematic data organization
- ✅ **Unified pipelines** for complete analysis workflows
- ✅ **Comprehensive documentation** and setup guides

## 🎯 **Quick Start**

### **Option 1: Complete Analysis Pipeline (Recommended)**

```bash
# Run everything for a video (pat3.mp4)
python vapor_complete_pipeline.py --video pat3.mp4

# Run with selective steps
python vapor_complete_pipeline.py --video pat3.mp4 --skip-blur --skip-reconstruction
```

### **Option 2: Individual Components**

```bash
# 1. Generate blurred frames and videos (tested with pat3.mp4, stride 60)
python blur/simple_blur_pipeline.py --video pat3.mp4 --stride 60

# 2. Calculate comprehensive quality metrics with new structure
python blur/metrics/updated_calculator.py --video pat3.mp4

# 3. Run 3D reconstruction analysis using maploc integration
python reconstruction/reconstruction_pipeline.py --video pat3.mp4
```

## 📁 **Directory Structure**

VAPOR now organizes data systematically with comprehensive separation of concerns:

```
data/
├── videos/
│   ├── original/           # Input videos (pat3.mp4, etc.)
│   └── blurred/           # Generated blurred videos
├── frames/
│   ├── original/{video}/   # Original extracted frames
│   ├── blurred/{video}/    # Blurred frame variations (gaussian_low, motion_blur_high, etc.)
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

## 🔧 **Installation & Setup**

### **1. Basic Setup**

```bash
# Clone the repository
git clone <repository-url>
cd VAPOR

# Install basic dependencies
pip install -r requirements.txt
```

### **2. Setup 3D Reconstruction (Maploc Integration)**

```bash
# Setup maploc integration for 3D reconstruction
python reconstruction/setup_vapor_maploc.py
```

### **3. Setup Deblurring Models (Optional)**

```bash
# Setup all 6 deblurring models with conda environments
python blur/fx_02_deblur/setup_repositories.py --all

# See docs/Setup_Guides/Deblur_Models_Setup.md for detailed instructions
```

## 🚀 **Usage Examples**

### **Complete Analysis (Recommended)**

```bash
# Run complete analysis pipeline on pat3.mp4
python vapor_complete_pipeline.py --video pat3.mp4

# Skip blur generation if frames already exist
python vapor_complete_pipeline.py --video pat3.mp4 --skip-blur

# Run only metrics and reconstruction
python vapor_complete_pipeline.py --video pat3.mp4 --skip-blur --skip-metrics
```

### **Individual Components**

```bash
# Generate blur effects (tested configuration)
python blur/simple_blur_pipeline.py --video pat3.mp4 --stride 60

# Calculate comprehensive metrics
python blur/metrics/updated_calculator.py --video pat3.mp4

# Run 3D reconstruction with specific algorithms
python reconstruction/reconstruction_pipeline.py --video pat3.mp4 --feature disk --matcher disk+lightglue

# Setup deblurring environments
python blur/fx_02_deblur/setup_repositories.py --method Restormer

# Run deblurring with specific method
python blur/fx_02_deblur/deblur_cli.py --input blurred_frames/ --output deblurred/ --method Restormer
```

### **Advanced Configuration**

```bash
# Use different feature detectors for 3D reconstruction
python reconstruction/reconstruction_pipeline.py --video pat3.mp4 --feature superpoint --matcher superglue

# Process with custom stride for faster processing
python blur/simple_blur_pipeline.py --video pat3.mp4 --stride 120

# Setup specific deblurring method only
python blur/fx_02_deblur/setup_repositories.py --method DeblurGANv2
```

## 🚀 **Core Features**

### **1. Comprehensive Blur Processing**

**Deterministic Blur Generation (blur/fx_01_blur/):**

- **Gaussian Blur**: Standard kernel-based blur with configurable sigma
- **Motion Blur**: Linear motion simulation with configurable angles
- **Defocus Blur**: Circular out-of-focus effects
- **Haze Blur**: Atmospheric scattering simulation
- **Combined Blur**: Sequential application of multiple effects
- **Deterministic Kernels**: Reproducible blur generation with fixed seeds

**Advanced Deblurring (blur/fx_02_deblur/):**

- **6 Integrated Models**: DeblurGANv2, Restormer, Uformer, MPRNet, DPIR, DeblurDiNAT
- **Unified CLI**: `blur/fx_02_deblur/deblur_cli.py` with conda environment switching
- **Environment Management**: Automatic conda environment setup and switching
- **Batch Processing**: Process entire directories with multiple methods

### **2. Quality Assessment System (blur/metrics/)**

**Multi-Level Metrics:**

- **No-Reference**: BRISQUE, NIQE (works on any image)
- **Full-Reference**: SSIM, PSNR (compares to original)
- **Sharpness Analysis**: Laplacian variance, gradient magnitude, total variation

**Organized Output:**

- Frame-by-frame detailed analysis
- Summary statistics by frame type (original/blurred/deblurred)
- Cross-comparison between blur methods and intensities

### **3. 3D Reconstruction Integration**

**Structure from Motion (SfM) with Maploc:**

- **Feature Detectors**: DISK, SuperPoint, SIFT, ALIKED
- **Feature Matchers**: LightGlue, SuperGlue, NN-ratio
- **Quality Metrics**: Point count, reprojection error, track length
- **COLMAP Backend**: Professional-grade 3D reconstruction

**Reconstruction Analysis:**

- Compare 3D quality across frame types (original/blurred/deblurred)
- Quantify blur impact on reconstruction quality
- Export point clouds (.ply format) for visualization

### **4. Comprehensive Logging System (blur/runs/)**

**Reproducible Experiments:**

- **Complete Tracking**: All parameters, seeds, and configurations logged
- **Performance Monitoring**: CPU, memory, and timing metrics
- **Environment Recording**: System information and dependency versions
- **Result Comparison**: Tools for analyzing multiple experimental runs

## 📊 **Analysis Workflow**

### **Step 1: Frame Processing & Blur Generation**

Extract frames from video and apply systematic blur effects using deterministic kernels:

```bash
python blur/simple_blur_pipeline.py --video pat3.mp4 --stride 60
```

- Tested with pat3.mp4 using stride 60 for efficient processing
- Generates 5 blur types × 2 intensities = 10 blur variations per frame
- Uses deterministic kernel generation for reproducible results

### **Step 2: Quality Metrics Calculation**

Calculate comprehensive quality metrics for all frame types using the new organized structure:

```bash
python blur/metrics/updated_calculator.py --video pat3.mp4
```

- Processes original, blurred, and deblurred frames separately
- Saves detailed and summary metrics in organized directory structure
- Supports both no-reference and full-reference quality assessment

### **Step 3: 3D Reconstruction Analysis**

Run SfM reconstruction on original, blurred, and deblurred frames using maploc integration:

```bash
python reconstruction/reconstruction_pipeline.py --video pat3.mp4 --feature disk --matcher disk+lightglue
```

- Compares reconstruction quality across different blur conditions
- Exports point clouds and quality metrics for each frame type
- Quantifies how blur affects 3D reconstruction capabilities

### **Step 4: Comprehensive Analysis**

Results are automatically organized and compared across:

- **Image Quality**: Sharpness degradation, perceptual quality (BRISQUE, SSIM, PSNR)
- **3D Reconstruction**: Point density, geometric accuracy, feature matching quality
- **Comparative Analysis**: Optimal deblurring method identification for 3D applications

## 🎯 **Key Improvements & Features**

### **Organized Data Structure**

- Systematic organization by video → frame type → method hierarchy
- Clear separation of original, blurred, and deblurred data across all analysis types
- Comprehensive logging and experiment tracking for reproducibility

### **Integrated 3D Analysis**

- **Maploc Integration**: Professional SfM reconstruction pipeline with COLMAP backend
- **Quality Assessment**: Quantitative 3D reconstruction analysis with multiple metrics
- **Blur Impact Measurement**: Direct measurement of blur effects on 3D reconstruction quality

### **Unified Processing & Reproducibility**

- **Single Command**: Complete analysis from video to 3D reconstruction
- **Flexible Execution**: Skip individual steps as needed with command-line flags
- **Resumable Processing**: Continue interrupted analyses with existing data
- **Deterministic Processing**: Fixed seeds and reproducible kernel generation
- **Comprehensive Logging**: Complete experiment tracking with performance monitoring

## 📖 **Documentation**

Complete documentation is organized in the `docs/` directory:

- **[3D Reconstruction Integration](docs/3D_Reconstruction_Integration.md)**: Complete guide to 3D analysis with maploc
- **[Directory Structure](docs/Directory_Structure.md)**: Detailed file organization and navigation guide
- **[Deblur Models Setup](docs/Setup_Guides/Deblur_Models_Setup.md)**: Complete setup guide for all 6 deblurring models

## 🔬 **Technical Architecture**

### **Core Modules**

- **`blur/`**: Complete blur processing pipeline with deterministic effects
  - `fx_01_blur/`: Blur generation with deterministic kernel system (`kernels/generator.py`)
  - `fx_02_deblur/`: 6 integrated deblurring models with unified CLI (`deblur_cli.py`)
  - `metrics/`: Comprehensive quality assessment system (`calculator.py`)
  - `runs/`: Advanced experiment logging and tracking (`experiment_logger.py`, `vapor_logger.py`)
- **`reconstruction/`**: 3D reconstruction pipeline with maploc integration
- **`utils/`**: Shared utilities for video processing, ROI detection, and file management (`core_utilities.py`)
- **`data/`**: Organized data storage with systematic hierarchy by video/type/method

### **Key Scripts & Their Status**

- **`vapor_complete_pipeline.py`**: ✅ Master controller for complete analysis - **TESTED & WORKING**
- **`blur/simple_blur_pipeline.py`**: ✅ Simplified blur processing pipeline - **TESTED with pat3.mp4, stride 60**
- **`metrics_calculator.py`**: ✅ Updated metrics calculation with new structure - **COMPLETE**
- **`reconstruction_pipeline.py`**: ✅ 3D reconstruction with maploc integration - **COMPLETE**
- **`setup_vapor_maploc.py`**: ✅ Setup script for maploc integration - **COMPLETE**

### **Blur Processing Pipeline (blur/fx_01_blur/)**

**Deterministic Kernel Generation:**

- Reproducible blur effects using fixed random seeds
- Multiple blur types: Gaussian, Motion, Defocus, Haze, Combined
- Configurable intensity levels with consistent parameters
- Complete metadata logging for experiment reproducibility

**Enhanced Blur Engine:**

- Scientific-grade blur application with proper normalization
- Advanced motion blur with configurable angles and lengths
- Atmospheric effects simulation for realistic degradation
- Batch processing with progress tracking

### **Comprehensive Metrics System (blur/metrics/)**

**Unified Calculator (`calculator.py`):**

- Integrates all metrics types in single interface
- Organized output with detailed and summary statistics
- Support for comparative analysis across blur methods

**Full-Reference Metrics (`full_reference.py`):**

- SSIM (Structural Similarity Index) with multiple scales
- PSNR (Peak Signal-to-Noise Ratio) calculation
- MSE (Mean Squared Error) analysis

**No-Reference Metrics (`no_reference.py`):**

- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
- NIQE (Natural Image Quality Evaluator)
- Advanced perceptual quality assessment

**Sharpness Metrics (`sharpness.py`):**

- Laplacian variance (primary sharpness measure)
- Gradient magnitude analysis
- Total variation calculation
- Multi-scale sharpness assessment

### **Advanced Deblurring Integration (blur/fx_02_deblur/)**

**Unified CLI System (`deblur_cli.py`):**

- Single interface for 6 different deblurring models
- Automatic conda environment switching
- Batch processing with progress tracking
- Performance comparison across methods

**Supported Models:**

- **DeblurGANv2**: GAN-based deblurring with adversarial training
- **Restormer**: Transformer-based restoration with multi-scale processing
- **Uformer**: U-Net with transformer blocks for efficient deblurring
- **DeblurDiNAT**: Dilated neighborhood attention transformer
- **DPIR**: Deep plug-and-play image restoration
- **MPRNet**: Multi-patch relationship network

**Environment Management (`setup_repositories.py`):**

- Automated repository cloning and environment setup
- Dependency management for each model
- Pre-trained model downloading assistance
- Environment documentation and verification

## 🏃‍♂️ **Example Workflows**

### **Complete Video Analysis**

```bash
# Process pat3.mp4 with complete pipeline
python vapor_complete_pipeline.py --video pat3.mp4

# Output structure:
# data/
# ├── frames/pat3/{original,blurred,deblurred}/
# ├── metrics/pat3/{original,blurred,deblurred}/
# ├── point_clouds/pat3/{original,blurred,deblurred}/
# └── reports/pat3_complete_analysis_report.txt
```

### **Selective Processing**

```bash
# Only blur processing (skip 3D reconstruction)
python vapor_complete_pipeline.py --video pat3.mp4 --skip-reconstruction

# Only metrics and reconstruction (frames already exist)
python vapor_complete_pipeline.py --video pat3.mp4 --skip-blur
```

### **Manual Step-by-Step**

```bash
# 1. Generate blurred frames
python blur/simple_blur_pipeline.py --video pat3.mp4 --stride 60

# 2. Calculate quality metrics
python blur/metrics/updated_calculator.py --video pat3.mp4

# 3. Run 3D reconstruction
python reconstruction/reconstruction_pipeline.py --video pat3.mp4 --feature disk
```

## 💡 **Understanding Results**

### **Quality Metrics Interpretation**

- **Higher Laplacian Variance** = Sharper images
- **Lower BRISQUE Score** = Better perceptual quality
- **Higher SSIM/PSNR** = Better similarity to original

### **3D Reconstruction Quality**

- **More 3D Points** = Denser reconstruction
- **Lower Reprojection Error** = More accurate geometry
- **Longer Track Length** = More robust feature matching

### **Blur Impact Assessment**

Compare across:

1. **Original frames** (baseline quality)
2. **Blurred variants** (degradation analysis)
3. **Deblurred frames** (restoration effectiveness)

## 🛠 **Development & Contributing**

### **Project Structure**

- `blur/`: Core blur processing and deblurring
- `reconstruction/`: 3D reconstruction with maploc
- `utils/`: Shared utilities and core functions
- `data/`: Organized data storage hierarchy
- `docs/`: Comprehensive documentation

### **Key Design Principles**

- **Modular Architecture**: Each component is self-contained
- **Reproducible Processing**: All operations use fixed seeds
- **Organized Output**: Systematic data hierarchy
- **Flexible Execution**: Skip/resume individual steps

## 📚 **Research Applications**

VAPOR is designed for research in:

- **Image Quality Assessment**: Quantitative blur impact measurement
- **Deblurring Algorithm Evaluation**: Systematic comparison framework
- **3D Reconstruction Analysis**: SfM quality under varying conditions
- **Medical Imaging**: CT scan reconstruction quality assessment

## ✅ **Recent Achievements & Testing Status**

### **Successfully Tested Components**

- **✅ Blur Pipeline**: Successfully tested with pat3.mp4 using stride 60
- **✅ Deterministic Kernels**: Reproducible blur generation with fixed seeds implemented and working
- **✅ Metrics Calculation**: Comprehensive quality assessment system operational with new directory structure
- **✅ 3D Reconstruction**: Maploc integration framework established and functional
- **✅ Directory Organization**: New structure implemented and validated (data/metrics/{video}/{original/blurred/deblurred})
- **✅ Documentation**: Complete documentation system organized in docs/ directory

### **Current Pipeline Status**

| Component              | Status                 | Details                                              |
| ---------------------- | ---------------------- | ---------------------------------------------------- |
| Blur Generation        | ✅ **Working**         | Deterministic kernel system operational              |
| Quality Metrics        | ✅ **Working**         | Full-reference, no-reference, and sharpness metrics  |
| Deblurring Integration | ✅ **Ready**           | 6 models with unified CLI and environment management |
| 3D Reconstruction      | ✅ **Framework Ready** | Maploc integration established                       |
| Experiment Logging     | ✅ **Operational**     | Comprehensive tracking and reproducibility system    |
| File Organization      | ✅ **Implemented**     | Systematic data organization by video/type/method    |

### **Recent Validation & Testing**

The VAPOR system has been comprehensively tested and validated with:

- **pat3.mp4 video processing** with stride 60 frame extraction
- **Complete blur generation pipeline** with all 5 blur types and 2 intensities
- **Deterministic kernel generation** ensuring reproducible results across runs
- **Organized directory structure** separating original, blurred, and deblurred data
- **Comprehensive setup guides** and documentation system
- **File cleanup and organization** removing duplicates and establishing clear structure

### **System Readiness**

VAPOR is now ready for:

- **Research Applications**: Quantitative blur impact studies
- **Algorithm Evaluation**: Systematic deblurring method comparison
- **3D Analysis**: SfM reconstruction quality assessment
- **Production Use**: Reproducible video analysis workflows

## 🔬 **Citation**

If you use VAPOR in your research, please cite:

```bibtex
@software{vapor2025,
  title={VAPOR: Video Analysis Processing for Object Recognition},
  author={VAPOR Development Team},
  year={2025},
  url={https://github.com/KesneyFerro/VAPOR},
  note={Integrated blur processing and 3D reconstruction analysis system}
}
```

## 📞 **Support**

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Report bugs and feature requests via GitHub issues
- **Setup Help**: See `docs/Setup_Guides/` for detailed setup instructions

---

_VAPOR provides a comprehensive framework for analyzing the impact of blur on both 2D image quality and 3D reconstruction capabilities, enabling quantitative evaluation of deblurring methods for computer vision applications._
