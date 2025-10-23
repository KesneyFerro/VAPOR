# VAPOR Directory Structure

This document describes the organized directory structure of the VAPOR (Video Analysis Processing for Object Recognition) project after cleanup and reorganization.

## 📁 **Root Directory Structure**

```
VAPOR/
├── blur/                           # Blur processing pipeline
│   ├── fx_01_blur/                # Blur generation effects
│   │   ├── kernels/               # Deterministic kernel generation
│   │   └── effects/               # Blur engines and effects
│   ├── fx_02_deblur/             # Deblurring models integration
│   │   ├── DeblurGANv2/          # DeblurGAN v2 model
│   │   ├── Restormer/            # Restormer model
│   │   ├── Uformer/              # Uformer model
│   │   ├── MPRNet/               # MPRNet model
│   │   ├── DPIR/                 # DPIR model
│   │   ├── DeblurDiNAT/          # DeblurDiNAT model
│   │   ├── deblur_example_usage.py  # Example deblurring usage
│   │   └── setup_repositories.py # Repository setup script
│   ├── metrics/                  # Quality metrics calculation
│   │   ├── metrics_calculator.py # Unified metrics calculator
│   │   ├── full_reference.py     # Full-reference metrics (SSIM, PSNR)
│   │   ├── no_reference.py       # No-reference metrics (BRISQUE)
│   │   └── sharpness.py          # Sharpness metrics
│   ├── runs/                     # Experiment logging
│   │   ├── experiment_logger.py  # Experiment tracking
│   │   └── vapor_logger.py       # VAPOR-specific logging
│   ├── blur_example_usage.py     # Example blur pipeline usage
│   └── blur_generator.py         # Main blur generation pipeline
├── data/                         # Data organization (NEW STRUCTURE)
│   ├── videos/
│   │   ├── original/             # Input videos
│   │   └── blurred/              # Generated blurred videos
│   ├── frames/
│   │   ├── original/{video}/     # Original extracted frames
│   │   ├── blurred/{video}/      # Blurred frame variations
│   │   └── deblurred/{video}/    # Deblurred frames
│   ├── metrics/{video}/
│   │   ├── original/             # Metrics for original frames
│   │   ├── blurred/              # Metrics for blurred frames
│   │   └── deblurred/            # Metrics for deblurred frames
│   ├── point_clouds/{video}/
│   │   ├── original/             # 3D reconstructions from original
│   │   ├── blurred/              # 3D reconstructions from blurred
│   │   └── deblurred/            # 3D reconstructions from deblurred
│   ├── reports/                  # Analysis reports
│   ├── logs/                     # Pipeline execution logs
│   └── scripts/                  # Data processing utilities
│       ├── frame_extractor.py    # Frame extraction utilities
│       └── video_reconstructor.py # Video reconstruction utilities
├── docs/                         # Documentation (ORGANIZED)
│   ├── 3D_Reconstruction_Integration.md # 3D reconstruction guide
│   ├── Directory_Structure.md    # This file
│   └── Setup_Guides/             # Setup documentation
├── floating_objects/             # Object detection module
├── preprocessing/                # Preprocessing utilities
├── reconstruction/               # 3D reconstruction pipeline
│   ├── maploc/                   # Maploc SfM integration
│   ├── reconstruction_pipeline.py # 3D reconstruction pipeline
│   ├── setup_vapor_maploc.py     # Maploc integration setup
│   └── __init__.py
├── specularity/                  # Specularity detection
├── utils/                        # Core utilities
│   └── core_utilities.py         # Shared utility functions
├── vapor_complete_pipeline.py    # Master pipeline controller
├── README.md                     # Main project documentation
├── requirements.txt              # Python dependencies
└── roadmap.md                    # Development roadmap
```

## 🔄 **Data Flow Structure**

### **Input → Processing → Output**

```
INPUT:
data/videos/original/{video}.mp4

PROCESSING:
1. Frame Extraction → data/frames/original/{video}/
2. Blur Application → data/frames/blurred/{video}/{blur_type}_{intensity}/
3. Metrics Calculation → data/metrics/{video}/{original/blurred/deblurred}/
4. 3D Reconstruction → data/point_clouds/{video}/{original/blurred/deblurred}/

OUTPUT:
- Blurred videos: data/videos/blurred/
- Quality metrics: data/metrics/{video}/
- 3D point clouds: data/point_clouds/{video}/
- Analysis reports: data/reports/
```

## 📊 **Key Directory Changes**

### **Before (Old Structure)**

```
- Scattered metrics files
- Mixed frame types in same directories
- No organized 3D reconstruction output
- Limited experiment tracking
```

### **After (New Structure)**

```
+ Organized by video → frame type → method
+ Separate metrics directories by frame type
+ Dedicated point cloud organization
+ Comprehensive logging and reporting
+ Clear separation of concerns
```

## 🎯 **Directory Purpose**

| Directory            | Purpose                  | Key Files                                        |
| -------------------- | ------------------------ | ------------------------------------------------ |
| `blur/`              | Blur processing pipeline | `blur_generator.py`, `blur_example_usage.py`     |
| `blur/fx_01_blur/`   | Blur generation          | `kernels/generator.py`, `effects/blur_engine.py` |
| `blur/fx_02_deblur/` | Deblurring models        | `deblur_example_usage.py`, model directories     |
| `blur/metrics/`      | Quality assessment       | `metrics_calculator.py`, metric modules          |
| `data/`              | Organized data storage   | Frame, metric, and point cloud files             |
| `docs/`              | Documentation            | Setup guides, integration docs                   |
| `reconstruction/`    | 3D reconstruction        | `maploc/` integration                            |
| `utils/`             | Shared utilities         | `core_utilities.py`                              |

## 🚀 **Quick Navigation**

### **For Blur Processing:**

- Main pipeline: `blur/blur_generator.py`
- Example usage: `blur/blur_example_usage.py`
- Kernel generation: `blur/fx_01_blur/kernels/generator.py`
- Blur effects: `blur/fx_01_blur/effects/blur_engine.py`

### **For Deblurring:**

- Example usage: `blur/fx_02_deblur/deblur_example_usage.py`
- Setup guide: `blur/fx_02_deblur/DEBLUR_SETUP_GUIDE.md`

### **For Metrics:**

- Calculator: `blur/metrics/metrics_calculator.py`

### **For 3D Reconstruction:**

- Pipeline: `reconstruction/reconstruction_pipeline.py`
- Setup: `reconstruction/setup_vapor_maploc.py`

### **For Complete Analysis:**

- Master pipeline: `vapor_complete_pipeline.py`

## 📋 **File Organization Principles**

1. **Separation by Function**: Each module has its dedicated directory
2. **Data Organization**: Structured by video → type → method hierarchy
3. **Documentation Centralization**: All docs in `docs/` directory
4. **Utility Sharing**: Common functions in `utils/`
5. **Clear Entry Points**: Main scripts at root level for easy access

This organized structure enables efficient navigation, maintenance, and extension of the VAPOR system.
