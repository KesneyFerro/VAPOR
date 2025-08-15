# VAPOR - Video Analysis Processing for Object Recognition

A comprehensive video analysis system designed for advanced medical imaging processing, blur detection, and object recognition. The project provides a modular framework for systematic analysis of video quality, focusing on CT scan reconstruction quality assessment.

## Technical Overview

VAPOR is structured as a modular system for processing and analyzing video data, with an initial focus on the impact of video quality (specifically blur) on 3D reconstruction quality. The system implements several key components:

### 1. Blur Generation and Analysis Pipeline

The blur module provides a systematic approach to generating controlled blur effects on video frames. This is essential for evaluating deblurring algorithms and analyzing how different types of blur affect reconstruction quality. The implemented blur types include:

- **Gaussian Blur**: Simulates lens defocus with a normal distribution kernel
- **Motion Blur**: Simulates camera or object movement during exposure
- **Out-of-Focus Blur**: Simulates optical defocus with a circular kernel
- **Average Blur**: Simple box filter blur
- **Median Blur**: Non-linear blur that preserves edges
- **Combined Blur**: Sequential application of motion, out-of-focus, and median blur

Each blur type is implemented with both low and high intensity settings to provide a range of testing conditions.

### 2. Content Detection System

The system incorporates automatic content detection to:

1. Identify regions of interest in video frames
2. Apply optimized cropping to focus processing on relevant areas
3. Maintain proper aspect ratios and sizing during processing
4. Efficiently handle multi-frame processing with consistent parameters

Content detection utilizes adaptive thresholding and contour analysis to determine optimal crop boundaries across video frames.

### 3. Frame Extraction and Processing

VAPOR implements robust frame extraction with:

- Configurable frame stride for variable processing density
- Parallel extraction of original and processed frames
- Proper frame indexing for reconstruction
- Structured directory organization for efficient data management

### 4. Video Reconstruction

The system preserves video quality characteristics through:

- Original frame size maintenance with black padding
- Consistent framerate and codec application
- Proper metadata transfer from source to output
- Quality preservation during compression

## Project Structure

```
VAPOR/
├── README.md
├── requirements.txt
├── main.py                        # Project entry point
├── blur/                          # Blur generation module
│   ├── main.py                    # Blur pipeline implementation
│   ├── blurring/                  # Blur effect implementations
│   │   ├── batch_blur.py          # Batch processing functions
│   │   └── blur_effects.py        # Individual blur algorithms
│   └── deblurring/                # Future deblurring algorithms
├── specularity/                   # Specularity detection module
│   ├── core/                      # Core processing classes
│   │   ├── video_processor.py     # Video frame processing
│   │   ├── frame_operations.py    # Image manipulation operations
│   │   └── display_manager.py     # Grid display and UI management
│   ├── utils/                     # Utility functions
│   │   ├── content_detection.py   # Content bounds detection
│   │   └── pipeline_manager.py    # Pipeline management
│   ├── examples/                  # Usage examples
│   └── tests/                     # Test modules
├── floating_objects/              # Future floating objects detection
├── data/                          # Data storage and processing
│   ├── videos/                    # Video files storage
│   │   ├── original/              # Original video files
│   │   ├── blurred/               # Artificially blurred videos
│   │   └── deblurred/             # Processed deblurred videos
│   ├── extracted_frames/          # Processed frames storage
│   │   ├── original/              # Frames from original videos
│   │   ├── blurred/               # Artificially blurred frames
│   │   └── deblurred/             # Processed deblurred frames
│   ├── utils/                     # Data utilities
│   │   └── filename_normalizer.py # Filename normalization
│   └── scripts/                   # Processing scripts
│       ├── frame_extractor.py     # Frame extraction utility
│       └── video_reconstructor.py # Video reconstruction from frames
```

## Technical Implementation Details

### Blur Generation Pipeline

The blur generation pipeline processes videos through several stages:

1. **Video Selection and Configuration**:
   - Interactive user selection of source videos
   - Configurable frame stride for processing density
   - Automatic video property extraction (resolution, FPS, frame count)

2. **Content Detection**:
   - Adaptive sampling of video frames for content bounds detection
   - Statistical analysis to determine optimal crop region
   - Single-pass detection to ensure consistent processing

3. **Frame Processing**:
   - Systematic extraction of frames at specified stride
   - Storage of original frames for reference
   - Application of 6 blur types at 2 intensity levels (12 variations)
   - Preservation of frame numbering for reconstruction

4. **Video Reconstruction**:
   - Assembly of processed frames into videos
   - Maintenance of original video properties
   - Proper padding to preserve aspect ratio

### Implementation Technologies

The system uses several key technologies:

- **OpenCV**: Core image and video processing
- **NumPy**: Efficient numerical operations on frame data
- **SciPy**: Advanced filter implementations for specialized blur effects
- **Pathlib**: Modern file system operations
- **Multi-processing**: Parallel processing capabilities for performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KesneyFerro/VAPOR.git
cd VAPOR
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Blur Generation Pipeline

To generate blurred videos with various effects:

```bash
# Process all frames
python blur/main.py

# Process every 5th frame (faster)
python blur/main.py --stride 5

# Process every 10th frame (fastest)
python blur/main.py --stride 10
```

The pipeline will:
1. Present available videos from `data/videos/original/`
2. Let you choose a video and processing density
3. Extract frames and apply multiple blur effects
4. Create blurred videos and store extracted frames

### Frame Management

Extracted frames are stored in:
- Original frames: `data/extracted_frames/original/{video_name}/`
- Blurred frames: `data/extracted_frames/blurred/{video_name}/{blur_type}_{intensity}/`

Resulting videos are stored in:
- `data/videos/blurred/`

## Evaluation Metrics and Analysis

The current implementation includes the groundwork for comprehensive blur quality analysis. The system processes videos through controlled blur introduction to establish a basis for evaluating deblurring algorithms and their impact on 3D reconstruction.

### Current Blur Assessment Framework

1. **Visual Comparison**:
   - Side-by-side comparison of original, blurred, and deblurred frames
   - Evaluation of detail preservation across different blur types
   - Analysis of blur intensity effects on feature detection

2. **Quality Control**:
   - Consistent blur application across video frames
   - Preserved video properties (resolution, aspect ratio)
   - Frame alignment maintenance for reconstruction comparison

## Future Work

The VAPOR project roadmap includes several significant enhancements:

### 1. Advanced Deblurring Integration

Future development will implement more than 10 different deblurring techniques to process blurred videos, including:

- Deep learning-based approaches (DeblurGAN, SRN-Deblur)
- Traditional deconvolution methods (Wiener, Richardson-Lucy)
- Blind deblurring algorithms
- Multi-scale deblurring approaches
- Frequency domain deblurring methods

These implementations will provide a comprehensive evaluation framework for deblurring effectiveness.

### 2. Quantitative Blur Metrics Implementation

The system will integrate advanced blur measurement metrics:

- **Fast Fourier Transform (FFT)** analysis for frequency domain evaluation
- **Laplacian operator** for edge detection and sharpness assessment
- **Variance of Laplacian** for focus measure
- **Image gradient magnitude** for sharpness estimation
- **Power spectrum analysis** for frequency distribution evaluation

These metrics will enable objective quality assessment and comparison across different blur types and deblurring methods.

### 3. Reconstruction Quality Assessment

A key future objective is to correlate blur metrics with 3D reconstruction quality:

- Compare point clouds from original, blurred, and deblurred video reconstructions
- Assess point cloud density, accuracy, and feature preservation
- Evaluate reconstruction against CT ground truth data
- Develop predictive models for estimating reconstruction quality from blur metrics
- Establish threshold values for acceptable blur levels in medical imaging applications

### 4. Marginal Gain Analysis Framework

The system will implement analysis to determine the marginal benefit of deblurring:

- Quantify improvement in reconstruction quality per unit of deblurring effort
- Establish cost-benefit models for computational investment in deblurring
- Develop predictive algorithms to identify frames where deblurring will yield significant improvements
- Create automated decision systems for selective deblurring based on expected gains

### 5. Extended Analysis Tools

Additional planned features include:

- Comprehensive reporting and visualization tools for blur metrics
- Batch processing capabilities for large datasets
- Statistical analysis framework for comparing deblurring methods
- Integration with 3D reconstruction pipelines
- Real-time blur assessment for live video processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for image processing libraries
- SciPy project for advanced mathematical operations
- Research contributors in the fields of image processing and medical imaging

1. Clone the repository:

```bash
git clone <repository-url>
cd VAPOR
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\\Scripts\\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Place your video files in the `data/videos/original/` directory
2. Run the main application:

```bash
python main.py
```

3. Select a video from the numbered list when prompted

The application will automatically normalize video filenames (replacing spaces with underscores and removing illegal characters) before displaying the selection menu.

### Video Selection

When you run the program, it will:

- Automatically normalize any video filenames in `data/videos/original/`
- Display a numbered list of available videos
- Ask you to select a video by entering its number
- If no videos are found, prompt you to add videos to the original directory

### Controls

Once a video is loaded, you can use various keyboard controls to navigate and process frames. Refer to the module-specific documentation for detailed control information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code structure and naming conventions
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request
