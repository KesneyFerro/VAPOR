# VAPOR - Video Analysis Processing for Object Recognition

A modular video analysis system designed for advanced object recognition and processing. The project supports multiple analysis modules including specularity detection, with planned extensions for blur detection and floating object recognition.

## Project Structure

```
VAPOR/
├── README.md
├── requirements.txt
├── main.py                     # Main entry point
├── .gitignore
├──
├── blur/                       # Future blur detection module
├── specularity/                # Specularity detection module
│   ├── core/                   # Core processing classes
│   │   ├── video_processor.py  # Main VideoFrameProcessor class
│   │   ├── frame_operations.py # Image manipulation operations
│   │   └── display_manager.py  # Grid display and UI management
│   ├── utils/                  # Utility functions
│   │   ├── content_detection.py # Content bounds detection
│   │   └── pipeline_manager.py  # Pipeline management
│   ├── examples/               # Usage examples
│   │   ├── basic_usage.py      # Basic usage example
│   │   └── rgb_test_generator.py # RGB test video generator
│   └── tests/                  # Test modules
│       ├── test_core_functionality.py
│       ├── test_display_system.py
│       ├── test_responsive_layout.py
│       └── test_pipeline_system.py
├── floating_objects/           # Future floating objects detection
├── data/                       # Data storage and processing
│   ├── videos/                 # Video files storage
│   ├── extracted_frames/       # Processed frames storage
│   ├── utils/                  # Data utilities
│   │   └── filename_normalizer.py # Video filename normalization
│   └── scripts/
│       └── frame_extractor.py  # Frame extraction utility
└── venv/                       # Virtual environment
```

## Installation

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

1. Place your video files in the `data/videos/` directory
2. Run the main application:

```bash
python main.py
```

3. Select a video from the numbered list when prompted

The application will automatically normalize video filenames (replacing spaces with underscores and removing illegal characters) before displaying the selection menu.

### Video Selection

When you run the program, it will:

- Automatically normalize any video filenames in `data/videos/`
- Display a numbered list of available videos
- Ask you to select a video by entering its number
- If no videos are found, prompt you to add videos to the directory

### Controls

Once a video is loaded, you can use various keyboard controls to navigate and process frames. Refer to the module-specific documentation for detailed control information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code structure and naming conventions
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request
