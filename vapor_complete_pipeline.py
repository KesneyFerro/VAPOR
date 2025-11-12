"""
VAPOR Complete Analysis Pipeline
Master script that runs the complete VAPOR analysis pipeline:
1. Blur generation (if needed)
2. Metrics calculation with new directory structure  
3. 3D reconstruction on all frame types
4. Comprehensive comparison and analysis

Usage:
    python vapor_complete_pipeline.py --video pat3.mp4 [--skip-blur] [--skip-metrics] [--skip-reconstruction]
"""

import os
# Fix OpenMP runtime conflict that can cause 3D reconstruction crashes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
import subprocess
from pathlib import Path
import logging
from datetime import datetime
import yaml

class VAPORCompletePipeline:
    """Master pipeline controller for VAPOR analysis."""
    
    def __init__(self, video_name: str, skip_blur: bool = False, skip_metrics: bool = False, 
                 skip_reconstruction: bool = False, manual_crop: bool = False):
        """Initialize the complete pipeline.
        
        Args:
            video_name: Name of video file (e.g., 'pat3.mp4')
            skip_blur: Skip blur generation if frames already exist
            skip_metrics: Skip metrics calculation
            skip_reconstruction: Skip 3D reconstruction
            manual_crop: Use manual 4-corner or two-point crop selection
        """
        self.video_name = video_name
        self.video_stem = Path(video_name).stem
        self.skip_blur = skip_blur
        self.skip_metrics = skip_metrics
        self.skip_reconstruction = skip_reconstruction
        self.manual_crop = manual_crop
        
        # Generate unique run ID for this pipeline execution
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.base_dir = Path(__file__).parent
        self.load_config()
        self.setup_logging()
        
    def load_config(self):
        """Load configuration from YAML file."""
        config_file = self.base_dir / "config" / "pipeline_config.yaml"
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            # Default configuration
            self.config = {
                'video': {
                    'frame_extraction': {'stride': 5}
                }
            }
        
    def setup_logging(self):
        """Setup logging for the pipeline."""
        log_dir = self.base_dir / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"vapor_complete_{self.video_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Clear any existing handlers to avoid duplicates
        logging.getLogger().handlers.clear()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True  # Force reconfiguration
        )
        self.logger = logging.getLogger(__name__)
        
        # Log the setup
        self.logger.info(f"Logging initialized - log file: {log_file}")
        
    def check_prerequisites(self):
        """Check if all required components are available."""
        self.logger.info("Checking prerequisites...")
        
        # Check GPU status and show initial info
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                self.logger.info(f"[GPU STATUS] {gpu_count} CUDA device(s) available: {gpu_name}")
            else:
                self.logger.info("[GPU STATUS] CUDA not available, using CPU")
        except ImportError:
            self.logger.info("[GPU STATUS] PyTorch not available")
            
        # Also show nvidia-smi info if available
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,utilization.gpu', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 2:
                    self.logger.info(f"[NVIDIA-SMI] {gpu_info[0]}, {gpu_info[1]} memory")
        except Exception:
            pass
        
        # Check if video exists
        video_path = self.base_dir / "data" / "videos" / "original" / self.video_name
        if not video_path.exists():
            self.logger.error(f"Video not found: {video_path}")
            return False
            
        # Check if blur pipeline components exist
        blur_script = self.base_dir / "blur" / "blur_generator.py"
        if not self.skip_blur and not blur_script.exists():
            self.logger.error(f"Blur generator script not found: {blur_script}")
            return False
            
        # Check if metrics calculator exists
        metrics_script = self.base_dir / "blur" / "metrics" / "metrics_calculator.py"
        if not self.skip_metrics and not metrics_script.exists():
            self.logger.error(f"Metrics calculator not found: {metrics_script}")
            return False
            
        # Check if reconstruction pipeline exists
        recon_script = self.base_dir / "reconstruction" / "reconstruction_pipeline.py"
        if not self.skip_reconstruction and not recon_script.exists():
            self.logger.error(f"Reconstruction pipeline not found: {recon_script}")
            return False
            
        self.logger.info("[OK] All prerequisites checked")
        return True
        
    def run_blur_generation(self):
        """Run blur generation pipeline."""
        if self.skip_blur:
            self.logger.info("Skipping blur generation (--skip-blur)")
            return True
        
        try:
            # Check if frames already exist and if we should skip
            frames_dir = self.base_dir / "data" / "frames" / "original" / self.video_stem
            force_regenerate = self.config.get('video', {}).get('frame_extraction', {}).get('force_regenerate', True)
            skip_existing = self.config.get('pipeline', {}).get('skip_existing', False)
            
            # If --skip-blur is specified, always skip if frames exist
            if self.skip_blur and frames_dir.exists() and any(frames_dir.iterdir()):
                self.logger.info("Original frames already exist, skipping blur generation (--skip-blur)")
                return True
            
            # If skip_existing is enabled and frames exist, skip
            if skip_existing and frames_dir.exists() and any(frames_dir.iterdir()) and not force_regenerate:
                self.logger.info("[SKIP] Original frames already exist (skip_existing=True, force_regenerate=False)")
                return True
            elif frames_dir.exists() and any(frames_dir.iterdir()):
                if force_regenerate:
                    self.logger.info("Original frames exist but force_regenerate=True, will regenerate frames")
                else:
                    self.logger.info("Original frames exist and skip_existing=False, will regenerate frames")
                
            # Run blur generator
            blur_script = self.base_dir / "blur" / "blur_generator.py"
            stride = self.config.get('video', {}).get('frame_extraction', {}).get('stride', 5)
            max_frames = self.config.get('video', {}).get('frame_extraction', {}).get('max_frames', None)
            blur_types = self.config.get('blur', {}).get('types', ['motion_blur'])
            intensities = self.config.get('blur', {}).get('intensities', ['high'])
            
            # Get time cropping parameters from config
            time_crop_config = self.config.get('video', {}).get('time_crop', {})
            start_time = None
            duration = None
            if time_crop_config.get('enabled', False):
                start_time = time_crop_config.get('start_time', 0.0)
                duration = time_crop_config.get('duration', None)
                self.logger.info(f"Time cropping enabled: start={start_time}s, duration={duration}s")
            
            cmd = [sys.executable, str(blur_script), "--video", self.video_name, "--stride", str(stride)]
            if max_frames is not None:
                cmd.extend(["--max-frames", str(max_frames)])
            if blur_types:
                cmd.extend(["--blur-types"] + blur_types)
            if intensities:
                cmd.extend(["--intensities"] + intensities)
            if start_time is not None:
                cmd.extend(["--start-time", str(start_time)])
            if duration is not None:
                cmd.extend(["--duration", str(duration)])
            if self.manual_crop:
                cmd.append("--manual-crop")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                self.logger.info("[OK] Blur generation completed successfully")
                return True
            else:
                self.logger.error(f"Blur generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running blur generation: {e}")
            return False
    
    def _run_deblur_on_directory(self, input_dir, output_dir, method):
        """Helper method to run deblur processing on a directory.
        
        Args:
            input_dir: Path to input frames directory
            output_dir: Path to output directory
            method: Deblur method name (Restormer, MPRNet, or Uformer)
        """
        # Map method names to module script names and conda environments
        method_info = {
            'Restormer': {'script': 'restormer_module.py', 'conda_env': 'restormer'},
            'MPRNet': {'script': 'mprnet_module.py', 'conda_env': 'base'}, 
            'Uformer': {'script': 'uformer_module.py', 'conda_env': 'uformer'}
        }
        
        method_config = method_info.get(method)
        if not method_config:
            self.logger.warning(f"Unknown method: {method}, skipping...")
            return False
        
        # Build command to run in correct conda environment
        module_path = self.base_dir / "blur" / "fx_02_deblur" / "modules" / "blur_modules" / method_config['script']
        
        # Use conda run to execute in the correct environment
        if method_config['conda_env'] == 'base':
            # For MPRNet, use current python environment
            cmd = [sys.executable, str(module_path),
                   "--input_dir", str(input_dir),
                   "--output_dir", str(output_dir)]
        else:
            # For Restormer and Uformer, use conda run with specific environment
            cmd = ["conda", "run", "-n", method_config['conda_env'], "python",
                   str(module_path),
                   "--input_dir", str(input_dir),
                   "--output_dir", str(output_dir)]
        
        # Set PYTHONPATH to include project root
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.base_dir)
        
        # Run the module with real-time output
        self.logger.info(f"Executing: {' '.join(cmd)}")
        self.logger.info(f"Conda environment: {method_config['conda_env']}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            cwd=self.base_dir, 
            env=env,
            bufsize=1,
            universal_newlines=True,
            shell=(method_config['conda_env'] != 'base')  # Use shell for conda run commands
        )
        
        # Stream output in real-time
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line.strip():  # Only print non-empty lines
                print(f"[{method}] {line.rstrip()}")  # Print to console immediately
                output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            self.logger.info(f"[OK] {method} processing completed")
            self.logger.info(f"Results saved to: {output_dir}")
            
            # Verify output files were created
            output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
            self.logger.info(f"Generated {len(output_files)} output files")
            return True
        else:
            self.logger.error(f"{method} processing failed with return code {return_code}")
            
            # Check for common conda environment issues
            if method_config['conda_env'] != 'base':
                if "No module named" in ''.join(output_lines):
                    self.logger.error(f"Missing dependencies in conda environment '{method_config['conda_env']}'")
                elif "conda: command not found" in ''.join(output_lines):
                    self.logger.error("Conda is not available in PATH.")
                elif f"Could not find conda environment: {method_config['conda_env']}" in ''.join(output_lines):
                    self.logger.error(f"Conda environment '{method_config['conda_env']}' does not exist")
            
            # Log the captured output for debugging
            if output_lines:
                self.logger.error(f"Output: {''.join(output_lines[-10:])}")  # Last 10 lines
            return False
            
    def run_deblur_processing(self):
        """Run deblur processing using direct module approach."""
        try:
            # Get deblur configuration based on pipeline mode
            if self.config.get('pipeline', {}).get('mode') == 'full':
                deblur_methods = self.config.get('full_mode', {}).get('deblur', {}).get('methods', ['Restormer', 'MPRNet', 'Uformer'])
            else:
                deblur_methods = self.config.get('deblur', {}).get('test_mode', {}).get('methods', ['Restormer'])
            
            # Only use available methods (current available: Restormer, MPRNet, Uformer)
            available_methods = ['Restormer', 'MPRNet', 'Uformer']
            deblur_methods = [method for method in deblur_methods if method in available_methods]
            
            if not deblur_methods:
                self.logger.warning("No available deblur methods found in configuration")
                return True
            
            self.logger.info(f"Running deblur processing with methods: {deblur_methods}")
            
            # Find all blur type directories
            blur_base_dir = self.base_dir / "data" / "frames" / "blurred" / self.video_stem
            
            if not blur_base_dir.exists():
                self.logger.error(f"Blur base directory not found: {blur_base_dir}")
                return False
            
            blur_dirs = [d for d in blur_base_dir.iterdir() if d.is_dir()]
            
            if not blur_dirs:
                self.logger.error(f"No blur directories found in {blur_base_dir}")
                return False
            
            # Process each blur directory with each method
            for input_dir in blur_dirs:
                blur_method_name = input_dir.name  # Extract blur method from directory name
                self.logger.info(f"Processing blur type: {blur_method_name}")
                
                for method in deblur_methods:
                    self.logger.info(f"Running {method} on {blur_method_name}...")
                    
                    # Set correct output directory: data/frames/deblurred/{video_stem}/{method_name}/
                    output_dir = self.base_dir / "data" / "frames" / "deblurred" / self.video_stem / method
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Check if deblurred frames already exist - skip if skip_existing is enabled
                    skip_existing = self.config.get('pipeline', {}).get('skip_existing', False)
                    if skip_existing:
                        existing_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
                        input_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
                        
                        if existing_files and len(existing_files) >= len(input_files):
                            self.logger.info(f"[SKIP] {method} on {blur_method_name} already processed ({len(existing_files)} files exist, skip_existing=True)")
                            continue
                        elif existing_files:
                            self.logger.info(f"[PARTIAL] {method} on {blur_method_name} has {len(existing_files)}/{len(input_files)} files, will reprocess")
                    
                    # Run deblur on this directory
                    self._run_deblur_on_directory(input_dir, output_dir, method)
            
            # Process original frames with deblur if configured
            process_original = self.config.get('deblur', {}).get('process_original_frames', False)
            original_method = self.config.get('deblur', {}).get('original_deblur_method', 'Restormer')
            
            if process_original and original_method in available_methods:
                self.logger.info(f"Processing original frames with {original_method}...")
                
                # Path to original frames
                original_frames_dir = self.base_dir / "data" / "frames" / "original" / self.video_stem
                
                if not original_frames_dir.exists():
                    self.logger.warning(f"Original frames directory not found: {original_frames_dir}")
                else:
                    # Output directory for deblurred original frames
                    output_dir = self.base_dir / "data" / "frames" / "deblurred" / self.video_stem / f"{original_method}_original"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Check if already processed
                    skip_existing = self.config.get('pipeline', {}).get('skip_existing', False)
                    if skip_existing:
                        existing_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
                        input_files = list(original_frames_dir.glob("*.png")) + list(original_frames_dir.glob("*.jpg"))
                        
                        if existing_files and len(existing_files) >= len(input_files):
                            self.logger.info(f"[SKIP] {original_method} on original frames already processed ({len(existing_files)} files exist, skip_existing=True)")
                        else:
                            # Process original frames
                            self._run_deblur_on_directory(original_frames_dir, output_dir, original_method)
                    else:
                        # Process original frames
                        self._run_deblur_on_directory(original_frames_dir, output_dir, original_method)
            
            return True
                    
        except Exception as e:
            self.logger.error(f"Error running deblur processing: {e}")
            return False
            
    def run_metrics_calculation(self):
        """Run metrics calculation with new directory structure."""
        if self.skip_metrics:
            self.logger.info("Skipping metrics calculation (--skip-metrics)")
            return True
            
        try:
            metrics_script = self.base_dir / "blur" / "metrics" / "metrics_calculator.py"
            cmd = [
                sys.executable, str(metrics_script), 
                "--video", self.video_name,
                "--run-id", self.run_id,
                "--pipeline-mode"
            ]
            
            # Set PYTHONPATH to include project root
            import os
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.base_dir)
            
            self.logger.info(f"Running metrics calculation with run_id: {self.run_id}")
            self.logger.info("This may take several minutes depending on the number of frames...")
            
            # Use Popen for real-time output instead of capture_output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.base_dir,
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    print(line.rstrip())
            
            return_code = process.wait()
            
            if return_code == 0:
                self.logger.info("[OK] Metrics calculation completed successfully")
                return True
            else:
                self.logger.error(f"Metrics calculation failed with return code {return_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running metrics calculation: {e}")
            return False
            
    def run_3d_reconstruction(self):
        """Run 3D reconstruction pipeline on all frame types."""
        if self.skip_reconstruction:
            self.logger.info("Skipping 3D reconstruction (--skip-reconstruction)")
            return True
            
        try:
            # Get frame types to process from configuration
            frame_types = self.config.get('reconstruction', {}).get('test_mode', {}).get('frame_types', ['original', 'blurred', 'deblurred'])
            if self.config.get('pipeline', {}).get('mode') == 'full':
                frame_types = self.config.get('full_mode', {}).get('reconstruction', {}).get('frame_types', ['original', 'blurred', 'deblurred'])
            
            self.logger.info(f"Running 3D reconstruction on frame types: {frame_types}")
            
            recon_script = self.base_dir / "reconstruction" / "reconstruction_pipeline.py"
            
            # Use video mode which processes all frame types automatically
            # Pass run_id and pipeline_mode to enable data manager
            cmd = [
                sys.executable, str(recon_script), 
                "--video", self.video_name,
                "--feature", "disk",
                "--matcher", "disk+lightglue",
                "--run-id", self.run_id,
                "--pipeline-mode"
            ]
            
            self.logger.info(f"Running reconstruction with run_id: {self.run_id}")
            self.logger.info(f"Running reconstruction command: {' '.join(cmd)}")
            self.logger.info("This may take several minutes depending on the number of frames...")
            
            # Use Popen for real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.base_dir,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    print(line.rstrip())
            
            return_code = process.wait()
            
            if return_code == 0:
                self.logger.info("[OK] 3D reconstruction completed successfully")
                return True
            else:
                self.logger.error(f"3D reconstruction failed with return code {return_code}")
                # Don't fail the entire pipeline if reconstruction fails
                return True
                
        except Exception as e:
            self.logger.error(f"Error running 3D reconstruction: {e}")
            return True  # Don't fail entire pipeline
    
    def run_registration(self):
        """Run point cloud registration and comparison."""
        try:
            # Check if reconstruction was skipped
            if self.skip_reconstruction:
                self.logger.info("Skipping registration (reconstruction was skipped)")
                return True
            
            # Check if ground truth exists
            gt_patterns = [
                f"{self.video_stem.split('_')[0]}_{self.video_stem.split('_')[1]}_segmentation.ply",
                "ground_truth.ply"
            ]
            
            gt_dir = self.base_dir / "data" / "point_clouds" / self.video_stem
            ground_truth_path = None
            
            for pattern in gt_patterns:
                test_path = gt_dir / pattern
                if test_path.exists():
                    ground_truth_path = test_path
                    break
            
            if not ground_truth_path:
                self.logger.warning(f"Ground truth PLY not found in {gt_dir}, skipping registration")
                self.logger.info(f"Looked for: {gt_patterns}")
                return True
            
            self.logger.info(f"Ground truth found: {ground_truth_path}")
            
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info("POINT CLOUD REGISTRATION REQUIRES INITIAL TRANSFORMATION")
            self.logger.info("="*80)
            self.logger.info("")
            self.logger.info("The ICP registration step requires an initial transformation matrix.")
            self.logger.info("Please create:")
            self.logger.info(f"  {self.data_manager.point_clouds_dir}/registration/initial_transformation.txt")
            self.logger.info("")
            self.logger.info("See INITIAL_TRANSFORMATION_SETUP.md for detailed instructions.")
            self.logger.info("")
            self.logger.info("To proceed with registration, run:")
            self.logger.info(f"  python reconstruction/registration_calculator.py \\")
            self.logger.info(f"    --video {self.video_stem} \\")
            self.logger.info(f"    --run run_{self.run_id}")
            self.logger.info("")
            self.logger.info("="*80)
                
        except Exception as e:
            self.logger.error(f"Error in registration step: {e}")
            return True  # Don't fail entire pipeline
    
    def run_complete_pipeline(self):
        """Run the complete VAPOR analysis pipeline."""
        self.logger.info("VAPOR Pipeline Starting - %s", self.video_name)
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.logger.error("Prerequisites failed")
            return False
            
        # Step 1: Blur generation
        self.logger.info("[1/5] Generating blur frames...")
        if not self.run_blur_generation():
            self.logger.error("Blur generation failed")
            return False
            
        # Step 2: Deblur processing
        self.logger.info("[2/5] Processing deblur methods...")
        if not self.run_deblur_processing():
            self.logger.error("Deblur processing failed")
            return False
            
        # Step 3: Metrics calculation
        self.logger.info("[3/5] Calculating metrics...")
        if not self.run_metrics_calculation():
            self.logger.error("Metrics calculation failed")
            return False
            
        # Step 4: 3D reconstruction
        self.logger.info("[4/5] Running 3D reconstruction...")
        if not self.run_3d_reconstruction():
            self.logger.warning("3D reconstruction issues, continuing...")
        
        # Step 5: Point cloud registration
        self.logger.info("[5/5] Running point cloud registration...")
        if not self.run_registration():
            self.logger.warning("Registration issues, continuing...")
        
        self.logger.info("VAPOR Pipeline Completed Successfully")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="VAPOR Complete Analysis Pipeline"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Video filename (e.g., pat3.mp4)"
    )
    parser.add_argument(
        "--skip-blur",
        action="store_true",
        help="Skip blur generation if frames already exist"
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true", 
        help="Skip metrics calculation"
    )
    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Skip 3D reconstruction"
    )
    parser.add_argument(
        "--manual-crop",
        action="store_true",
        help="Use interactive manual crop selection (select 4 corners on 3 frames)"
    )
    # Removed interactive crop option
    
    args = parser.parse_args()
    
    # Initialize and run complete pipeline
    pipeline = VAPORCompletePipeline(
        video_name=args.video,
        skip_blur=args.skip_blur,
        skip_metrics=args.skip_metrics,
        skip_reconstruction=args.skip_reconstruction,
        manual_crop=args.manual_crop
    )
    
    success = pipeline.run_complete_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())