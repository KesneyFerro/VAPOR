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

import argparse
import sys
import subprocess
from pathlib import Path
import logging
from datetime import datetime

class VAPORCompletePipeline:
    """Master pipeline controller for VAPOR analysis."""
    
    def __init__(self, video_name: str, skip_blur: bool = False, skip_metrics: bool = False, skip_reconstruction: bool = False):
        """Initialize the complete pipeline.
        
        Args:
            video_name: Name of video file (e.g., 'pat3.mp4')
            skip_blur: Skip blur generation if frames already exist
            skip_metrics: Skip metrics calculation
            skip_reconstruction: Skip 3D reconstruction
        """
        self.video_name = video_name
        self.video_stem = Path(video_name).stem
        self.skip_blur = skip_blur
        self.skip_metrics = skip_metrics
        self.skip_reconstruction = skip_reconstruction
        
        self.base_dir = Path(__file__).parent
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the pipeline."""
        log_dir = self.base_dir / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"vapor_complete_{self.video_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_prerequisites(self):
        """Check if all required components are available."""
        self.logger.info("Checking prerequisites...")
        
        # Check if video exists
        video_path = self.base_dir / "data" / "videos" / "original" / self.video_name
        if not video_path.exists():
            self.logger.error(f"Video not found: {video_path}")
            return False
            
        # Check if blur pipeline components exist
        blur_script = self.base_dir / "blur" / "simple_blur_pipeline.py"
        if not self.skip_blur and not blur_script.exists():
            self.logger.error(f"Blur pipeline script not found: {blur_script}")
            return False
            
        # Check if metrics calculator exists
        metrics_script = self.base_dir / "blur" / "metrics" / "updated_calculator.py"
        if not self.skip_metrics and not metrics_script.exists():
            self.logger.error(f"Metrics calculator not found: {metrics_script}")
            return False
            
        # Check if reconstruction pipeline exists
        recon_script = self.base_dir / "reconstruction" / "reconstruction_pipeline.py"
        if not self.skip_reconstruction and not recon_script.exists():
            self.logger.error(f"Reconstruction pipeline not found: {recon_script}")
            return False
            
        self.logger.info("✓ All prerequisites checked")
        return True
        
    def run_blur_generation(self):
        """Run blur generation pipeline."""
        if self.skip_blur:
            self.logger.info("Skipping blur generation (--skip-blur)")
            return True
            
        self.logger.info("Running blur generation pipeline...")
        
        try:
            # Check if frames already exist
            frames_dir = self.base_dir / "data" / "frames" / "original" / self.video_stem
            if frames_dir.exists() and any(frames_dir.iterdir()):
                self.logger.info("Original frames already exist, skipping blur generation")
                return True
                
            # Run blur pipeline
            blur_script = self.base_dir / "blur" / "simple_blur_pipeline.py"
            cmd = [sys.executable, str(blur_script), "--video", self.video_name, "--stride", "60"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                self.logger.info("✓ Blur generation completed successfully")
                return True
            else:
                self.logger.error(f"Blur generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running blur generation: {e}")
            return False
            
    def run_metrics_calculation(self):
        """Run metrics calculation with new directory structure."""
        if self.skip_metrics:
            self.logger.info("Skipping metrics calculation (--skip-metrics)")
            return True
            
        self.logger.info("Running metrics calculation...")
        
        try:
            metrics_script = self.base_dir / "blur" / "metrics" / "updated_calculator.py"
            cmd = [sys.executable, str(metrics_script), "--video", self.video_name]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                self.logger.info("✓ Metrics calculation completed successfully")
                return True
            else:
                self.logger.error(f"Metrics calculation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running metrics calculation: {e}")
            return False
            
    def run_3d_reconstruction(self):
        """Run 3D reconstruction pipeline."""
        if self.skip_reconstruction:
            self.logger.info("Skipping 3D reconstruction (--skip-reconstruction)")
            return True
            
        self.logger.info("Running 3D reconstruction pipeline...")
        
        try:
            recon_script = self.base_dir / "reconstruction" / "reconstruction_pipeline.py"
            cmd = [sys.executable, str(recon_script), "--video", self.video_name]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                self.logger.info("✓ 3D reconstruction completed successfully")
                return True
            else:
                self.logger.error(f"3D reconstruction failed: {result.stderr}")
                # Don't fail the entire pipeline if reconstruction fails
                self.logger.warning("Continuing pipeline despite reconstruction failure")
                return True
                
        except Exception as e:
            self.logger.error(f"Error running 3D reconstruction: {e}")
            return True  # Don't fail entire pipeline
            
    def generate_final_report(self):
        """Generate a comprehensive analysis report."""
        self.logger.info("Generating final analysis report...")
        
        try:
            report_dir = self.base_dir / "data" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"{self.video_stem}_complete_analysis_report.txt"
            
            with open(report_file, 'w') as f:
                f.write(f"VAPOR Complete Analysis Report\\n")
                f.write(f"Video: {self.video_name}\\n")
                f.write(f"Generated: {datetime.now().isoformat()}\\n")
                f.write(f"{'='*60}\\n\\n")
                
                # Check what was generated
                frames_dir = self.base_dir / "data" / "frames"
                metrics_dir = self.base_dir / "data" / "metrics" / self.video_stem
                point_clouds_dir = self.base_dir / "data" / "point_clouds" / self.video_stem
                
                f.write("Generated Data:\\n")
                f.write(f"- Original frames: {self._count_files(frames_dir / 'original' / self.video_stem, '*.png')} files\\n")
                f.write(f"- Blurred frame sets: {self._count_directories(frames_dir / 'blurred' / self.video_stem)} sets\\n")
                f.write(f"- Metrics files: {self._count_files(metrics_dir, '*.csv')} CSV files\\n")
                f.write(f"- Point clouds: {self._count_files(point_clouds_dir, '*.ply', recursive=True)} PLY files\\n")
                
                f.write(f"\\nOutput Directories:\\n")
                f.write(f"- Frames: {frames_dir}\\n")
                f.write(f"- Metrics: {metrics_dir}\\n")
                f.write(f"- Point Clouds: {point_clouds_dir}\\n")
                f.write(f"- Reports: {report_dir}\\n")
                
            self.logger.info(f"✓ Analysis report saved: {report_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return False
            
    def _count_files(self, directory: Path, pattern: str, recursive: bool = False):
        """Count files matching pattern in directory."""
        if not directory.exists():
            return 0
        if recursive:
            return len(list(directory.rglob(pattern)))
        else:
            return len(list(directory.glob(pattern)))
            
    def _count_directories(self, directory: Path):
        """Count subdirectories."""
        if not directory.exists():
            return 0
        return len([d for d in directory.iterdir() if d.is_dir()])
        
    def run_complete_pipeline(self):
        """Run the complete VAPOR analysis pipeline."""
        self.logger.info("="*80)
        self.logger.info("VAPOR COMPLETE ANALYSIS PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"Video: {self.video_name}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.logger.error("Prerequisites check failed")
            return False
            
        # Step 1: Blur generation
        if not self.run_blur_generation():
            self.logger.error("Pipeline failed at blur generation")
            return False
            
        # Step 2: Metrics calculation
        if not self.run_metrics_calculation():
            self.logger.error("Pipeline failed at metrics calculation")
            return False
            
        # Step 3: 3D reconstruction
        if not self.run_3d_reconstruction():
            self.logger.warning("3D reconstruction had issues but continuing")
            
        # Step 4: Generate report
        self.generate_final_report()
        
        self.logger.info("\\n" + "="*80)
        self.logger.info("VAPOR COMPLETE PIPELINE FINISHED")
        self.logger.info("="*80)
        
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
    
    args = parser.parse_args()
    
    # Initialize and run complete pipeline
    pipeline = VAPORCompletePipeline(
        video_name=args.video,
        skip_blur=args.skip_blur,
        skip_metrics=args.skip_metrics,
        skip_reconstruction=args.skip_reconstruction
    )
    
    success = pipeline.run_complete_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())