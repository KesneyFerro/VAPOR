"""
VAPOR 3D Reconstruction Pipeline
Unified script for running Structure from Motion (SfM) reconstruction on original, blurred, and deblurred frames.

This script integrates the maploc SfM pipeline with VAPOR's blur processing system to:
1. Run 3D reconstruction on original frames
2. Run 3D reconstruction on all blurred frame variations
3. Run 3D reconstruction on deblurred frames (when available)
4. Compare reconstruction quality across different blur conditions
5. Save results in organized directory structure

Usage:
    python reconstruction_pipeline.py --video pat3.mp4 [--feature disk] [--matcher disk+lightglue]
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import shutil
import json
import pandas as pd
import numpy as np

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory (VAPOR root)
sys.path.append(str(Path(__file__).parent / "maploc"))  # Add maploc directory

from reconstruction.maploc.BatchRunUtils import TrialConfig
from reconstruction.maploc.hloc import extract_features, match_features
import pycolmap


class VAPORReconstructionPipeline:
    """Unified 3D reconstruction pipeline for VAPOR blur analysis."""
    
    def __init__(self, video_name: str, feature: str = "disk", matcher: str = "disk+lightglue"):
        """Initialize the reconstruction pipeline.
        
        Args:
            video_name: Name of video file (e.g., 'pat3.mp4')
            feature: Feature detector to use ('disk', 'superpoint', etc.)
            matcher: Feature matcher to use ('disk+lightglue', 'superglue', etc.)
        """
        self.video_name = video_name
        self.video_stem = Path(video_name).stem  # Remove .mp4 extension
        self.feature = feature
        self.matcher = matcher
        
        # Setup base paths
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        
        # Input paths - frame directories
        self.frames_base = self.data_dir / "frames"
        self.frames_original = self.frames_base / "original" / self.video_stem
        self.frames_blurred = self.frames_base / "blurred" / self.video_stem
        self.frames_deblurred = self.frames_base / "deblurred" / self.video_stem
        
        # Output paths - point clouds and metrics
        self.point_clouds_base = self.data_dir / "point_clouds" / self.video_stem
        self.metrics_base = self.data_dir / "metrics" / self.video_stem
        
        # Reconstruction settings
        self.reconstruction_settings = {
            "n_matches": 50,
            "retrieval_algo": "netvlad",
            "geometric_verification_options": dict(
                min_num_inliers=15,  # Reduced for smaller frame sets
                ransac=dict(max_num_trials=10000, min_inlier_ratio=0.3)
            )
        }
        
        # Initialize directories
        self._setup_directories()
        
        # Configure logging
        self._setup_logging()
        
    def _setup_directories(self):
        """Create necessary output directories."""
        # Point cloud directories
        self.point_clouds_base.mkdir(parents=True, exist_ok=True)
        (self.point_clouds_base / "original").mkdir(exist_ok=True)
        (self.point_clouds_base / "blurred").mkdir(exist_ok=True)
        (self.point_clouds_base / "deblurred").mkdir(exist_ok=True)
        
        # Metrics directories
        self.metrics_base.mkdir(parents=True, exist_ok=True)
        (self.metrics_base / "original").mkdir(exist_ok=True)
        (self.metrics_base / "blurred").mkdir(exist_ok=True)
        (self.metrics_base / "deblurred").mkdir(exist_ok=True)
        
    def _setup_logging(self):
        """Configure logging for the reconstruction pipeline."""
        log_file = self.metrics_base / f"reconstruction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_available_frame_sets(self):
        """Discover available frame sets for reconstruction."""
        frame_sets = {}
        
        # Check original frames
        if self.frames_original.exists() and any(self.frames_original.iterdir()):
            frame_count = len(list(self.frames_original.glob("*.png")))
            frame_sets['original'] = {
                'path': self.frames_original,
                'count': frame_count,
                'type': 'original'
            }
            
        # Check blurred frame variants
        if self.frames_blurred.exists():
            for blur_dir in self.frames_blurred.iterdir():
                if blur_dir.is_dir() and any(blur_dir.iterdir()):
                    frame_count = len(list(blur_dir.glob("*.png")))
                    frame_sets[f'blurred_{blur_dir.name}'] = {
                        'path': blur_dir,
                        'count': frame_count,
                        'type': 'blurred',
                        'blur_method': blur_dir.name
                    }
                    
        # Check deblurred frames
        if self.frames_deblurred.exists():
            for deblur_dir in self.frames_deblurred.iterdir():
                if deblur_dir.is_dir() and any(deblur_dir.iterdir()):
                    frame_count = len(list(deblur_dir.glob("*.png")))
                    frame_sets[f'deblurred_{deblur_dir.name}'] = {
                        'path': deblur_dir,
                        'count': frame_count,
                        'type': 'deblurred',
                        'deblur_method': deblur_dir.name
                    }
                    
        return frame_sets
    
    def run_sfm_reconstruction(self, frames_path: Path, output_name: str, frame_type: str):
        """Run SfM reconstruction on a set of frames.
        
        Args:
            frames_path: Path to directory containing frames
            output_name: Name for output directory
            frame_type: Type of frames ('original', 'blurred', 'deblurred')
            
        Returns:
            dict: Reconstruction results and statistics
        """
        self.logger.info(f"Starting SfM reconstruction for {output_name}")
        self.logger.info(f"  Frames path: {frames_path}")
        self.logger.info(f"  Frame count: {len(list(frames_path.glob('*.png')))}")
        
        try:
            # Create output directory for this reconstruction
            if frame_type == "original":
                output_dir = self.point_clouds_base / "original" / output_name
            elif frame_type == "blurred":
                output_dir = self.point_clouds_base / "blurred" / output_name
            elif frame_type == "deblurred":
                output_dir = self.point_clouds_base / "deblurred" / output_name
            else:
                output_dir = self.point_clouds_base / output_name
                
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create TrialConfig for maploc pipeline
            trial_config = TrialConfig(
                feature=self.feature,
                matcher=self.matcher,
                retrievalAlgo=self.reconstruction_settings["retrieval_algo"],
                isEndToEnd=False,
                imagesDir=frames_path,
                outputsDir=output_dir,
                outputsDirIsParent=False,
                geometric_verification_options=self.reconstruction_settings["geometric_verification_options"]
            )
            
            # Run the reconstruction pipeline
            self.logger.info(f"  Running feature extraction and matching...")
            trial_config.pairFromRetrieval(nMatches=self.reconstruction_settings["n_matches"])
            trial_config.extractAndMatchFeatures()
            
            self.logger.info(f"  Running SfM reconstruction...")
            trial_config.reconstruction()
            
            # Export results
            if hasattr(trial_config, 'model') and trial_config.model is not None:
                ply_path = output_dir / "reconstruction.ply"
                trial_config.model.export_PLY(str(ply_path))
                self.logger.info(f"  Exported point cloud: {ply_path}")
                
                # Calculate reconstruction statistics
                stats = self._calculate_reconstruction_stats(trial_config.model, output_name, frame_type)
                
                # Save statistics
                stats_path = output_dir / "reconstruction_stats.json"
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                    
                self.logger.info(f"  Reconstruction completed successfully")
                return stats
            else:
                self.logger.warning(f"  Reconstruction failed - no model generated")
                return None
                
        except Exception as e:
            self.logger.error(f"  Reconstruction failed with error: {e}")
            return None
    
    def _calculate_reconstruction_stats(self, model, output_name: str, frame_type: str):
        """Calculate reconstruction quality statistics."""
        try:
            stats = {
                'reconstruction_name': output_name,
                'frame_type': frame_type,
                'timestamp': datetime.now().isoformat(),
                'feature_detector': self.feature,
                'feature_matcher': self.matcher,
                'num_registered_images': len(model.reg_image_ids()),
                'num_3d_points': model.num_points3D(),
                'mean_observations_per_image': model.compute_mean_observations_per_reg_image(),
                'mean_reprojection_error': model.compute_mean_reprojection_error(),
                'mean_track_length': model.compute_mean_track_length(),
            }
            
            # Add summary string
            stats['summary'] = model.summary()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate reconstruction stats: {e}")
            return {
                'reconstruction_name': output_name,
                'frame_type': frame_type,
                'error': str(e)
            }
    
    def run_complete_pipeline(self):
        """Run the complete reconstruction pipeline on all available frame sets."""
        self.logger.info("="*80)
        self.logger.info(f"VAPOR 3D Reconstruction Pipeline - {self.video_name}")
        self.logger.info("="*80)
        
        # Discover available frame sets
        frame_sets = self.get_available_frame_sets()
        
        if not frame_sets:
            self.logger.error("No frame sets found for reconstruction!")
            return False
            
        self.logger.info(f"Found {len(frame_sets)} frame sets:")
        for name, info in frame_sets.items():
            self.logger.info(f"  - {name}: {info['count']} frames ({info['type']})")
        
        # Run reconstruction on each frame set
        all_results = {}
        
        for set_name, set_info in frame_sets.items():
            self.logger.info(f"\n--- Processing {set_name} ---")
            
            results = self.run_sfm_reconstruction(
                frames_path=set_info['path'],
                output_name=set_name,
                frame_type=set_info['type']
            )
            
            if results:
                all_results[set_name] = results
            else:
                self.logger.warning(f"Reconstruction failed for {set_name}")
        
        # Create summary comparison
        self._create_reconstruction_summary(all_results)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("RECONSTRUCTION PIPELINE COMPLETED")
        self.logger.info("="*80)
        
        return len(all_results) > 0
    
    def _create_reconstruction_summary(self, all_results: dict):
        """Create a summary comparison of all reconstructions."""
        if not all_results:
            return
            
        self.logger.info("\n--- Reconstruction Quality Summary ---")
        
        # Create comparison DataFrame
        summary_data = []
        for name, stats in all_results.items():
            if 'error' not in stats:
                summary_data.append({
                    'reconstruction': name,
                    'frame_type': stats['frame_type'],
                    'registered_images': stats['num_registered_images'],
                    'num_3d_points': stats['num_3d_points'],
                    'mean_reprojection_error': stats['mean_reprojection_error'],
                    'mean_track_length': stats['mean_track_length'],
                    'mean_observations_per_image': stats['mean_observations_per_image']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Save summary CSV
            summary_path = self.metrics_base / f"{self.video_stem}_reconstruction_summary.csv"
            df.to_csv(summary_path, index=False)
            
            # Print summary table
            self.logger.info(f"\nReconstruction Summary (saved to {summary_path}):")
            self.logger.info(df.to_string(index=False))
            
            # Find best reconstruction by number of 3D points
            if len(df) > 1:
                best_idx = df['num_3d_points'].idxmax()
                best_recon = df.iloc[best_idx]
                self.logger.info(f"\nBest reconstruction: {best_recon['reconstruction']}")
                self.logger.info(f"  3D Points: {best_recon['num_3d_points']}")
                self.logger.info(f"  Reprojection Error: {best_recon['mean_reprojection_error']:.3f}")


def main():
    """Main function for running the reconstruction pipeline."""
    parser = argparse.ArgumentParser(
        description="VAPOR 3D Reconstruction Pipeline"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Video filename (e.g., pat3.mp4)"
    )
    parser.add_argument(
        "--feature",
        default="disk",
        choices=["disk", "superpoint", "aliked-n16", "sift"],
        help="Feature detector to use (default: disk)"
    )
    parser.add_argument(
        "--matcher", 
        default="disk+lightglue",
        choices=["disk+lightglue", "superglue", "aliked+lightglue", "nn-ratio"],
        help="Feature matcher to use (default: disk+lightglue)"
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = VAPORReconstructionPipeline(
        video_name=args.video,
        feature=args.feature,
        matcher=args.matcher
    )
    
    success = pipeline.run_complete_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())