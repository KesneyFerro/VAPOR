"""
VAPOR 3D Reconstruction Pipeline
Unified script for running Structure from Motion (SfM) reconstruction on frames.

This script integrates the maploc SfM pipeline with VAPOR's system to:
1. Run 3D reconstruction on any folder containing PNG images
2. Run 3D reconstruction on original frames
3. Run 3D reconstruction on all blurred frame variations
4. Run 3D reconstruction on deblurred frames (when available)
5. Compare reconstruction quality across different blur conditions
6. Save results in organized directory structure

Usage:
    # Process a specific folder of PNG images:
    python reconstruction_pipeline.py --folder "S:\Kesney\VAPOR\data\frames\original\pat3" [--feature disk] [--matcher disk+lightglue]
    
    # Process all frame sets for a video:
    python reconstruction_pipeline.py --video pat3.mp4 [--feature disk] [--matcher disk+lightglue]
"""

import os
# Fix OpenMP runtime conflict before importing other modules
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
from utils.data_manager import VAPORDataManager
import pycolmap
import h5py


class VAPORReconstructionPipeline:
    """Unified 3D reconstruction pipeline for VAPOR blur analysis."""
    
    def __init__(self, video_name: str = None, folder_path: str = None, feature: str = "disk", 
                 matcher: str = "disk+lightglue", skip_cleanup: bool = False, pipeline_mode: bool = False,
                 run_id: str = None):
        """Initialize the reconstruction pipeline.
        
        Args:
            video_name: Name of video file (e.g., 'pat3.mp4') - for processing all frame sets
            folder_path: Direct path to folder containing PNG images - for single folder processing
            feature: Feature detector to use ('disk', 'superpoint', etc.)
            matcher: Feature matcher to use ('disk+lightglue', 'superglue', etc.)
            skip_cleanup: Skip cleaning previous outputs (useful if permission issues)
            pipeline_mode: If True, use pipeline data manager; if False, use standalone
            run_id: Specific run ID to use (optional, will be generated if None)
        """
        self.folder_mode = folder_path is not None
        self.video_name = video_name
        self.folder_path = Path(folder_path) if folder_path else None
        self.video_stem = Path(video_name).stem if video_name else None
        self.feature = feature
        self.matcher = matcher
        self.skip_cleanup = skip_cleanup
        self.pipeline_mode = pipeline_mode
        self.run_id = run_id
        
        # Setup base paths
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir.parent / "data"
        
        # Initialize data manager
        if self.pipeline_mode:
            # Use pipeline data manager for persistent, timestamped outputs
            self.data_manager = VAPORDataManager(
                video_name=self.video_name,
                mode="pipeline",
                run_id=self.run_id  # Pass run_id to coordinate with other modules
            )
            # Use data manager paths
            self.point_clouds_base = self.data_manager.point_clouds_dir
            self.metrics_base = self.data_manager.metrics_dir
            self.timestamp = self.data_manager.timestamp if not self.run_id else self.run_id
        else:
            # Use standalone mode for testing (overwritable outputs)
            self.data_manager = VAPORDataManager(
                video_name=self.video_name if self.video_name else "test.mp4",
                mode="standalone",
                module_name="reconstruction"
            )
            # Use data manager paths
            self.point_clouds_base = self.data_manager.point_clouds_dir
            self.metrics_base = self.data_manager.metrics_dir
            self.timestamp = self.data_manager.timestamp
        
        # Parse folder information if in folder mode
        if self.folder_mode:
            path_parts = self.folder_path.parts
            if 'frames' in path_parts:
                frames_idx = path_parts.index('frames')
                if frames_idx < len(path_parts) - 2:
                    self.frame_type = path_parts[frames_idx + 1]
                    self.video_name_from_path = path_parts[frames_idx + 2]
                else:
                    self.frame_type = "unknown"
                    self.video_name_from_path = self.folder_path.name
            else:
                self.frame_type = "custom"
                self.video_name_from_path = self.folder_path.name
            
            if self.frame_type in ['blurred', 'deblurred'] and len(path_parts) > frames_idx + 3:
                self.method_name = path_parts[frames_idx + 3]
            else:
                self.method_name = None
        else:
            # Video mode - process all frame sets
            self.frames_base = self.data_dir / "frames"
            self.frames_original = self.frames_base / "original" / self.video_stem
            self.frames_blurred = self.frames_base / "blurred" / self.video_stem
            self.frames_deblurred = self.frames_base / "deblurred" / self.video_stem
        
        # Reconstruction settings (matching the notebook exactly)
        self.reconstruction_settings = {
            "n_matches": 50,
            "retrieval_algo": "netvlad",
            "geometric_verification_options": dict(
                min_num_inliers=250,
                ransac=dict(max_num_trials=20000, min_inlier_ratio=0.9)
            )
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_directories(self):
        """Create necessary output directories - now handled by data manager."""
        pass  # Data manager handles directory creation
    
    
    def _setup_logging(self):
        """Configure logging for the reconstruction pipeline."""
        # Use data manager to get log file path
        log_file = self.data_manager.get_log_file_path("reconstruction")
        
        # Clear any existing handlers to avoid duplicates
        logging.getLogger().handlers.clear()
        
        # Create a new logger with both file and console handlers
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),  # 'w' mode to overwrite existing logs
                logging.StreamHandler()
            ],
            force=True  # Force reconfiguration
        )
        self.logger = logging.getLogger(__name__)
        
        # Log the setup
        self.logger.info(f"Logging initialized - log file: {log_file}")
        
    def run_single_folder_reconstruction(self):
        """Run reconstruction on a single folder of PNG images."""
        if not self.folder_mode:
            raise ValueError("This method requires folder_mode to be enabled")
            
        self.logger.info("="*80)
        self.logger.info(f"VAPOR Single Folder Reconstruction - {self.folder_path}")
        self.logger.info("="*80)
        
        # Setup directories if cleanup was skipped
        if self.skip_cleanup:
            self.logger.info("Cleanup was skipped during initialization (--skip-cleanup)")
            self._setup_directories()  # Still ensure directories exist
        
        # Verify folder exists and contains PNG files
        if not self.folder_path.exists():
            self.logger.error(f"Folder does not exist: {self.folder_path}")
            return False
            
        png_files = list(self.folder_path.glob("*.png"))
        if not png_files:
            self.logger.error(f"No PNG files found in: {self.folder_path}")
            return False
            
        self.logger.info(f"Found {len(png_files)} PNG files")
        self.logger.info(f"Using retrieval algorithm: {self.reconstruction_settings['retrieval_algo']}")
        self.logger.info(f"Number of matches per image: {self.reconstruction_settings['n_matches']}")
        
        # Log what will be generated and where
        self.logger.info("="*60)
        self.logger.info("RECONSTRUCTION OUTPUTS:")
        self.logger.info(f"Video: {self.video_name_from_path}")
        self.logger.info(f"Frame type: {self.frame_name}")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Main output directory: {self.outputs_base}")
        self.logger.info(f"Point clouds will be saved to: {self.point_clouds_base}")
        self.logger.info(f"Metrics will be saved to: {self.metrics_base}")
        self.logger.info("Files that will be generated:")
        self.logger.info("  - reconstruction.ply (3D point cloud)")
        self.logger.info("  - reconstruction_stats.json (statistics)")
        self.logger.info("  - pairs-*.txt (image pairs)")
        self.logger.info("  - feats-*.h5 (features)")
        self.logger.info("  - matches-*.h5 (matches)")
        self.logger.info("  - models/ (COLMAP model files)")
        self.logger.info("="*60)
        
        # Run reconstruction
        output_name = f"reconstruction_{self.frame_name}"
        results = self.run_sfm_reconstruction(
            frames_path=self.folder_path,
            output_name=output_name,
            frame_type=self.frame_type  # Use detected frame type
        )
        
        if results:
            self.logger.info("\n--- Reconstruction Results ---")
            basic = results.get('basic_metrics', {})
            self.logger.info(f"Registered Images: {basic.get('num_registered_images', 0)}")
            self.logger.info(f"3D Points: {basic.get('num_3d_points', 0)}")
            self.logger.info(f"Mean Reprojection Error: {basic.get('mean_reprojection_error', 0):.4f}")
            self.logger.info(f"Mean Track Length: {basic.get('mean_track_length', 0):.2f}")
            
            # Find and copy largest reconstruction if multiple exist
            self._process_multiple_reconstructions()
            
            self.logger.info("\n--- Output Summary ---")
            self.logger.info(f"All outputs saved to: {self.outputs_base}")
            self.logger.info(f"Point cloud (.ply): {self.point_clouds_base}")
            self.logger.info(f"Statistics (.json): {self.metrics_base}")
            
            self.logger.info("="*80)
            self.logger.info("SINGLE FOLDER RECONSTRUCTION COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            return True
        else:
            self.logger.error("Reconstruction failed!")
            return False
            
    def _process_multiple_reconstructions(self):
        """Process multiple reconstructions and select the largest one (from notebook logic)."""
        try:
            sfm_dirs = list(self.point_clouds_base.glob("sfm_*"))
            if not sfm_dirs:
                return
                
            for sfm_dir in sfm_dirs:
                models_dir = sfm_dir / "models"
                if not models_dir.exists():
                    continue
                    
                model_subdirs = [p for p in models_dir.iterdir() if p.is_dir()]
                
                if len(model_subdirs) > 1:
                    self.logger.info("Multiple reconstructions found, selecting largest...")
                    
                    # Find reconstruction with largest points3D.bin file
                    point_files = []
                    sizes = []
                    
                    for model_dir in model_subdirs:
                        point_file = model_dir / "points3D.bin"
                        if point_file.exists():
                            point_files.append(point_file)
                            sizes.append(point_file.stat().st_size)
                    
                    if sizes:
                        largest_idx = np.argmax(sizes)
                        largest_recon = point_files[largest_idx].parent
                        
                        self.logger.info(f"Largest reconstruction: {largest_recon.name}")
                        self.logger.info(f"Point file sizes: {sizes}")
                        
                        # Export PLY for each reconstruction
                        for model_dir in model_subdirs:
                            try:
                                import pycolmap
                                reconstruction = pycolmap.Reconstruction()
                                reconstruction.read_binary(str(model_dir))
                                ply_path = model_dir / f"reconstruction_{model_dir.name}.ply"
                                reconstruction.export_PLY(str(ply_path))
                                self.logger.info(f"Exported: {ply_path}")
                            except Exception as e:
                                self.logger.warning(f"Failed to export {model_dir}: {e}")
                                
        except Exception as e:
            self.logger.warning(f"Error processing multiple reconstructions: {e}")
        
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
            if self.folder_mode:
                # In folder mode, output goes directly to the frame-specific directory
                output_dir = self.point_clouds_base / output_name
            else:
                # In video mode, organize by frame type under the timestamp
                if frame_type == "original":
                    frame_dir = "original"
                elif frame_type == "blurred":
                    # Extract blur method from output_name (e.g., "blurred_motion_blur_high" -> "motion_blur_high")
                    frame_dir = output_name.replace("blurred_", "") if "blurred_" in output_name else "blurred"
                elif frame_type == "deblurred":
                    # Extract deblur method from output_name (e.g., "deblurred_uformer" -> "uformer")
                    frame_dir = output_name.replace("deblurred_", "") if "deblurred_" in output_name else "deblurred"
                else:
                    frame_dir = frame_type
                
                output_dir = self.point_clouds_base / frame_dir / output_name
                
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"  Output directory: {output_dir}")
            
            # Create TrialConfig for maploc pipeline (using notebook settings)
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
            
            # Use fixed n_matches like the notebook (with fallback for errors)
            try:
                trial_config.pairFromRetrieval(nMatches=self.reconstruction_settings["n_matches"])
            except Exception as e:
                self.logger.warning(f"  Retrieval failed with {self.reconstruction_settings['n_matches']} matches: {e}")
                # Fall back to exhaustive matching if retrieval fails
                self.logger.info(f"  Falling back to exhaustive matching...")
                trial_config.retrievalConfig = "EXHAUSTIVE"
                trial_config.pairFromRetrieval(nMatches=self.reconstruction_settings["n_matches"])
            
            trial_config.extractAndMatchFeatures()

            self.logger.info(f"  Running SfM reconstruction...")
            # Simple reconstruction call matching the notebook exactly
            try:
                trial_config.reconstruction()
            except Exception as e:
                self.logger.error(f"  Reconstruction call failed: {e}")
                import traceback
                self.logger.error(f"  Stack trace: {traceback.format_exc()}")
                return None
            
            # Export results
            if hasattr(trial_config, 'model') and trial_config.model is not None:
                # Simple success logging like the notebook
                total_images = len(list(frames_path.glob('*.png')))
                registered_images = len(trial_config.model.reg_image_ids())
                num_points = trial_config.model.num_points3D()
                
                self.logger.info(f"  Registration: {registered_images}/{total_images} images")
                self.logger.info(f"  3D points: {num_points}")
                
                # Check if reconstruction is valid (has registered images and points)
                if registered_images == 0:
                    self.logger.warning(f"  No images were registered - reconstruction failed")
                    return None
                
                if num_points == 0:
                    self.logger.warning(f"  No 3D points reconstructed - reconstruction failed")
                    return None
                
                # Export PLY with error handling
                ply_path = output_dir / "reconstruction.ply"
                try:
                    trial_config.model.export_PLY(str(ply_path))
                    self.logger.info(f"  Exported point cloud: {ply_path}")
                except Exception as e:
                    self.logger.error(f"  Failed to export PLY: {e}")
                    # Continue anyway - we can still save metrics even if PLY export fails
                    ply_path = None
                
                # Also copy to main data directory for pipeline compatibility
                if ply_path and ply_path.exists() and hasattr(self, 'data_point_clouds_base'):
                    try:
                        data_ply_dir = self.data_point_clouds_base / output_name
                        data_ply_dir.mkdir(parents=True, exist_ok=True)
                        data_ply_path = data_ply_dir / "reconstruction.ply"
                        
                        import shutil
                        shutil.copy2(ply_path, data_ply_path)
                        self.logger.info(f"  Also saved to data directory: {data_ply_path}")
                    except Exception as e:
                        self.logger.error(f"  Failed to copy PLY to data directory: {e}")
                
                # Generate per-image contribution report if in pipeline mode
                if self.pipeline_mode:
                    self.generate_per_image_contribution_report(
                        model=trial_config.model,
                        features_path=trial_config.featurePath,
                        matches_path=trial_config.matchPath,
                        output_name=output_name,
                        frame_type=frame_type
                    )
                
                # Calculate reconstruction statistics with enhanced metrics
                stats = self._calculate_reconstruction_stats(
                    model=trial_config.model,
                    output_name=output_name,
                    frame_type=frame_type,
                    features_path=trial_config.featurePath,
                    matches_path=trial_config.matchPath
                )
                
                # Check if stats calculation had an error
                if 'error' in stats.get('basic_metrics', {}):
                    self.logger.error(f"  Reconstruction stats calculation failed, skipping metrics save")
                    return stats
                
                # Save using data manager
                if self.pipeline_mode:
                    # Extract method name from output_name or frame_type
                    if frame_type == "blurred" or frame_type == "deblurred":
                        # output_name format: "blurred_motion_blur_high" or "deblurred_Restormer"
                        method_name = output_name.replace(f"{frame_type}_", "")
                    else:
                        method_name = None
                    
                    self.data_manager.save_reconstruction_metrics(
                        frame_type=frame_type,
                        method_name=method_name,
                        reconstruction_settings=self.reconstruction_settings,
                        basic_metrics=stats['basic_metrics'],
                        per_image_metrics=stats.get('per_image_metrics'),
                        matching_metrics=stats.get('matching_metrics'),
                        point_cloud_metrics=stats.get('point_cloud_metrics'),
                        processing_info=stats.get('processing_info'),
                        file_references=stats.get('file_references')
                    )
                else:
                    # Standalone mode - save to test_run folder
                    stats_path = output_dir / "reconstruction_metrics.json"
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    
                self.logger.info(f"  Reconstruction completed successfully")
                return stats
            else:
                # Simple failure logging
                self.logger.warning(f"  Reconstruction failed - no model generated")
                return None
                
        except Exception as e:
            self.logger.error(f"  Reconstruction failed with error: {e}")
            return None
    
    def _calculate_reconstruction_stats(self, model, output_name: str, frame_type: str,
                                         features_path: Path = None, matches_path: Path = None):
        """Calculate comprehensive reconstruction quality statistics."""
        try:
            # Get set of registered image IDs for efficient lookup
            registered_ids = set(model.reg_image_ids())
            
            # Basic metrics (always collected - no performance impact)
            basic_metrics = {
                'num_input_images': len(list(model.images.values())),
                'num_registered_images': len(registered_ids),
                'registration_rate': len(registered_ids) / len(model.images) if len(model.images) > 0 else 0,
                'num_3d_points': model.num_points3D(),
                'num_observations': sum(len(img.points2D) for img_id, img in model.images.items() if img_id in registered_ids),
                'mean_track_length': model.compute_mean_track_length(),
                'mean_observations_per_image': model.compute_mean_observations_per_reg_image(),
                'mean_reprojection_error': model.compute_mean_reprojection_error(),
            }
            
            # Collect enhanced metrics if in pipeline mode
            per_image_metrics = None
            matching_metrics = None
            point_cloud_metrics = None
            
            if self.pipeline_mode:
                # LOW IMPACT: Per-image metrics (~5% overhead)
                per_image_metrics = self._collect_per_image_metrics(model, features_path)
                
                # MODERATE IMPACT: Matching metrics (~10% overhead)
                if matches_path and matches_path.exists():
                    matching_metrics = self._collect_matching_metrics(matches_path)
                
                # MODERATE IMPACT: Point cloud metrics (~10% overhead)
                point_cloud_metrics = self._collect_point_cloud_metrics(model)
            
            # Compile all stats
            stats = {
                'basic_metrics': basic_metrics,
                'per_image_metrics': per_image_metrics,
                'matching_metrics': matching_metrics,
                'point_cloud_metrics': point_cloud_metrics,
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'reconstruction_name': output_name,
                    'frame_type': frame_type,
                    'feature_detector': self.feature,
                    'feature_matcher': self.matcher,
                    'model_summary': model.summary() if model else None
                },
                'file_references': {
                    'features': str(features_path.name) if features_path else None,
                    'matches': str(matches_path.name) if matches_path else None,
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate reconstruction stats: {e}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Return minimal valid stats structure
            return {
                'basic_metrics': {
                    'error': str(e),
                    'num_input_images': 0,
                    'num_registered_images': 0,
                    'registration_rate': 0.0,
                    'num_3d_points': 0,
                    'num_observations': 0,
                    'mean_track_length': 0.0,
                    'mean_observations_per_image': 0.0,
                    'mean_reprojection_error': 0.0,
                },
                'per_image_metrics': None,
                'matching_metrics': None,
                'point_cloud_metrics': None,
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'reconstruction_name': output_name,
                    'frame_type': frame_type,
                    'error': str(e)
                },
                'file_references': None
            }
    
    def _collect_per_image_metrics(self, model, features_path: Path = None) -> dict:
        """Collect per-image quality metrics (LOW computational cost)."""
        try:
            feature_counts = []
            per_image_errors = []
            per_image_points = []
            
            # Load features if available
            features = {}
            if features_path and features_path.exists():
                try:
                    with h5py.File(features_path, 'r') as f:
                        for key in f.keys():
                            if 'keypoints' in f[key]:
                                features[key] = len(f[key]['keypoints'][()])
                except Exception as e:
                    self.logger.warning(f"Could not load features from {features_path}: {e}")
            
            # Collect per-image statistics
            for img_id in model.reg_image_ids():
                image = model.images[img_id]
                
                # Feature counts
                if image.name in features:
                    feature_counts.append(features[image.name])
                
                # Points visible in this image
                num_points = sum(1 for p2D in image.points2D if p2D.point3D_id != -1)
                per_image_points.append(num_points)
                
                # Calculate mean error for this image's observations
                errors = []
                for point2D in image.points2D:
                    if point2D.point3D_id != -1 and point2D.point3D_id in model.points3D:
                        point3D = model.points3D[point2D.point3D_id]
                        errors.append(point3D.error)
                
                if errors:
                    per_image_errors.append(np.mean(errors))
                else:
                    per_image_errors.append(0.0)
            
            return {
                'feature_detection': {
                    'mean_features_per_image': np.mean(feature_counts) if feature_counts else 0,
                    'std_features_per_image': np.std(feature_counts) if feature_counts else 0,
                    'min_features': np.min(feature_counts) if feature_counts else 0,
                    'max_features': np.max(feature_counts) if feature_counts else 0,
                },
                'registration_quality': {
                    'mean_error': np.mean(per_image_errors) if per_image_errors else 0,
                    'std_error': np.std(per_image_errors) if per_image_errors else 0,
                    'mean_registered_points': np.mean(per_image_points) if per_image_points else 0,
                    'std_registered_points': np.std(per_image_points) if per_image_points else 0,
                }
            }
        except Exception as e:
            self.logger.warning(f"Could not collect per-image metrics: {e}")
            return {}
    
    def _collect_matching_metrics(self, matches_path: Path) -> dict:
        """Collect feature matching quality metrics (MODERATE cost)."""
        try:
            total_pairs = 0
            successful_pairs = 0
            match_counts = []
            
            with h5py.File(matches_path, 'r') as f:
                # Handle nested structure: image1/image2/matches0
                for img1_name in f.keys():
                    img1_group = f[img1_name]
                    
                    # Check if this is a group containing other images
                    if isinstance(img1_group, h5py.Group):
                        for img2_name in img1_group.keys():
                            total_pairs += 1
                            img2_group = img1_group[img2_name]
                            
                            # Look for match data
                            if isinstance(img2_group, h5py.Group) and 'matches0' in img2_group:
                                matches = img2_group['matches0'][()]
                                num_matches = np.sum(matches >= 0)
                                
                                if num_matches > 0:
                                    successful_pairs += 1
                                    match_counts.append(num_matches)
                            elif isinstance(img2_group, h5py.Dataset):
                                # Sometimes matches are stored directly as dataset
                                matches = img2_group[()]
                                if matches.size > 0:
                                    num_matches = np.sum(matches >= 0) if matches.ndim > 0 else (1 if matches >= 0 else 0)
                                    if num_matches > 0:
                                        successful_pairs += 1
                                        match_counts.append(num_matches)
            
            return {
                'total_pairs_attempted': total_pairs,
                'successful_pairs': successful_pairs,
                'pair_success_rate': successful_pairs / total_pairs if total_pairs > 0 else 0,
                'match_statistics': {
                    'mean_matches_per_pair': float(np.mean(match_counts)) if match_counts else 0,
                    'std_matches_per_pair': float(np.std(match_counts)) if match_counts else 0,
                    'min_matches': int(np.min(match_counts)) if match_counts else 0,
                    'max_matches': int(np.max(match_counts)) if match_counts else 0,
                }
            }
        except Exception as e:
            self.logger.warning(f"Could not collect matching metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {}
    
    def _collect_point_cloud_metrics(self, model) -> dict:
        """Collect point cloud quality metrics (MODERATE cost)."""
        try:
            depths = []
            triangulation_angles = []
            point_errors = []
            
            for point3D in model.points3D.values():
                # Depth calculation
                depth = np.linalg.norm(point3D.xyz)
                depths.append(depth)
                
                # Reprojection error
                point_errors.append(point3D.error)
                
                # Triangulation angle (for points with 2+ observations)
                if len(point3D.track.elements) >= 2:
                    cameras = []
                    for track_element in list(point3D.track.elements)[:2]:
                        if track_element.image_id in model.images:
                            image = model.images[track_element.image_id]
                            cam_center = image.projection_center()
                            cameras.append(cam_center)
                    
                    if len(cameras) == 2:
                        v1 = cameras[0] - point3D.xyz
                        v2 = cameras[1] - point3D.xyz
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        if norm1 > 0 and norm2 > 0:
                            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle_deg = np.degrees(np.arccos(cos_angle))
                            triangulation_angles.append(angle_deg)
            
            # Quality classification
            high_quality = sum(1 for e in point_errors if e < 1.0)
            medium_quality = sum(1 for e in point_errors if 1.0 <= e < 2.0)
            low_quality = sum(1 for e in point_errors if e >= 2.0)
            
            return {
                'depth_statistics': {
                    'mean_depth': float(np.mean(depths)) if depths else 0,
                    'std_depth': float(np.std(depths)) if depths else 0,
                    'min_depth': float(np.min(depths)) if depths else 0,
                    'max_depth': float(np.max(depths)) if depths else 0,
                    'median_depth': float(np.median(depths)) if depths else 0,
                },
                'triangulation_quality': {
                    'mean_triangulation_angle_degrees': float(np.mean(triangulation_angles)) if triangulation_angles else 0,
                    'min_triangulation_angle': float(np.min(triangulation_angles)) if triangulation_angles else 0,
                    'max_triangulation_angle': float(np.max(triangulation_angles)) if triangulation_angles else 0,
                    'median_triangulation_angle': float(np.median(triangulation_angles)) if triangulation_angles else 0,
                    'optimal_angle_percentage': float(sum(1 for a in triangulation_angles if 5 < a < 30) / len(triangulation_angles)) if triangulation_angles else 0,
                },
                'point_quality_distribution': {
                    'high_quality_points': high_quality,
                    'medium_quality_points': medium_quality,
                    'low_quality_points': low_quality,
                    'high_quality_percentage': high_quality / len(point_errors) if point_errors else 0,
                }
            }
        except Exception as e:
            self.logger.warning(f"Could not collect point cloud metrics: {e}")
            return {}
    
    def generate_per_image_contribution_report(self, model, features_path: Path, matches_path: Path, 
                                               output_name: str, frame_type: str) -> Path:
        """
        Generate a comprehensive per-image contribution analysis report.
        
        For each image, provides:
        - Registration status (was it successfully registered?)
        - Feature detection (how many features were detected?)
        - Matching statistics (how many matches with other images?)
        - 3D contribution (how many 3D points does it observe?)
        - Reprojection quality (what's the average reprojection error?)
        - Pose information (camera position and orientation)
        - Contribution ranking (how important is this image?)
        
        Returns:
            Path to the saved CSV report
        """
        try:
            self.logger.info(f"  Generating per-image contribution report...")
            
            per_image_data = []
            
            # Load features
            features = {}
            if features_path and features_path.exists():
                with h5py.File(features_path, 'r') as f:
                    for img_name in f.keys():
                        if 'keypoints' in f[img_name]:
                            features[img_name] = len(f[img_name]['keypoints'])
            
            # Load matches (nested structure)
            matches_per_image = {}  # image_name -> list of (partner_name, num_matches)
            if matches_path and matches_path.exists():
                with h5py.File(matches_path, 'r') as f:
                    for img1_name in f.keys():
                        if img1_name not in matches_per_image:
                            matches_per_image[img1_name] = []
                        
                        img1_group = f[img1_name]
                        if isinstance(img1_group, h5py.Group):
                            for img2_name in img1_group.keys():
                                img2_group = img1_group[img2_name]
                                
                                # Count matches
                                num_matches = 0
                                if isinstance(img2_group, h5py.Group) and 'matches0' in img2_group:
                                    matches = img2_group['matches0'][()]
                                    num_matches = int(np.sum(matches >= 0))
                                elif isinstance(img2_group, h5py.Dataset):
                                    matches = img2_group[()]
                                    if matches.size > 0:
                                        num_matches = int(np.sum(matches >= 0)) if matches.ndim > 0 else (1 if matches >= 0 else 0)
                                
                                if num_matches > 0:
                                    matches_per_image[img1_name].append((img2_name, num_matches))
                                    
                                    # Add reverse relationship
                                    if img2_name not in matches_per_image:
                                        matches_per_image[img2_name] = []
                                    matches_per_image[img2_name].append((img1_name, num_matches))
            
            # Get all image names from model
            registered_images = {img.name: img for img in model.images.values()}
            
            # Process each image
            all_image_names = set(features.keys()) | set(registered_images.keys())
            
            for img_name in sorted(all_image_names):
                row = {
                    'image_name': img_name,
                    'frame_number': self._extract_frame_number(img_name),
                }
                
                # Registration status
                is_registered = img_name in registered_images
                row['is_registered'] = is_registered
                
                # Feature detection
                row['num_features_detected'] = features.get(img_name, 0)
                
                # Matching statistics
                image_matches = matches_per_image.get(img_name, [])
                row['num_matched_pairs'] = len(image_matches)
                row['total_matches'] = sum(m[1] for m in image_matches)
                row['avg_matches_per_pair'] = row['total_matches'] / row['num_matched_pairs'] if row['num_matched_pairs'] > 0 else 0
                row['max_matches_with_single_image'] = max([m[1] for m in image_matches], default=0)
                
                if is_registered:
                    image = registered_images[img_name]
                    
                    # 3D point observations
                    observed_points = [pid for pid, point in model.points3D.items() 
                                     if any(elem.image_id == image.image_id for elem in point.track.elements)]
                    row['num_3d_points_observed'] = len(observed_points)
                    
                    # Reprojection error
                    errors = []
                    for pid in observed_points:
                        point = model.points3D[pid]
                        errors.append(point.error)
                    row['mean_reprojection_error'] = float(np.mean(errors)) if errors else 0.0
                    row['std_reprojection_error'] = float(np.std(errors)) if errors else 0.0
                    
                    # Pose information
                    try:
                        if hasattr(image.cam_from_world, 'inverse'):
                            pose = image.cam_from_world.inverse()
                        else:
                            pose = image.cam_from_world
                        
                        translation = pose.translation
                        rotation = pose.rotation.quat
                        
                        row['camera_x'] = float(translation[0])
                        row['camera_y'] = float(translation[1])
                        row['camera_z'] = float(translation[2])
                        row['quat_w'] = float(rotation[0])
                        row['quat_x'] = float(rotation[1])
                        row['quat_y'] = float(rotation[2])
                        row['quat_z'] = float(rotation[3])
                    except Exception as e:
                        self.logger.debug(f"Could not extract pose for {img_name}: {e}")
                        row['camera_x'] = row['camera_y'] = row['camera_z'] = 0.0
                        row['quat_w'] = row['quat_x'] = row['quat_y'] = row['quat_z'] = 0.0
                    
                    # Contribution score (weighted combination of metrics)
                    # Higher is better
                    contribution_score = (
                        row['num_3d_points_observed'] * 10.0 +  # Primary contribution
                        row['num_matched_pairs'] * 5.0 +         # Connectivity
                        row['total_matches'] * 0.1 -             # Match quality
                        row['mean_reprojection_error'] * 100.0  # Accuracy penalty
                    )
                    row['contribution_score'] = float(contribution_score)
                    
                else:
                    # Not registered
                    row['num_3d_points_observed'] = 0
                    row['mean_reprojection_error'] = 0.0
                    row['std_reprojection_error'] = 0.0
                    row['camera_x'] = row['camera_y'] = row['camera_z'] = 0.0
                    row['quat_w'] = row['quat_x'] = row['quat_y'] = row['quat_z'] = 0.0
                    row['contribution_score'] = 0.0
                
                per_image_data.append(row)
            
            # Sort by contribution score (descending)
            per_image_data.sort(key=lambda x: x['contribution_score'], reverse=True)
            
            # Add rank
            for i, row in enumerate(per_image_data):
                row['contribution_rank'] = i + 1
            
            # Save to CSV - should be in data/metrics/{video_stem}/per_image_reports/
            # In pipeline mode: metrics_dir = data/metrics/{video_stem}/run_{timestamp}
            # We want: data/metrics/{video_stem}/per_image_reports/
            if self.pipeline_mode:
                # Go up one level from run_XXX to video_stem, then into per_image_reports
                output_dir = self.data_manager.metrics_dir.parent / "per_image_reports"
            else:
                # In standalone mode, use the test output location
                output_dir = self.data_manager.metrics_dir / "per_image_reports"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = output_dir / f"{output_name}_per_image_contribution.csv"
            
            import csv
            fieldnames = [
                'contribution_rank', 'image_name', 'frame_number', 'is_registered',
                'num_features_detected', 'num_matched_pairs', 'total_matches', 
                'avg_matches_per_pair', 'max_matches_with_single_image',
                'num_3d_points_observed', 'mean_reprojection_error', 'std_reprojection_error',
                'camera_x', 'camera_y', 'camera_z', 
                'quat_w', 'quat_x', 'quat_y', 'quat_z',
                'contribution_score'
            ]
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_image_data)
            
            self.logger.info(f"  Per-image contribution report saved: {csv_path}")
            
            # Generate summary statistics
            registered_count = sum(1 for row in per_image_data if row['is_registered'])
            avg_contribution = np.mean([row['contribution_score'] for row in per_image_data if row['is_registered']])
            
            self.logger.info(f"  Summary: {registered_count}/{len(per_image_data)} images registered")
            self.logger.info(f"  Average contribution score: {avg_contribution:.2f}")
            
            # Identify problematic images (not registered but have features)
            problematic = [row for row in per_image_data 
                          if not row['is_registered'] and row['num_features_detected'] > 0]
            if problematic:
                self.logger.warning(f"  {len(problematic)} images have features but failed to register:")
                for row in problematic[:5]:  # Show first 5
                    self.logger.warning(f"    - {row['image_name']}: {row['num_features_detected']} features, "
                                      f"{row['num_matched_pairs']} matched pairs")
            
            return csv_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate per-image contribution report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _extract_frame_number(self, image_name: str) -> int:
        """Extract frame number from image filename."""
        import re
        match = re.search(r'(\d+)\.(?:png|jpg|jpeg)$', image_name)
        return int(match.group(1)) if match else 0
    
    def run_complete_pipeline(self):
        """Run the complete reconstruction pipeline."""
        if self.folder_mode:
            return self.run_single_folder_reconstruction()
        else:
            return self.run_video_pipeline()
            
    def _check_initial_transformation(self) -> bool:
        """
        Prompt user to create/place initial transformation matrix and verify it exists.
        
        Returns:
            True if initial transformation is found, False otherwise
        """
        # Determine where the transformation should be
        initial_transform_file = self.point_clouds_base / "initial_transformation.txt"
        
        self.logger.info("\n" + "="*80)
        self.logger.info("INITIAL TRANSFORMATION REQUIRED FOR ICP REGISTRATION")
        self.logger.info("="*80)
        self.logger.info(f"\nPlease create the initial transformation matrix file:")
        self.logger.info(f"  Location: {initial_transform_file}")
        self.logger.info(f"\nThe file should contain a 4x4 transformation matrix in the format:")
        self.logger.info(f"  -1.324484109879 -1.483755350113 0.880392253399 115.265403747559")
        self.logger.info(f"  -1.550405263901 1.510512351990 0.213249132037 174.122238159180")
        self.logger.info(f"  -0.756877481937 -0.497696936131 -1.977451205254 -81.323654174805")
        self.logger.info(f"  0.000000000000 0.000000000000 0.000000000000 1.000000000000")
        
        # Retry loop for user confirmation
        while True:
            input("\nPress ENTER once you have created the initial_transformation.txt file...")
            
            # Check if file exists
            if initial_transform_file.exists():
                # Try to load and validate it
                try:
                    transformation = np.loadtxt(str(initial_transform_file))
                    if transformation.shape != (4, 4):
                        self.logger.error(f"ERROR: Transformation matrix must be 4x4, got {transformation.shape}")
                        self.logger.info("Please fix the file and try again.")
                        continue
                    
                    self.logger.info(f"✓ Initial transformation found and validated!")
                    self.logger.info(f"  Location: {initial_transform_file}")
                    return True
                except Exception as e:
                    self.logger.error(f"ERROR: Failed to load transformation matrix: {e}")
                    self.logger.info("Please check the file format and try again.")
                    continue
            else:
                self.logger.error(f"ERROR: File not found: {initial_transform_file}")
                self.logger.info("Please create the file and try again.")
                continue

    def run_video_pipeline(self):
        """Run the complete reconstruction pipeline on all available frame sets for a video."""
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
        self.logger.info("RECONSTRUCTION COMPLETED - PROCEED WITH REGISTRATION")
        self.logger.info("="*80)
        
        # Check for initial transformation before registration
        if not self._check_initial_transformation():
            self.logger.error("Initial transformation not found - skipping ICP registration")
            return len(all_results) > 0
        
        # Proceed with ICP registration using the initial transformation
        self.logger.info("\nStarting ICP registration with initial transformation...")
        # Registration will be called by main pipeline if successful
        
        return len(all_results) > 0
    
    def _create_reconstruction_summary(self, all_results: dict):
        """Create a summary comparison of all reconstructions."""
        if not all_results:
            return
            
        self.logger.info("\n--- Reconstruction Quality Summary ---")
        
        # Create comparison DataFrame
        summary_data = []
        for name, stats in all_results.items():
            basic = stats.get('basic_metrics', {})
            processing = stats.get('processing_info', {})
            
            if 'error' not in basic:
                summary_data.append({
                    'reconstruction': name,
                    'frame_type': processing.get('frame_type', 'unknown'),
                    'registered_images': basic.get('num_registered_images', 0),
                    'num_3d_points': basic.get('num_3d_points', 0),
                    'mean_reprojection_error': basic.get('mean_reprojection_error', 0),
                    'mean_track_length': basic.get('mean_track_length', 0),
                    'mean_observations_per_image': basic.get('mean_observations_per_image', 0)
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Save summary CSV with timestamp
            summary_path = self.metrics_base / f"{self.video_stem}_reconstruction_summary_{self.timestamp}.csv"
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
        description="VAPOR 3D Reconstruction Pipeline",
        epilog="""
Examples:
  # Process a specific folder of PNG images:
  python reconstruction_pipeline.py --folder "S:\\Kesney\\VAPOR\\data\\frames\\original\\pat3"
  
  # Process all frame sets for a video:
  python reconstruction_pipeline.py --video pat3.mp4
        """
    )
    
    # Create mutually exclusive group for folder or video mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--folder",
        help="Path to folder containing PNG images for reconstruction"
    )
    mode_group.add_argument(
        "--video",
        help="Video filename (e.g., pat3.mp4) - processes all frame sets"
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
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleaning previous reconstruction outputs (useful for permission issues)"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique run identifier (timestamp). If not provided, uses 'test_run' for standalone mode."
    )
    parser.add_argument(
        "--pipeline-mode",
        action="store_true",
        help="Enable pipeline mode for timestamped persistent storage (vs test_run that can be overwritten)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline based on mode
    if args.folder:
        pipeline = VAPORReconstructionPipeline(
            folder_path=args.folder,
            feature=args.feature,
            matcher=args.matcher,
            skip_cleanup=args.skip_cleanup,
            run_id=args.run_id,
            pipeline_mode=args.pipeline_mode
        )
    else:
        pipeline = VAPORReconstructionPipeline(
            video_name=args.video,
            feature=args.feature,
            matcher=args.matcher,
            skip_cleanup=args.skip_cleanup,
            run_id=args.run_id,
            pipeline_mode=args.pipeline_mode
        )
    
    success = pipeline.run_complete_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())