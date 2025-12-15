"""
Point Cloud Registration and Comparison Calculator for VAPOR
Comprehensive 3D reconstruction evaluation with ICP registration and all metrics.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import csv
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import open3d as o3d
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Error: Required library not installed: {e}")
    print("Please install: pip install open3d numpy scipy pandas matplotlib")
    sys.exit(1)

from reconstruction.registration_icp import ICPRegistration
from reconstruction.registration_metrics import PointCloudMetrics

# Import all comprehensive metric modules
from registration_metrics_modules.core.ct_mesh_preprocessing import CTMeshPreprocessor
from registration_metrics_modules.core.roi_masking import ROIMaskGenerator
from registration_metrics_modules.core.distance_metrics import DistanceMetricsCalculator
from registration_metrics_modules.core.normal_consistency import NormalConsistencyCalculator
from registration_metrics_modules.core.coverage_sampling import CoverageAnalyzer
from registration_metrics_modules.core.noise_smoothness import NoiseSmoothnessAnalyzer
from registration_metrics_modules.core.registration_sanity import RegistrationSanityChecker
from registration_metrics_modules.core.statistical_analysis import StatisticalAnalyzer
from registration_metrics_modules.core.visualization import VisualizationGenerator


class RegistrationCalculator:
    """ICP-based registration calculator requiring initial transformation matrix."""
    
    def __init__(self, 
                 video_name: str,
                 run_name: str,
                 ground_truth_name: str = None):
        """
        Initialize registration calculator.
        
        Args:
            video_name: Video name (e.g., '4_32_slow_run_20250403_094922')
            run_name: Run timestamp (e.g., 'run_20251023_071408')
            ground_truth_name: Name of ground truth PLY file (without .ply extension)
        """
        self.video_name = video_name
        self.run_name = run_name
        self.ground_truth_name = ground_truth_name or "ground_truth"
        
        # Setup paths
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        
        # Ground truth path - handle both segmentation and regular ground truth
        # Try segmentation file first (e.g., 4_32_segmentation.ply)
        segmentation_name = f"{video_name.split('_')[0]}_{video_name.split('_')[1]}_segmentation.ply"
        self.ground_truth_path = self.data_dir / "point_clouds" / video_name / segmentation_name
        
        # If segmentation doesn't exist, try the provided ground truth name
        if not self.ground_truth_path.exists():
            self.ground_truth_path = self.data_dir / "point_clouds" / video_name / f"{self.ground_truth_name}.ply"
        
        # Reconstruction base path
        self.recon_base = self.data_dir / "point_clouds" / video_name / run_name
        
        # Metrics base path
        self.metrics_base = self.data_dir / "metrics" / video_name / run_name / "reconstruction_metrics"
        
        # Setup logging
        self._setup_logging()
        
        # Initialize ICP registrator
        self.registrator = ICPRegistration(
            max_correspondence_distance=0.2,
            relative_fitness=1e-4,
            relative_rmse=1e-4,
            max_iterations=300
        )
        
        self.metrics_calculator = PointCloudMetrics()
        
        # Storage for CSV export
        self.all_results = []
        
        # Storage for comprehensive metrics (new modules)
        self.comprehensive_metrics = {}
        self.ct_mesh_processed = None
        self.h = None  # Characteristic length
        self.raycasting_scene = None
        
        # Verify ground truth exists
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_path}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    log_dir / f"registration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def find_reconstruction_methods(self) -> list:
        """
        Find all reconstruction methods in the run directory.
        
        Returns:
            List of method directories
        """
        methods = []
        
        # Scan the run directory for all subdirectories
        if not self.recon_base.exists():
            self.logger.error(f"Reconstruction base directory not found: {self.recon_base}")
            return methods
        
        for method_dir in self.recon_base.iterdir():
            if not method_dir.is_dir():
                continue
            
            # Check for direct reconstruction.ply
            reconstruction_file = method_dir / "reconstruction.ply"
            if reconstruction_file.exists():
                # Determine type from name or default to 'original'
                if 'deblur' in method_dir.name.lower() or method_dir.name in ['MPRNet', 'Restormer', 'Uformer']:
                    method_type = 'deblurred'
                elif 'blur' in method_dir.name.lower():
                    method_type = 'blurred'
                else:
                    method_type = 'original'
                
                methods.append({
                    'type': method_type,
                    'name': method_dir.name,
                    'path': method_dir,
                    'reconstruction_file': reconstruction_file
                })
            else:
                # Check subdirectories (e.g., motion_blur_high/blurred_motion_blur_high/)
                for subdir in method_dir.iterdir():
                    if subdir.is_dir():
                        reconstruction_file = subdir / "reconstruction.ply"
                        if reconstruction_file.exists():
                            # Determine type from subdirectory name
                            if 'deblur' in subdir.name.lower():
                                method_type = 'deblurred'
                            elif 'blur' in subdir.name.lower():
                                method_type = 'blurred'
                            else:
                                method_type = method_dir.name
                            
                            # Use parent directory name as the method name
                            methods.append({
                                'type': method_type,
                                'name': method_dir.name,
                                'path': subdir,  # Use subdirectory path
                                'reconstruction_file': reconstruction_file
                            })
        
        return methods
    
    def _load_initial_transformation(self, method_info: Dict[str, Any]) -> np.ndarray:
        """
        Load the required initial transformation matrix for ICP.
        Interactive prompt if file is missing or invalid.
        
        Args:
            method_info: Dictionary with method information
            
        Returns:
            4x4 transformation matrix
        """
        initial_transform_file = self.recon_base / "initial_transformation.txt"
        
        while True:
            # Check if file exists
            if not initial_transform_file.exists():
                self.logger.error("="*80)
                self.logger.error("INITIAL TRANSFORMATION FILE NOT FOUND")
                self.logger.error("="*80)
                self.logger.error(f"Expected location: {initial_transform_file}")
                self.logger.error("")
                self.logger.error("Please create the file with a 4x4 transformation matrix.")
                self.logger.error("Example format:")
                self.logger.error("  -1.324484109879 -1.483755350113 0.880392253399 115.265403747559")
                self.logger.error("  -1.550405263901 1.510512351990 0.213249132037 174.122238159180")
                self.logger.error("  -0.756877481937 -0.497696936131 -1.977451205254 -81.323654174805")
                self.logger.error("  0.000000000000 0.000000000000 0.000000000000 1.000000000000")
                self.logger.error("="*80)
                
                input("\nPress ENTER after creating the file to try again (or Ctrl+C to abort)...")
                continue
            
            # Try to load and validate the file
            try:
                # Try loading with different encodings and skip empty lines
                with open(initial_transform_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    raise ValueError("File is empty")
                
                # Parse the matrix
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                if len(lines) != 4:
                    raise ValueError(f"Expected 4 rows, got {len(lines)}")
                
                matrix_data = []
                for i, line in enumerate(lines):
                    values = line.split()
                    if len(values) != 4:
                        raise ValueError(f"Row {i+1} has {len(values)} values, expected 4")
                    matrix_data.append([float(v) for v in values])
                
                transformation = np.array(matrix_data)
                
                if transformation.shape != (4, 4):
                    raise ValueError(f"Transformation matrix must be 4x4, got {transformation.shape}")
                
                # Successfully loaded!
                self.logger.info("="*80)
                self.logger.info("SUCCESS: Initial transformation loaded successfully!")
                self.logger.info("="*80)
                self.logger.info(f"Location: {initial_transform_file}")
                self.logger.info(f"Matrix shape: {transformation.shape}")
                self.logger.info("Transformation matrix:")
                for row in transformation:
                    self.logger.info(f"  {' '.join(f'{v:>15.9f}' for v in row)}")
                self.logger.info("="*80)
                return transformation
                
            except ValueError as e:
                self.logger.error("="*80)
                self.logger.error("INVALID TRANSFORMATION FILE FORMAT")
                self.logger.error("="*80)
                self.logger.error(f"File: {initial_transform_file}")
                self.logger.error(f"Error: {e}")
                self.logger.error("")
                self.logger.error("The file must contain exactly 4 rows with 4 numeric values each.")
                self.logger.error("Example format:")
                self.logger.error("  -1.324484109879 -1.483755350113 0.880392253399 115.265403747559")
                self.logger.error("  -1.550405263901 1.510512351990 0.213249132037 174.122238159180")
                self.logger.error("  -0.756877481937 -0.497696936131 -1.977451205254 -81.323654174805")
                self.logger.error("  0.000000000000 0.000000000000 0.000000000000 1.000000000000")
                self.logger.error("="*80)
                
                input("\nPress ENTER after fixing the file to try again (or Ctrl+C to abort)...")
                continue
                
            except Exception as e:
                self.logger.error("="*80)
                self.logger.error("ERROR READING TRANSFORMATION FILE")
                self.logger.error("="*80)
                self.logger.error(f"File: {initial_transform_file}")
                self.logger.error(f"Error: {e}")
                self.logger.error(f"File size: {initial_transform_file.stat().st_size if initial_transform_file.exists() else 'N/A'} bytes")
                self.logger.error("="*80)
                
                input("\nPress ENTER after fixing the file to try again (or Ctrl+C to abort)...")
                continue
    
    def process_method(self, method_info: Dict[str, Any]) -> bool:
        """
        Process a single reconstruction method with ICP registration.
        
        Args:
            method_info: Dictionary with method information
            
        Returns:
            True if successful, False otherwise
        """
        method_name = method_info['name']
        method_type = method_info['type']
        reconstruction_file = method_info['reconstruction_file']
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Processing: {method_type}/{method_name}")
        self.logger.info(f"{'='*80}")
        
        # Check if ground truth exists
        if not self.ground_truth_path.exists():
            self.logger.error(f"Ground truth not found: {self.ground_truth_path}")
            return False
        
        # Setup output directory
        registration_dir = method_info['path'] / "registration"
        registration_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load required initial transformation matrix
            init_transform = self._load_initial_transformation(method_info)
            self.registrator.init_transformation = init_transform
            
            # Run ICP registration
            self.logger.info(f"Running ICP registration with initial transformation...")
            registered_cloud, registration_metadata = self.registrator.register(
                source_path=reconstruction_file,
                target_path=self.ground_truth_path,
                output_dir=registration_dir
            )
            
            # Load ground truth for metrics
            ground_truth = o3d.io.read_point_cloud(str(self.ground_truth_path))
            
            # Compute all metrics
            self.logger.info("Computing comparison metrics...")
            comparison_metrics = self.metrics_calculator.compute_all_metrics(
                source=registered_cloud,
                target=ground_truth,
                thresholds=[0.001, 0.005, 0.01]  # 1mm, 5mm, 10mm
            )
            
            # Generate distance heatmap
            heatmap_path = registration_dir / "distance_heatmap.ply"
            heatmap_info = self.metrics_calculator.generate_distance_heatmap(
                source=registered_cloud,
                target=ground_truth,
                output_path=heatmap_path,
                max_distance=0.01  # 10mm
            )
            
            # Generate alignment visualization (combined view)
            self.logger.info("Generating alignment visualization...")
            alignment_viz_path = registration_dir / "alignment_visualization.ply"
            self._generate_alignment_visualization(
                source=registered_cloud,
                target=ground_truth,
                output_path=alignment_viz_path
            )
            
            # Update existing reconstruction metrics JSON
            metrics_file = self.metrics_base / f"{method_name}.json"
            
            if metrics_file.exists():
                self.logger.info(f"Updating existing metrics file: {metrics_file}")
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            else:
                self.logger.warning(f"Metrics file not found, creating new: {metrics_file}")
                existing_metrics = {
                    'metric_type': 'reconstruction',
                    'frame_type': method_type,
                    'method_name': method_name
                }
            
            # Add registration metrics
            existing_metrics['registration'] = {
                'timestamp': datetime.now().isoformat(),
                'registration_method': 'ICP',
                'initial_transformation_file': str(self.recon_base / "initial_transformation.txt"),
                'ground_truth_file': str(self.ground_truth_path),
                'registration_parameters': registration_metadata.get('parameters', {}),
                'registration_result': registration_metadata.get('result', registration_metadata.get('final_result', {})),
                'outputs': {
                    'registered_cloud': str(registration_dir / "registered_cloud.ply"),
                    'transformation_matrix': str(registration_dir / "transformation.txt"),
                    'metadata': str(registration_dir / "registration_metadata.json"),
                    'distance_heatmap': str(heatmap_path),
                    'alignment_visualization': str(alignment_viz_path)
                }
            }
            
            # Compute numerical quality assessment
            quality_metrics = self._compute_numerical_assessment(comparison_metrics)
            
            # Add comparison metrics
            existing_metrics['point_cloud_comparison'] = {
                'timestamp': datetime.now().isoformat(),
                'ground_truth_points': comparison_metrics['point_counts']['target_points'],
                'reconstruction_points': comparison_metrics['point_counts']['source_points'],
                'coverage_ratio': comparison_metrics['point_counts']['source_points'] / comparison_metrics['point_counts']['target_points'],
                'primary_metrics': {
                    'chamfer_distance_m': comparison_metrics['chamfer']['chamfer_distance'],
                    'chamfer_distance_mm': comparison_metrics['chamfer']['chamfer_distance'] * 1000,
                    'hausdorff_distance_m': comparison_metrics['hausdorff']['hausdorff_distance'],
                    'hausdorff_distance_mm': comparison_metrics['hausdorff']['hausdorff_distance'] * 1000,
                    'rms_error_m': comparison_metrics['rms']['rms_error'],
                    'rms_error_mm': comparison_metrics['rms']['rms_error'] * 1000,
                    'mean_absolute_error_m': comparison_metrics['mae']['mean_absolute_error'],
                    'mean_absolute_error_mm': comparison_metrics['mae']['mean_absolute_error'] * 1000,
                    'median_absolute_error_m': comparison_metrics['mae']['median_absolute_error'],
                    'median_absolute_error_mm': comparison_metrics['mae']['median_absolute_error'] * 1000
                },
                'completeness': comparison_metrics['completeness'],
                'distance_statistics': comparison_metrics['distance_statistics'],
                'numerical_quality_metrics': quality_metrics,
                'heatmap_visualization': heatmap_info
            }
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
            
            self.logger.info(f"Updated metrics saved to: {metrics_file}")
            
            # Store results for CSV export
            self._store_csv_row(method_name, method_type, comparison_metrics, quality_metrics, registration_metadata)
            
            # Print summary
            self._print_summary(method_name, comparison_metrics)
            
            # Compute comprehensive metrics using new modules (Steps 2-7, 9-10)
            if self.ct_mesh_processed is not None:
                method_comprehensive_dir = self.metrics_base / method_name
                method_comprehensive_dir.mkdir(parents=True, exist_ok=True)
                
                self._compute_comprehensive_metrics(method_name, registered_cloud, method_comprehensive_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {method_name}: {e}", exc_info=True)
            return False
    
    def _compute_numerical_assessment(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute numerical quality metrics (no subjective labels).
        
        Args:
            metrics: Computed metrics
            
        Returns:
            Numerical quality metrics dictionary
        """
        mae_mm = metrics['mae']['mean_absolute_error'] * 1000
        chamfer_mm = metrics['chamfer']['chamfer_distance'] * 1000
        rms_mm = metrics['rms']['rms_error'] * 1000
        hausdorff_mm = metrics['hausdorff']['hausdorff_distance'] * 1000
        median_mm = metrics['mae']['median_absolute_error'] * 1000
        
        completeness_1mm = metrics['completeness']['threshold_1mm']['completeness_percent']
        completeness_5mm = metrics['completeness']['threshold_5mm']['completeness_percent']
        completeness_10mm = metrics['completeness']['threshold_10mm']['completeness_percent']
        
        # Distance statistics (percentiles are already in mm in distance_statistics)
        dist_stats = metrics['distance_statistics']
        p95_mm = dist_stats['percentiles']['p95']
        p99_mm = dist_stats['percentiles']['p99']
        
        return {
            'primary_metrics_mm': {
                'mean_absolute_error': round(mae_mm, 4),
                'median_absolute_error': round(median_mm, 4),
                'rms_error': round(rms_mm, 4),
                'chamfer_distance': round(chamfer_mm, 4),
                'hausdorff_distance': round(hausdorff_mm, 4)
            },
            'completeness_thresholds_percent': {
                'within_1mm': round(completeness_1mm, 2),
                'within_5mm': round(completeness_5mm, 2),
                'within_10mm': round(completeness_10mm, 2)
            },
            'distance_percentiles_mm': {
                'p95': round(p95_mm, 4),
                'p99': round(p99_mm, 4)
            },
            'point_coverage': {
                'source_points': metrics['point_counts']['source_points'],
                'target_points': metrics['point_counts']['target_points'],
                'coverage_ratio': round(metrics['point_counts']['source_points'] / metrics['point_counts']['target_points'], 4)
            }
        }
    
    def _generate_alignment_visualization(self, source: o3d.geometry.PointCloud, 
                                          target: o3d.geometry.PointCloud, 
                                          output_path: Path):
        """
        Generate alignment visualization with colored point clouds.
        
        Args:
            source: Registered source point cloud
            target: Target ground truth point cloud
            output_path: Path to save visualization
        """
        # Create copies
        source_viz = o3d.geometry.PointCloud(source)
        target_viz = o3d.geometry.PointCloud(target)
        
        # Color source (reconstruction) in red
        source_viz.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        
        # Color target (ground truth) in blue
        target_viz.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        
        # Combine point clouds
        combined = source_viz + target_viz
        
        # Save
        o3d.io.write_point_cloud(str(output_path), combined)
        self.logger.info(f"Saved alignment visualization to: {output_path}")
        self.logger.info("  Red = Registered reconstruction, Blue = Ground truth")
    
    def _store_csv_row(self, method_name: str, method_type: str, 
                       comparison_metrics: Dict[str, Any], 
                       quality_metrics: Dict[str, Any],
                       registration_metadata: Dict[str, Any]):
        """Store results for CSV export."""
        row = {
            'video_name': self.video_name,
            'run_name': self.run_name,
            'method_name': method_name,
            'method_type': method_type,
            'timestamp': datetime.now().isoformat(),
            
            # ICP Registration metrics
            'icp_fitness': registration_metadata.get('result', {}).get('fitness', 0),
            'icp_inlier_rmse': registration_metadata.get('result', {}).get('inlier_rmse', 0),
            
            # Point counts
            'source_points': comparison_metrics['point_counts']['source_points'],
            'target_points': comparison_metrics['point_counts']['target_points'],
            'coverage_ratio': round(comparison_metrics['point_counts']['source_points'] / comparison_metrics['point_counts']['target_points'], 4),
            
            # Primary distance metrics (mm)
            'mae_mm': quality_metrics['primary_metrics_mm']['mean_absolute_error'],
            'median_ae_mm': quality_metrics['primary_metrics_mm']['median_absolute_error'],
            'rms_error_mm': quality_metrics['primary_metrics_mm']['rms_error'],
            'chamfer_distance_mm': quality_metrics['primary_metrics_mm']['chamfer_distance'],
            'hausdorff_distance_mm': quality_metrics['primary_metrics_mm']['hausdorff_distance'],
            
            # Completeness percentages
            'completeness_1mm_percent': quality_metrics['completeness_thresholds_percent']['within_1mm'],
            'completeness_5mm_percent': quality_metrics['completeness_thresholds_percent']['within_5mm'],
            'completeness_10mm_percent': quality_metrics['completeness_thresholds_percent']['within_10mm'],
            
            # Distance percentiles (mm)
            'p95_distance_mm': quality_metrics['distance_percentiles_mm']['p95'],
            'p99_distance_mm': quality_metrics['distance_percentiles_mm']['p99'],
        }
        
        self.all_results.append(row)
    
    def _export_csv_summary(self):
        """Export all results to a CSV file."""
        if not self.all_results:
            self.logger.warning("No results to export to CSV")
            return
        
        # Create CSV directory in data/metrics/
        csv_dir = self.data_dir / "metrics" / "reconstruction_metrics"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        csv_filename = f"registration_summary_{self.video_name}_{self.run_name}.csv"
        csv_path = csv_dir / csv_filename
        
        # Write CSV
        fieldnames = list(self.all_results[0].keys())
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_results)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"CSV SUMMARY EXPORTED")
        self.logger.info(f"Location: {csv_path}")
        self.logger.info(f"Methods: {len(self.all_results)}")
        self.logger.info(f"{'='*80}")
    
    def _print_summary(self, method_name: str, metrics: Dict[str, Any]):
        """Print summary of metrics."""
        print(f"\n{'='*80}")
        print(f"SUMMARY: {method_name}")
        print(f"{'='*80}")
        print(f"Chamfer Distance:    {metrics['chamfer']['chamfer_distance']*1000:.3f} mm")
        print(f"MAE:                 {metrics['mae']['mean_absolute_error']*1000:.3f} mm")
        print(f"Median AE:           {metrics['mae']['median_absolute_error']*1000:.3f} mm")
        print(f"RMS Error:           {metrics['rms']['rms_error']*1000:.3f} mm")
        print(f"Hausdorff Distance:  {metrics['hausdorff']['hausdorff_distance']*1000:.3f} mm")
        print(f"Completeness (1mm):  {metrics['completeness']['threshold_1mm']['completeness_percent']:.1f}%")
        print(f"Completeness (5mm):  {metrics['completeness']['threshold_5mm']['completeness_percent']:.1f}%")
        print(f"Completeness (10mm): {metrics['completeness']['threshold_10mm']['completeness_percent']:.1f}%")
        print(f"{'='*80}\n")
    
    def _preprocess_ground_truth(self):
        """Preprocess ground truth mesh for comprehensive metrics (Step 1)."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 1: Preprocessing Ground Truth CT Mesh")
        self.logger.info("="*80)
        
        # Load CT mesh
        ct_mesh = o3d.io.read_triangle_mesh(str(self.ground_truth_path))
        
        # Create output directory for CT validation
        ct_output_dir = self.metrics_base / 'GT_sanity_check'
        ct_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocess
        preprocessor = CTMeshPreprocessor(ct_output_dir, self.logger)
        report = preprocessor.preprocess(ct_mesh, clean=True)
        preprocessor.save_health_report()
        preprocessor.save_preview()
        
        self.ct_mesh_processed = preprocessor.mesh
        self.h = preprocessor.characteristic_length_h
        self.raycasting_scene = preprocessor.raycasting_scene
        
        self.logger.info(f"✓ Characteristic length h = {self.h*1000:.3f} mm")
        self.logger.info(f"✓ CT mesh preprocessed: {len(self.ct_mesh_processed.vertices)} vertices")
    
    def _compute_comprehensive_metrics(self, method_name: str, registered_cloud: o3d.geometry.PointCloud, 
                                      method_output_dir: Path):
        """
        Compute comprehensive metrics using all new modules (Steps 2-7, 9-10).
        
        Args:
            method_name: Name of the method
            registered_cloud: Registered point cloud
            method_output_dir: Output directory for this method
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Computing Comprehensive Metrics: {method_name}")
        self.logger.info(f"{'='*80}")
        
        metrics = {}
        
        # Step 2: Generate ROI
        self.logger.info(f"\n--- Step 2: ROI Generation ---")
        roi_generator = ROIMaskGenerator(method_output_dir, self.h, self.logger)
        roi_mask = roi_generator.generate_proximity_based_roi(self.ct_mesh_processed, registered_cloud)
        roi_mesh = roi_generator.extract_roi_mesh(self.ct_mesh_processed, roi_mask)
        roi_samples = roi_generator.sample_roi_poisson_disk(roi_mesh)
        roi_generator.save_roi_mask(method_name)
        roi_generator.save_roi_visualization(self.ct_mesh_processed, roi_mask, method_name)
        
        metrics['roi_info'] = roi_generator.roi_info
        
        # Step 3: Distance Metrics
        self.logger.info(f"\n--- Step 3: Distance Metrics ---")
        dist_calc = DistanceMetricsCalculator(method_output_dir, self.h, self.logger)
        dist_metrics = dist_calc.compute_all_metrics(
            registered_cloud, self.raycasting_scene, roi_samples,
            thresholds=[self.h, 2*self.h, 3*self.h]
        )
        dist_calc.save_distance_arrays(method_name)
        dist_calc.save_metrics_table(method_name)
        
        metrics['distance_metrics'] = dist_metrics
        
        # Step 4: Normal Consistency
        self.logger.info(f"\n--- Step 4: Normal Consistency ---")
        normal_calc = NormalConsistencyCalculator(method_output_dir, self.h, self.logger)
        normal_stats = normal_calc.compute_normal_consistency(
            registered_cloud, self.ct_mesh_processed, self.raycasting_scene
        )
        normal_calc.save_results(method_name)
        
        metrics['normal_consistency'] = normal_stats
        
        # Step 5: Coverage & Sampling
        self.logger.info(f"\n--- Step 5: Coverage & Sampling ---")
        coverage_analyzer = CoverageAnalyzer(method_output_dir, self.h, self.logger)
        
        coverage_results = coverage_analyzer.compute_coverage_at_thresholds(
            roi_samples, registered_cloud, [self.h, 2*self.h, 3*self.h]
        )
        
        observed_area = coverage_analyzer.compute_observed_area(
            self.ct_mesh_processed, roi_mask, dist_calc.s2p_distances
        )
        
        holes = coverage_analyzer.analyze_holes(
            self.ct_mesh_processed, roi_mask, dist_calc.s2p_distances
        )
        
        density = coverage_analyzer.compute_density_uniformity(
            roi_mesh, registered_cloud
        )
        
        coverage_results_all = {
            'coverage': coverage_results,
            'observed_area': observed_area,
            'holes': holes,
            'density': density,
        }
        coverage_analyzer.save_results(method_name, coverage_results_all)
        
        metrics['coverage_sampling'] = coverage_results_all
        
        # Step 6: Noise & Smoothness
        self.logger.info(f"\n--- Step 6: Noise & Smoothness ---")
        smoothness_analyzer = NoiseSmoothnessAnalyzer(method_output_dir, self.logger)
        
        roughness, roughness_stats = smoothness_analyzer.compute_local_roughness(registered_cloud)
        shape_descriptors = smoothness_analyzer.compute_shape_descriptors(registered_cloud)
        smoothness_analyzer.save_results(method_name, roughness, roughness_stats, shape_descriptors)
        
        metrics['noise_smoothness'] = {
            'roughness': roughness_stats,
            'shape_descriptors': shape_descriptors,
        }
        
        # Step 7: Registration Sanity
        self.logger.info(f"\n--- Step 7: Registration Sanity Check ---")
        reg_checker = RegistrationSanityChecker(method_output_dir, self.h, self.logger)
        reg_stats = reg_checker.run_icp_sanity_check(registered_cloud, roi_mesh)
        reg_checker.save_results(method_name, reg_stats)
        
        metrics['registration_sanity'] = reg_stats
        
        # Step 9: Statistical Analysis
        self.logger.info(f"\n--- Step 9: Statistical Analysis ---")
        stats_analyzer = StatisticalAnalyzer(method_output_dir, self.logger)
        
        bootstrap_p2s = stats_analyzer.bootstrap_confidence_intervals(dist_calc.p2s_distances)
        bootstrap_s2p = stats_analyzer.bootstrap_confidence_intervals(dist_calc.s2p_distances)
        
        bootstrap_results = {
            'p2s_bootstrap_ci': bootstrap_p2s,
            's2p_bootstrap_ci': bootstrap_s2p,
        }
        stats_analyzer.save_results(method_name, bootstrap_results)
        
        metrics['statistical_analysis'] = bootstrap_results
        
        # Step 10: Visualizations
        self.logger.info(f"\n--- Step 10: Visualizations ---")
        viz_gen = VisualizationGenerator(method_output_dir, self.logger)
        
        viz_gen.generate_distance_heatmap(
            registered_cloud, dist_calc.p2s_distances, 
            f'p2s_{method_name}', max_distance=3*self.h
        )
        
        viz_gen.generate_histogram(
            dist_calc.p2s_distances, f'p2s_{method_name}',
            title=f'P2S Distance Distribution - {method_name}'
        )
        
        viz_gen.generate_histogram(
            dist_calc.s2p_distances, f's2p_{method_name}',
            title=f'S2P Distance Distribution - {method_name}'
        )
        
        self.comprehensive_metrics[method_name] = metrics
        self.logger.info(f"✓ Comprehensive metrics complete for {method_name}")
        
        return metrics
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive markdown report (Step 11)."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 11: Generating Comprehensive Report")
        self.logger.info("="*80)
        
        report_path = self.metrics_base / 'comprehensive_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# VAPOR 3D Reconstruction Comprehensive Metrics Report\n\n")
            f.write(f"**Video:** {self.video_name}  \n")
            f.write(f"**Run:** {self.run_name}  \n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Characteristic Length (h):** {self.h*1000:.3f} mm  \n\n")
            
            f.write("---\n\n")
            
            # Per-method comprehensive summaries
            for method_name, metrics in self.comprehensive_metrics.items():
                dist_metrics = metrics['distance_metrics']
                
                f.write(f"## {method_name}\n\n")
                f.write(f"**ROI Coverage:** {metrics['roi_info']['roi_percentage']:.1f}%  \n")
                f.write(f"**Points:** {dist_metrics['n_reconstruction_points']:,}  \n\n")
                
                f.write("### Distance Metrics\n\n")
                f.write(f"- P2S Median: {dist_metrics['p2s_statistics']['median']*1000:.3f} mm\n")
                f.write(f"- P2S Mean: {dist_metrics['p2s_statistics']['mean']*1000:.3f} mm\n")
                f.write(f"- P2S P95: {dist_metrics['p2s_statistics']['p95']*1000:.3f} mm\n")
                f.write(f"- S2P Median: {dist_metrics['s2p_statistics']['median']*1000:.3f} mm\n")
                f.write(f"- Chamfer: {dist_metrics['chamfer']['chamfer_distance']:.8f}\n")
                f.write(f"- HD95: {dist_metrics['hausdorff']['hd95']*1000:.3f} mm\n")
                f.write(f"- ASSD: {dist_metrics['assd']['assd']*1000:.3f} mm\n\n")
                
                f.write("### Quality Metrics\n\n")
                f.write(f"- Normal Consistency: {metrics['normal_consistency']['median_angle_deg']:.2f}°\n")
                f.write(f"- ICP Fitness: {metrics['registration_sanity']['fitness']:.3f}\n")
                f.write(f"- RMSE: {metrics['registration_sanity']['rmse_mm']:.3f} mm\n")
                f.write(f"- Roughness: {metrics['noise_smoothness']['roughness']['median']*1000:.4f} mm\n")
                f.write(f"- Density Gini: {metrics['coverage_sampling']['density']['gini_coefficient']:.3f}\n\n")
                
                f.write("### Coverage Analysis\n\n")
                for thresh_key, thresh_data in metrics['coverage_sampling']['coverage'].items():
                    f.write(f"- {thresh_key}: {thresh_data['coverage_percentage']:.1f}%\n")
                f.write(f"\n- Observed Area: {metrics['coverage_sampling']['observed_area']['observed_percentage']:.1f}%\n")
                f.write(f"- Holes Detected: {metrics['coverage_sampling']['holes']['n_holes']}\n\n")
                
                f.write("---\n\n")
            
            # Ranking
            f.write("## Model Ranking (by P2S median)\n\n")
            ranked = sorted(
                self.comprehensive_metrics.items(),
                key=lambda x: x[1]['distance_metrics']['p2s_statistics']['median']
            )
            
            for rank, (method_name, metrics) in enumerate(ranked, 1):
                p2s = metrics['distance_metrics']['p2s_statistics']['median'] * 1000
                f.write(f"{rank}. **{method_name}** - {p2s:.3f} mm\n")
        
        self.logger.info(f"✓ Comprehensive report saved: {report_path}")
    
    def _export_comprehensive_csv(self):
        """Export comprehensive metrics to CSV."""
        if not self.comprehensive_metrics:
            return
        
        rows = []
        
        for method_name, metrics in self.comprehensive_metrics.items():
            dist_metrics = metrics['distance_metrics']
            row = {
                'video': self.video_name,
                'run': self.run_name,
                'method': method_name,
                'n_points': dist_metrics['n_reconstruction_points'],
                'roi_coverage_pct': metrics['roi_info']['roi_percentage'],
                
                # Distance metrics (mm)
                'p2s_median_mm': dist_metrics['p2s_statistics']['median'] * 1000,
                'p2s_mean_mm': dist_metrics['p2s_statistics']['mean'] * 1000,
                'p2s_p95_mm': dist_metrics['p2s_statistics']['p95'] * 1000,
                's2p_median_mm': dist_metrics['s2p_statistics']['median'] * 1000,
                's2p_mean_mm': dist_metrics['s2p_statistics']['mean'] * 1000,
                'chamfer': dist_metrics['chamfer']['chamfer_distance'],
                'hd95_mm': dist_metrics['hausdorff']['hd95'] * 1000,
                'assd_mm': dist_metrics['assd']['assd'] * 1000,
                
                # F-scores
                **{f"fscore_{key}_f1": fmetrics['f1_score'] 
                   for key, fmetrics in dist_metrics['fscores'].items()},
                
                # Quality metrics
                'normal_median_deg': metrics['normal_consistency']['median_angle_deg'],
                'normal_p95_deg': metrics['normal_consistency']['p95_angle_deg'],
                'icp_fitness': metrics['registration_sanity']['fitness'],
                'icp_rmse_mm': metrics['registration_sanity']['rmse_mm'],
                'roughness_median_mm': metrics['noise_smoothness']['roughness']['median'] * 1000,
                'density_gini': metrics['coverage_sampling']['density']['gini_coefficient'],
                
                # Coverage
                'observed_area_pct': metrics['coverage_sampling']['observed_area']['observed_percentage'],
                'n_holes': metrics['coverage_sampling']['holes']['n_holes'],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.metrics_base / 'comprehensive_metrics_summary.csv'
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"✓ Comprehensive CSV saved: {csv_path}")
    
    def process_all(self):
        """Process all reconstruction methods in the run with comprehensive metrics."""
        self.logger.info(f"Starting comprehensive evaluation for {self.video_name}/{self.run_name}")
        self.logger.info(f"Ground truth: {self.ground_truth_path}")
        
        # Step 1: Preprocess ground truth (once for all methods)
        self._preprocess_ground_truth()
        
        # Find all methods
        methods = self.find_reconstruction_methods()
        
        if not methods:
            self.logger.error("No reconstruction methods found!")
            return
        
        self.logger.info(f"Found {len(methods)} reconstruction methods to process")
        
        # Process each method
        success_count = 0
        for method_info in methods:
            if self.process_method(method_info):
                success_count += 1
        
        # Export summaries
        self._export_csv_summary()
        self._export_comprehensive_csv()
        self._generate_comprehensive_report()
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"PROCESSING COMPLETE")
        self.logger.info(f"Successfully processed: {success_count}/{len(methods)}")
        self.logger.info(f"{'='*80}")
    
    def process_single(self, method_name: str):
        """
        Process a single reconstruction method by name with comprehensive metrics.
        
        Args:
            method_name: Name of the method to process
        """
        # Preprocess ground truth first
        self._preprocess_ground_truth()
        
        methods = self.find_reconstruction_methods()
        
        for method_info in methods:
            if method_info['name'] == method_name:
                if self.process_method(method_info):
                    # Export summaries for single method
                    self._export_csv_summary()
                    self._export_comprehensive_csv()
                    self._generate_comprehensive_report()
                return
        
        self.logger.error(f"Method not found: {method_name}")
        self.logger.info(f"Available methods: {[m['name'] for m in methods]}")


def main():
    parser = argparse.ArgumentParser(
        description='VAPOR Comprehensive 3D Reconstruction Evaluation with ICP Registration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMPREHENSIVE EVALUATION PIPELINE (11 Steps):
  Step 0: ICP Registration with initial transformation
  Step 1: CT mesh preprocessing and validation
  Step 2: ROI generation (proximity-based)
  Step 3: Distance metrics (P2S, S2P, Chamfer, Hausdorff, F-scores, ASSD)
  Step 4: Normal consistency analysis
  Step 5: Coverage, holes, and density uniformity
  Step 6: Noise and smoothness analysis
  Step 7: Registration sanity check (ICP validation)
  Step 8: Pairwise model comparison (if multiple methods)
  Step 9: Statistical analysis (bootstrap CI)
  Step 10: Visualizations (heatmaps, histograms)
  Step 11: Comprehensive reports (CSV + Markdown)

IMPORTANT: This script requires an initial transformation matrix!

Before running, create initial_transformation.txt in:
  data/point_clouds/{video}/{run}/initial_transformation.txt

The file should contain a 4x4 transformation matrix:
  -1.324484109879 -1.483755350113 0.880392253399 115.265403747559
  -1.550405263901 1.510512351990 0.213249132037 174.122238159180
  -0.756877481937 -0.497696936131 -1.977451205254 -81.323654174805
  0.000000000000 0.000000000000 0.000000000000 1.000000000000

OUTPUT:
  data/metrics/{video}/{run}/reconstruction_metrics/
    ├── registration_summary_{video}_{run}.csv       # ICP & basic metrics
    ├── comprehensive_metrics_summary.csv            # All 11-step metrics
    ├── comprehensive_report.md                      # Detailed report
    ├── GT_sanity_check/                            # CT validation
    └── {method}/                                    # Per-method detailed results
        ├── roi/                                     # ROI visualization
        ├── distance_metrics/                        # Distance arrays & tables
        ├── normal_consistency/                      # Normal angles
        ├── coverage_sampling/                       # Coverage analysis
        ├── noise_smoothness/                        # Roughness, shape descriptors
        ├── registration_sanity/                     # ICP validation
        ├── statistical_analysis/                    # Bootstrap CI
        └── visualizations/                          # Heatmaps, histograms

Examples:
  # Process all methods in a run (recommended)
  python registration_calculator.py --video 4_32_slow_run_20250403_094922 --run run_20251023_071408
  
  # Process only a specific method
  python registration_calculator.py --video 4_32_slow_run_20250403_094922 --run run_20251023_071408 --method-name motion_blur_high
  
  # Specify custom ground truth file
  python registration_calculator.py --video 4_32_slow_run_20250403_094922 --run run_20251023_071408 --ground-truth ct_scan
        """
    )
    
    parser.add_argument('--video', required=True, help='Video name (e.g., 4_32_slow_run_20250403_094922)')
    parser.add_argument('--run', required=True, help='Run name (e.g., run_20251023_071408)')
    parser.add_argument('--ground-truth', default='ground_truth',
                       help='Ground truth PLY filename (without .ply extension)')
    parser.add_argument('--method-name', help='Process only this specific method name')
    
    args = parser.parse_args()
    
    # Create calculator
    calculator = RegistrationCalculator(
        video_name=args.video,
        run_name=args.run,
        ground_truth_name=args.ground_truth
    )
    
    # Process
    if args.method_name:
        calculator.process_single(args.method_name)
    else:
        calculator.process_all()


if __name__ == "__main__":
    main()
