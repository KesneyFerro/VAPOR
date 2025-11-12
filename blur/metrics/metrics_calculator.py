"""
VAPOR Metrics Calculator
Calculates comprehensive quality metrics with organized directory structure: data/metrics/{video_name}/{original/blurred/deblurred}
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import argparse
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from blur.metrics.no_reference import NoReferenceMetrics
from blur.metrics.full_reference import FullReferenceMetrics
from blur.metrics.sharpness import SharpnessMetrics
from utils.core_utilities import get_image_files
from utils.data_manager import VAPORDataManager


class VAPORMetricsCalculator:
    """Updated metrics calculator with new directory structure."""
    
    def __init__(self, video_name: str, run_id: str = None, pipeline_mode: bool = False):
        """Initialize calculator for specific video.
        
        Args:
            video_name: Name of video file (e.g., 'pat3.mp4')
            run_id: Unique identifier for this pipeline run (e.g., timestamp or 'test_run')
            pipeline_mode: If True, use data manager for timestamped persistent storage.
                          If False, use test_run folder that can be overwritten.
        """
        self.video_name = video_name
        self.video_stem = Path(video_name).stem
        self.pipeline_mode = pipeline_mode
        self.run_id = run_id
        
        # Setup paths
        self.base_dir = Path(__file__).parent.parent.parent
        
        # Initialize data manager
        if pipeline_mode:
            self.data_manager = VAPORDataManager(
                video_name=video_name,
                mode="pipeline",
                run_id=run_id  # Pass run_id to coordinate with other modules
            )
        else:
            self.data_manager = VAPORDataManager(
                video_name=video_name,
                mode="standalone",
                module_name="metrics"
            )
        
        # Input paths
        self.frames_base = self.base_dir / "data" / "frames"
        self.frames_original = self.frames_base / "original" / self.video_stem
        self.frames_blurred = self.frames_base / "blurred" / self.video_stem
        self.frames_deblurred = self.frames_base / "deblurred" / self.video_stem
        
        # Output paths - use data manager's metrics_dir
        if pipeline_mode:
            # Use data manager paths for pipeline mode
            self.metrics_base = self.data_manager.metrics_dir
            self.metrics_original = self.metrics_base / "original"
            self.metrics_blurred = self.metrics_base / "blurred"
            self.metrics_deblurred = self.metrics_base / "deblurred"
        else:
            # Use old paths for standalone mode
            self.metrics_base = self.base_dir / "data" / "metrics" / self.video_stem
            self.metrics_original = self.metrics_base / "original"
            self.metrics_blurred = self.metrics_base / "blurred"
            self.metrics_deblurred = self.metrics_base / "deblurred"
            
        # Create output directories
        self._setup_directories()
        
        # Initialize metrics calculators
        self._initialize_calculators()
        
    def _setup_directories(self):
        """Create necessary output directories."""
        for dir_path in [self.metrics_base, self.metrics_original, self.metrics_blurred, self.metrics_deblurred]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _initialize_calculators(self):
        """Initialize metrics calculation engines."""
        try:
            self.sharpness_calculator = SharpnessMetrics()
            self.full_ref_calculator = FullReferenceMetrics()
            
            try:
                import signal
                
                # Set a timeout for no-reference calculator initialization
                def timeout_handler(signum, frame):
                    raise TimeoutError("No-reference calculator initialization timed out")
                
                # Try to initialize with timeout (Windows doesn't support SIGALRM, so skip timeout on Windows)
                import platform
                if platform.system() != 'Windows':
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(60)  # 60 second timeout
                
                self.no_ref_calculator = NoReferenceMetrics()
                
                if platform.system() != 'Windows':
                    signal.alarm(0)  # Cancel the alarm
                    
            except (TimeoutError, Exception) as e:
                self.logger.warning(f"No-reference calculator failed to initialize: {e}")
                self.logger.warning("Continuing without BRISQUE/NIQE metrics")
                self.no_ref_calculator = None
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics calculators: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_available_frame_sets(self):
        """Discover available frame sets for metrics calculation."""
        frame_sets = {}
        
        # Check original frames
        if self.frames_original.exists() and any(self.frames_original.iterdir()):
            frame_count = len(list(self.frames_original.glob("*.png")))
            frame_sets['original'] = {
                'path': self.frames_original,
                'count': frame_count,
                'type': 'original',
                'output_dir': self.metrics_original
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
                        'method': blur_dir.name,
                        'output_dir': self.metrics_blurred
                    }
                    
        # Check deblurred frames
        if self.frames_deblurred.exists():
            for deblur_dir in self.frames_deblurred.iterdir():
                if deblur_dir.is_dir():
                    # Check if frames are directly in this directory
                    direct_frames = list(deblur_dir.glob("*.png"))
                    if direct_frames:
                        frame_count = len(direct_frames)
                        frame_sets[f'deblurred_{deblur_dir.name}'] = {
                            'path': deblur_dir,
                            'count': frame_count,
                            'type': 'deblurred',
                            'method': deblur_dir.name,
                            'output_dir': self.metrics_deblurred
                        }
                    else:
                        # Check subdirectories (new structure: method/blur_type/)
                        for sub_dir in deblur_dir.iterdir():
                            if sub_dir.is_dir() and any(sub_dir.iterdir()):
                                frame_count = len(list(sub_dir.glob("*.png")))
                                if frame_count > 0:
                                    frame_sets[f'deblurred_{deblur_dir.name}_{sub_dir.name}'] = {
                                        'path': sub_dir,
                                        'count': frame_count,
                                        'type': 'deblurred',
                                        'method': f'{deblur_dir.name}_{sub_dir.name}',
                                        'output_dir': self.metrics_deblurred
                                    }
                    
        return frame_sets
    
    def calculate_metrics_for_frame_set(self, set_name: str, set_info: dict):
        """Calculate metrics for a specific frame set with optimized batch processing."""
        
        metrics_data = []
        
        # Get all frame files
        frame_files = get_image_files(set_info['path'])
        if not frame_files:
            return []
        
        
        # Pre-allocate arrays for batch processing
        batch_size = min(10, len(frame_files))  # Process in smaller batches
        processed_count = 0
        
        # Track processed frames to prevent duplicates
        processed_frames = set()
        
        
        for i in range(0, len(frame_files), batch_size):
            batch_files = frame_files[i:i + batch_size]
            
            for frame_file in batch_files:
                # Skip if frame already processed (safeguard against duplicates)
                if frame_file.name in processed_frames:
                    continue
                
                try:
                    processed_frames.add(frame_file.name)
                    # Load frame
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        continue
                    
                    # Pre-check for problematic images (too small, constant, etc.)
                    if frame.size < 100:  # Skip very small images
                        continue
                    
                    # Check for constant images that cause issues
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    if np.std(gray_frame) < 1e-8:
                        # Still add a row with default values for constant images
                        metrics_row = {
                            'frame': frame_file.name,
                            'frame_set': set_name,
                            'frame_type': set_info['type'],
                            'brisque': 100.0,  # Poor quality for constant image
                            'niqe': 100.0,
                            'acutance': 0.0,
                            'mean_gradient': 0.0,
                            'fft_high_freq': 0.0,
                            'laplacian_variance': 0.0
                        }
                        metrics_data.append(metrics_row)
                        continue
                    
                    # Calculate no-reference and sharpness metrics with error handling
                    try:
                        no_ref_metrics = self.no_ref_calculator.calculate_all(frame)
                    except Exception as e:
                        no_ref_metrics = {'brisque': 100.0, 'niqe': 100.0}
                    
                    try:
                        sharpness_metrics = self.sharpness_calculator.calculate_all(frame)
                    except Exception as e:
                        sharpness_metrics = {'acutance': 0.0, 'mean_gradient': 0.0, 'fft_high_freq': 0.0, 'laplacian_variance': 0.0}
                    
                    # Prepare metrics row
                    metrics_row = {
                        'frame': frame_file.name,
                        'frame_set': set_name,
                        'frame_type': set_info['type'],
                        **no_ref_metrics,
                        **sharpness_metrics
                    }
                
                    # Add method-specific information with consistent column naming
                    if 'method' in set_info:
                        if set_info['type'] == 'blurred':
                            parts = set_info['method'].split('_')
                            if len(parts) >= 2:
                                metrics_row['method'] = set_info['method']  # Keep original method name
                                metrics_row['blur_type'] = '_'.join(parts[:-1])
                                metrics_row['intensity'] = parts[-1]  # Use 'intensity' not 'blur_intensity' for consistency
                            else:
                                metrics_row['method'] = set_info['method']
                                metrics_row['blur_type'] = set_info['method']
                                metrics_row['intensity'] = 'unknown'
                        elif set_info['type'] == 'deblurred':
                            metrics_row['method'] = set_info['method']  # Use consistent 'method' column
                            metrics_row['deblur_method'] = set_info['method']
                        elif set_info['type'] == 'original':
                            metrics_row['method'] = 'original'
                            metrics_row['blur_type'] = 'none'
                            metrics_row['intensity'] = 'none'
                    
                    # Calculate full-reference metrics if original frame exists
                    if set_info['type'] != 'original':
                        original_frame_path = self.frames_original / frame_file.name
                        if original_frame_path.exists():
                            try:
                                original_frame = cv2.imread(str(original_frame_path))
                                if original_frame is not None:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        full_ref_metrics = self.full_ref_calculator.calculate_all(frame, original_frame)
                                        metrics_row.update(full_ref_metrics)
                            except Exception as e:
                                self.logger.error(f"Error calculating full-reference metrics for {frame_file.name}: {e}")
                    
                    metrics_data.append(metrics_row)
                    
                    # Progress update - show every 5 frames for more frequent feedback
                    processed_count += 1
                    if processed_count % 5 == 0 or processed_count == len(frame_files):
                        progress_pct = (processed_count / len(frame_files)) * 100
                        
                except Exception as e:
                    processed_count += 1  # Still count failed frames in progress
        
        return metrics_data
    
    def save_metrics(self, all_metrics_data: list):
        """Save metrics data using data manager in pipeline mode or CSV in standalone mode."""
        if not all_metrics_data:
            return
        
        
        # Create comprehensive DataFrame
        df = pd.DataFrame(all_metrics_data)
        
        # Remove duplicates based on frame, method, blur_type, and intensity
        duplicate_columns = ['frame']
        if 'method' in df.columns:
            duplicate_columns.append('method')
        if 'blur_type' in df.columns:
            duplicate_columns.append('blur_type')
        if 'intensity' in df.columns:
            duplicate_columns.append('intensity')
        if 'deblur_method' in df.columns:
            duplicate_columns.append('deblur_method')
        if 'frame_type' in df.columns:
            duplicate_columns.append('frame_type')
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=duplicate_columns, keep='first')
        final_count = len(df)
        
        if initial_count != final_count:
            duplicates_removed = initial_count - final_count
        
        # ALWAYS save detailed per-frame CSV data (for detailed analysis)
        self._save_detailed_csv(df)
        
        if self.pipeline_mode and self.data_manager:
            # ALSO save aggregated summaries to data manager (for organized pipeline storage)
            self._save_aggregated_summaries(df)
        else:
            # In standalone mode, also save summary CSVs
            self._save_summary_csv(df)
    
    def _aggregate_sharpness_metrics(self, df: pd.DataFrame) -> Dict:
        """Aggregate sharpness metrics from per-frame data.
        
        Returns mean, std, min, max for each sharpness metric.
        """
        sharpness_cols = ['acutance', 'mean_gradient', 'fft_high_freq', 'laplacian_variance']
        metrics = {}
        
        for col in sharpness_cols:
            if col in df.columns:
                metrics[f'{col}_mean'] = float(df[col].mean())
                metrics[f'{col}_std'] = float(df[col].std())
                metrics[f'{col}_min'] = float(df[col].min())
                metrics[f'{col}_max'] = float(df[col].max())
        
        return metrics
    
    def _aggregate_no_reference_metrics(self, df: pd.DataFrame) -> Dict:
        """Aggregate no-reference quality metrics from per-frame data.
        
        Returns mean, std, min, max for BRISQUE and NIQE scores.
        """
        no_ref_cols = ['brisque', 'niqe']
        metrics = {}
        
        for col in no_ref_cols:
            if col in df.columns:
                metrics[f'{col}_mean'] = float(df[col].mean())
                metrics[f'{col}_std'] = float(df[col].std())
                metrics[f'{col}_min'] = float(df[col].min())
                metrics[f'{col}_max'] = float(df[col].max())
        
        return metrics
    
    def _aggregate_full_reference_metrics(self, df: pd.DataFrame) -> Dict:
        """Aggregate full-reference quality metrics from per-frame data.
        
        Returns mean, std, min, max for PSNR, SSIM, MSE, MAE.
        """
        full_ref_cols = ['psnr', 'ssim', 'mse', 'mae']
        metrics = {}
        
        for col in full_ref_cols:
            if col in df.columns:
                metrics[f'{col}_mean'] = float(df[col].mean())
                metrics[f'{col}_std'] = float(df[col].std())
                metrics[f'{col}_min'] = float(df[col].min())
                metrics[f'{col}_max'] = float(df[col].max())
        
        return metrics if metrics else None
    
    def _save_with_data_manager(self, df: pd.DataFrame):
        """Save metrics using the data manager (pipeline mode).
        
        This method aggregates per-frame metrics into summary statistics per method,
        matching the data manager's API which expects aggregated metrics, not per-frame data.
        """
        
        # Group by frame type and method
        frame_types = df['frame_type'].unique()
        
        for frame_type in frame_types:
            type_df = df[df['frame_type'] == frame_type]
            
            if frame_type == 'original':
                # Original frames - aggregate and save sharpness metrics
                try:
                    sharpness_metrics = self._aggregate_sharpness_metrics(type_df)
                    processing_info = {
                        'frame_count': len(type_df),
                        'frame_names': type_df['frame'].tolist()
                    }
                    
                    self.data_manager.save_sharpness_metrics(
                        frame_type='original',
                        method_name=None,
                        metrics=sharpness_metrics,
                        processing_info=processing_info
                    )
                except Exception as e:
                    self.logger.error(f"Error saving original metrics: {e}")
                
            elif frame_type == 'blurred':
                # Blurred frames - group by method and save aggregated metrics
                methods = type_df['method'].unique() if 'method' in type_df.columns else [None]
                for method in methods:
                    method_df = type_df[type_df['method'] == method] if method else type_df
                    
                    try:
                        # Aggregate and save no-reference metrics
                        no_ref_metrics = self._aggregate_no_reference_metrics(method_df)
                        processing_info = {
                            'frame_count': len(method_df),
                            'blur_type': method_df.iloc[0].get('blur_type') if 'blur_type' in method_df.columns else None,
                            'intensity': method_df.iloc[0].get('intensity') if 'intensity' in method_df.columns else None
                        }
                        
                        self.data_manager.save_no_reference_metrics(
                            frame_type='blurred',
                            method_name=method,
                            metrics=no_ref_metrics,
                            processing_info=processing_info
                        )
                        
                        # Aggregate and save sharpness metrics
                        sharpness_metrics = self._aggregate_sharpness_metrics(method_df)
                        self.data_manager.save_sharpness_metrics(
                            frame_type='blurred',
                            method_name=method,
                            metrics=sharpness_metrics,
                            processing_info=processing_info
                        )
                        
                        # Aggregate and save full-reference metrics if available
                        if any(col in method_df.columns for col in ['psnr', 'ssim', 'mse', 'mae']):
                            full_ref_metrics = self._aggregate_full_reference_metrics(method_df)
                            if full_ref_metrics:
                                self.data_manager.save_full_reference_metrics(
                                    reference_type='original',
                                    degraded_type='blurred',
                                    degraded_method=method,
                                    metrics=full_ref_metrics,
                                    processing_info=processing_info
                                )
                        
                    except Exception as e:
                        self.logger.error(f"Error saving blurred metrics: {e}")
                    
            elif frame_type == 'deblurred':
                # Deblurred frames - group by method and save aggregated metrics
                methods = type_df['method'].unique() if 'method' in type_df.columns else [None]
                for method in methods:
                    method_df = type_df[type_df['method'] == method] if method else type_df
                    
                    try:
                        # Aggregate and save no-reference metrics
                        no_ref_metrics = self._aggregate_no_reference_metrics(method_df)
                        processing_info = {
                            'frame_count': len(method_df),
                            'deblur_method': method
                        }
                        
                        self.data_manager.save_no_reference_metrics(
                            frame_type='deblurred',
                            method_name=method,
                            metrics=no_ref_metrics,
                            processing_info=processing_info
                        )
                        
                        # Aggregate and save sharpness metrics
                        sharpness_metrics = self._aggregate_sharpness_metrics(method_df)
                        self.data_manager.save_sharpness_metrics(
                            frame_type='deblurred',
                            method_name=method,
                            metrics=sharpness_metrics,
                            processing_info=processing_info
                        )
                        
                        # Aggregate and save full-reference metrics if available
                        if any(col in method_df.columns for col in ['psnr', 'ssim', 'mse', 'mae']):
                            full_ref_metrics = self._aggregate_full_reference_metrics(method_df)
                            if full_ref_metrics:
                                self.data_manager.save_full_reference_metrics(
                                    reference_type='original',
                                    degraded_type='deblurred',
                                    degraded_method=method,
                                    metrics=full_ref_metrics,
                                    processing_info=processing_info
                                )
                        
                    except Exception as e:
                        self.logger.error(f"Error saving deblurred metrics: {e}")
    
    def _save_detailed_csv(self, df: pd.DataFrame):
        """Save detailed per-frame metrics to CSV files.
        
        This preserves ALL per-frame data for detailed analysis.
        Files are organized by frame type in the metrics directory.
        Redundant columns (method, blur_type, intensity, deblur_method) are removed
        as this information is already stored in the JSON files.
        """
        
        # Remove redundant columns that are already in JSON
        columns_to_remove = ['method', 'blur_type', 'intensity', 'deblur_method']
        df_clean = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
        
        # Save detailed metrics by type
        frame_types = df_clean['frame_type'].unique()
        
        for frame_type in frame_types:
            type_df = df_clean[df_clean['frame_type'] == frame_type]
            
            if frame_type == 'original':
                output_dir = self.metrics_original
            elif frame_type == 'blurred':
                output_dir = self.metrics_blurred
            elif frame_type == 'deblurred':
                output_dir = self.metrics_deblurred
            else:
                output_dir = self.metrics_base
            
            # Save detailed per-frame metrics for this type
            detailed_path = output_dir / f"{self.video_stem}_{frame_type}_detailed_per_frame.csv"
            type_df.to_csv(detailed_path, index=False)
        
        # Save overall comprehensive per-frame metrics
        overall_detailed = self.metrics_base / f"{self.video_stem}_all_frames_detailed.csv"
        df_clean.to_csv(overall_detailed, index=False)
    
    def _save_summary_csv(self, df: pd.DataFrame):
        """Save summary statistics to CSV files (standalone mode).
        
        Generates aggregated statistics from per-frame data.
        """
        
        # Generate summary statistics by frame type and method
        frame_types = df['frame_type'].unique()
        
        for frame_type in frame_types:
            type_df = df[df['frame_type'] == frame_type]
            
            if frame_type == 'original':
                output_dir = self.metrics_original
            elif frame_type == 'blurred':
                output_dir = self.metrics_blurred
            elif frame_type == 'deblurred':
                output_dir = self.metrics_deblurred
            else:
                output_dir = self.metrics_base
            
            # Save summary statistics for this type
            if len(type_df) > 1:
                numeric_columns = type_df.select_dtypes(include=[np.number]).columns
                summary_df = type_df.groupby(['frame_set'])[numeric_columns].agg(['mean', 'std', 'min', 'max']).round(4)
                
                summary_path = output_dir / f"{self.video_stem}_{frame_type}_summary_stats.csv"
                summary_df.to_csv(summary_path)
        
        # Create overall summary comparison
        if len(df) > 1:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            overall_summary = df.groupby(['frame_type', 'frame_set'])[numeric_columns].agg(['mean', 'std', 'min', 'max']).round(4)
            
            # Flatten multi-level column headers for CSV compatibility
            overall_summary.columns = ['_'.join(col).strip() for col in overall_summary.columns.values]
            
            overall_summary_path = self.metrics_base / f"{self.video_stem}_overall_summary_stats.csv"
            overall_summary.to_csv(overall_summary_path)
    
    def _save_aggregated_summaries(self, df: pd.DataFrame):
        """Save aggregated metric summaries to data manager (pipeline mode).
        
        This saves summary statistics to organized JSON files via data manager,
        while detailed per-frame CSVs are saved separately via _save_detailed_csv().
        """
        
        # Use the existing _save_with_data_manager logic
        self._save_with_data_manager(df)
    
    def print_summary(self, all_metrics_data: list):
        """Print a summary of calculated metrics."""
        if not all_metrics_data:
            return
            
        df = pd.DataFrame(all_metrics_data)
        
        print("METRICS CALCULATION SUMMARY")
        print("="*60)
        
        # Summary by frame type
        type_summary = df.groupby('frame_type').agg({
            'frame': 'count',
            'laplacian_variance': 'mean'
        }).round(4)
        
        
        # Quality comparison
        if 'laplacian_variance' in df.columns:
            sharpness_by_set = df.groupby('frame_set')['laplacian_variance'].mean().sort_values(ascending=False)
            for frame_set, sharpness in sharpness_by_set.head(10).items():
                pass  # Statistical analysis, not printed
        
        if 'brisque_score' in df.columns:
            quality_by_set = df.groupby('frame_set')['brisque_score'].mean().sort_values()
            for frame_set, quality in quality_by_set.head(10).items():
                pass  # Statistical analysis, not printed
    
    def run_complete_metrics_calculation(self):
        """Run metrics calculation on all available frame sets."""
        print("=" * 50, flush=True)
        
        # Discover available frame sets
        frame_sets = self.get_available_frame_sets()
        
        if not frame_sets:
            return False
        
        for name, info in frame_sets.items():
            pass  # Frame sets enumeration, not printed
        
        # Calculate metrics for each frame set
        all_metrics_data = []
        total_sets = len(frame_sets)
        
        for idx, (set_name, set_info) in enumerate(frame_sets.items(), 1):
            metrics_data = self.calculate_metrics_for_frame_set(set_name, set_info)
            all_metrics_data.extend(metrics_data)
            print(f"[Frame Set {idx}/{total_sets}] Completed {set_name} - {len(metrics_data)} metrics collected")
        
        # Save results
        self.save_metrics(all_metrics_data)
        
        # Print summary
        self.print_summary(all_metrics_data)
        
        print("METRICS CALCULATION COMPLETED")
        print("="*50)
        
        return len(all_metrics_data) > 0


def main():
    """Main function for metrics calculation."""
    # Print immediately to show the script has started
    print("Starting VAPOR Metrics Calculator...")
    
    parser = argparse.ArgumentParser(
        description="VAPOR Metrics Calculator with Updated Directory Structure"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Video filename (e.g., pat3.mp4)"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique run identifier (timestamp or 'test_run'). Required for pipeline mode."
    )
    parser.add_argument(
        "--pipeline-mode",
        action="store_true",
        help="Use data manager for organized timestamped storage (vs legacy CSV)"
    )
    
    args = parser.parse_args()
    
    
    # Initialize and run metrics calculation
    calculator = VAPORMetricsCalculator(
        video_name=args.video,
        run_id=args.run_id,
        pipeline_mode=args.pipeline_mode
    )
    success = calculator.run_complete_metrics_calculation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())