"""
Updated VAPOR Metrics Calculator
Calculates metrics with the new directory structure: data/metrics/{video_name}/{original/blurred/deblurred}
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import argparse
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from blur.metrics.no_reference import NoReferenceMetrics
from blur.metrics.full_reference import FullReferenceMetrics
from blur.metrics.sharpness import SharpnessMetrics
from utils.core_utilities import get_image_files


class VAPORMetricsCalculator:
    """Updated metrics calculator with new directory structure."""
    
    def __init__(self, video_name: str):
        """Initialize calculator for specific video.
        
        Args:
            video_name: Name of video file (e.g., 'pat3.mp4')
        """
        self.video_name = video_name
        self.video_stem = Path(video_name).stem
        
        # Setup paths
        self.base_dir = Path(__file__).parent.parent
        
        # Input paths
        self.frames_base = self.base_dir / "data" / "frames"
        self.frames_original = self.frames_base / "original" / self.video_stem
        self.frames_blurred = self.frames_base / "blurred" / self.video_stem
        self.frames_deblurred = self.frames_base / "deblurred" / self.video_stem
        
        # Output paths - new structure
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
            self.no_ref_calculator = NoReferenceMetrics()
            self.full_ref_calculator = FullReferenceMetrics()
            self.sharpness_calculator = SharpnessMetrics()
            print("  ✓ Metrics calculators initialized")
        except Exception as e:
            print(f"ERROR: Failed to initialize metrics calculators: {e}")
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
                if deblur_dir.is_dir() and any(deblur_dir.iterdir()):
                    frame_count = len(list(deblur_dir.glob("*.png")))
                    frame_sets[f'deblurred_{deblur_dir.name}'] = {
                        'path': deblur_dir,
                        'count': frame_count,
                        'type': 'deblurred',
                        'method': deblur_dir.name,
                        'output_dir': self.metrics_deblurred
                    }
                    
        return frame_sets
    
    def calculate_metrics_for_frame_set(self, set_name: str, set_info: dict):
        """Calculate metrics for a specific frame set."""
        print(f"\n--- Processing {set_name} ---")
        print(f"  Frames: {set_info['count']} images")
        print(f"  Path: {set_info['path']}")
        
        metrics_data = []
        
        # Get all frame files
        frame_files = get_image_files(set_info['path'])
        if not frame_files:
            print(f"  WARNING: No frame files found in {set_info['path']}")
            return []
        
        print(f"  Processing {len(frame_files)} frames...")
        
        for i, frame_file in enumerate(frame_files, 1):
            try:
                # Load frame
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    print(f"    WARNING: Could not load {frame_file.name}")
                    continue
                
                # Calculate no-reference and sharpness metrics
                no_ref_metrics = self.no_ref_calculator.calculate_all(frame)
                sharpness_metrics = self.sharpness_calculator.calculate_all(frame)
                
                # Prepare metrics row
                metrics_row = {
                    'frame': frame_file.name,
                    'frame_set': set_name,
                    'frame_type': set_info['type'],
                    **no_ref_metrics,
                    **sharpness_metrics
                }
                
                # Add method-specific information
                if 'method' in set_info:
                    if set_info['type'] == 'blurred':
                        parts = set_info['method'].split('_')
                        if len(parts) >= 2:
                            metrics_row['blur_type'] = '_'.join(parts[:-1])
                            metrics_row['blur_intensity'] = parts[-1]
                        else:
                            metrics_row['blur_type'] = set_info['method']
                            metrics_row['blur_intensity'] = 'unknown'
                    elif set_info['type'] == 'deblurred':
                        metrics_row['deblur_method'] = set_info['method']
                
                # Calculate full-reference metrics if original frame exists
                if set_info['type'] != 'original':
                    original_frame_path = self.frames_original / frame_file.name
                    if original_frame_path.exists():
                        try:
                            original_frame = cv2.imread(str(original_frame_path))
                            if original_frame is not None:
                                full_ref_metrics = self.full_ref_calculator.calculate_all(frame, original_frame)
                                metrics_row.update(full_ref_metrics)
                        except Exception as e:
                            print(f"    WARNING: Could not calculate full-reference metrics for {frame_file.name}: {e}")
                
                metrics_data.append(metrics_row)
                
                # Progress update
                if i % 10 == 0 or i == len(frame_files):
                    print(f"    Progress: {i}/{len(frame_files)} frames processed")
                    
            except Exception as e:
                print(f"    ERROR processing {frame_file.name}: {e}")
        
        print(f"  ✓ Completed processing {len(metrics_data)} frames")
        return metrics_data
    
    def save_metrics(self, all_metrics_data: list):
        """Save metrics data in the new directory structure."""
        if not all_metrics_data:
            print("WARNING: No metrics data to save")
            return
        
        print(f"\nSaving metrics to {self.metrics_base}...")
        
        # Create comprehensive DataFrame
        df = pd.DataFrame(all_metrics_data)
        
        # Save detailed metrics by type
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
            
            # Save detailed metrics for this type
            detailed_path = output_dir / f"{self.video_stem}_{frame_type}_detailed_metrics.csv"
            type_df.to_csv(detailed_path, index=False)
            print(f"  ✓ Saved {frame_type} detailed metrics: {detailed_path}")
            
            # Save summary statistics for this type
            if len(type_df) > 1:
                numeric_columns = type_df.select_dtypes(include=[np.number]).columns
                summary_df = type_df.groupby(['frame_set'])[numeric_columns].agg(['mean', 'std']).round(4)
                
                summary_path = output_dir / f"{self.video_stem}_{frame_type}_summary_metrics.csv"
                summary_df.to_csv(summary_path)
                print(f"  ✓ Saved {frame_type} summary metrics: {summary_path}")
        
        # Save overall comprehensive metrics
        overall_detailed = self.metrics_base / f"{self.video_stem}_all_detailed_metrics.csv"
        df.to_csv(overall_detailed, index=False)
        print(f"  ✓ Saved comprehensive detailed metrics: {overall_detailed}")
        
        # Create overall summary comparison
        if len(df) > 1:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            overall_summary = df.groupby(['frame_type', 'frame_set'])[numeric_columns].agg(['mean', 'std']).round(4)
            
            overall_summary_path = self.metrics_base / f"{self.video_stem}_overall_summary_metrics.csv"
            overall_summary.to_csv(overall_summary_path)
            print(f"  ✓ Saved overall summary metrics: {overall_summary_path}")
    
    def print_summary(self, all_metrics_data: list):
        """Print a summary of calculated metrics."""
        if not all_metrics_data:
            return
            
        df = pd.DataFrame(all_metrics_data)
        
        print("\n" + "="*60)
        print("METRICS CALCULATION SUMMARY")
        print("="*60)
        
        # Summary by frame type
        type_summary = df.groupby('frame_type').agg({
            'frame': 'count',
            'laplacian_variance': 'mean'
        }).round(4)
        
        print("\nFrames processed by type:")
        print(type_summary)
        
        # Quality comparison
        if 'laplacian_variance' in df.columns:
            print("\nSharpness Comparison (Laplacian Variance - higher is sharper):")
            sharpness_by_set = df.groupby('frame_set')['laplacian_variance'].mean().sort_values(ascending=False)
            for frame_set, sharpness in sharpness_by_set.head(10).items():
                print(f"  {frame_set:25}: {sharpness:.4f}")
        
        if 'brisque_score' in df.columns:
            print("\nQuality Comparison (BRISQUE Score - lower is better):")
            quality_by_set = df.groupby('frame_set')['brisque_score'].mean().sort_values()
            for frame_set, quality in quality_by_set.head(10).items():
                print(f"  {frame_set:25}: {quality:.4f}")
    
    def run_complete_metrics_calculation(self):
        """Run metrics calculation on all available frame sets."""
        print("VAPOR - Updated Metrics Calculator")
        print("=" * 50)
        print(f"Video: {self.video_name}")
        print(f"Output directory: {self.metrics_base}")
        
        # Discover available frame sets
        frame_sets = self.get_available_frame_sets()
        
        if not frame_sets:
            print("ERROR: No frame sets found for metrics calculation!")
            return False
        
        print(f"\nFound {len(frame_sets)} frame sets:")
        for name, info in frame_sets.items():
            print(f"  - {name}: {info['count']} frames ({info['type']})")
        
        # Calculate metrics for each frame set
        all_metrics_data = []
        
        for set_name, set_info in frame_sets.items():
            metrics_data = self.calculate_metrics_for_frame_set(set_name, set_info)
            all_metrics_data.extend(metrics_data)
        
        # Save results
        self.save_metrics(all_metrics_data)
        
        # Print summary
        self.print_summary(all_metrics_data)
        
        print("\n" + "="*50)
        print("METRICS CALCULATION COMPLETED")
        print("="*50)
        
        return len(all_metrics_data) > 0


def main():
    """Main function for metrics calculation."""
    parser = argparse.ArgumentParser(
        description="VAPOR Metrics Calculator with Updated Directory Structure"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Video filename (e.g., pat3.mp4)"
    )
    
    args = parser.parse_args()
    
    # Initialize and run metrics calculation
    calculator = VAPORMetricsCalculator(video_name=args.video)
    success = calculator.run_complete_metrics_calculation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())