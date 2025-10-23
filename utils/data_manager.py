"""
VAPOR Data Management Utility
Provides unified data saving/loading interface for all VAPOR modules.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import os


class VAPORDataManager:
    """Unified data management for VAPOR pipeline."""
    
    def __init__(self, video_name: str, mode: str = "pipeline", module_name: Optional[str] = None, run_id: Optional[str] = None):
        """Initialize data manager.
        
        Args:
            video_name: Name of video file (e.g., 'pat3.mp4')
            mode: 'pipeline' for full pipeline runs, 'standalone' for module testing
            module_name: Name of module for standalone mode ('blur', 'reconstruction', 'deblur')
            run_id: Specific run ID to use (optional, will be generated if None)
        """
        self.video_name = video_name
        self.video_stem = Path(video_name).stem
        self.mode = mode
        self.module_name = module_name
        self.timestamp = run_id if run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.base_dir = Path(__file__).parent.parent
        self._setup_paths()
        
    def _setup_paths(self):
        """Setup output paths based on mode."""
        if self.mode == "standalone":
            if not self.module_name:
                raise ValueError("module_name required for standalone mode")
            
            # Ephemeral test outputs - single folder that gets overwritten
            self.output_base = self.base_dir / self.module_name / "outputs" / "test_run"
            
            # Clear previous test run
            if self.output_base.exists():
                shutil.rmtree(self.output_base)
            
            self.output_base.mkdir(parents=True, exist_ok=True)
            self.metrics_dir = self.output_base / "metrics"
            self.point_clouds_dir = self.output_base / "point_clouds"
            self.frames_dir = self.output_base / "frames"
            
        elif self.mode == "pipeline":
            # Persistent pipeline outputs - timestamped runs
            self.run_id = f"{self.video_stem}_{self.timestamp}"
            self.run_dir = self.base_dir / "data" / "metrics" / self.video_stem / f"run_{self.timestamp}"
            
            self.metrics_dir = self.run_dir
            self.point_clouds_dir = self.base_dir / "data" / "point_clouds" / self.video_stem / f"run_{self.timestamp}"
            self.frames_dir = self.base_dir / "data" / "frames"
            self.logs_dir = self.base_dir / "data" / "logs"
            self.reports_dir = self.base_dir / "data" / "reports" / self.video_stem
            
            # Create all directories
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.point_clouds_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Create symlink to latest run (Windows requires admin privileges, so skip if fails)
            latest_link = self.base_dir / "data" / "metrics" / self.video_stem / "latest"
            try:
                if latest_link.exists() or latest_link.is_symlink():
                    latest_link.unlink()
                latest_link.symlink_to(f"run_{self.timestamp}", target_is_directory=True)
            except OSError as e:
                # Symlink creation failed (likely Windows permissions), skip it
                pass
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Create metric subdirectories
        self.sharpness_dir = self.metrics_dir / "sharpness_metrics"
        self.no_ref_dir = self.metrics_dir / "no_reference_metrics"
        self.full_ref_dir = self.metrics_dir / "full_reference_metrics"
        self.recon_dir = self.metrics_dir / "reconstruction_metrics"
        
        for d in [self.sharpness_dir, self.no_ref_dir, self.full_ref_dir, self.recon_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_manifest(self, config: Dict, processing_stages: Dict, frame_counts: Dict):
        """Save run manifest with metadata."""
        manifest = {
            "run_id": self.run_id if self.mode == "pipeline" else "test_run",
            "video_name": self.video_name,
            "video_stem": self.video_stem,
            "timestamp": datetime.now().isoformat(),
            "pipeline_mode": config.get('pipeline', {}).get('mode', 'test'),
            "configuration": config,
            "processing_stages": processing_stages,
            "frame_counts": frame_counts,
            "output_files": self._collect_output_files()
        }
        
        manifest_path = self.metrics_dir / "manifest.json"
        self._save_json(manifest_path, manifest)
        return manifest_path
    
    def save_sharpness_metrics(self, frame_type: str, method_name: Optional[str], 
                                metrics: Dict, processing_info: Dict):
        """Save sharpness metrics."""
        filename = f"{method_name}.json" if method_name else f"{frame_type}.json"
        
        data = {
            "metric_type": "sharpness",
            "frame_type": frame_type,
            "method_name": method_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "processing_info": processing_info
        }
        
        filepath = self.sharpness_dir / filename
        self._save_json(filepath, data)
        return filepath
    
    def save_no_reference_metrics(self, frame_type: str, method_name: Optional[str],
                                   metrics: Dict, processing_info: Dict):
        """Save no-reference quality metrics."""
        filename = f"{method_name}.json" if method_name else f"{frame_type}.json"
        
        data = {
            "metric_type": "no_reference",
            "frame_type": frame_type,
            "method_name": method_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "processing_info": processing_info
        }
        
        filepath = self.no_ref_dir / filename
        self._save_json(filepath, data)
        return filepath
    
    def save_full_reference_metrics(self, reference_type: str, degraded_type: str,
                                     degraded_method: str, metrics: Dict, processing_info: Dict):
        """Save full-reference quality metrics."""
        comparison_name = f"{degraded_method}_vs_{reference_type}"
        filename = f"{comparison_name}.json"
        
        data = {
            "metric_type": "full_reference",
            "comparison": comparison_name,
            "reference_type": reference_type,
            "degraded_type": degraded_type,
            "degraded_method": degraded_method,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "processing_info": processing_info
        }
        
        filepath = self.full_ref_dir / filename
        self._save_json(filepath, data)
        return filepath
    
    def save_reconstruction_metrics(self, frame_type: str, method_name: Optional[str],
                                     reconstruction_settings: Dict, basic_metrics: Dict,
                                     per_image_metrics: Optional[Dict] = None,
                                     matching_metrics: Optional[Dict] = None,
                                     point_cloud_metrics: Optional[Dict] = None,
                                     bundle_adjustment_stats: Optional[Dict] = None,
                                     processing_info: Optional[Dict] = None,
                                     file_references: Optional[Dict] = None):
        """Save comprehensive reconstruction metrics."""
        filename = f"{method_name}.json" if method_name else f"{frame_type}.json"
        
        data = {
            "metric_type": "reconstruction",
            "frame_type": frame_type,
            "method_name": method_name,
            "timestamp": datetime.now().isoformat(),
            "reconstruction_settings": reconstruction_settings,
            "basic_metrics": basic_metrics,
        }
        
        # Add optional enhanced metrics
        if per_image_metrics:
            data["per_image_metrics"] = per_image_metrics
        if matching_metrics:
            data["matching_metrics"] = matching_metrics
        if point_cloud_metrics:
            data["point_cloud_metrics"] = point_cloud_metrics
        if bundle_adjustment_stats:
            data["bundle_adjustment_stats"] = bundle_adjustment_stats
        if processing_info:
            data["processing_info"] = processing_info
        if file_references:
            data["file_references"] = file_references
        
        filepath = self.recon_dir / filename
        self._save_json(filepath, data)
        return filepath
    
    def save_point_cloud_metadata(self, frame_type: str, method_name: Optional[str],
                                   ply_filename: str, properties: Dict, bounding_box: Dict,
                                   quality_summary: Dict, source_paths: Dict):
        """Save point cloud metadata."""
        method_dir = self.point_clouds_dir / (method_name if method_name else frame_type)
        method_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            "point_cloud_file": ply_filename,
            "frame_type": frame_type,
            "method_name": method_name,
            "timestamp": datetime.now().isoformat(),
            "format": "PLY",
            "properties": properties,
            "bounding_box": bounding_box,
            "quality_summary": quality_summary,
            "source_metrics": source_paths.get('metrics'),
            "source_frames": source_paths.get('frames')
        }
        
        filepath = method_dir / "metadata.json"
        self._save_json(filepath, data)
        return filepath
    
    def load_metrics(self, metric_type: str, frame_type: str = None, 
                     method_name: str = None) -> Dict[str, Any]:
        """Load metrics by type."""
        metric_dirs = {
            'sharpness': self.sharpness_dir,
            'no_reference': self.no_ref_dir,
            'full_reference': self.full_ref_dir,
            'reconstruction': self.recon_dir
        }
        
        metric_dir = metric_dirs.get(metric_type)
        if not metric_dir:
            raise ValueError(f"Invalid metric_type: {metric_type}")
        
        if method_name:
            filepath = metric_dir / f"{method_name}.json"
        elif frame_type:
            filepath = metric_dir / f"{frame_type}.json"
        else:
            # Load all metrics of this type
            all_metrics = {}
            for json_file in metric_dir.glob("*.json"):
                all_metrics[json_file.stem] = self._load_json(json_file)
            return all_metrics
        
        return self._load_json(filepath)
    
    def get_comparison_data(self, metric_type: str = 'reconstruction') -> Dict:
        """Get data for cross-method comparison."""
        metric_dir = self.metrics_dir / f"{metric_type}_metrics"
        
        comparison_data = {}
        for json_file in metric_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                key = data.get('method_name') or data.get('frame_type')
                comparison_data[key] = data
        
        return comparison_data
    
    def _collect_output_files(self) -> Dict[str, List[str]]:
        """Collect paths of all output files."""
        output_files = {
            'sharpness_metrics': [],
            'no_reference_metrics': [],
            'full_reference_metrics': [],
            'reconstruction_metrics': [],
            'point_clouds': []
        }
        
        # Collect metric files
        for metric_type in ['sharpness', 'no_reference', 'full_reference', 'reconstruction']:
            metric_dir = self.metrics_dir / f"{metric_type}_metrics"
            if metric_dir.exists():
                output_files[f'{metric_type}_metrics'] = [
                    str(f.relative_to(self.metrics_dir)) 
                    for f in metric_dir.glob("*.json")
                ]
        
        # Collect point cloud files
        if self.point_clouds_dir.exists():
            output_files['point_clouds'] = [
                str(f.relative_to(self.point_clouds_dir))
                for f in self.point_clouds_dir.rglob("*.ply")
            ]
        
        return output_files
    
    def _save_json(self, filepath: Path, data: Dict, indent: int = 2):
        """Save JSON with pretty formatting."""
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
    
    def _load_json(self, filepath: Path) -> Dict:
        """Load JSON file."""
        with open(filepath) as f:
            return json.load(f)
    
    def get_log_file_path(self, log_type: str = "pipeline") -> Path:
        """Get log file path based on mode."""
        if self.mode == "standalone":
            return self.output_base / f"{self.module_name}_test_run.log"
        else:
            return self.logs_dir / f"vapor_{log_type}_{self.video_stem}_{self.timestamp}.log"


# Convenience functions for quick access
def get_data_manager(video_name: str, mode: str = "pipeline", module_name: str = None) -> VAPORDataManager:
    """Get a configured data manager instance."""
    return VAPORDataManager(video_name, mode, module_name)


def load_latest_metrics(video_stem: str, metric_type: str) -> Dict:
    """Load metrics from latest run for a video."""
    latest_dir = Path("data/metrics") / video_stem / "latest"
    if not latest_dir.exists():
        raise FileNotFoundError(f"No runs found for video: {video_stem}")
    
    metric_dir = latest_dir / f"{metric_type}_metrics"
    all_metrics = {}
    for json_file in metric_dir.glob("*.json"):
        with open(json_file) as f:
            all_metrics[json_file.stem] = json.load(f)
    
    return all_metrics


def compare_across_runs(video_stem: str, metric_type: str, metric_key: str) -> Dict:
    """Compare a specific metric across all runs."""
    metrics_base = Path("data/metrics") / video_stem
    all_runs = sorted([d for d in metrics_base.iterdir() 
                       if d.is_dir() and d.name.startswith("run_")])
    
    timeline = {}
    for run_dir in all_runs:
        run_name = run_dir.name
        metric_dir = run_dir / f"{metric_type}_metrics"
        
        run_metrics = {}
        for json_file in metric_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                method = data.get('method_name') or data.get('frame_type')
                
                # Extract nested metric value
                value = data
                for key in metric_key.split('.'):
                    value = value.get(key, {})
                
                run_metrics[method] = value
        
        timeline[run_name] = run_metrics
    
    return timeline
