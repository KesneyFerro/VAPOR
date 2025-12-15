"""
Environment and Input Loader (Step 0)
=====================================

Handles:
- Loading CT mesh (PLY) with Open3D
- Loading registered point clouds from multiple models
- Environment validation and logging
- Library version tracking
- Path resolution and validation

Production requirements:
- Open3D >= 0.17 for geometry I/O
- Robust exception handling
- Comprehensive logging
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

import numpy as np
import open3d as o3d

# Version tracking
try:
    import trimesh
    TRIMESH_VERSION = trimesh.__version__
except ImportError:
    TRIMESH_VERSION = "not installed"

try:
    import scipy
    SCIPY_VERSION = scipy.__version__
except ImportError:
    SCIPY_VERSION = "not installed"


class EnvironmentLoader:
    """
    Load and validate all inputs for 3D reconstruction QA.
    Ensures strong exception safety and comprehensive logging.
    """
    
    def __init__(self, output_dir: Path, log_dir: Path):
        """
        Initialize environment loader.
        
        Args:
            output_dir: Root output directory for all results
            log_dir: Directory for log files
            
        Raises:
            ValueError: If directories cannot be created
        """
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Failed to create directories: {e}")
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Track loaded data
        self.ct_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.ct_mesh_path: Optional[Path] = None
        self.point_clouds: Dict[str, o3d.geometry.PointCloud] = {}
        self.point_cloud_paths: Dict[str, Path] = {}
        self.environment_info: Dict[str, Any] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger('VAPOR.QA.Environment')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'environment_loader_{timestamp}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def collect_environment_info(self) -> Dict[str, Any]:
        """
        Collect comprehensive environment information.
        
        Returns:
            Dictionary with library versions, Python info, system info
        """
        info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'libraries': {
                'open3d': o3d.__version__,
                'numpy': np.__version__,
                'scipy': SCIPY_VERSION,
                'trimesh': TRIMESH_VERSION,
            },
            'system': {
                'platform': sys.platform,
            }
        }
        
        # Check for optional libraries
        try:
            import igl
            info['libraries']['igl'] = 'installed'
        except ImportError:
            info['libraries']['igl'] = 'not installed'
            self.logger.warning("pyigl not available - curvature analysis will be skipped")
        
        self.environment_info = info
        self.logger.info(f"Environment info collected: Open3D {o3d.__version__}, NumPy {np.__version__}")
        
        return info
    
    def find_ct_mesh(self, video_dir: Path) -> Path:
        """
        Find CT reference mesh (PLY) in video directory.
        
        Args:
            video_dir: Directory containing CT mesh (e.g., /data/point_clouds/{video_name}/)
            
        Returns:
            Path to CT mesh PLY file
            
        Raises:
            FileNotFoundError: If no PLY found or multiple PLYs found
        """
        video_dir = Path(video_dir)
        
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        # Find PLY files at this directory level only (not subdirectories)
        ply_files = list(video_dir.glob('*.ply'))
        
        if len(ply_files) == 0:
            raise FileNotFoundError(f"No PLY file found in {video_dir}")
        elif len(ply_files) > 1:
            raise FileNotFoundError(
                f"Multiple PLY files found in {video_dir}: {[p.name for p in ply_files]}. "
                "Expected exactly one CT reference mesh."
            )
        
        ct_path = ply_files[0]
        self.logger.info(f"Found CT reference mesh: {ct_path}")
        return ct_path
    
    def load_ct_mesh(self, ct_mesh_path: Path) -> o3d.geometry.TriangleMesh:
        """
        Load CT reference mesh with validation.
        
        Args:
            ct_mesh_path: Path to CT mesh PLY
            
        Returns:
            Open3D TriangleMesh
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be loaded or is empty
        """
        ct_mesh_path = Path(ct_mesh_path)
        
        if not ct_mesh_path.exists():
            raise FileNotFoundError(f"CT mesh file not found: {ct_mesh_path}")
        
        self.logger.info(f"Loading CT mesh from: {ct_mesh_path}")
        
        try:
            mesh = o3d.io.read_triangle_mesh(str(ct_mesh_path))
        except Exception as e:
            raise ValueError(f"Failed to load CT mesh: {e}")
        
        # Validate mesh
        if not mesh.has_vertices():
            raise ValueError(f"CT mesh has no vertices: {ct_mesh_path}")
        
        n_vertices = len(mesh.vertices)
        n_triangles = len(mesh.triangles)
        
        self.logger.info(f"CT mesh loaded: {n_vertices} vertices, {n_triangles} triangles")
        
        if n_vertices == 0:
            raise ValueError("CT mesh is empty (0 vertices)")
        
        # If no triangles, might be a point cloud - log warning
        if n_triangles == 0:
            self.logger.warning(
                "CT mesh has no triangles - treating as point cloud. "
                "Will convert to mesh via Poisson reconstruction if needed."
            )
        
        # Store
        self.ct_mesh = mesh
        self.ct_mesh_path = ct_mesh_path
        
        return mesh
    
    def find_registered_point_clouds(self, 
                                      video_dir: Path,
                                      run_timestamp: str,
                                      models: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Find all registered point cloud files.
        
        Expected structure:
        /data/point_clouds/{video_name}/{run_timestamp}/{model}/{model}/registration/registered_cloud.ply
        
        Args:
            video_dir: Root directory for video (e.g., /data/point_clouds/{video_name}/)
            run_timestamp: Timestamp of reconstruction run
            models: Optional list of model names to search for. If None, searches all.
            
        Returns:
            Dictionary mapping model_name -> registered_cloud.ply path
            
        Raises:
            FileNotFoundError: If no point clouds found
        """
        video_dir = Path(video_dir)
        run_dir = video_dir / run_timestamp
        
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        point_cloud_paths = {}
        
        # If models specified, search only those
        if models:
            search_dirs = [run_dir / model for model in models]
        else:
            # Search all subdirectories
            search_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        
        for model_dir in search_dirs:
            if not model_dir.exists():
                self.logger.warning(f"Model directory not found: {model_dir}")
                continue
            
            model_name = model_dir.name
            
            # Expected path: {model}/{model}/registration/registered_cloud.ply
            registered_path = model_dir / model_name / 'registration' / 'registered_cloud.ply'
            
            if registered_path.exists():
                point_cloud_paths[model_name] = registered_path
                self.logger.info(f"Found registered point cloud for '{model_name}': {registered_path}")
            else:
                self.logger.warning(
                    f"No registered_cloud.ply found for '{model_name}' at expected path: {registered_path}"
                )
        
        if not point_cloud_paths:
            raise FileNotFoundError(
                f"No registered point clouds found in {run_dir}. "
                f"Expected path pattern: {{model}}/{{model}}/registration/registered_cloud.ply"
            )
        
        self.logger.info(f"Found {len(point_cloud_paths)} registered point clouds")
        return point_cloud_paths
    
    def load_point_cloud(self, path: Path, model_name: str) -> o3d.geometry.PointCloud:
        """
        Load a single point cloud with validation.
        
        Args:
            path: Path to PLY point cloud
            model_name: Name of the model (for logging)
            
        Returns:
            Open3D PointCloud
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be loaded or is empty
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Point cloud not found: {path}")
        
        self.logger.info(f"Loading point cloud '{model_name}' from: {path}")
        
        try:
            pcd = o3d.io.read_point_cloud(str(path))
        except Exception as e:
            raise ValueError(f"Failed to load point cloud '{model_name}': {e}")
        
        # Validate
        if not pcd.has_points():
            raise ValueError(f"Point cloud '{model_name}' has no points: {path}")
        
        n_points = len(pcd.points)
        
        if n_points == 0:
            raise ValueError(f"Point cloud '{model_name}' is empty (0 points)")
        
        self.logger.info(f"Point cloud '{model_name}' loaded: {n_points} points")
        
        return pcd
    
    def load_all_point_clouds(self, 
                               point_cloud_paths: Dict[str, Path]) -> Dict[str, o3d.geometry.PointCloud]:
        """
        Load all point clouds.
        
        Args:
            point_cloud_paths: Dictionary mapping model_name -> path
            
        Returns:
            Dictionary mapping model_name -> PointCloud
            
        Raises:
            ValueError: If any point cloud fails to load
        """
        point_clouds = {}
        errors = []
        
        for model_name, path in point_cloud_paths.items():
            try:
                pcd = self.load_point_cloud(path, model_name)
                point_clouds[model_name] = pcd
                self.point_cloud_paths[model_name] = path
            except Exception as e:
                error_msg = f"Failed to load '{model_name}': {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        if errors:
            raise ValueError(f"Failed to load {len(errors)} point cloud(s):\n" + "\n".join(errors))
        
        self.point_clouds = point_clouds
        self.logger.info(f"Successfully loaded {len(point_clouds)} point clouds")
        
        return point_clouds
    
    def save_run_log(self) -> Path:
        """
        Save comprehensive run log with all environment and input information.
        
        Returns:
            Path to saved run_log.txt
        """
        log_path = self.output_dir / 'run_log.txt'
        
        with open(log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VAPOR 3D Reconstruction QA - Run Log\n")
            f.write("=" * 80 + "\n\n")
            
            # Timestamp
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            # Environment
            f.write("Environment Information:\n")
            f.write("-" * 40 + "\n")
            env = self.environment_info
            f.write(f"Python: {env.get('python_version', 'unknown')}\n")
            f.write("Libraries:\n")
            for lib, version in env.get('libraries', {}).items():
                f.write(f"  - {lib}: {version}\n")
            f.write(f"Platform: {env.get('system', {}).get('platform', 'unknown')}\n\n")
            
            # CT Mesh
            f.write("CT Reference Mesh:\n")
            f.write("-" * 40 + "\n")
            if self.ct_mesh_path:
                f.write(f"Path: {self.ct_mesh_path}\n")
                if self.ct_mesh:
                    f.write(f"Vertices: {len(self.ct_mesh.vertices)}\n")
                    f.write(f"Triangles: {len(self.ct_mesh.triangles)}\n")
            else:
                f.write("Not loaded\n")
            f.write("\n")
            
            # Point Clouds
            f.write("Registered Point Clouds:\n")
            f.write("-" * 40 + "\n")
            for model_name, pcd in self.point_clouds.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Path: {self.point_cloud_paths.get(model_name, 'unknown')}\n")
                f.write(f"  Points: {len(pcd.points)}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        # Also save as JSON
        json_path = self.output_dir / 'run_log.json'
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment_info,
            'ct_mesh': {
                'path': str(self.ct_mesh_path) if self.ct_mesh_path else None,
                'vertices': len(self.ct_mesh.vertices) if self.ct_mesh else 0,
                'triangles': len(self.ct_mesh.triangles) if self.ct_mesh else 0,
            },
            'point_clouds': {
                model: {
                    'path': str(self.point_cloud_paths.get(model, '')),
                    'points': len(pcd.points)
                }
                for model, pcd in self.point_clouds.items()
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Run log saved to: {log_path}")
        self.logger.info(f"Run log JSON saved to: {json_path}")
        
        return log_path
    
    def load_complete_pipeline(self,
                                video_name: str,
                                run_timestamp: str,
                                data_root: Path = Path('data/point_clouds'),
                                models: Optional[List[str]] = None) -> Tuple[
                                    o3d.geometry.TriangleMesh,
                                    Dict[str, o3d.geometry.PointCloud]
                                ]:
        """
        Complete pipeline: find and load all inputs.
        
        Args:
            video_name: Name of the video/sequence
            run_timestamp: Timestamp of reconstruction run
            data_root: Root data directory (default: data/point_clouds)
            models: Optional list of specific models to load
            
        Returns:
            Tuple of (ct_mesh, point_clouds_dict)
            
        Raises:
            FileNotFoundError: If inputs cannot be found
            ValueError: If inputs cannot be loaded
        """
        self.logger.info(f"Starting complete pipeline for video='{video_name}', run='{run_timestamp}'")
        
        video_dir = Path(data_root) / video_name
        
        # Collect environment info
        self.collect_environment_info()
        
        # Find and load CT mesh
        ct_path = self.find_ct_mesh(video_dir)
        ct_mesh = self.load_ct_mesh(ct_path)
        
        # Find and load point clouds
        pc_paths = self.find_registered_point_clouds(video_dir, run_timestamp, models)
        point_clouds = self.load_all_point_clouds(pc_paths)
        
        # Save run log
        self.save_run_log()
        
        self.logger.info("Complete pipeline loaded successfully")
        
        return ct_mesh, point_clouds


# Convenience function
def load_reconstruction_data(video_name: str,
                              run_timestamp: str,
                              output_dir: Path,
                              log_dir: Path,
                              data_root: Path = Path('data/point_clouds'),
                              models: Optional[List[str]] = None) -> Tuple[
                                  o3d.geometry.TriangleMesh,
                                  Dict[str, o3d.geometry.PointCloud],
                                  EnvironmentLoader
                              ]:
    """
    Convenience function to load all reconstruction data.
    
    Args:
        video_name: Name of video/sequence
        run_timestamp: Reconstruction run timestamp
        output_dir: Output directory for results
        log_dir: Log directory
        data_root: Root data directory
        models: Optional list of specific models
        
    Returns:
        Tuple of (ct_mesh, point_clouds_dict, loader)
    """
    loader = EnvironmentLoader(output_dir, log_dir)
    ct_mesh, point_clouds = loader.load_complete_pipeline(
        video_name, run_timestamp, data_root, models
    )
    return ct_mesh, point_clouds, loader
