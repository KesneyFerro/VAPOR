"""
CT Mesh Preprocessing Module (Step 1)
=====================================

Handles:
- Mesh validation and cleaning (watertightness, manifoldness, degeneracies)
- Normal computation
- Characteristic length calculation (h = median edge length)
- Acceleration structure creation (RaycastingScene, KDTree)

Production-grade implementation using:
- Open3D for mesh operations and raycasting
- trimesh for topology validation
- NumPy/SciPy for statistics
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class CTMeshPreprocessor:
    """
    Preprocess and validate CT reference mesh.
    Ensures strong exception safety and production-grade mesh health checks.
    """
    
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Output directory for CT sanity check results
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Results
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.characteristic_length_h: Optional[float] = None
        self.raycasting_scene: Optional[o3d.t.geometry.RaycastingScene] = None
        self.mesh_kdtree: Optional[o3d.geometry.KDTreeFlann] = None
        self.health_report: Dict[str, Any] = {}
        
    def compute_edge_lengths(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        """
        Compute all edge lengths in mesh.
        
        Args:
            mesh: Open3D TriangleMesh
            
        Returns:
            Array of edge lengths
        """
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        if len(triangles) == 0:
            self.logger.warning("Mesh has no triangles, cannot compute edge lengths")
            return np.array([])
        
        # Get all edges (3 edges per triangle)
        edges = []
        for tri in triangles:
            v0, v1, v2 = vertices[tri]
            edges.append(np.linalg.norm(v1 - v0))
            edges.append(np.linalg.norm(v2 - v1))
            edges.append(np.linalg.norm(v0 - v2))
        
        return np.array(edges)
    
    def compute_vertex_nn_distances(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        """
        Compute nearest-neighbor distances between vertices.
        
        Args:
            mesh: Open3D TriangleMesh
            
        Returns:
            Array of NN distances
        """
        vertices = np.asarray(mesh.vertices)
        
        if len(vertices) < 2:
            return np.array([])
        
        # Build KDTree
        tree = cKDTree(vertices)
        
        # Query k=2 (self + nearest neighbor)
        distances, _ = tree.query(vertices, k=2)
        
        # Return distances to nearest neighbor (excluding self)
        return distances[:, 1]
    
    def compute_characteristic_length(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, float]:
        """
        Compute characteristic length h from mesh.
        
        h = median(h_e, h_nn) where:
        - h_e: median edge length
        - h_nn: median nearest-neighbor vertex spacing
        
        Args:
            mesh: Open3D TriangleMesh
            
        Returns:
            Dictionary with h, h_e, h_nn (in meters)
            
        Raises:
            ValueError: If mesh is invalid
        """
        self.logger.info("Computing characteristic length h...")
        
        # Edge lengths
        edge_lengths = self.compute_edge_lengths(mesh)
        
        if len(edge_lengths) == 0:
            raise ValueError("Cannot compute characteristic length: no edges in mesh")
        
        h_e = float(np.median(edge_lengths))
        
        # Vertex NN distances
        nn_distances = self.compute_vertex_nn_distances(mesh)
        
        if len(nn_distances) == 0:
            raise ValueError("Cannot compute characteristic length: insufficient vertices")
        
        h_nn = float(np.median(nn_distances))
        
        # Characteristic length
        h = float(np.median([h_e, h_nn]))
        
        self.logger.info(f"Characteristic length: h={h:.6f}m (h_e={h_e:.6f}m, h_nn={h_nn:.6f}m)")
        
        self.characteristic_length_h = h
        
        return {
            'h': h,
            'h_e': h_e,
            'h_nn': h_nn,
            'units': 'meters',
            'edge_length_stats': {
                'min': float(np.min(edge_lengths)),
                'max': float(np.max(edge_lengths)),
                'mean': float(np.mean(edge_lengths)),
                'std': float(np.std(edge_lengths)),
            },
            'nn_distance_stats': {
                'min': float(np.min(nn_distances)),
                'max': float(np.max(nn_distances)),
                'mean': float(np.mean(nn_distances)),
                'std': float(np.std(nn_distances)),
            }
        }
    
    def validate_mesh_health_open3d(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
        """
        Validate mesh health using Open3D.
        
        Args:
            mesh: Open3D TriangleMesh
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating mesh health (Open3D)...")
        
        results = {
            'has_vertices': mesh.has_vertices(),
            'has_triangles': mesh.has_triangles(),
            'has_vertex_normals': mesh.has_vertex_normals(),
            'has_triangle_normals': mesh.has_triangle_normals(),
            'n_vertices': len(mesh.vertices),
            'n_triangles': len(mesh.triangles),
            'is_edge_manifold': mesh.is_edge_manifold(),
            'is_vertex_manifold': mesh.is_vertex_manifold(),
            'is_orientable': mesh.is_orientable(),
            'is_watertight': mesh.is_watertight(),
        }
        
        # Bounding box
        if mesh.has_vertices():
            vertices = np.asarray(mesh.vertices)
            bbox_min = vertices.min(axis=0).tolist()
            bbox_max = vertices.max(axis=0).tolist()
            bbox_size = (vertices.max(axis=0) - vertices.min(axis=0)).tolist()
            
            results['bounding_box'] = {
                'min': bbox_min,
                'max': bbox_max,
                'size': bbox_size,
            }
        
        return results
    
    def validate_mesh_health_trimesh(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
        """
        Validate mesh health using trimesh (if available).
        
        Args:
            mesh: Open3D TriangleMesh
            
        Returns:
            Dictionary with validation results
        """
        if not HAS_TRIMESH:
            self.logger.warning("trimesh not available, skipping trimesh validation")
            return {'available': False}
        
        self.logger.info("Validating mesh health (trimesh)...")
        
        try:
            # Convert to trimesh
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
            
            results = {
                'available': True,
                'is_watertight': tmesh.is_watertight,
                'is_winding_consistent': tmesh.is_winding_consistent,
                'euler_number': tmesh.euler_number,
                'volume': float(tmesh.volume) if tmesh.is_watertight else None,
                'area': float(tmesh.area),
                'n_connected_components': len(trimesh.graph.connected_components(tmesh.edges)),
            }
            
            # Check for degeneracies
            if hasattr(tmesh, 'face_adjacency'):
                results['n_face_adjacencies'] = len(tmesh.face_adjacency)
            
            return results
            
        except Exception as e:
            self.logger.error(f"trimesh validation failed: {e}")
            return {'available': True, 'error': str(e)}
    
    def clean_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Clean mesh: remove duplicates, degenerate faces, etc.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Cleaned mesh
        """
        self.logger.info("Cleaning mesh...")
        
        cleaned = mesh
        
        # Remove duplicated vertices
        cleaned = cleaned.remove_duplicated_vertices()
        self.logger.debug("Removed duplicated vertices")
        
        # Remove duplicated triangles
        cleaned = cleaned.remove_duplicated_triangles()
        self.logger.debug("Removed duplicated triangles")
        
        # Remove degenerate triangles
        cleaned = cleaned.remove_degenerate_triangles()
        self.logger.debug("Removed degenerate triangles")
        
        # Remove unreferenced vertices
        cleaned = cleaned.remove_unreferenced_vertices()
        self.logger.debug("Removed unreferenced vertices")
        
        self.logger.info(f"Mesh cleaned: {len(cleaned.vertices)} vertices, {len(cleaned.triangles)} triangles")
        
        return cleaned
    
    def compute_normals(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """
        Compute vertex and triangle normals.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Mesh with normals
        """
        self.logger.info("Computing normals...")
        
        if len(mesh.triangles) == 0:
            self.logger.warning("Cannot compute normals: mesh has no triangles")
            return mesh
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        # Compute triangle normals
        mesh.compute_triangle_normals()
        
        self.logger.info("Normals computed")
        
        return mesh
    
    def create_raycasting_scene(self, mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
        """
        Create Open3D RaycastingScene for exact point-to-triangle distances.
        
        Args:
            mesh: Open3D TriangleMesh
            
        Returns:
            RaycastingScene
            
        Raises:
            ValueError: If mesh has no triangles
        """
        self.logger.info("Creating RaycastingScene...")
        
        if len(mesh.triangles) == 0:
            raise ValueError("Cannot create RaycastingScene: mesh has no triangles")
        
        # Convert to tensor mesh
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        
        # Create scene
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        
        self.logger.info("RaycastingScene created")
        self.raycasting_scene = scene
        
        return scene
    
    def create_kdtree(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.KDTreeFlann:
        """
        Create KDTree for mesh vertices (for mesh-to-point queries).
        
        Args:
            mesh: Open3D TriangleMesh
            
        Returns:
            KDTreeFlann
        """
        self.logger.info("Creating KDTree for mesh vertices...")
        
        # Create point cloud from mesh vertices
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        
        # Build KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        self.logger.info("KDTree created")
        self.mesh_kdtree = kdtree
        
        return kdtree
    
    def preprocess(self, 
                   mesh: o3d.geometry.TriangleMesh,
                   clean: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            mesh: Input CT mesh
            clean: Whether to clean mesh (remove duplicates, degeneracies)
            
        Returns:
            Preprocessing report
            
        Raises:
            ValueError: If preprocessing fails
        """
        self.logger.info("Starting CT mesh preprocessing...")
        
        report = {}
        
        # Initial validation
        report['initial_health_open3d'] = self.validate_mesh_health_open3d(mesh)
        report['initial_health_trimesh'] = self.validate_mesh_health_trimesh(mesh)
        
        # Clean if requested
        if clean:
            mesh_cleaned = self.clean_mesh(mesh)
            report['cleaned'] = True
            report['post_clean_health'] = self.validate_mesh_health_open3d(mesh_cleaned)
        else:
            mesh_cleaned = mesh
            report['cleaned'] = False
        
        # Compute normals
        mesh_cleaned = self.compute_normals(mesh_cleaned)
        
        # Compute characteristic length
        try:
            char_length = self.compute_characteristic_length(mesh_cleaned)
            report['characteristic_length'] = char_length
        except Exception as e:
            self.logger.error(f"Failed to compute characteristic length: {e}")
            raise ValueError(f"Failed to compute characteristic length: {e}")
        
        # Create acceleration structures
        try:
            if len(mesh_cleaned.triangles) > 0:
                self.create_raycasting_scene(mesh_cleaned)
                report['raycasting_scene'] = 'created'
            else:
                self.logger.warning("No triangles, skipping RaycastingScene")
                report['raycasting_scene'] = 'skipped (no triangles)'
        except Exception as e:
            self.logger.error(f"Failed to create RaycastingScene: {e}")
            report['raycasting_scene'] = f'failed: {e}'
        
        try:
            self.create_kdtree(mesh_cleaned)
            report['kdtree'] = 'created'
        except Exception as e:
            self.logger.error(f"Failed to create KDTree: {e}")
            report['kdtree'] = f'failed: {e}'
        
        # Store processed mesh
        self.mesh = mesh_cleaned
        self.health_report = report
        
        self.logger.info("CT mesh preprocessing complete")
        
        return report
    
    def save_health_report(self) -> Path:
        """
        Save health report to JSON.
        
        Returns:
            Path to saved JSON
        """
        output_path = self.output_dir / 'ct_mesh_health.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.health_report, f, indent=2)
        
        self.logger.info(f"Health report saved to: {output_path}")
        
        return output_path
    
    def save_preview(self, output_name: str = 'ct_mesh_preview') -> Tuple[Path, Path]:
        """
        Save mesh preview (PLY and PNG).
        
        Args:
            output_name: Base name for output files
            
        Returns:
            Tuple of (ply_path, png_path)
        """
        if self.mesh is None:
            raise ValueError("No mesh to save. Run preprocess() first.")
        
        # Save PLY
        ply_path = self.output_dir / f'{output_name}.ply'
        o3d.io.write_triangle_mesh(str(ply_path), self.mesh)
        self.logger.info(f"Mesh saved to: {ply_path}")
        
        # Save PNG (visualization)
        png_path = self.output_dir / f'{output_name}.png'
        
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(self.mesh)
            vis.update_geometry(self.mesh)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(png_path))
            vis.destroy_window()
            self.logger.info(f"Preview image saved to: {png_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save preview image: {e}")
            png_path = None
        
        return ply_path, png_path


def preprocess_ct_mesh(mesh: o3d.geometry.TriangleMesh,
                        output_dir: Path,
                        clean: bool = True,
                        logger: Optional[logging.Logger] = None) -> Tuple[
                            o3d.geometry.TriangleMesh,
                            float,
                            o3d.t.geometry.RaycastingScene,
                            o3d.geometry.KDTreeFlann,
                            Dict[str, Any]
                        ]:
    """
    Convenience function for CT mesh preprocessing.
    
    Args:
        mesh: Input CT mesh
        output_dir: Output directory
        clean: Whether to clean mesh
        logger: Optional logger
        
    Returns:
        Tuple of (processed_mesh, h, raycasting_scene, kdtree, report)
    """
    preprocessor = CTMeshPreprocessor(output_dir, logger)
    report = preprocessor.preprocess(mesh, clean=clean)
    preprocessor.save_health_report()
    preprocessor.save_preview()
    
    return (
        preprocessor.mesh,
        preprocessor.characteristic_length_h,
        preprocessor.raycasting_scene,
        preprocessor.mesh_kdtree,
        report
    )
