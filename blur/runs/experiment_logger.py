"""
Comprehensive Experiment Logging System

This module provides a comprehensive logging system for tracking all experiments,
runs, and processing operations. It ensures complete reproducibility
by logging all random elements including seeds, angles, kernels, parameters,
and environmental settings.

Features:
- Deterministic run tracking with unique IDs
- Complete parameter logging for reproducibility
- Performance metrics and timing
- Error tracking and debugging information
- Configuration versioning
- Result comparison and analysis

Author: Kesney de Oliveira
"""

import logging
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import psutil
import platform
import sys
import os
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time


@dataclass
class SystemInfo:
    """System information for reproducibility."""
    platform: str
    python_version: str
    cpu_count: int
    memory_total: float
    gpu_info: List[str]
    environment_variables: Dict[str, str]
    timestamp: str


@dataclass
class RunConfiguration:
    """Configuration for an experiment run."""
    run_id: str
    run_type: str  # 'blur', 'deblur', 'analysis', etc.
    timestamp: str
    input_path: str
    output_path: str
    parameters: Dict[str, Any]
    random_seeds: Dict[str, int]
    system_info: SystemInfo


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    operation: str
    success: bool
    start_time: str
    end_time: str
    duration_seconds: float
    input_files: List[str]
    output_files: List[str]
    parameters_used: Dict[str, Any]
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None


class ExperimentLogger:
    """
    Comprehensive logging system for video processing experiments.
    
    Provides structured logging with automatic parameter tracking,
    performance monitoring, and reproducibility guarantees.
    """
    
    def __init__(self, run_type: str, base_dir: Path = None, run_id: str = None):
        """
        Initialize experiment logger.
        
        Args:
            run_type: Type of run ('blur', 'deblur', 'analysis', etc.)
            base_dir: Base directory for logs (defaults to blur/runs)
            run_id: Custom run ID (auto-generated if None)
        """
        self.run_type = run_type
        self.run_id = run_id or self._generate_run_id()
        
        # Set up directories
        if base_dir is None:
            base_dir = Path(__file__).parent
        
        self.run_dir = base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.system_info = self._collect_system_info()
        self.configuration = None
        self.results = []
        self.current_operation = None
        
        # Set up logging
        self._setup_logging()
        
        self.logger.info(f"Initialized experiment logger for {run_type} run: {self.run_id}")
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add hash for uniqueness within the same second
        unique_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"{self.run_type}_{timestamp}_{unique_hash}"
    
    def _collect_system_info(self) -> SystemInfo:
        """Collect comprehensive system information."""
        # GPU information
        gpu_info = []
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_info = [f"{gpu.name} ({gpu.memoryTotal}MB)" for gpu in gpus]
        except ImportError:
            try:
                import nvidia_ml_py3 as pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    gpu_info.append(name.decode('utf-8'))
            except ImportError:
                gpu_info = ["GPU info not available"]
        
        # Environment variables (filter sensitive ones)
        safe_env_vars = {
            k: v for k, v in os.environ.items() 
            if not any(sensitive in k.upper() for sensitive in ['PASSWORD', 'TOKEN', 'KEY', 'SECRET'])
        }
        
        return SystemInfo(
            platform=platform.platform(),
            python_version=sys.version,
            cpu_count=psutil.cpu_count(),
            memory_total=psutil.virtual_memory().total / (1024**3),  # GB
            gpu_info=gpu_info,
            environment_variables=safe_env_vars,
            timestamp=datetime.now().isoformat()
        )
    
    def _setup_logging(self):
        """Set up structured logging."""
        self.logger = logging.getLogger(f'experiment_{self.run_id}')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for detailed logs
        log_file = self.run_dir / "run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def start_run(self, input_path: str, output_path: str, parameters: Dict[str, Any], 
                  random_seeds: Dict[str, int] = None):
        """
        Start a new run with full configuration logging.
        
        Args:
            input_path: Input data path
            output_path: Output data path  
            parameters: All parameters for the run
            random_seeds: Random seeds used for deterministic operations
        """
        self.configuration = RunConfiguration(
            run_id=self.run_id,
            run_type=self.run_type,
            timestamp=datetime.now().isoformat(),
            input_path=input_path,
            output_path=output_path,
            parameters=parameters,
            random_seeds=random_seeds or {},
            system_info=self.system_info
        )
        
        # Save configuration
        config_file = self.run_dir / "configuration.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.configuration), f, indent=2, default=str)
        
        # Save parameters separately for easy access
        params_file = self.run_dir / "parameters.json"
        with open(params_file, 'w') as f:
            json.dump(parameters, f, indent=2, default=str)
        
        # Save random seeds
        if random_seeds:
            seeds_file = self.run_dir / "random_seeds.json"
            with open(seeds_file, 'w') as f:
                json.dump(random_seeds, f, indent=2)
        
        self.logger.info(f"Started run with configuration saved to {config_file}")
        
    @contextmanager
    def log_operation(self, operation_name: str, parameters: Dict[str, Any] = None):
        """
        Context manager for logging individual operations.
        
        Args:
            operation_name: Name of the operation
            parameters: Operation-specific parameters
        """
        start_time = datetime.now()
        self.current_operation = operation_name
        
        self.logger.info(f"Starting operation: {operation_name}")
        if parameters:
            self.logger.debug(f"Operation parameters: {json.dumps(parameters, default=str)}")
        
        result = ProcessingResult(
            operation=operation_name,
            success=False,
            start_time=start_time.isoformat(),
            end_time="",
            duration_seconds=0,
            input_files=[],
            output_files=[],
            parameters_used=parameters or {},
            performance_metrics={}
        )
        
        try:
            yield result
            result.success = True
            self.logger.info(f"Operation completed successfully: {operation_name}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"Operation failed: {operation_name} - {e}")
            raise
            
        finally:
            end_time = datetime.now()
            result.end_time = end_time.isoformat()
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            # Add performance metrics
            result.performance_metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'duration_seconds': result.duration_seconds
            }
            
            self.results.append(result)
            self.current_operation = None
            
            # Save result immediately
            self._save_result(result)
            
            self.logger.info(f"Operation '{operation_name}' completed in {result.duration_seconds:.2f}s")
    
    def _save_result(self, result: ProcessingResult):
        """Save individual result to file."""
        results_dir = self.run_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        result_file = results_dir / f"{result.operation}_{datetime.now().strftime('%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
    
    def log_file_processing(self, input_file: str, output_file: str, 
                           success: bool, metadata: Dict[str, Any] = None):
        """
        Log individual file processing.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            success: Whether processing succeeded
            metadata: Additional metadata (kernels used, parameters, etc.)
        """
        if self.current_operation and self.results:
            # Add to current operation result
            current_result = self.results[-1]
            current_result.input_files.append(input_file)
            if success:
                current_result.output_files.append(output_file)
        
        # Log the file processing
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"File processing {status}: {Path(input_file).name} -> {Path(output_file).name}")
        
        if metadata:
            self.logger.debug(f"File metadata: {json.dumps(metadata, default=str)}")
    
    def log_kernel_usage(self, kernel_info: Dict[str, Any]):
        """
        Log kernel usage for blur operations.
        
        Args:
            kernel_info: Information about kernels used
        """
        kernels_dir = self.run_dir / "kernels_used"
        kernels_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%H%M%S_%f")
        kernel_file = kernels_dir / f"kernel_{timestamp}.json"
        
        with open(kernel_file, 'w') as f:
            json.dump(kernel_info, f, indent=2, default=str)
        
        self.logger.debug(f"Kernel usage logged: {kernel_info.get('type', 'unknown')}")
    
    def save_numpy_data(self, data: np.ndarray, name: str, metadata: Dict[str, Any] = None):
        """
        Save numpy data with metadata.
        
        Args:
            data: Numpy array to save
            name: Name for the data
            metadata: Additional metadata
        """
        data_dir = self.run_dir / "numpy_data"
        data_dir.mkdir(exist_ok=True)
        
        # Save data
        data_file = data_dir / f"{name}.npz"
        np.savez_compressed(data_file, data=data)
        
        # Save metadata
        if metadata:
            meta_file = data_dir / f"{name}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        self.logger.debug(f"Numpy data saved: {name} (shape: {data.shape})")
    
    def finalize_run(self):
        """Finalize the run and save all results."""
        # Save complete results
        results_file = self.run_dir / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)
        
        # Generate summary
        summary = {
            'run_id': self.run_id,
            'run_type': self.run_type,
            'total_operations': len(self.results),
            'successful_operations': len([r for r in self.results if r.success]),
            'failed_operations': len([r for r in self.results if not r.success]),
            'total_duration': sum(r.duration_seconds for r in self.results),
            'total_files_processed': sum(len(r.input_files) for r in self.results),
            'run_directory': str(self.run_dir)
        }
        
        summary_file = self.run_dir / "run_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Run finalized: {summary['successful_operations']}/{summary['total_operations']} operations successful")
        self.logger.info(f"Total duration: {summary['total_duration']:.2f} seconds")
        self.logger.info(f"All data saved to: {self.run_dir}")


def create_logger(run_type: str, base_dir: Path = None) -> ExperimentLogger:
    """
    Create an experiment logger instance.
    
    Args:
        run_type: Type of run ('blur', 'deblur', 'analysis', etc.)
        base_dir: Base directory for logs
        
    Returns:
        Configured ExperimentLogger instance
    """
    return ExperimentLogger(run_type, base_dir)


def load_run_results(run_dir: Path) -> Dict[str, Any]:
    """
    Load results from a completed run.
    
    Args:
        run_dir: Directory containing run data
        
    Returns:
        Dictionary with run configuration and results
    """
    config_file = run_dir / "configuration.json"
    results_file = run_dir / "complete_results.json"
    summary_file = run_dir / "run_summary.json"
    
    data = {}
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            data['configuration'] = json.load(f)
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            data['results'] = json.load(f)
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)
    
    return data


def compare_runs(run_dirs: List[Path]) -> Dict[str, Any]:
    """
    Compare multiple runs for analysis.
    
    Args:
        run_dirs: List of run directories to compare
        
    Returns:
        Comparison analysis
    """
    runs_data = []
    for run_dir in run_dirs:
        try:
            data = load_run_results(run_dir)
            data['run_dir'] = str(run_dir)
            runs_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load run from {run_dir}: {e}")
    
    comparison = {
        'total_runs': len(runs_data),
        'run_types': list(set(r['configuration']['run_type'] for r in runs_data if 'configuration' in r)),
        'runs': runs_data,
        'comparison_timestamp': datetime.now().isoformat()
    }
    
    return comparison
