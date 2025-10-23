#!/usr/bin/env python3
"""
Deblur Models Runner
Main script to run all deblur models on a given input directory.

Usage:
    python run_all_models.py --input_dir "S:/Kesney/VAPOR/data/frames/blurred/pat3/motion_blur_high"
    python run_all_models.py --input_dir "S:/Kesney/VAPOR/data/frames/blurred/pat3/motion_blur_high" --models mprnet restormer
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List


class DeblurModelsRunner:
    """Runner for all deblur models."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.blur_modules_dir = self.script_dir / "blur_modules"
        self.outputs_dir = self.script_dir / "outputs"
        
        # Available models and their script files with conda environments
        self.available_models = {
            "mprnet": {
                "script": "mprnet_module.py",
                "conda_env": "base"  # MPRNet works in base environment
            },
            "restormer": {
                "script": "restormer_module.py",
                "conda_env": "restormer"
            },
            "uformer": {
                "script": "uformer_module.py",
                "conda_env": "uformer"
            }
        }
    
    def run_single_model(self, model_name: str, input_dir: str) -> bool:
        """Run a single deblur model in its dedicated conda environment."""
        if model_name not in self.available_models:
            print(f"Error: Model '{model_name}' not available.")
            print(f"Available models: {list(self.available_models.keys())}")
            return False
        
        model_info = self.available_models[model_name]
        script_path = self.blur_modules_dir / model_info["script"]
        conda_env = model_info["conda_env"]
        
        if not script_path.exists():
            print(f"Error: Script not found: {script_path}")
            return False
        
        print(f"\n{'='*60}")
        print(f"Running {model_name.upper()}")
        print(f"{'='*60}")
        print(f"Conda environment: {conda_env}")
        
        # Prepare command with conda environment activation
        if conda_env == "base":
            # Use current python for base environment (MPRNet)
            cmd = [sys.executable, str(script_path), "--input_dir", input_dir]
        else:
            # Use conda run to execute in specific environment
            cmd = [
                "conda", "run", "-n", conda_env, "python", 
                str(script_path), "--input_dir", input_dir
            ]
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run the model script
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"Stderr:")
                print(result.stderr)
            
            if result.returncode == 0:
                print(f"[OK] {model_name} completed successfully")
                return True
            else:
                print(f"[ERROR]  {model_name} failed with return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[ERROR]  {model_name} timed out after 1 hour")
            return False
        except Exception as e:
            print(f"[ERROR]  {model_name} failed with error: {e}")
            return False
    
    def run_all_models(self, input_dir: str, models: List[str] = None) -> dict:
        """Run all specified models or all available models."""
        if models is None:
            models = list(self.available_models.keys())
        
        # Validate input directory
        if not Path(input_dir).exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            return {}
        
        print(f"Input directory: {input_dir}")
        print(f"Models to run: {models}")
        print(f"Output directory: {self.outputs_dir}")
        
        # Ensure outputs directory exists
        self.outputs_dir.mkdir(exist_ok=True)
        
        results = {}
        
        for model_name in models:
            try:
                success = self.run_single_model(model_name, input_dir)
                results[model_name] = success
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted while running {model_name}")
                results[model_name] = False
                break
            except Exception as e:
                print(f"[ERROR]  Unexpected error running {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    def print_summary(self, results: dict):
        """Print summary of results."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        for model_name, success in results.items():
            status = "[OK] SUCCESS" if success else "[ERROR]  FAILED"
            output_dir = self.outputs_dir / model_name
            print(f"{model_name:15} {status:10} Output: {output_dir}")
        
        print(f"\nOverall: {success_count}/{total_count} models completed successfully")
        
        if success_count > 0:
            print(f"\nResults saved to: {self.outputs_dir}")
            print("You can now compare the outputs from different models.")


def main():
    parser = argparse.ArgumentParser(description="Run deblur models on input directory")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"S:\Kesney\VAPOR\data\frames\blurred\pat3\motion_blur_high",
        help="Input directory containing blurred images"
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=None,
        choices=["mprnet", "restormer", "uformer"],
        help="List of models to run (default: all models)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    runner = DeblurModelsRunner()
    
    if args.list_models:
        print("Available models:")
        for model_name, model_info in runner.available_models.items():
            print(f"  - {model_name} (conda env: {model_info['conda_env']})")
        return 0
    
    try:
        # Run models
        results = runner.run_all_models(args.input_dir, args.models)
        
        # Print summary
        runner.print_summary(results)
        
        # Return appropriate exit code
        success_count = sum(1 for success in results.values() if success)
        return 0 if success_count > 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR]  Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())