#!/usr/bin/env python3
"""
Setup Deblur Model Environments
Script to create and set up conda environments for each deblur model.

Usage:
    python setup_environments.py
    python setup_environments.py --models deblurgan_v2 restormer
"""

import os
import sys
import argparse
import subprocess
from typing import List


class EnvironmentSetup:
    """Setup conda environments for deblur models."""
    
    def __init__(self):
        # Environment configurations
        self.environments = {
            "restormer": {
                "python_version": "3.8",
                "packages": [
                    "pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch",
                    "opencv",
                    "numpy",
                    "pillow",
                    "tqdm",
                    "lmdb",
                    "basicsr"
                ]
            },
            "uformer": {
                "python_version": "3.8",
                "packages": [
                    "pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch",
                    "opencv",
                    "numpy",
                    "pillow",
                    "tqdm",
                    "timm",
                    "einops"
                ]
            }
        }
    
    def check_conda_installed(self):
        """Check if conda is installed and available."""
        try:
            result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[OK] Conda found: {result.stdout.strip()}")
                return True
            else:
                print("[ERROR]  Conda not found or not working")
                return False
        except Exception as e:
            print(f"[ERROR]  Error checking conda: {e}")
            return False
    
    def environment_exists(self, env_name: str) -> bool:
        """Check if conda environment already exists."""
        try:
            result = subprocess.run(
                ["conda", "env", "list"], 
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return env_name in result.stdout
            return False
        except Exception:
            return False
    
    def create_environment(self, env_name: str) -> bool:
        """Create a conda environment."""
        if self.environment_exists(env_name):
            print(f"⚠️  Environment '{env_name}' already exists")
            return True
        
        env_config = self.environments[env_name]
        python_version = env_config["python_version"]
        
        print(f"📦 Creating environment '{env_name}' with Python {python_version}...")
        
        cmd = ["conda", "create", "-n", env_name, f"python={python_version}", "-y"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[OK] Environment '{env_name}' created successfully")
                return True
            else:
                print(f"[ERROR]  Failed to create environment '{env_name}'")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR]  Error creating environment '{env_name}': {e}")
            return False
    
    def install_packages(self, env_name: str) -> bool:
        """Install packages in the conda environment."""
        env_config = self.environments[env_name]
        packages = env_config["packages"]
        
        print(f"📦 Installing packages in environment '{env_name}'...")
        
        for package_spec in packages:
            print(f"  Installing: {package_spec}")
            
            # Determine if this is a conda or pip install
            if "-c pytorch" in package_spec or package_spec in ["opencv", "numpy", "pillow"]:
                # Use conda install
                cmd = ["conda", "install", "-n", env_name] + package_spec.split() + ["-y"]
            else:
                # Use pip install within the conda environment
                cmd = ["conda", "run", "-n", env_name, "pip", "install"] + package_spec.split()
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"    [OK] Installed: {package_spec}")
                else:
                    print(f"    [ERROR]  Failed to install: {package_spec}")
                    print(f"    Error: {result.stderr}")
                    return False
            except Exception as e:
                print(f"    [ERROR]  Error installing {package_spec}: {e}")
                return False
        
        print(f"[OK] All packages installed in environment '{env_name}'")
        return True
    
    def setup_environment(self, env_name: str) -> bool:
        """Set up a complete environment."""
        if env_name not in self.environments:
            print(f"[ERROR]  Unknown environment: {env_name}")
            print(f"Available environments: {list(self.environments.keys())}")
            return False
        
        print(f"\n{'='*60}")
        print(f"Setting up environment: {env_name}")
        print(f"{'='*60}")
        
        # Create environment
        if not self.create_environment(env_name):
            return False
        
        # Install packages
        if not self.install_packages(env_name):
            return False
        
        print(f"[OK] Environment '{env_name}' setup complete!")
        return True
    
    def setup_all_environments(self, env_names: List[str] = None) -> dict:
        """Set up all specified environments."""
        if env_names is None:
            env_names = list(self.environments.keys())
        
        if not self.check_conda_installed():
            print("Please install conda/miniconda first")
            return {}
        
        results = {}
        
        for env_name in env_names:
            try:
                success = self.setup_environment(env_name)
                results[env_name] = success
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted while setting up {env_name}")
                results[env_name] = False
                break
            except Exception as e:
                print(f"[ERROR]  Unexpected error setting up {env_name}: {e}")
                results[env_name] = False
        
        return results
    
    def print_summary(self, results: dict):
        """Print setup summary."""
        print(f"\n{'='*60}")
        print("SETUP SUMMARY")
        print(f"{'='*60}")
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        for env_name, success in results.items():
            status = "[OK] SUCCESS" if success else "[ERROR]  FAILED"
            print(f"{env_name:15} {status}")
        
        print(f"\nOverall: {success_count}/{total_count} environments set up successfully")
        
        if success_count > 0:
            print(f"\n🎉 Ready to run deblur models!")
            print("You can now use: python run_all_models.py")


def main():
    parser = argparse.ArgumentParser(description="Set up conda environments for deblur models")
    parser.add_argument(
        "--models",
        nargs='+',
        default=None,
        choices=["restormer", "uformer"],
        help="List of model environments to set up (default: all models)"
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List available environments and exit"
    )
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup()
    
    if args.list_envs:
        print("Available environments:")
        for env_name, config in setup.environments.items():
            python_ver = config["python_version"]
            pkg_count = len(config["packages"])
            print(f"  - {env_name} (Python {python_ver}, {pkg_count} packages)")
        return 0
    
    try:
        # Set up environments
        results = setup.setup_all_environments(args.models)
        
        # Print summary
        setup.print_summary(results)
        
        # Return appropriate exit code
        success_count = sum(1 for success in results.values() if success)
        return 0 if success_count > 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR]  Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())