"""
VAPOR-Maploc Setup and Integration Script
Ensures maploc dependencies are installed and adapts the pipeline for VAPOR.
"""

import subprocess
import sys
from pathlib import Path
import os

def install_maploc_dependencies():
    """Install maploc requirements in the current environment."""
    print("Installing maploc dependencies...")
    
    maploc_dir = Path(__file__).parent / "maploc"
    requirements_path = maploc_dir / "requirements.txt"
    
    if not requirements_path.exists():
        print(f"ERROR: Requirements file not found: {requirements_path}")
        return False
    
    try:
        # Install maploc package in development mode
        print("Installing maploc package...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", str(maploc_dir)
        ], check=True)
        
        print("✓ Maploc installation completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install maploc dependencies: {e}")
        return False

def setup_vapor_maploc_integration():
    """Setup integration between VAPOR and maploc."""
    print("Setting up VAPOR-maploc integration...")
    
    # Add maploc to Python path
    maploc_dir = Path(__file__).parent / "maploc"
    
    if maploc_dir.exists():
        # Create __init__.py files if needed
        init_files = [
            maploc_dir / "__init__.py",
            maploc_dir / "hloc" / "__init__.py"
        ]
        
        for init_file in init_files:
            if not init_file.exists():
                init_file.touch()
                print(f"Created {init_file}")
        
        print("✓ VAPOR-maploc integration setup completed")
        return True
    else:
        print(f"ERROR: Maploc directory not found: {maploc_dir}")
        return False

def verify_installation():
    """Verify that maploc is properly installed."""
    print("Verifying maploc installation...")
    
    try:
        # Test basic imports
        sys.path.append(str(Path(__file__).parent / "reconstruction" / "maploc"))
        
        import pycolmap
        print("  ✓ pycolmap imported successfully")
        
        from hloc import extract_features, match_features, reconstruction
        print("  ✓ hloc modules imported successfully")
        
        from BatchRunUtils import TrialConfig
        print("  ✓ BatchRunUtils imported successfully")
        
        print("✓ All maploc dependencies verified")
        return True
        
    except ImportError as e:
        print(f"ERROR: Import failed: {e}")
        return False

def main():
    """Main setup function."""
    print("VAPOR-Maploc Setup Script")
    print("=" * 40)
    
    # Step 1: Install dependencies
    if not install_maploc_dependencies():
        print("Setup failed at dependency installation")
        return 1
    
    # Step 2: Setup integration
    if not setup_vapor_maploc_integration():
        print("Setup failed at integration setup")
        return 1
    
    # Step 3: Verify installation
    if not verify_installation():
        print("Setup failed at verification")
        return 1
    
    print("\n" + "=" * 40)
    print("VAPOR-MAPLOC SETUP COMPLETED")
    print("=" * 40)
    print("\nYou can now run:")
    print("  python reconstruction_pipeline.py --video pat3.mp4")
    print("  python metrics_calculator.py --video pat3.mp4")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())