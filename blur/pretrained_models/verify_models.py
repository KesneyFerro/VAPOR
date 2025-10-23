#!/usr/bin/env python3
"""
VAPOR Model Verification Script
==============================
Verifies all 5 deblurring models are properly downloaded and accessible.
"""

import os
from pathlib import Path

def verify_models():
    """Verify all 5 models in the VAPOR pipeline."""
    
    base_dir = Path(".")
    models = {
        "Restormer": {
            "path": base_dir / "restormer" / "motion_deblurring.pth",
            "expected_size_mb": 100
        },
        "MPRNet": {
            "path": base_dir / "mprnet" / "mprnet_deblur.pth", 
            "expected_size_mb": 77
        },
        "Uformer": {
            "path": base_dir / "uformer" / "Uformer_B.pth",
            "expected_size_mb": 584
        },
        "DeblurDiNAT": {
            "path": base_dir / "deblurdinat" / "vgg19-dcbb9e9d.pth",
            "expected_size_mb": 548
        },
        "DeblurGANv2": {
            "path": base_dir / "deblurgan_v2" / "fpn_inception.h5",
            "expected_size_mb": 233
        }
    }
    
    print("VAPOR Model Verification")
    print("=" * 40)
    
    total_size = 0
    completed = 0
    issues = []
    
    for name, info in models.items():
        path = info["path"]
        expected_size = info["expected_size_mb"]
        
        if path.exists():
            actual_size_mb = path.stat().st_size / 1024 / 1024
            total_size += actual_size_mb
            
            # Check if size is reasonable (within 20% of expected)
            size_diff = abs(actual_size_mb - expected_size) / expected_size
            if size_diff > 0.2:
                status = "WARNING"
                issues.append(f"{name}: Size mismatch ({actual_size_mb:.1f}MB vs expected {expected_size}MB)")
            else:
                status = "OK"
                completed += 1
                
            print(f"[{status}] {name}: {path.name} ({actual_size_mb:.1f}MB)")
        else:
            print(f"[MISSING] {name}: {path}")
            issues.append(f"{name}: Model file not found at {path}")
    
    print("=" * 40)
    print(f"Models Ready: {completed}/5")
    print(f"Total Size: {total_size:.1f}MB ({total_size/1024:.2f}GB)")
    
    if completed == 5:
        print("STATUS: All models verified successfully")
        return True
    else:
        print(f"STATUS: {len(issues)} issues found")
        for issue in issues:
            print(f"  - {issue}")
        return False

if __name__ == "__main__":
    success = verify_models()
    exit(0 if success else 1)