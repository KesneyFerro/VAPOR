#!/usr/bin/env python3
"""
MPRNet Module Script
Processes blurred images using MPRNet model.

Usage:
    python mprnet_module.py --input_dir "S:/Kesney/VAPOR/data/frames/blurred/pat3/motion_blur_high"
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from glob import glob
from tqdm import tqdm

# Add MPRNet to path
current_dir = Path(__file__).parent.parent
mprnet_dir = current_dir.parent / "MPRNet"
sys.path.insert(0, str(mprnet_dir))
sys.path.insert(0, str(mprnet_dir / "Deblurring"))

try:
    # Try to import MPRNet modules
    from MPRNet import MPRNet
except ImportError as e:
    print(f"Error importing MPRNet modules: {e}")
    print(f"Please ensure MPRNet is properly set up in {mprnet_dir}")
    sys.exit(1)


class MPRNetProcessor:
    def __init__(self, weights_path: str = None):
        """Initialize MPRNet processor."""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default weights path if not provided
        if weights_path is None:
            # Try multiple possible locations
            possible_paths = [
                current_dir.parent.parent / "pretrained_models" / "mprnet" / "mprnet_deblur.pth",
                mprnet_dir / "Deblurring" / "pretrained_models" / "model_deblurring.pth",
                mprnet_dir / "Deblurring" / "pretrained_models" / "mprnet_deblur.pth"
            ]
            
            weights_path = None
            for path in possible_paths:
                if path.exists():
                    weights_path = str(path)
                    break
        
        if weights_path is None or not os.path.exists(weights_path):
            print("Warning: No pre-trained weights found!")
            print("Searched locations:")
            for path in possible_paths:
                print(f"  - {path}")
            print("Running with random weights - results will be poor!")
            self.weights_path = None
        else:
            self.weights_path = weights_path
            print(f"Using weights: {weights_path}")
        
        # Initialize model
        try:
            # MPRNet parameters (from original paper/code)
            self.model = MPRNet(
                in_c=3,
                out_c=3,
                n_feat=96,
                scale_unetfeats=48,
                scale_orsnetfeats=32,
                num_cab=8,
                kernel_size=3,
                reduction=4,
                bias=False
            ).to(self.device)
            
            # Load weights if available
            if self.weights_path and os.path.exists(self.weights_path):
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Assume the checkpoint is the state dict
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                print("Successfully loaded pre-trained weights")
            
            self.model.eval()
            
        except Exception as e:
            print(f"Error initializing MPRNet: {e}")
            raise
    
    def predict_single_image(self, image_path: str) -> np.ndarray:
        """Process a single image."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor (CHW format)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        # Ensure dimensions are multiples of 8 (required by MPRNet)
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Process with MPRNet
        with torch.no_grad():
            restored = self.model(img_tensor)
            
            # MPRNet returns a list of outputs, take the final one
            if isinstance(restored, list):
                restored = restored[-1]
            
            # Ensure tensor is on CPU and has valid range
            restored = restored.squeeze(0).cpu()
            
            # Check for invalid values
            if torch.isnan(restored).any() or torch.isinf(restored).any():
                print("Warning: Invalid values detected in output, using input image")
                restored = img_tensor.squeeze(0).cpu()
        
        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            restored = restored[:, :h, :w]
        
        # Convert back to numpy
        restored = restored.numpy()
        restored = restored.transpose(1, 2, 0)  # CHW to HWC
        
        # Ensure output is in valid range [0, 1]
        restored = np.clip(restored, 0, 1)
        
        # Convert to uint8
        restored = (restored * 255).astype(np.uint8)
        
        # Convert back to BGR for saving
        restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        
        return restored
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all images in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
        
        print(f"Processing {len(image_files)} images...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Device: {self.device}")
        print(f"Weights: {self.weights_path or 'None (random weights)'}")
        
        # Process each image
        success_count = 0
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                # Process image
                result = self.predict_single_image(str(image_file))
                
                # Save result
                output_file = output_path / image_file.name
                cv2.imwrite(str(output_file), result)
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
        
        print(f"Successfully processed {success_count}/{len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(description="MPRNet Image Processing Module")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"S:\Kesney\VAPOR\data\frames\blurred\pat3\motion_blur_high",
        help="Input directory containing blurred images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for deblurred images"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to model weights file"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(__file__).parent.parent / "outputs" / "mprnet")
    
    try:
        # Initialize processor
        print("Initializing MPRNet...")
        processor = MPRNetProcessor(args.weights_path)
        
        # Process directory
        processor.process_directory(args.input_dir, output_dir)
        
        print(f"Processing complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())