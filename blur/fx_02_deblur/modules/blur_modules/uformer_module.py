#!/usr/bin/env python3
"""
Uformer Module Script
Processes blurred images using Uformer model.

Usage:
    python uformer_module.py --input_dir "S:/Kesney/VAPOR/data/frames/blurred/pat3/motion_blur_high"
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import math
from pathlib import Path
from glob import glob
from tqdm import tqdm

# Add Uformer to path
current_dir = Path(__file__).parent.parent
uformer_dir = current_dir.parent / "Uformer"
sys.path.insert(0, str(uformer_dir))
sys.path.insert(0, str(uformer_dir / "deblurring"))

try:
    # Import Uformer modules
    from model import Uformer
except ImportError as e:
    print(f"Error importing Uformer modules: {e}")
    if "timm" in str(e):
        print("Missing dependency: timm")
        print("Please run: conda activate uformer && pip install timm einops")
    elif "einops" in str(e):
        print("Missing dependency: einops")
        print("Please run: conda activate uformer && pip install einops")
    print(f"Please ensure Uformer is properly set up in {uformer_dir}")
    sys.exit(1)


class UformerProcessor:
    def __init__(self, weights_path: str = None):
        """Initialize Uformer processor."""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default weights path if not provided
        if weights_path is None:
            # Try multiple possible locations
            possible_paths = [
                current_dir.parent.parent / "pretrained_models" / "uformer" / "Uformer_B.pth",
                uformer_dir / "deblurring" / "logs" / "GoPro" / "models" / "model_best.pth",
                uformer_dir / "logs" / "GoPro" / "models" / "model_best.pth",
                current_dir.parent.parent / "pretrained_models" / "uformer" / "uformer_deblur.pth"
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
        
        # Initialize model with Uformer parameters
        try:
            # Default Uformer_B configuration for deblurring (matching original implementation)
            from model import Uformer
            self.model = Uformer(
                img_size=128,  # Use training patch size
                embed_dim=32,
                win_size=8,
                token_projection='linear',
                token_mlp='leff',
                depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
                modulator=True,
                dd_in=3
            ).to(self.device)
            
            # Load weights if available
            if self.weights_path and os.path.exists(self.weights_path):
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                
                # Handle different checkpoint formats
                state_dict = None
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        # Assume the checkpoint is the state dict
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Handle DataParallel checkpoints (remove 'module.' prefix)
                if state_dict and any(key.startswith('module.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('module.'):
                            new_key = key[7:]  # Remove 'module.' prefix
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    state_dict = new_state_dict
                
                # Load the cleaned state dict
                self.model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded pre-trained weights")
            
            self.model.eval()
            
        except Exception as e:
            print(f"Error initializing Uformer: {e}")
            raise
    
    def predict_single_image(self, image_path: str) -> np.ndarray:
        """Process a single image."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor (CHW format)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        # Store original dimensions
        _, _, orig_h, orig_w = img_tensor.shape
        
        # Expand to square as done in original Uformer test (using factor=128)
        factor = 128.0
        max_dim = max(orig_h, orig_w)
        target_size = int(math.ceil(max_dim / factor) * factor)
        
        # Pad to square
        pad_h = target_size - orig_h
        pad_w = target_size - orig_w
        
        # Pad to square (pad equally on both sides when possible)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        img_tensor = torch.nn.functional.pad(
            img_tensor, 
            (pad_left, pad_right, pad_top, pad_bottom), 
            mode='reflect'
        )
        
        # Process with Uformer
        with torch.no_grad():
            restored = self.model(img_tensor)
            
            # Uformer might return multiple outputs, take the first one
            if isinstance(restored, (list, tuple)):
                restored = restored[0]
            
            # Ensure tensor is on CPU and has valid range
            restored = restored.squeeze(0).cpu()
            
            # Check for invalid values
            if torch.isnan(restored).any() or torch.isinf(restored).any():
                print("Warning: Invalid values detected in output, using input image")
                restored = img_tensor.squeeze(0).cpu()
        
        # Remove padding to restore original dimensions
        restored = restored[:, pad_top:pad_top+orig_h, pad_left:pad_left+orig_w]
        
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
    parser = argparse.ArgumentParser(description="Uformer Image Processing Module")
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
        output_dir = str(Path(__file__).parent.parent / "outputs" / "uformer")
    
    try:
        # Initialize processor
        print("Initializing Uformer...")
        processor = UformerProcessor(args.weights_path)
        
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