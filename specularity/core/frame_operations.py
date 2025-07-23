"""
Frame operations for specularity detection.
Contains all image manipulation and processing functions.
"""

import cv2
import numpy as np
from skimage import exposure
from skimage.filters import unsharp_mask


class FrameOperations:
    """Container for all frame manipulation operations."""
    
    def __init__(self):
        """Initialize frame operations with available manipulations."""
        self.available_manipulations = {
            '1': 'Original',
            '2': 'Grayscale', 
            '3': 'R Channel',
            '4': 'G Channel',
            '5': 'B Channel',
            '6': 'Histogram Normalization',
            '7': 'Grayscale + CLAHE',
            '8': 'Gamma Correction (γ=1.5)',
            '9': 'Bilateral Filtering',
            '0': 'Unsharp Masking',
            'a': 'Logarithmic Transformation'
        }
    
    def apply_single_operation(self, frame, key):
        """
        Apply a single operation to a frame.
        
        Args:
            frame: Input frame
            key: Operation key
            
        Returns:
            numpy.ndarray: Processed frame
        """
        try:
            if key == '1':  # Original
                return frame
            elif key == '2':  # Grayscale
                if len(frame.shape) == 3:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return frame
            elif key == '3':  # R Channel
                if len(frame.shape) == 3:
                    return frame[:, :, 2]
                return frame
            elif key == '4':  # G Channel
                if len(frame.shape) == 3:
                    return frame[:, :, 1]
                return frame
            elif key == '5':  # B Channel
                if len(frame.shape) == 3:
                    return frame[:, :, 0]
                return frame
            elif key == '6':  # Histogram Normalization
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                normalized = exposure.equalize_hist(gray)
                return (normalized * 255).astype(np.uint8)
            elif key == '7':  # CLAHE
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                return clahe.apply(gray)
            elif key == '8':  # Gamma Correction
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                corrected = exposure.adjust_gamma(gray, 1.5)
                return (corrected * 255).astype(np.uint8)
            elif key == '9':  # Bilateral Filtering
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                return cv2.bilateralFilter(gray, 15, 80, 80)
            elif key == '0':  # Unsharp Masking
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                unsharp = unsharp_mask(gray, radius=1, amount=1)
                return (unsharp * 255).astype(np.uint8)
            elif key == 'a':  # Logarithmic Transformation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                gray_float = gray.astype(np.float32)
                c = 255 / np.log(1 + np.max(gray_float))
                log_transform = c * np.log(1 + gray_float)
                return np.clip(log_transform, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Warning: Operation {key} failed: {e}")
            return frame
        
        return frame
    
    def apply_pipeline(self, frame, pipeline_keys):
        """
        Apply a pipeline of operations in sequence.
        
        Args:
            frame: Input frame
            pipeline_keys: List of operation keys
            
        Returns:
            tuple: (processed_frame, pipeline_name)
        """
        current_frame = frame.copy()
        pipeline_name_parts = []
        
        for key in pipeline_keys:
            if key not in self.available_manipulations:
                continue
                
            manipulation_name = self.available_manipulations[key]
            pipeline_name_parts.append(manipulation_name)
            current_frame = self.apply_single_operation(current_frame, key)
        
        pipeline_name = ' + '.join(pipeline_name_parts)
        return current_frame, pipeline_name
    
    def apply_all_manipulations(self, frame, selected_keys, pipelines, pipeline_manager):
        """
        Apply all selected manipulations to the frame.
        
        Args:
            frame: Input frame
            selected_keys: List of selected manipulation keys
            pipelines: List of pipeline strings
            pipeline_manager: PipelineManager instance
            
        Returns:
            dict: Dictionary of manipulation names to processed frames
        """
        from .content_detection import crop_to_content
        
        manipulations = {}
        
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Apply content cropping preprocessing to all operations
        cropped_frame = crop_to_content(frame)
        
        # Apply individual manipulations
        for key in selected_keys:
            if key in self.available_manipulations:
                name = self.available_manipulations[key]
                if key == '1':  # Original uses cropped frame
                    manipulations[name] = cropped_frame
                else:
                    manipulations[name] = self.apply_single_operation(cropped_frame, key)
        
        # Apply pipeline results
        for pipeline_str in pipelines:
            pipeline_keys = pipeline_manager.parse_pipeline(pipeline_str)
            pipeline_result, pipeline_name = self.apply_pipeline(cropped_frame, pipeline_keys)
            manipulations[f"Pipeline: {pipeline_name}"] = pipeline_result
        
        return manipulations
    
    def normalize_image_for_display(self, img, target_size):
        """
        Normalize image for display while preserving exact aspect ratio.
        
        Args:
            img: Input image
            target_size: (height, width) tuple
            
        Returns:
            numpy.ndarray: Normalized image
        """
        target_h, target_w = target_size
        
        # Ensure proper data type and range
        if img.dtype == np.float64 or img.dtype == np.float32:
            # Normalize to 0-1 range if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Convert to 3-channel if needed
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Use BGR for OpenCV consistency
        
        # Simply resize to exact target dimensions maintaining aspect ratio
        # This will resize the image to fit exactly within the specified dimensions
        resized = cv2.resize(img, (target_w, target_h))
        
        return resized
