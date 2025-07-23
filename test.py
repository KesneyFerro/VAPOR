import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import exposure
from skimage.filters import unsharp_mask
import os
import math

class VideoFrameProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Available manipulations with their keys
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
        
        # Default selected manipulations (all)
        self.selected_manipulations = list(self.available_manipulations.keys())
        
        # Pipeline functionality
        self.pipelines = []  # List of pipeline strings like ['57', '5', '7']
        
        # Ask user for initial selection
        self._get_initial_selection()
        
        print(f"Video loaded: {os.path.basename(video_path)}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print("Press 'c' to go to next frame, 'q' to quit")
        
    def _get_initial_selection(self):
        """Get initial selection from user"""
        print("\nAvailable manipulations:")
        for key, name in self.available_manipulations.items():
            print(f"  {key}: {name}")
        
        print("\nSelect manipulations to display:")
        print("Enter keys separated by spaces (e.g., '1 2 3 4 5' for Original, Grayscale, R, G, B)")
        print("For pipelines, use combined keys (e.g., '57' for B Channel + CLAHE)")
        print("You can mix individual and pipeline operations (e.g., '5 7 57')")
        print("Press Enter for all individual manipulations, or type 'quick' for R,G,B channels only")
        
        user_input = input("Selection: ").strip().lower()
        
        if user_input == "":
            # Keep all individual selected (default)
            pass
        elif user_input == "quick":
            # Quick selection for R, G, B channels
            self.selected_manipulations = ['3', '4', '5']
        else:
            # Parse user input for both individual operations and pipelines
            self.selected_manipulations = []
            self.pipelines = []
            
            input_parts = user_input.split()
            
            for part in input_parts:
                if len(part) == 1 and part in self.available_manipulations:
                    # Single operation
                    self.selected_manipulations.append(part)
                elif len(part) > 1:
                    # Check if all characters are valid operations
                    if all(c in self.available_manipulations for c in part):
                        # Pipeline operation
                        self.pipelines.append(part)
                    else:
                        print(f"Warning: Invalid pipeline '{part}' - contains invalid operations")
                else:
                    print(f"Warning: Invalid operation '{part}'")
            
            # Ensure at least something is selected
            if not self.selected_manipulations and not self.pipelines:
                print("No valid selections found. Using all individual manipulations.")
                self.selected_manipulations = list(self.available_manipulations.keys())
        
        print(f"\nSelected {len(self.selected_manipulations)} individual manipulations:")
        for key in self.selected_manipulations:
            print(f"  {key}: {self.available_manipulations[key]}")
        
        if self.pipelines:
            print(f"\nConfigured {len(self.pipelines)} pipelines:")
            for pipeline in self.pipelines:
                pipeline_keys = self.parse_pipeline(pipeline)
                pipeline_names = [self.available_manipulations[k] for k in pipeline_keys if k in self.available_manipulations]
                print(f"  {pipeline}: {' + '.join(pipeline_names)}")
        print()
        
    def get_frame(self, frame_number):
        """Get a specific frame from the video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            # Keep BGR format for OpenCV display (don't convert to RGB)
            return frame
        return None
    
    def find_content_bounds(self, frame):
        """Find the bounds of non-black content from center edges"""
        # Convert to grayscale for edge detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        h, w = gray.shape
        
        # Find first non-black pixel from each direction (from center edges)
        # Top edge going down
        top_bound = 0
        for y in range(h):
            if gray[y, w//2] > 0:  # Non-black pixel
                top_bound = y
                break
        
        # Bottom edge going up
        bottom_bound = h - 1
        for y in range(h-1, -1, -1):
            if gray[y, w//2] > 0:  # Non-black pixel
                bottom_bound = y
                break
        
        # Left edge going right
        left_bound = 0
        for x in range(w):
            if gray[h//2, x] > 0:  # Non-black pixel
                left_bound = x
                break
        
        # Right edge going left
        right_bound = w - 1
        for x in range(w-1, -1, -1):
            if gray[h//2, x] > 0:  # Non-black pixel
                right_bound = x
                break
        
        # Add small padding to ensure we don't crop too aggressively
        padding = 5
        top_bound = max(0, top_bound - padding)
        bottom_bound = min(h - 1, bottom_bound + padding)
        left_bound = max(0, left_bound - padding)
        right_bound = min(w - 1, right_bound + padding)
        
        return top_bound, bottom_bound, left_bound, right_bound
    
    def crop_to_content(self, frame):
        """Crop frame to content bounds"""
        top, bottom, left, right = self.find_content_bounds(frame)
        
        # Crop the frame
        if len(frame.shape) == 3:
            cropped = frame[top:bottom+1, left:right+1, :]
        else:
            cropped = frame[top:bottom+1, left:right+1]
        
        return cropped
    
    def parse_pipeline(self, pipeline_str):
        """Parse pipeline string like '57' into ['5', '7'] or '5 7' into ['5', '7']"""
        pipeline_str = pipeline_str.strip()
        
        # If contains spaces, split by spaces
        if ' ' in pipeline_str:
            return pipeline_str.split()
        
        # Otherwise, treat each character as a separate operation
        return list(pipeline_str)
    
    def _apply_single_operation(self, frame, key):
        """Apply a single operation to a frame"""
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
        """Apply a pipeline of operations in sequence"""
        current_frame = frame.copy()
        pipeline_name_parts = []
        
        for key in pipeline_keys:
            if key not in self.available_manipulations:
                continue
                
            manipulation_name = self.available_manipulations[key]
            pipeline_name_parts.append(manipulation_name)
            current_frame = self._apply_single_operation(current_frame, key)
        
        pipeline_name = ' + '.join(pipeline_name_parts)
        return current_frame, pipeline_name

    def apply_manipulations(self, frame):
        """Apply all image manipulations to the frame"""
        manipulations = {}
        
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Apply content cropping preprocessing to all operations
        cropped_frame = self.crop_to_content(frame)
        
        # 1. Original frame (cropped)
        manipulations['Original'] = cropped_frame
        
        # 2. Grayscale
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        manipulations['Grayscale'] = gray
        
        # 3. R channel (BGR format, so R is index 2)
        r_channel = cropped_frame[:, :, 2]
        manipulations['R Channel'] = r_channel
        
        # 4. G channel  
        g_channel = cropped_frame[:, :, 1]
        manipulations['G Channel'] = g_channel
        
        # 5. B channel (BGR format, so B is index 0)
        b_channel = cropped_frame[:, :, 0]
        manipulations['B Channel'] = b_channel
        
        # 6. Histogram Normalization (Global)
        try:
            hist_norm = exposure.equalize_hist(gray)
            # Convert back to uint8
            hist_norm = (hist_norm * 255).astype(np.uint8)
            manipulations['Histogram Normalization'] = hist_norm
        except Exception as e:
            print(f"Warning: Histogram normalization failed: {e}")
            manipulations['Histogram Normalization'] = gray
        
        # 7. Grayscale + CLAHE (Contrast Limited Adaptive Histogram Equalization)
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            manipulations['Grayscale + CLAHE'] = clahe_img
        except Exception as e:
            print(f"Warning: CLAHE failed: {e}")
            manipulations['Grayscale + CLAHE'] = gray
        
        # 8. Gamma Correction
        try:
            gamma = 1.5
            gamma_corrected = exposure.adjust_gamma(gray, gamma)
            # Convert back to uint8
            gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
            manipulations['Gamma Correction (γ=1.5)'] = gamma_corrected
        except Exception as e:
            print(f"Warning: Gamma correction failed: {e}")
            manipulations['Gamma Correction (γ=1.5)'] = gray
        
        # 9. Bilateral Filtering
        try:
            bilateral = cv2.bilateralFilter(gray, 15, 80, 80)
            manipulations['Bilateral Filtering'] = bilateral
        except Exception as e:
            print(f"Warning: Bilateral filtering failed: {e}")
            manipulations['Bilateral Filtering'] = gray
        
        # 10. Unsharp Masking
        try:
            unsharp = unsharp_mask(gray, radius=1, amount=1)
            # Convert back to uint8
            unsharp = (unsharp * 255).astype(np.uint8)
            manipulations['Unsharp Masking'] = unsharp
        except Exception as e:
            print(f"Warning: Unsharp masking failed: {e}")
            manipulations['Unsharp Masking'] = gray
        
        # 11. Logarithmic Transformation for lighting corrections
        try:
            # Avoid log(0) by adding 1 to all pixels
            gray_float = gray.astype(np.float32)
            c = 255 / np.log(1 + np.max(gray_float))
            log_transform = c * np.log(1 + gray_float)
            log_transform = np.clip(log_transform, 0, 255).astype(np.uint8)
            manipulations['Logarithmic Transformation'] = log_transform
        except Exception as e:
            print(f"Warning: Logarithmic transformation failed: {e}")
            manipulations['Logarithmic Transformation'] = gray
        
        # Add pipeline results
        for pipeline_str in self.pipelines:
            pipeline_keys = self.parse_pipeline(pipeline_str)
            pipeline_result, pipeline_name = self.apply_pipeline(cropped_frame, pipeline_keys)
            manipulations[f"Pipeline: {pipeline_name}"] = pipeline_result
        
        return manipulations
    
    def _normalize_image_for_display(self, img, target_size):
        """Normalize image for display while preserving exact aspect ratio"""
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

    def _get_manipulation_names(self):
        """Get list of manipulation names based on selected keys and pipelines"""
        names = []
        
        # Add individual manipulations
        for key in self.selected_manipulations:
            names.append(self.available_manipulations[key])
        
        # Add pipeline manipulations
        for pipeline_str in self.pipelines:
            pipeline_keys = self.parse_pipeline(pipeline_str)
            pipeline_names = [self.available_manipulations[k] for k in pipeline_keys if k in self.available_manipulations]
            pipeline_name = f"Pipeline: {' + '.join(pipeline_names)}"
            names.append(pipeline_name)
        
        return names
    
    def _get_optimal_grid_size(self, num_items):
        """Calculate optimal grid size for given number of items"""
        if num_items <= 1:
            return 1, 1
        elif num_items <= 2:
            return 1, 2
        elif num_items <= 4:
            return 2, 2
        elif num_items <= 6:
            return 2, 3
        elif num_items <= 9:
            return 3, 3
        elif num_items <= 12:
            return 3, 4
        else:
            # For more than 12 items, use 4 columns
            rows = (num_items + 3) // 4
            return rows, 4

    def _get_screen_dimensions(self):
        """Get screen dimensions for responsive display"""
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            return screen_width, screen_height
        except Exception:
            # Fallback to common resolution
            return 1920, 1080
    
    def create_grid_image(self, manipulations):
        """Create a single image with all manipulations arranged in a responsive grid"""
        # Get selected manipulations
        selected_names = self._get_manipulation_names()
        num_items = len(selected_names)
        
        # Calculate optimal grid size
        rows, cols = self._get_optimal_grid_size(num_items)
        
        # Define target screen dimensions (get actual screen size)
        target_screen_width, target_screen_height = self._get_screen_dimensions()
        # Use 90% of screen size to leave room for window decorations
        target_screen_width = int(target_screen_width * 0.9)
        target_screen_height = int(target_screen_height * 0.9)
        
        # Calculate dimensions with responsive sizing that adapts to content
        padding = 15  # Reduced padding for more space
        text_height = 80  # Increased space for multi-line text
        header_height = 120
        footer_height = 60
        
        # Calculate available space for images
        available_width = target_screen_width - (cols + 1) * padding
        available_height = target_screen_height - header_height - footer_height - (rows * text_height) - (rows + 1) * padding
        
        # Calculate maximum cell size that fits the grid
        max_cell_width = available_width // cols
        max_cell_height = available_height // rows
        
        # Use a more conservative maximum image size to prevent stretching
        max_image_dimension = min(max_cell_width - 20, max_cell_height - 20, 350)  # Reduced cap and added margin
        
        # Calculate actual grid dimensions
        cell_width = max_image_dimension + 20  # Add margin around images
        cell_height = max_image_dimension + text_height
        
        grid_width = cols * cell_width + (cols + 1) * padding
        grid_height = rows * cell_height + (rows + 1) * padding + header_height + footer_height
        
        # Create the grid image
        grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add header with video information
        video_name = os.path.basename(self.video_path)
        header_text = f"Video: {video_name} | Frame: {self.current_frame + 1}/{self.total_frames} | FPS: {self.fps:.1f}"
        # Scale font size based on grid width but ensure readability
        font_scale = max(1.2, min(2.5, grid_width / 800))
        font_thickness = max(3, int(font_scale * 2))
        cv2.putText(grid_img, header_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 128), font_thickness)
        
        # Add selection controls
        selection_text = "Toggle: 1=Orig 2=Gray 3=R 4=G 5=B 6=Hist 7=CLAHE 8=Gamma 9=Bilat 0=Unsharp A=Log | P=Pipelines"
        font_scale_small = max(0.8, min(1.4, grid_width / 1000))
        font_thickness_small = max(2, int(font_scale_small * 2))
        cv2.putText(grid_img, selection_text, (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 120, 0), font_thickness_small)
        
        # Show currently selected
        selected_keys = ''.join(self.selected_manipulations)
        pipeline_info = f" | Pipelines: {len(self.pipelines)}" if self.pipelines else ""
        current_selection = f"Selected: [{selected_keys}] ({len(self.selected_manipulations)} items){pipeline_info}"
        cv2.putText(grid_img, current_selection, (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 150, 0), font_thickness_small)
        
        # Display only selected manipulations
        for i, name in enumerate(selected_names):
            if i >= num_items:
                break
                
            # Skip if manipulation not available
            if name not in manipulations:
                continue
                
            row = i // cols
            col = i % cols
            
            # Get the actual image to determine its individual aspect ratio
            actual_img = manipulations[name]
            if len(actual_img.shape) == 3:
                actual_h, actual_w, _ = actual_img.shape
            else:
                actual_h, actual_w = actual_img.shape
            
            # Calculate the aspect ratio for this specific image
            img_aspect_ratio = actual_w / actual_h
            
            # Calculate display size that fits within the cell while preserving aspect ratio
            max_width = max_image_dimension
            max_height = max_image_dimension
            
            if img_aspect_ratio > 1.0:  # Wider than tall
                display_width = max_width
                display_height = int(max_width / img_aspect_ratio)
                if display_height > max_height:
                    display_height = max_height
                    display_width = int(max_height * img_aspect_ratio)
            else:  # Taller than wide or square
                display_height = max_height
                display_width = int(max_height * img_aspect_ratio)
                if display_width > max_width:
                    display_width = max_width
                    display_height = int(max_width / img_aspect_ratio)
            
            # Ensure minimum readable size
            min_size = 120
            if display_width < min_size or display_height < min_size:
                if img_aspect_ratio > 1.0:
                    display_width = min_size
                    display_height = int(min_size / img_aspect_ratio)
                else:
                    display_height = min_size
                    display_width = int(min_size * img_aspect_ratio)
            
            # Calculate position within the cell (centered)
            cell_x = col * (cell_width + padding) + padding
            cell_y = row * cell_height + padding + header_height
            
            # Center the image within the cell
            x = cell_x + (cell_width - display_width) // 2
            y = cell_y + (max_image_dimension - display_height) // 2
            
            # Resize the image to exact display dimensions (maintains aspect ratio)
            img = self._normalize_image_for_display(manipulations[name], (display_height, display_width))
            
            # Place the image
            grid_img[y:y+display_height, x:x+display_width] = img
            
            # Add title text with responsive font size and better readability
            title_y = cell_y + max_image_dimension + 25
            
            # Wrap long names by breaking at common separators
            display_name = name
            if len(name) > 25:  # If name is too long
                # Try to break at common separators
                if ' + ' in name:
                    parts = name.split(' + ')
                    if len(parts) == 2:
                        display_name = parts[0] + '\n+ ' + parts[1]
                elif ': ' in name and 'Pipeline' in name:
                    display_name = name.replace('Pipeline: ', 'Pipeline:\n')
                elif len(name) > 30:
                    # Force wrap very long names
                    mid = len(name) // 2
                    space_pos = name.find(' ', mid)
                    if space_pos != -1:
                        display_name = name[:space_pos] + '\n' + name[space_pos+1:]
            
            # Calculate font size based on cell width and text length
            base_font_scale = min(0.5, cell_width / 400)
            if len(display_name) > 20:
                title_font_scale = max(0.4, base_font_scale * 0.7)
            else:
                title_font_scale = max(0.5, base_font_scale)
            
            title_thickness = max(1, int(title_font_scale * 2))
            
            # Handle multi-line text
            lines = display_name.split('\n')
            line_height = int(title_font_scale * 30)
            total_text_height = len(lines) * line_height
            
            # Calculate background size for all lines
            max_text_width = 0
            for line in lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness)[0]
                max_text_width = max(max_text_width, text_size[0])
            
            # Center the text block
            text_start_x = cell_x + (cell_width - max_text_width) // 2
            
            # Add background rectangle for better text readability
            bg_padding = 3
            cv2.rectangle(grid_img, 
                         (text_start_x - bg_padding, title_y - line_height - bg_padding), 
                         (text_start_x + max_text_width + bg_padding, title_y + total_text_height - line_height + bg_padding), 
                         (255, 255, 255), -1)
            cv2.rectangle(grid_img, 
                         (text_start_x - bg_padding, title_y - line_height - bg_padding), 
                         (text_start_x + max_text_width + bg_padding, title_y + total_text_height - line_height + bg_padding), 
                         (0, 0, 0), 1)
            
            # Draw each line
            for i, line in enumerate(lines):
                line_y = title_y + (i * line_height)
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness)[0]
                line_x = cell_x + (cell_width - text_size[0]) // 2  # Center each line
                cv2.putText(grid_img, line, (line_x, line_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (0, 0, 0), title_thickness)
        
        # Add control instructions at the bottom with better readability
        control_info = "Controls: 'c'/Space=Next | 'v'=Prev | 'm'=+1sec | 'n'=-1sec | 1-9,0,A=Toggle | 'p'=Pipelines | 'q'/Esc=Quit"
        footer_font_scale = max(0.8, min(1.2, grid_width / 1200))
        footer_thickness = max(2, int(footer_font_scale * 2))
        
        # Add background for footer text
        footer_y = grid_height - 35
        footer_text_size = cv2.getTextSize(control_info, cv2.FONT_HERSHEY_SIMPLEX, footer_font_scale, footer_thickness)[0]
        cv2.rectangle(grid_img, (15, footer_y - footer_text_size[1] - 5), (footer_text_size[0] + 25, footer_y + 5), (255, 255, 255), -1)
        cv2.rectangle(grid_img, (15, footer_y - footer_text_size[1] - 5), (footer_text_size[0] + 25, footer_y + 5), (0, 0, 255), 2)
        
        cv2.putText(grid_img, control_info, (20, footer_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, footer_font_scale, (0, 0, 255), footer_thickness)
        
        return grid_img
    
    def _handle_frame_navigation(self, key):
        """Handle frame navigation keys"""
        if key == ord('c') or key == 32:  # 'c' or Space
            if self.current_frame < self.total_frames - 1:
                self.current_frame += 1
                return 'next'
            else:
                print("Already at the last frame!")
                return 'stay'
        elif key == ord('v'):  # 'v' for previous
            if self.current_frame > 0:
                self.current_frame -= 1
                return 'prev'
            else:
                print("Already at the first frame!")
                return 'stay'
        return None

    def _handle_time_navigation(self, key):
        """Handle time-based navigation keys"""
        frames_per_second = int(self.fps)
        
        if key == ord('m'):  # 'm' for next second
            new_frame = min(self.current_frame + frames_per_second, self.total_frames - 1)
            if new_frame != self.current_frame:
                self.current_frame = new_frame
                return 'next_second'
            else:
                print("Already at or near the last frame!")
                return 'stay'
        elif key == ord('n'):  # 'n' for previous second
            new_frame = max(self.current_frame - frames_per_second, 0)
            if new_frame != self.current_frame:
                self.current_frame = new_frame
                return 'prev_second'
            else:
                print("Already at or near the first frame!")
                return 'stay'
        return None

    def _handle_selection_toggle(self, key):
        """Handle selection toggle keys"""
        key_char = chr(key).lower()
        
        if key_char in self.available_manipulations:
            if key_char in self.selected_manipulations:
                # Remove from selection
                self.selected_manipulations.remove(key_char)
                print(f"Removed: {self.available_manipulations[key_char]}")
            else:
                # Add to selection
                self.selected_manipulations.append(key_char)
                print(f"Added: {self.available_manipulations[key_char]}")
            
            # Ensure at least one manipulation is selected
            if not self.selected_manipulations:
                self.selected_manipulations = ['1']  # Default to Original
                print("At least one manipulation must be selected. Added Original.")
            
            return 'toggle'
        
        return None

    def _handle_key_press(self, key):
        """Handle key press events"""
        if key == ord('q') or key == 27:  # 'q' or Esc
            return 'quit'
        
        # Handle pipeline input (p key)
        if key == ord('p'):
            return self._handle_pipeline_input()
        
        # Try selection toggle first
        action = self._handle_selection_toggle(key)
        if action:
            return action
        
        # Try frame navigation
        action = self._handle_frame_navigation(key)
        if action:
            return action
        
        # Try time navigation
        action = self._handle_time_navigation(key)
        if action:
            return action
        
        # Unknown key
        print("Unknown key pressed. Use 'c' for next, 'v' for previous, 'm' for +1sec, 'n' for -1sec, 1-9,0,A for toggle, 'p' for pipeline management, 'q' to quit.")
        return 'stay'
    
    def _handle_pipeline_input(self):
        """Handle pipeline input from user"""
        print("\nPipeline Management:")
        print("Current pipelines:", self.pipelines if self.pipelines else "None")
        print("Enter pipeline commands:")
        print("  Add pipeline: '+57' (adds B Channel + CLAHE)")
        print("  Remove pipeline: '-57' (removes that pipeline)")
        print("  Multiple operations: '+57 +68 -12' (adds 57, 68 and removes 12)")
        print("  Remove multiple: '-1 -2 -3 -4 -5' (removes multiple pipelines)")
        print("  Clear all: 'clear'")
        print("  Cancel: just press Enter")
        
        user_input = input("Pipeline command: ").strip()
        
        if not user_input:
            return 'stay'
        
        if user_input.lower() == 'clear':
            self.pipelines = []
            print("All pipelines cleared.")
            return 'update'
        
        # Parse multiple commands separated by spaces
        commands = user_input.split()
        changes_made = False
        
        for command in commands:
            if command.startswith('+'):
                # Add pipeline
                pipeline = command[1:]
                if all(c in self.available_manipulations for c in pipeline):
                    if pipeline not in self.pipelines:
                        self.pipelines.append(pipeline)
                        pipeline_keys = self.parse_pipeline(pipeline)
                        pipeline_names = [self.available_manipulations[k] for k in pipeline_keys]
                        print(f"Added pipeline: {pipeline} ({' + '.join(pipeline_names)})")
                        changes_made = True
                    else:
                        print(f"Pipeline {pipeline} already exists.")
                else:
                    print(f"Invalid pipeline {pipeline} - contains invalid operations.")
                    
            elif command.startswith('-'):
                # Remove pipeline
                pipeline = command[1:]
                if pipeline in self.pipelines:
                    self.pipelines.remove(pipeline)
                    pipeline_keys = self.parse_pipeline(pipeline)
                    pipeline_names = [self.available_manipulations[k] for k in pipeline_keys if k in self.available_manipulations]
                    print(f"Removed pipeline: {pipeline} ({' + '.join(pipeline_names)})")
                    changes_made = True
                else:
                    print(f"Pipeline {pipeline} not found.")
            else:
                print(f"Invalid command '{command}'. Use +pipeline or -pipeline format.")
        
        if changes_made:
            print(f"\nUpdated pipelines: {self.pipelines if self.pipelines else 'None'}")
            return 'update'
        else:
            return 'stay'

    def _print_instructions(self):
        """Print usage instructions"""
        print(f"Video loaded: {os.path.basename(self.video_path)}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print("\nAvailable Manipulations:")
        for key, name in self.available_manipulations.items():
            status = "✓" if key in self.selected_manipulations else " "
            print(f"  {key}: [{status}] {name}")
        
        if self.pipelines:
            print("\nActive Pipelines:")
            for pipeline in self.pipelines:
                pipeline_keys = self.parse_pipeline(pipeline)
                pipeline_names = [self.available_manipulations[k] for k in pipeline_keys if k in self.available_manipulations]
                print(f"  {pipeline}: {' + '.join(pipeline_names)}")
        
        print("\nControls:")
        print("  Navigation:")
        print("    'c' or Space: Next frame")
        print("    'v': Previous frame")
        print("    'n': Skip forward 1 second")
        print("    'm': Skip backward 1 second")
        print("  Selection:")
        print("    1-9, 0, A: Toggle manipulations on/off")
        print("    'p': Manage pipelines (+pipeline, -pipeline, multiple commands, clear)")
        print("  Other:")
        print("    'q' or Esc: Quit")
        print("    Mouse wheel: Zoom in/out")

    def process_video(self):
        """Main processing loop using OpenCV for display"""
        window_name = "Video Frame Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # Get screen dimensions and set window size
        screen_width, screen_height = self._get_screen_dimensions()
        cv2.resizeWindow(window_name, int(screen_width * 0.9), int(screen_height * 0.9))
        
        self._print_instructions()
        
        while self.current_frame < self.total_frames:
            # Get current frame
            frame = self.get_frame(self.current_frame)
            if frame is None:
                print(f"Could not read frame {self.current_frame}")
                break
            
            # Apply manipulations
            manipulations = self.apply_manipulations(frame)
            
            # Create grid image
            grid_img = self.create_grid_image(manipulations)
            
            # Display the grid
            cv2.imshow(window_name, grid_img)
            
            # Wait for key press and handle it
            key = cv2.waitKey(0) & 0xFF
            action = self._handle_key_press(key)
            
            if action == 'quit':
                break
            elif action == 'update':
                # Re-display current frame with updated settings
                continue
        
        cv2.destroyAllWindows()
        self.cap.release()
        print("Video processing completed!")

def main():
    # Get video path from user
    video_path = input("Enter the path to your video file: ").strip().strip('"')
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    try:
        processor = VideoFrameProcessor(video_path)
        processor.process_video()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()