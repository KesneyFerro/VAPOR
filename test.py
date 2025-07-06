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
        print("Press Enter for all manipulations, or type 'quick' for R,G,B channels only")
        
        user_input = input("Selection: ").strip().lower()
        
        if user_input == "":
            # Keep all selected (default)
            pass
        elif user_input == "quick":
            # Quick selection for R, G, B channels
            self.selected_manipulations = ['3', '4', '5']
        else:
            # Parse user input
            selected_keys = user_input.split()
            valid_keys = [key for key in selected_keys if key in self.available_manipulations]
            
            if valid_keys:
                self.selected_manipulations = valid_keys
            else:
                print("No valid selections found. Using all manipulations.")
        
        print(f"\nSelected {len(self.selected_manipulations)} manipulations:")
        for key in self.selected_manipulations:
            print(f"  {key}: {self.available_manipulations[key]}")
        print()
        
    def get_frame(self, frame_number):
        """Get a specific frame from the video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            # Keep BGR format for OpenCV display (don't convert to RGB)
            return frame
        return None
    
    def apply_manipulations(self, frame):
        """Apply all image manipulations to the frame"""
        manipulations = {}
        
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # 1. Original frame
        manipulations['Original'] = frame
        
        # 2. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        manipulations['Grayscale'] = gray
        
        # 3. R channel (BGR format, so R is index 2)
        r_channel = frame[:, :, 2]
        manipulations['R Channel'] = r_channel
        
        # 4. G channel  
        g_channel = frame[:, :, 1]
        manipulations['G Channel'] = g_channel
        
        # 5. B channel (BGR format, so B is index 0)
        b_channel = frame[:, :, 0]
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
        
        return manipulations
    
    def _normalize_image_for_display(self, img, target_size):
        """Normalize image for display in grid"""
        h, w = target_size
        
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
        
        # Ensure we have the right shape
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
            
        return img

    def _get_manipulation_names(self):
        """Get list of manipulation names based on selected keys"""
        return [self.available_manipulations[key] for key in self.selected_manipulations]
    
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

    def create_grid_image(self, manipulations):
        """Create a single image with all manipulations arranged in a responsive grid"""
        # Get image dimensions from the first manipulation
        sample_img = list(manipulations.values())[0]
        if len(sample_img.shape) == 3:
            h, w, _ = sample_img.shape
        else:
            h, w = sample_img.shape
        
        # Get selected manipulations
        selected_names = self._get_manipulation_names()
        num_items = len(selected_names)
        
        # Calculate optimal grid size
        rows, cols = self._get_optimal_grid_size(num_items)
        
        # Calculate grid dimensions with padding (adjusted for very large text)
        padding = 30
        text_height = 250  # Increased significantly for more padding between image and title
        header_height = 180  # Increased space for selection controls
        
        grid_width = cols * w + (cols + 1) * padding
        grid_height = rows * (h + text_height) + (rows + 1) * padding + header_height + 80  # Extra space for footer
        
        # Create the grid image
        grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Add header with video information
        video_name = os.path.basename(self.video_path)
        header_text = f"Video: {video_name} | Frame: {self.current_frame + 1}/{self.total_frames} | FPS: {self.fps:.1f}"
        cv2.putText(grid_img, header_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (128, 0, 0), 8)  # Navy blue header (BGR: 128,0,0)
        
        # Add selection controls
        selection_text = "Toggle: 1=Orig 2=Gray 3=R 4=G 5=B 6=Hist 7=CLAHE 8=Gamma 9=Bilat 0=Unsharp A=Log"
        cv2.putText(grid_img, selection_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 0), 3)  # Green selection text
        
        # Show currently selected
        selected_keys = ''.join(self.selected_manipulations)
        current_selection = f"Selected: [{selected_keys}] ({num_items} items)"
        cv2.putText(grid_img, current_selection, (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 150, 0), 4)  # Darker green
        
        # Display only selected manipulations
        for i, name in enumerate(selected_names):
            if i >= num_items:
                break
                
            # Skip if manipulation not available
            if name not in manipulations:
                continue
                
            row = i // cols
            col = i % cols
            
            # Calculate position (adjusted for header)
            x = col * (w + padding) + padding
            y = row * (h + text_height + padding) + padding + header_height
            
            # Get and normalize the image
            img = self._normalize_image_for_display(manipulations[name], (h, w))
            
            # Place the image
            grid_img[y:y+h, x:x+w] = img
            
            # Add larger title text with more padding
            title_y = y + h + 80  # Increased spacing significantly (was 50, now 80)
            cv2.putText(grid_img, name, (x, title_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 0, 0), 10)  # Keep your large font size
        
        # Add control instructions at the bottom
        control_info = "Controls: 'c'/Space=Next | 'v'=Prev | 'm'=+1sec | 'n'=-1sec | 1-9,0,A=Toggle | 'q'/Esc=Quit"
        cv2.putText(grid_img, control_info, (20, grid_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 10)  # Red control text (BGR: 0,0,255)
        
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
        print("Unknown key pressed. Use 'c' for next, 'v' for previous, 'm' for +1sec, 'n' for -1sec, 1-9,0,A for toggle, 'q' to quit.")
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
        print("\nControls:")
        print("  Navigation:")
        print("    'c' or Space: Next frame")
        print("    'v': Previous frame")
        print("    'n': Skip forward 1 second")
        print("    'm': Skip backward 1 second")
        print("  Selection:")
        print("    1-9, 0, A: Toggle manipulations on/off")
        print("  Other:")
        print("    'q' or Esc: Quit")
        print("    Mouse wheel: Zoom in/out")

    def process_video(self):
        """Main processing loop using OpenCV for display"""
        window_name = "Video Frame Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
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