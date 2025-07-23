"""
Display manager for VAPOR project.
Handles grid layout, UI display, and screen management.
"""

import cv2
import numpy as np
import os


class DisplayManager:
    """Manager for displaying manipulated frames in a grid layout."""
    
    def __init__(self):
        """Initialize display manager."""
        pass
    
    def get_screen_dimensions(self):
        """
        Get screen dimensions for responsive display.
        
        Returns:
            tuple: (screen_width, screen_height)
        """
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
    
    def get_optimal_grid_size(self, num_items):
        """
        Calculate optimal grid size for given number of items.
        
        Args:
            num_items: Number of items to display
            
        Returns:
            tuple: (rows, cols)
        """
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
    
    def create_grid_image(self, manipulations, selected_names, video_path, 
                         current_frame, total_frames, fps, selected_manipulations, 
                         pipelines, frame_operations):
        """
        Create a single image with all manipulations arranged in a responsive grid.
        
        Args:
            manipulations: Dict of manipulation results
            selected_names: List of selected manipulation names
            video_path: Path to the video file
            current_frame: Current frame number
            total_frames: Total number of frames
            fps: Video FPS
            selected_manipulations: List of selected manipulation keys
            pipelines: List of pipeline strings
            frame_operations: FrameOperations instance
            
        Returns:
            numpy.ndarray: Grid image
        """
        num_items = len(selected_names)
        
        # Calculate optimal grid size
        rows, cols = self.get_optimal_grid_size(num_items)
        
        # Define target screen dimensions (get actual screen size)
        target_screen_width, target_screen_height = self.get_screen_dimensions()
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
        
        # Add header information
        self._add_header(grid_img, video_path, current_frame, total_frames, fps, 
                        selected_manipulations, pipelines, grid_width)
        
        # Display manipulations in grid
        self._add_manipulation_grid(grid_img, manipulations, selected_names, 
                                   rows, cols, padding, header_height, 
                                   max_image_dimension, cell_width, cell_height,
                                   frame_operations)
        
        # Add footer controls
        self._add_footer(grid_img, grid_width, grid_height)
        
        return grid_img
    
    def _add_header(self, grid_img, video_path, current_frame, total_frames, 
                   fps, selected_manipulations, pipelines, grid_width):
        """Add header information to grid image."""
        video_name = os.path.basename(video_path)
        header_text = f"Video: {video_name} | Frame: {current_frame + 1}/{total_frames} | FPS: {fps:.1f}"
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
        selected_keys = ''.join(selected_manipulations)
        pipeline_info = f" | Pipelines: {len(pipelines)}" if pipelines else ""
        current_selection = f"Selected: [{selected_keys}] ({len(selected_manipulations)} items){pipeline_info}"
        cv2.putText(grid_img, current_selection, (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 150, 0), font_thickness_small)
    
    def _add_manipulation_grid(self, grid_img, manipulations, selected_names, 
                              rows, cols, padding, header_height, max_image_dimension,
                              cell_width, cell_height, frame_operations):
        """Add manipulation images to grid."""
        for i, name in enumerate(selected_names):
            if i >= len(selected_names):
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
            display_width, display_height = self._calculate_display_size(
                img_aspect_ratio, max_image_dimension
            )
            
            # Calculate position within the cell (centered)
            cell_x = col * (cell_width + padding) + padding
            cell_y = row * cell_height + padding + header_height
            
            # Center the image within the cell
            x = cell_x + (cell_width - display_width) // 2
            y = cell_y + (max_image_dimension - display_height) // 2
            
            # Resize the image to exact display dimensions (maintains aspect ratio)
            img = frame_operations.normalize_image_for_display(
                manipulations[name], (display_height, display_width)
            )
            
            # Place the image
            grid_img[y:y+display_height, x:x+display_width] = img
            
            # Add title text
            self._add_image_title(grid_img, name, cell_x, cell_y + max_image_dimension + 25, 
                                 cell_width)
    
    def _calculate_display_size(self, img_aspect_ratio, max_image_dimension):
        """Calculate display size maintaining aspect ratio."""
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
        
        return display_width, display_height
    
    def _add_image_title(self, grid_img, name, cell_x, title_y, cell_width):
        """Add title text for each image."""
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
    
    def _add_footer(self, grid_img, grid_width, grid_height):
        """Add footer controls to grid image."""
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
