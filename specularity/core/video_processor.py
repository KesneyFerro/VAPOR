"""
Main video frame processor for VAPOR project.
Core class for handling video processing and user interaction.
"""

import cv2
import os
from .frame_operations import FrameOperations
from .display_manager import DisplayManager
from ..utils.pipeline_manager import PipelineManager


class VideoFrameProcessor:
    """Main class for processing video frames and detecting specularity."""
    
    def __init__(self, video_path):
        """
        Initialize video frame processor.
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize components
        self.frame_operations = FrameOperations()
        self.display_manager = DisplayManager()
        self.pipeline_manager = PipelineManager(self.frame_operations.available_manipulations)
        
        # Default selected manipulations (all)
        self.selected_manipulations = list(self.frame_operations.available_manipulations.keys())
        
        # Ask user for initial selection
        self._get_initial_selection()
        
        print(f"Video loaded: {os.path.basename(video_path)}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print("Press 'c' to go to next frame, 'q' to quit")
    
    def _get_initial_selection(self):
        """Get initial selection from user."""
        available_manipulations = self.frame_operations.available_manipulations
        
        print("\nAvailable manipulations:")
        for key, name in available_manipulations.items():
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
            
            input_parts = user_input.split()
            
            for part in input_parts:
                if len(part) == 1 and part in available_manipulations:
                    # Single operation
                    self.selected_manipulations.append(part)
                elif len(part) > 1:
                    # Check if all characters are valid operations
                    if all(c in available_manipulations for c in part):
                        # Pipeline operation
                        self.pipeline_manager.add_pipeline(part)
                    else:
                        print(f"Warning: Invalid pipeline '{part}' - contains invalid operations")
                else:
                    print(f"Warning: Invalid operation '{part}'")
            
            # Ensure at least something is selected
            if not self.selected_manipulations and not self.pipeline_manager.pipelines:
                print("No valid selections found. Using all individual manipulations.")
                self.selected_manipulations = list(available_manipulations.keys())
        
        print(f"\nSelected {len(self.selected_manipulations)} individual manipulations:")
        for key in self.selected_manipulations:
            print(f"  {key}: {available_manipulations[key]}")
        
        if self.pipeline_manager.pipelines:
            print(f"\nConfigured {len(self.pipeline_manager.pipelines)} pipelines:")
            for pipeline in self.pipeline_manager.pipelines:
                pipeline_keys = self.pipeline_manager.parse_pipeline(pipeline)
                pipeline_names = [available_manipulations[k] for k in pipeline_keys if k in available_manipulations]
                print(f"  {pipeline}: {' + '.join(pipeline_names)}")
        print()
    
    def get_frame(self, frame_number):
        """
        Get a specific frame from the video.
        
        Args:
            frame_number: Frame number to retrieve
            
        Returns:
            numpy.ndarray: Frame data or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            # Keep BGR format for OpenCV display (don't convert to RGB)
            return frame
        return None
    
    def apply_manipulations(self, frame):
        """
        Apply all image manipulations to the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            dict: Dictionary of manipulation results
        """
        return self.frame_operations.apply_all_manipulations(
            frame, self.selected_manipulations, 
            self.pipeline_manager.pipelines, self.pipeline_manager
        )
    
    def _get_manipulation_names(self):
        """
        Get list of manipulation names based on selected keys and pipelines.
        
        Returns:
            list: List of manipulation names
        """
        names = []
        
        # Add individual manipulations
        for key in self.selected_manipulations:
            names.append(self.frame_operations.available_manipulations[key])
        
        # Add pipeline manipulations
        names.extend(self.pipeline_manager.get_pipeline_names())
        
        return names
    
    def create_grid_image(self, manipulations):
        """
        Create a single image with all manipulations arranged in a responsive grid.
        
        Args:
            manipulations: Dictionary of manipulation results
            
        Returns:
            numpy.ndarray: Grid image
        """
        selected_names = self._get_manipulation_names()
        
        return self.display_manager.create_grid_image(
            manipulations, selected_names, self.video_path,
            self.current_frame, self.total_frames, self.fps,
            self.selected_manipulations, self.pipeline_manager.pipelines,
            self.frame_operations
        )
    
    def _handle_frame_navigation(self, key):
        """Handle frame navigation keys."""
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
        """Handle time-based navigation keys."""
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
        """Handle selection toggle keys."""
        key_char = chr(key).lower()
        available_manipulations = self.frame_operations.available_manipulations
        
        if key_char in available_manipulations:
            if key_char in self.selected_manipulations:
                # Remove from selection
                self.selected_manipulations.remove(key_char)
                print(f"Removed: {available_manipulations[key_char]}")
            else:
                # Add to selection
                self.selected_manipulations.append(key_char)
                print(f"Added: {available_manipulations[key_char]}")
            
            # Ensure at least one manipulation is selected
            if not self.selected_manipulations:
                self.selected_manipulations = ['1']  # Default to Original
                print("At least one manipulation must be selected. Added Original.")
            
            return 'toggle'
        
        return None

    def _handle_key_press(self, key):
        """Handle key press events."""
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
        """Handle pipeline input from user."""
        print("\nPipeline Management:")
        print("Current pipelines:", self.pipeline_manager.pipelines if self.pipeline_manager.pipelines else "None")
        print("Enter pipeline commands:")
        print("  Add pipeline: '+57' (adds B Channel + CLAHE)")
        print("  Remove pipeline: '-57' (removes that pipeline)")
        print("  Multiple operations: '+57 +68 -12' (adds 57, 68 and removes 12)")
        print("  Remove multiple: '-1 -2 -3 -4 -5' (removes multiple pipelines)")
        print("  Clear all: 'clear'")
        print("  Cancel: just press Enter")
        
        user_input = input("Pipeline command: ").strip()
        
        if self.pipeline_manager.process_commands(user_input):
            print(f"\nUpdated pipelines: {self.pipeline_manager.pipelines if self.pipeline_manager.pipelines else 'None'}")
            return 'update'
        else:
            return 'stay'

    def _print_instructions(self):
        """Print usage instructions."""
        available_manipulations = self.frame_operations.available_manipulations
        
        print(f"Video loaded: {os.path.basename(self.video_path)}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print("\nAvailable Manipulations:")
        for key, name in available_manipulations.items():
            status = "✓" if key in self.selected_manipulations else " "
            print(f"  {key}: [{status}] {name}")
        
        if self.pipeline_manager.pipelines:
            print("\nActive Pipelines:")
            for pipeline in self.pipeline_manager.pipelines:
                pipeline_keys = self.pipeline_manager.parse_pipeline(pipeline)
                pipeline_names = [available_manipulations[k] for k in pipeline_keys if k in available_manipulations]
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
        """Main processing loop using OpenCV for display."""
        window_name = "Video Frame Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # Get screen dimensions and set window size
        screen_width, screen_height = self.display_manager.get_screen_dimensions()
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
