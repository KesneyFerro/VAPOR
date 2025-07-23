"""
Pipeline management utilities for VAPOR project.
Contains functions for parsing and managing image processing pipelines.
"""


class PipelineManager:
    """Manager for image processing pipelines."""
    
    def __init__(self, available_manipulations):
        """
        Initialize pipeline manager.
        
        Args:
            available_manipulations: Dict of available manipulation operations
        """
        self.available_manipulations = available_manipulations
        self.pipelines = []
    
    def parse_pipeline(self, pipeline_str):
        """
        Parse pipeline string like '57' into ['5', '7'] or '5 7' into ['5', '7'].
        
        Args:
            pipeline_str: String representing the pipeline
            
        Returns:
            list: List of operation keys
        """
        pipeline_str = pipeline_str.strip()
        
        # If contains spaces, split by spaces
        if ' ' in pipeline_str:
            return pipeline_str.split()
        
        # Otherwise, treat each character as a separate operation
        return list(pipeline_str)
    
    def add_pipeline(self, pipeline_str):
        """
        Add a pipeline to the manager.
        
        Args:
            pipeline_str: String representing the pipeline
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        if all(c in self.available_manipulations for c in pipeline_str):
            if pipeline_str not in self.pipelines:
                self.pipelines.append(pipeline_str)
                return True
        return False
    
    def remove_pipeline(self, pipeline_str):
        """
        Remove a pipeline from the manager.
        
        Args:
            pipeline_str: String representing the pipeline
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        if pipeline_str in self.pipelines:
            self.pipelines.remove(pipeline_str)
            return True
        return False
    
    def clear_pipelines(self):
        """Clear all pipelines."""
        self.pipelines.clear()
    
    def get_pipeline_names(self):
        """
        Get formatted names for all pipelines.
        
        Returns:
            list: List of formatted pipeline names
        """
        names = []
        for pipeline_str in self.pipelines:
            pipeline_keys = self.parse_pipeline(pipeline_str)
            pipeline_names = [
                self.available_manipulations[k] 
                for k in pipeline_keys 
                if k in self.available_manipulations
            ]
            pipeline_name = f"Pipeline: {' + '.join(pipeline_names)}"
            names.append(pipeline_name)
        return names
    
    def process_commands(self, user_input):
        """
        Process pipeline commands like '+57 +68 -12'.
        
        Args:
            user_input: String with pipeline commands
            
        Returns:
            bool: True if any changes were made, False otherwise
        """
        if not user_input:
            return False
        
        if user_input.lower() == 'clear':
            self.clear_pipelines()
            return True
        
        # Parse multiple commands separated by spaces
        commands = user_input.split()
        changes_made = False
        
        for command in commands:
            if command.startswith('+'):
                # Add pipeline
                pipeline = command[1:]
                if self.add_pipeline(pipeline):
                    changes_made = True
            elif command.startswith('-'):
                # Remove pipeline
                pipeline = command[1:]
                if self.remove_pipeline(pipeline):
                    changes_made = True
        
        return changes_made
