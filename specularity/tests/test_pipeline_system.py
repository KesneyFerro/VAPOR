#!/usr/bin/env python3
"""
Test script for pipeline management functionality.
Tests the enhanced pipeline management system.
"""

import sys
from pathlib import Path

# Add parent directories to path to import specularity modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from specularity.utils.pipeline_manager import PipelineManager

def test_pipeline_parsing():
    """Test the pipeline command parsing"""
    
    class MockProcessor:
        def __init__(self):
            self.pipelines = []
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
        
        def parse_pipeline(self, pipeline_str):
            """Parse pipeline string like '57' into ['5', '7'] or '5 7' into ['5', '7']"""
            pipeline_str = pipeline_str.strip()
            
            # If contains spaces, split by spaces
            if ' ' in pipeline_str:
                return pipeline_str.split()
            
            # Otherwise, treat each character as a separate operation
            return list(pipeline_str)
        
        def process_pipeline_commands(self, user_input):
            """Process pipeline commands like the real handler would"""
            if not user_input:
                return False
            
            if user_input.lower() == 'clear':
                self.pipelines = []
                print("All pipelines cleared.")
                return True
            
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
                print(f"Updated pipelines: {self.pipelines if self.pipelines else 'None'}")
            
            return changes_made

    processor = MockProcessor()
    
    # Test cases
    test_cases = [
        # Single additions
        ("+57", "Add single pipeline: B Channel + CLAHE"),
        ("+68", "Add single pipeline: G Channel + Gamma Correction"),
        ("+90", "Add single pipeline: Bilateral Filtering + Unsharp Masking"),
        
        # Multiple additions
        ("+12 +34 +56", "Add multiple pipelines at once"),
        
        # Single removals
        ("-57", "Remove single pipeline"),
        
        # Multiple removals
        ("-12 -34 -56", "Remove multiple pipelines at once"),
        
        # Mixed operations
        ("+abc -68 +123", "Mixed add and remove operations"),
        
        # Clear all
        ("clear", "Clear all pipelines"),
        
        # Invalid operations
        ("+xyz", "Invalid pipeline with non-existent operations"),
        ("invalid", "Invalid command format"),
    ]
    
    print("Testing Pipeline Management Commands:")
    print("=" * 50)
    
    for command, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Command: '{command}'")
        print(f"Current pipelines before: {processor.pipelines}")
        
        processor.process_pipeline_commands(command)
        
        print(f"Current pipelines after: {processor.pipelines}")
        print("-" * 30)

if __name__ == "__main__":
    test_pipeline_parsing()
    
    print("\n" + "=" * 50)
    print("USAGE EXAMPLES for the main application:")
    print("=" * 50)
    print("\n1. At startup selection prompt:")
    print("   - Type '5 7 57' for: B Channel, CLAHE, B+CLAHE pipeline")
    print("   - Type '123 456' for: multiple pipelines")
    
    print("\n2. During runtime (press 'p'):")
    print("   Single operations:")
    print("   - '+57' → Add B Channel + CLAHE pipeline")
    print("   - '-57' → Remove that pipeline")
    
    print("\n   Multiple operations:")
    print("   - '+57 +68 +90' → Add multiple pipelines")
    print("   - '-1 -2 -3 -4 -5' → Remove multiple pipelines") 
    print("   - '+abc -123 +456' → Mixed add/remove operations")
    
    print("\n   Other commands:")
    print("   - 'clear' → Remove all pipelines")
    print("   - '' (empty) → Cancel/no changes")
    
    print("\n3. Pipeline examples:")
    print("   - '57' = B Channel + CLAHE")
    print("   - '68' = G Channel + Gamma Correction")
    print("   - '90' = Bilateral Filtering + Unsharp Masking")
    print("   - '123' = Original + Grayscale + R Channel")
    print("   - 'abc' = Logarithmic + Unsharp + Grayscale + CLAHE")
