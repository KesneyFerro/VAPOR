"""
Update VAPOR pipeline config for time-cropped video processing.
Run this to set time segment, then use vapor_complete_pipeline.py
"""

from pathlib import Path

# ============================================================================
# TIME SEGMENT CONFIGURATION - EDIT THESE VALUES
# ============================================================================
START_TIME = 40.0   # Start time in seconds
DURATION = 20.0     # Duration in seconds

# ============================================================================

config_path = Path(__file__).parent / "config" / "pipeline_config.yaml"

# Read config
with open(config_path, 'r') as f:
    lines = f.readlines()

# Update time cropping settings
new_lines = []
in_time_crop = False
for line in lines:
    if 'time_crop:' in line:
        in_time_crop = True
        new_lines.append(line)
    elif in_time_crop:
        if 'enabled:' in line:
            new_lines.append('    enabled: true\n')
        elif 'start_time:' in line:
            new_lines.append(f'    start_time: {START_TIME}\n')
        elif 'end_time:' in line:
            new_lines.append('    end_time: null\n')
        elif 'duration:' in line:
            new_lines.append(f'    duration: {DURATION}\n')
            in_time_crop = False
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

# Write updated config
with open(config_path, 'w') as f:
    f.writelines(new_lines)

print("✓ Config updated for time segment:")
print(f"  Start: {START_TIME}s")
print(f"  Duration: {DURATION}s")
print(f"  End: {START_TIME + DURATION}s")
print(f"\nNext: python vapor_complete_pipeline.py --video YOUR_VIDEO.avi")
