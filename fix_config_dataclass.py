import os

with open('agentic_reliability_framework/config.py', 'r') as f:
    lines = f.readlines()

# Find the dataclass definition and add the file paths
new_lines = []
in_dataclass = False
added = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Check if we're in the dataclass definition
    if '@dataclass' in line and not added:
        in_dataclass = True
    
    # Add the file paths after the other attributes
    if in_dataclass and 'base_users:' in line and not added:
        # Add file path attributes
        new_lines.append('\n    # File Paths\n')
        new_lines.append('    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")\n')
        new_lines.append('    data_file: str = os.path.join(data_dir, "demo_incidents.json")\n')
        new_lines.append('    index_file: str = os.path.join(data_dir, "incident_vectors.index")\n')
        new_lines.append('    incident_texts_file: str = os.path.join(data_dir, "incident_texts.json")\n')
        added = True
        in_dataclass = False

with open('agentic_reliability_framework/config.py', 'w') as f:
    f.writelines(new_lines)

print("Added file path attributes to config dataclass")
