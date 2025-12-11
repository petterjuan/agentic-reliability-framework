#!/usr/bin/env python3
"""
Manual fix for BusinessMetricsTracker
"""
import re

with open('agentic_reliability_framework/app.py', 'r') as f:
    content = f.read()

# Find the BusinessMetricsTracker section
# We need to move def get_business_metrics() outside the class
# and fix indentation of class methods

# The pattern: from "class BusinessMetricsTracker:" to somewhere
# Let's find where the class ends (before next class or top-level def)
lines = content.split('\n')

# Find start and end of the problematic section
start = -1
for i in range(len(lines)):
    if 'class BusinessMetricsTracker:' in lines[i]:
        start = i
        break

if start == -1:
    print("❌ Could not find class")
    exit(1)

# Look for where to end (next class or top-level def after some methods)
end = -1
found_methods = False
for i in range(start + 1, len(lines)):
    stripped = lines[i].lstrip()
    if stripped.startswith('def ') and not 'get_business_metrics' in stripped:
        found_methods = True
    if found_methods and (stripped.startswith('class ') or stripped.startswith('def get_') or stripped.startswith('# ====')):
        end = i
        break

if end == -1:
    end = len(lines)

print(f"Section from line {start+1} to {end+1}")

# Now extract and fix
section = lines[start:end]
new_section = []

# Reconstruct with proper indentation
i = 0
while i < len(section):
    line = section[i]
    stripped = line.lstrip()
    
    if i == 0:
        # Class definition
        new_section.append(line)
    elif 'def get_business_metrics():' in line:
        # Skip this line and its body - we'll add it later outside class
        # Skip until we find a line at same or less indentation
        indent = len(line) - len(line.lstrip())
        i += 1
        while i < len(section):
            next_line = section[i]
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_indent <= indent and next_line.strip():
                break
            i += 1
        i -= 1  # Adjust for loop increment
    elif stripped.startswith('def '):
        # Class method - ensure 4-space indentation
        new_section.append('    ' + stripped)
    elif line.strip() == '':
        # Empty line - preserve
        new_section.append('')
    else:
        # Other line in class (docstring, etc.)
        new_section.append(line)
    
    i += 1

# Add get_business_metrics function outside class
new_section.append('')
new_section.append('# Global business metrics tracker')
new_section.append('business_metrics = BusinessMetricsTracker()')
new_section.append('')
new_section.append('def get_business_metrics():')
new_section.append('    """Get the global BusinessMetricsTracker instance"""')
new_section.append('    return business_metrics')

# Replace the section
new_lines = lines[:start] + new_section + lines[end:]

with open('agentic_reliability_framework/app.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("✅ Fixed BusinessMetricsTracker structure")
