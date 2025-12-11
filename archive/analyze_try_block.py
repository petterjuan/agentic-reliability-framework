#!/usr/bin/env python3
"""
Analyze the problematic try block
"""

with open('arf/app.py', 'r') as f:
    lines = f.readlines()

# Find the try block starting at line 268
in_try = False
try_start = 268
try_end = None
indent_level = None

for i in range(267, len(lines)):  # Start at line 268 (0-indexed)
    line = lines[i]
    stripped = line.strip()
    
    if i == 267 and stripped.startswith('try:'):  # line 268
        in_try = True
        indent_level = len(line) - len(line.lstrip())
        print(f"Found try block starting at line {i+1}")
        print(f"Indent level: {indent_level}")
    
    if in_try and stripped.startswith('except'):
        # Check if it's at same indent level
        current_indent = len(line) - len(line.lstrip())
        if current_indent == indent_level:
            try_end = i
            print(f"Found except at line {i+1}: {stripped[:50]}")
            break

if try_end:
    print(f"\nTry block spans lines {try_start}-{try_end}")
    print("\nContent:")
    for j in range(try_start-1, min(try_end+3, len(lines))):
        print(f"{j+1:4}: {lines[j].rstrip()}")
