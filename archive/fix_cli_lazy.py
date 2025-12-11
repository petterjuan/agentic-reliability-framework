#!/usr/bin/env python3
"""
Make CLI commands lazy - don't load FAISS for simple commands like --version
"""

with open('arf/cli.py', 'r') as f:
    lines = f.readlines()

# Find the serve command and doctor command
for i, line in enumerate(lines):
    if '@main.command()' in line and i+1 < len(lines):
        next_line = lines[i+1]
        if 'def serve' in next_line or 'def doctor' in next_line:
            # These commands need the engine, that's OK
            pass
        elif 'def version' in next_line:
            # Version command should NOT load engine
            # Check if it imports anything heavy
            print(f"Found version command at line {i+2}")
            
            # Look ahead to see what it does
            for j in range(i+2, min(i+10, len(lines))):
                if lines[j].strip().startswith('def '):
                    break
                if 'import' in lines[j] and 'app' in lines[j]:
                    print(f"  Line {j+1}: {lines[j].strip()}")
