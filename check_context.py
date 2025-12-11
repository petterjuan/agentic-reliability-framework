#!/usr/bin/env python3
"""
Check context of each get_business_metrics() call
"""
import re

with open('agentic_reliability_framework/app.py', 'r') as f:
    lines = f.readlines()

line_numbers = [1678, 2043, 2083, 2104, 2121, 2139, 2238]

for line_num in line_numbers:
    print(f"\n{'='*60}")
    print(f"Line {line_num}:")
    print(f"{'='*60}")
    
    # Show context (5 lines before, the line, 5 lines after)
    start = max(0, line_num - 6)
    end = min(len(lines), line_num + 4)
    
    for i in range(start, end):
        prefix = ">>> " if i == line_num - 1 else "    "
        print(f"{prefix}{i+1:4d}: {lines[i].rstrip()}")
