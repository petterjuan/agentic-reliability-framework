#!/usr/bin/env python3
"""
Fix the get_business_metrics() call to use self.business_calculator
"""
import re

# Read app.py
with open('agentic_reliability_framework/app.py', 'r') as f:
    content = f.read()

# Find and replace get_business_metrics() with self.business_calculator
# But only within the EnhancedReliabilityEngine class methods

# First, let's find where get_business_metrics() is called
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'get_business_metrics()' in line:
        print(f"Line {i+1}: {line}")
        # Check if we're in a method (look for def above)
        for j in range(i, max(-1, i-20), -1):
            if 'def ' in lines[j] and 'self' in lines[j]:
                print(f"  In method: {lines[j]}")
                # This is in a method, so replace with self.business_calculator
                lines[i] = lines[i].replace('get_business_metrics()', 'self.business_calculator')
                break

# Write back
with open('agentic_reliability_framework/app.py', 'w') as f:
    f.write('\n'.join(lines))

print("✅ Fixed get_business_metrics() calls")
