#!/usr/bin/env python3
"""
Manual fix for get_business_metrics() calls
"""
import re

with open('agentic_reliability_framework/app.py', 'r') as f:
    lines = f.readlines()

print("Fixing get_business_metrics() calls...")

# Fix each occurrence based on context
fixes = {
    1678: 'self.business_calculator',  # In process_event_enhanced method
    2043: 'get_business_metrics()',    # In reset_metrics method (needs global)
    2083: 'get_business_metrics()',    # In create_enhanced_ui function
    2104: 'get_business_metrics()',    # In create_enhanced_ui function  
    2121: 'get_business_metrics()',    # In create_enhanced_ui function
    2139: 'get_business_metrics()',    # In create_enhanced_ui function
    2238: 'get_business_metrics()',    # In get_engine function
}

for line_num, replacement in fixes.items():
    idx = line_num - 1  # Convert to 0-based index
    if idx < len(lines):
        old_line = lines[idx]
        if 'get_business_metrics()' in old_line:
            new_line = old_line.replace('get_business_metrics()', replacement)
            lines[idx] = new_line
            print(f"  Fixed line {line_num}: {replacement}")

print("\nAdding global business_metrics instance...")

# Find BusinessMetricsTracker class
tracker_start = -1
for i, line in enumerate(lines):
    if 'class BusinessMetricsTracker' in line:
        tracker_start = i
        break

if tracker_start != -1:
    # Find end of class (next class or empty line followed by class/def)
    for i in range(tracker_start + 1, len(lines)):
        if lines[i].strip() == '' and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line.strip().startswith(('class ', 'def ', '@')):
                # Insert after this empty line
                insert_idx = i + 1
                global_code = '''# Global business metrics tracker
business_metrics = BusinessMetricsTracker()

def get_business_metrics():
    """Get the global BusinessMetricsTracker instance"""
    return business_metrics
'''
                lines.insert(insert_idx, global_code)
                print("  Added global business_metrics instance and get_business_metrics() function")
                break
else:
    print("  Warning: BusinessMetricsTracker class not found")

# Write back
with open('agentic_reliability_framework/app.py', 'w') as f:
    f.writelines(lines)

print("\n✅ All fixes applied!")
