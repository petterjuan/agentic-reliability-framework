#!/usr/bin/env python3
"""
Replace direct references with lazy function calls
"""

import re

def replace_in_file():
    with open('arf/app.py', 'r') as f:
        content = f.read()
    
    # Replace thread_safe_index with get_faiss_index()
    # But be careful not to replace in comments or strings
    lines = content.split('\n')
    replaced = []
    changes = []
    
    for i, line in enumerate(lines, 1):
        original = line
        
        # Skip comment lines
        if line.strip().startswith('#'):
            replaced.append(line)
            continue
        
        # Replace thread_safe_index with get_faiss_index()
        if 'thread_safe_index' in line and not 'def thread_safe_index' in line:
            # Simple replacement for most cases
            new_line = line.replace('thread_safe_index', 'get_faiss_index()')
            
            # Special case: if it's thread_safe_index. (with dot)
            if 'thread_safe_index.' in line:
                new_line = line.replace('thread_safe_index.', 'get_faiss_index().')
            
            if new_line != original:
                changes.append((i, 'thread_safe_index', 'get_faiss_index()'))
                replaced.append(new_line)
            else:
                replaced.append(original)
        
        # Replace business_metrics with get_business_metrics()
        elif 'business_metrics' in line and not 'def business_metrics' in line:
            new_line = line.replace('business_metrics', 'get_business_metrics()')
            
            if 'business_metrics.' in line:
                new_line = line.replace('business_metrics.', 'get_business_metrics().')
            
            if new_line != original:
                changes.append((i, 'business_metrics', 'get_business_metrics()'))
                replaced.append(new_line)
            else:
                replaced.append(original)
        
        else:
            replaced.append(original)
    
    # Write the updated file
    with open('arf/app.py.updated', 'w') as f:
        f.write('\n'.join(replaced))
    
    return changes

if __name__ == '__main__':
    print("Replacing direct references with lazy function calls...")
    changes = replace_in_file()
    
    if changes:
        print(f"\n✅ Made {len(changes)} replacements:")
        for lineno, old, new in changes[:10]:  # Show first 10
            print(f"  Line {lineno}: {old} → {new}")
        
        if len(changes) > 10:
            print(f"  ... and {len(changes)-10} more")
        
        print("\n📋 Created arf/app.py.updated")
        print("   Review then: mv arf/app.py.updated arf/app.py")
    else:
        print("✅ No replacements needed")
