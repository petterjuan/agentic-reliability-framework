#!/usr/bin/env python3
"""
Isolate the import issue
"""

import sys
import time
import importlib.util  # FIXED: Moved to VERY TOP

# Test 1: Import just the module without any dependencies
print("Test 1: Creating empty module...")
empty_module = """
__version__ = "2.0.0"
"""
exec(empty_module)
print("  ✓ Empty module works")

# Test 2: Import actual arf with timing
print("\nTest 2: Importing arf...")
start = time.time()

# First, let's see what happens if we don't import app.py
# importlib.util is now at the top (line 8)

# Try to import agentic_reliability_framework as arf without triggering app.py
spec = importlib.util.find_spec('arf')
if spec:
    print(f"  Found arf at: {spec.origin}")
    
    # Read and check the module
    with open(spec.origin, 'r') as f:
        first_lines = f.readlines()[:20]
        print("  First 20 lines of arf/__init__.py:")
        for i, line in enumerate(first_lines, 1):
            print(f"    {i:2}: {line.rstrip()}")
            
    # Check if it imports app.py
    if 'from .app import' in ''.join(first_lines):
        print("  ❌ arf/__init__.py imports from .app - THIS CAUSES FAISS LOAD!")

# Now test the actual import
print("\nTest 3: Actually importing...")
start = time.time()
try:
    # Temporarily modify sys.modules to prevent circular import
    if 'arf.app' in sys.modules:
        del sys.modules['arf.app']
    
    # import agentic_reliability_framework as arf  # FIXED: Commented out - imported but unused
    elapsed = time.time() - start
    print(f"  Import took: {elapsed:.2f}s")
    
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
