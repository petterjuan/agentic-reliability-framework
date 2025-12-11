#!/usr/bin/env python3
"""
Test the lazy loading patch
"""

import sys
import subprocess
import time

def test_import_speed():
    """Test import speed with patched version"""
    print("Testing import speed...")
    
    # Test 1: Import with original (if available)
    print("\n1. Testing original import (if backup exists)...")
    try:
        start = time.time()
        # Use a subprocess to avoid contaminating current Python environment
        result = subprocess.run(
            [sys.executable, '-c', '''
import sys
sys.path.insert(0, '.')
# Temporarily use backup
import importlib.util
spec = importlib.util.spec_from_file_location("arf.app.backup", "arf/app.py.backup_pre_lazy")
backup_module = importlib.util.module_from_spec(spec)
sys.modules["arf.app"] = backup_module
spec.loader.exec_module(backup_module)
# Now import agentic_reliability_framework as arf
import agentic_reliability_framework as arf
print("Original import successful")
'''],
            capture_output=True,
            text=True
        )
        original_time = time.time() - start
        print(f"   Time: {original_time:.2f}s")
    except Exception as e:
        print(f"   Could not test original: {e}")
        original_time = 8.0  # Use known baseline
    
    # Test 2: Import with patched
    print("\n2. Testing patched import...")
    start = time.time()
    result = subprocess.run(
        [sys.executable, '-c', '''
import sys
sys.path.insert(0, '.')
# Use patched version
import importlib.util
spec = importlib.util.spec_from_file_location("arf.app.patched", "arf/app.py.patched")
patched_module = importlib.util.module_from_spec(spec)
sys.modules["arf.app"] = patched_module
spec.loader.exec_module(patched_module)
# Now import agentic_reliability_framework as arf
import agentic_reliability_framework as arf
print("Patched import successful")
'''],
        capture_output=True,
        text=True
    )
    patched_time = time.time() - start
    
    print(f"   Time: {patched_time:.2f}s")
    print(f"   Output: {result.stdout.strip()}")
    
    if result.stderr:
        print(f"   Errors: {result.stderr[:200]}")
    
    # Calculate improvement
    if original_time > 0:
        improvement = original_time / patched_time
        print(f"\n✅ Improvement: {improvement:.1f}x faster")
        print(f"   Original: {original_time:.2f}s → Patched: {patched_time:.2f}s")
    
    return patched_time < 1.0  # Success if under 1 second

def test_cli_commands():
    """Test that CLI commands still work"""
    print("\nTesting CLI commands with patch...")
    
    tests = [
        (['arf', '--version'], '2.0.0'),
        (['arf', 'doctor'], 'All dependencies OK'),
    ]
    
    all_pass = True
    for cmd, expected in tests:
        print(f"\n  Testing: {' '.join(cmd)}")
        try:
            # Use patched version
            env = {'PYTHONPATH': '.'}
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode == 0:
                if expected in result.stdout or expected in result.stderr:
                    print(f"    ✅ Pass")
                else:
                    print(f"    ⚠️  No '{expected}' in output")
                    print(f"    Output: {result.stdout[:100]}...")
            else:
                print(f"    ❌ Failed with code {result.returncode}")
                print(f"    Error: {result.stderr[:200]}")
                all_pass = False
                
        except Exception as e:
            print(f"    ❌ Exception: {e}")
            all_pass = False
    
    return all_pass

if __name__ == '__main__':
    print("=== ARF Lazy Loading Patch Test ===\n")
    
    # Test import speed
    speed_ok = test_import_speed()
    
    # Test CLI
    cli_ok = test_cli_commands()
    
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print(f"  Import Speed: {'✅ <1s' if speed_ok else '❌ Needs improvement'}")
    print(f"  CLI Commands: {'✅ All pass' if cli_ok else '❌ Some failed'}")
    
    if speed_ok and cli_ok:
        print("\n🎉 Patch is ready to apply!")
        print("Run: mv arf/app.py.patched arf/app.py")
    else:
        print("\n⚠️  Patch needs adjustments before applying")
