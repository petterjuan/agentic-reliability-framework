#!/usr/bin/env python3
"""
Verify that circular import fixes are working
Run this after updating __init__.py files

This script can be run directly in GitHub UI by clicking "Run" button
"""

import sys
import traceback
import importlib
import warnings


def clear_module_cache():
    """Clear cached modules to test fresh imports"""
    modules_to_clear = [
        'agentic_reliability_framework',
        'agentic_reliability_framework.arf_core',
        'agentic_reliability_framework.arf_core.engine.simple_mcp_client',
        'agentic_reliability_framework.arf_core.models.healing_intent',
        'agentic_reliability_framework.arf_core.engine.oss_mcp_client',
        'agentic_reliability_framework.config',
        'agentic_reliability_framework.engine',
        'agentic_reliability_framework.lazy',
        'agentic_reliability_framework.app',
    ]
    
    cleared = []
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
            cleared.append(module)
    
    return cleared


def test_import_no_circular():
    """Test that imports work without circular dependencies"""
    print("🔍 Testing imports for circular dependencies...")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Import main package
    try:
        import agentic_reliability_framework as arf
        version = getattr(arf, '__version__', 'unknown')
        tests.append(("Main package", True, f"v{version}"))
    except RecursionError as e:
        tests.append(("Main package", False, f"❌ RecursionError: {e}"))
        traceback.print_exc()
        return False, tests  # Critical failure
    except Exception as e:
        tests.append(("Main package", False, f"❌ Error: {type(e).__name__}: {e}"))
        traceback.print_exc()
    
    # Test 2: Import arf_core directly
    try:
        from agentic_reliability_framework import arf_core
        tests.append(("arf_core module", True, "✅ Import successful"))
    except RecursionError as e:
        tests.append(("arf_core module", False, f"❌ RecursionError: {e}"))
        traceback.print_exc()
        return False, tests
    except Exception as e:
        tests.append(("arf_core module", False, f"❌ Error: {type(e).__name__}: {e}"))
        traceback.print_exc()
    
    # Test 3: Import OSS components
    try:
        from agentic_reliability_framework import HealingIntent
        tests.append(("HealingIntent", True, "✅ Import successful"))
    except RecursionError as e:
        tests.append(("HealingIntent", False, f"❌ RecursionError: {e}"))
        traceback.print_exc()
        return False, tests
    except Exception as e:
        tests.append(("HealingIntent", False, f"❌ Error: {type(e).__name__}: {e}"))
        traceback.print_exc()
    
    # Test 4: Import OSSMCPClient
    try:
        from agentic_reliability_framework import OSSMCPClient
        tests.append(("OSSMCPClient", True, "✅ Import successful"))
    except RecursionError as e:
        tests.append(("OSSMCPClient", False, f"❌ RecursionError: {e}"))
        traceback.print_exc()
        return False, tests
    except Exception as e:
        tests.append(("OSSMCPClient", False, f"❌ Error: {type(e).__name__}: {e}"))
        traceback.print_exc()
    
    # Test 5: Test the problematic chain
    try:
        import agentic_reliability_framework.arf_core.constants
        tests.append(("Constants module", True, "✅ Import successful"))
    except RecursionError as e:
        tests.append(("Constants module", False, f"❌ RecursionError: {e}"))
        traceback.print_exc()
        return False, tests
    except Exception as e:
        tests.append(("Constants module", False, f"❌ Error: {type(e).__name__}: {e}"))
        traceback.print_exc()
    
    # Test 6: Test simple_mcp_client import
    try:
        import agentic_reliability_framework.arf_core.engine.simple_mcp_client
        tests.append(("simple_mcp_client", True, "✅ Import successful"))
    except RecursionError as e:
        tests.append(("simple_mcp_client", False, f"❌ RecursionError: {e}"))
        traceback.print_exc()
        return False, tests
    except Exception as e:
        tests.append(("simple_mcp_client", False, f"❌ Error: {type(e).__name__}: {e}"))
        traceback.print_exc()
    
    # Test 7: Test healing_intent import
    try:
        import agentic_reliability_framework.arf_core.models.healing_intent
        tests.append(("healing_intent", True, "✅ Import successful"))
    except RecursionError as e:
        tests.append(("healing_intent", False, f"❌ RecursionError: {e}"))
        traceback.print_exc()
        return False, tests
    except Exception as e:
        tests.append(("healing_intent", False, f"❌ Error: {type(e).__name__}: {e}"))
        traceback.print_exc()
    
    # Test 8: Test factory functions
    try:
        from agentic_reliability_framework import create_rollback_intent
        tests.append(("Factory functions", True, "✅ Available"))
    except Exception as e:
        tests.append(("Factory functions", False, f"⚠️ Not available: {e}"))
    
    return True, tests


def check_import_paths():
    """Check import paths are correct"""
    print("\n📁 Checking import paths...")
    print("-" * 40)
    
    try:
        # Check simple_mcp_client.py import
        import agentic_reliability_framework.arf_core.engine.simple_mcp_client as smc
        source_file = getattr(smc, '__file__', 'unknown')
        print(f"✅ simple_mcp_client.py: {source_file}")
        
        # Check it can instantiate
        client = smc.OSSMCPClient()
        print(f"✅ OSSMCPClient instantiated: {client.mode} mode")
        
    except Exception as e:
        print(f"❌ simple_mcp_client check failed: {e}")
    
    try:
        # Check healing_intent import path
        import agentic_reliability_framework.arf_core.models.healing_intent as hi
        source_file = getattr(hi, '__file__', 'unknown')
        print(f"✅ healing_intent.py: {source_file}")
        
        # Check it can create an intent
        from datetime import datetime
        intent = hi.HealingIntent(
            action="restart",
            component="test-service",
            parameters={},
            justification="test",
            confidence=0.8,
            incident_id="test-123",
            detected_at=datetime.now().timestamp()
        )
        print(f"✅ HealingIntent created: {intent.action}")
        
    except Exception as e:
        print(f"❌ healing_intent check failed: {e}")


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🔄 Circular Import Fix Verification")
    print("=" * 60)
    
    # Clear cache first
    print("\n🗑️  Clearing module cache...")
    cleared = clear_module_cache()
    if cleared:
        print(f"   Cleared: {len(cleared)} modules")
    
    # Run main test
    success, test_results = test_import_no_circular()
    
    print("\n📊 Import Test Results:")
    print("=" * 60)
    
    all_pass = True
    for name, test_success, details in test_results:
        if test_success:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_pass = False
        print(f"{status} {name}: {details}")
    
    print("=" * 60)
    
    if all_pass and success:
        print("\n🎉 SUCCESS: No circular imports detected!")
        print("\n✅ Verification complete. All critical imports working.")
        
        # Show additional info
        check_import_paths()
        
        print("\n" + "=" * 60)
        print("📋 NEXT STEPS:")
        print("1. Commit all changes with message:")
        print('   "fix: resolve circular import dependencies in __init__.py files"')
        print("2. Push to main branch")
        print("3. Monitor CI/CD pipeline in GitHub Actions")
        print("4. Run: python -m pytest Test/test_basic.py")
        print("5. Test package installation if possible")
        
        return True
    else:
        print("\n⚠️  FAILURE: Circular import issues detected!")
        print("\n🔧 Debugging required:")
        print("1. Check arf_core/__init__.py - ensure no lazy loading of engine modules")
        print("2. Check simple_mcp_client.py - line 17 should be 'from ..models.healing_intent'")
        print("3. Check main __init__.py - OSS components should be imported directly")
        print("4. Clear Python cache and retest")
        
        check_import_paths()
        
        return False


def quick_test():
    """Quick test that can be run in GitHub UI"""
    print("⚡ Quick Import Test")
    print("=" * 40)
    
    test_code = """
import sys
# Clear cache
modules = ['agentic_reliability_framework', 'agentic_reliability_framework.arf_core']
for m in modules:
    sys.modules.pop(m, None)

# Test imports
import agentic_reliability_framework as arf
print(f"✅ Main package: v{arf.__version__}")

from agentic_reliability_framework import HealingIntent
print(f"✅ HealingIntent: {HealingIntent}")

from agentic_reliability_framework import OSSMCPClient
print(f"✅ OSSMCPClient: {OSSMCPClient}")

print("\\n🎉 All imports successful!")
    """
    
    print("Test code to run:")
    print("-" * 40)
    print(test_code)
    print("-" * 40)
    
    try:
        # Clear cache
        modules = ['agentic_reliability_framework', 'agentic_reliability_framework.arf_core']
        for m in modules:
            sys.modules.pop(m, None)
        
        # Test imports
        import agentic_reliability_framework as arf
        print(f"✅ Main package: v{arf.__version__}")
        
        from agentic_reliability_framework import HealingIntent
        print(f"✅ HealingIntent: {HealingIntent}")
        
        from agentic_reliability_framework import OSSMCPClient
        print(f"✅ OSSMCPClient: {OSSMCPClient}")
        
        print("\n🎉 All imports successful!")
        return True
    except RecursionError as e:
        print(f"❌ CIRCULAR IMPORT: {e}")
        return False
    except Exception as e:
        print(f"❌ Import error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify circular import fixes")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test()
        sys.exit(0 if success else 1)
    else:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
