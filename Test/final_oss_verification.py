# Test/final_oss_verification.py
"""
FINAL OSS VERIFICATION - ARF 3.3.5 OSS Edition
Comprehensive verification that all OSS boundary fixes are working
"""

import sys
import subprocess
import json
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and return success/failure"""
    print(f"\n🔍 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"   ✅ SUCCESS")
            if result.stdout.strip():
                # Show first line of output
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    print(f"   Output: {lines[0][:80]}...")
            return True
        else:
            print(f"   ❌ FAILED (exit code: {result.returncode})")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()[:150]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  TIMEOUT after 30 seconds")
        return False
    except Exception as e:
        print(f"   ⚠️  EXCEPTION: {e}")
        return False


def run_python_test(code, description):
    """Run Python code snippet"""
    print(f"\n🔍 {description}")
    print(f"   Code: {code[:80]}..." if len(code) > 80 else f"   Code: {code}")
    
    try:
        exec_globals = {}
        exec(code, exec_globals)
        print(f"   ✅ SUCCESS")
        return True
    except RecursionError as e:
        print(f"   ❌ CIRCULAR IMPORT DETECTED: {e}")
        return False
    except ImportError as e:
        print(f"   ❌ IMPORT ERROR: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️  RUNTIME ERROR: {type(e).__name__}: {e}")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists"""
    print(f"\n🔍 {description}")
    print(f"   File: {filepath}")
    
    if Path(filepath).exists():
        print(f"   ✅ EXISTS")
        return True
    else:
        print(f"   ❌ MISSING")
        return False


def check_file_content(filepath, pattern, description):
    """Check if file contains/doesn't contain a pattern"""
    print(f"\n🔍 {description}")
    print(f"   File: {filepath}")
    print(f"   Pattern: '{pattern}'")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if pattern in content:
            print(f"   ✅ CONTAINS pattern")
            return True
        else:
            print(f"   ❌ DOES NOT contain pattern")
            return False
    except Exception as e:
        print(f"   ⚠️  ERROR reading file: {e}")
        return False


def main():
    print("=" * 80)
    print("🚀 ARF OSS EDITION FINAL VERIFICATION - v3.3.5")
    print("=" * 80)
    print("Verifying all OSS boundary fixes and import stability")
    
    test_results = []
    
    # ================== FILE STRUCTURE VERIFICATION ==================
    print("\n📁 FILE STRUCTURE VERIFICATION")
    
    file_checks = [
        ("agentic_reliability_framework/arf_core/__init__.py", "arf_core __init__.py exists"),
        ("agentic_reliability_framework/arf_core/constants.py", "OSS constants file exists"),
        ("agentic_reliability_framework/arf_core/engine/oss_mcp_client.py", "OSS MCP client exists"),
        ("agentic_reliability_framework/arf_core/models/healing_intent.py", "HealingIntent module exists"),
        ("scripts/oss_boundary_check.py", "OSS boundary checker exists"),
        ("scripts/verify_circular_fix.py", "Circular import verifier exists"),
        (".pre-commit-config.yaml", "Pre-commit config exists"),
        ("Test/verify_import_fix.py", "Import verification test exists"),
    ]
    
    for filepath, desc in file_checks:
        success = check_file_exists(filepath, desc)
        test_results.append(("File Structure", desc, success))
    
    # ================== OSS BOUNDARY CHECKS ==================
    print("\n🔒 OSS BOUNDARY CHECKS")
    
    # Check that deleted file is not referenced
    success = not check_file_content(
        "agentic_reliability_framework/arf_core/__init__.py",
        "simple_mcp_client",
        "No references to deleted simple_mcp_client.py"
    )
    test_results.append(("OSS Boundary", "No simple_mcp_client references", success))
    
    # Check that license_key pattern is not present
    success = not check_file_content(
        "agentic_reliability_framework/arf_core/constants.py",
        "license_key =",
        "No license_key variable patterns"
    )
    test_results.append(("OSS Boundary", "No license_key patterns", success))
    
    # Check for OSS edition markers
    success = check_file_content(
        "agentic_reliability_framework/arf_core/constants.py",
        "OSS_EDITION",
        "OSS edition constants defined"
    )
    test_results.append(("OSS Boundary", "OSS edition constants", success))
    
    # ================== IMPORT STABILITY TESTS ==================
    print("\n🔄 IMPORT STABILITY TESTS")
    
    import_tests = [
        # Core package
        ("import agentic_reliability_framework as arf", "Main package import"),
        ("from agentic_reliability_framework import __version__; print(f'Version: {__version__}')", "Version check"),
        
        # OSS constants
        ("from agentic_reliability_framework import OSS_EDITION, OSS_LICENSE, EXECUTION_ALLOWED, MCP_MODES_ALLOWED", "OSS constants import"),
        
        # HealingIntent
        ("from agentic_reliability_framework import HealingIntent, HealingIntentSerializer", "HealingIntent import"),
        
        # OSSMCPClient
        ("from agentic_reliability_framework import OSSMCPClient, create_mcp_client", "OSSMCPClient import"),
        
        # Factory functions
        ("from agentic_reliability_framework import create_rollback_intent, create_restart_intent, create_scale_out_intent", "Factory functions import"),
        
        # Engine factory
        ("from agentic_reliability_framework import EngineFactory, create_engine, get_engine", "Engine factory import"),
        
        # OSS validation
        ("from agentic_reliability_framework import OSSBoundaryError, validate_oss_config, get_oss_capabilities", "OSS validation imports"),
        
        # Core models
        ("from agentic_reliability_framework import ReliabilityEvent, EventSeverity, create_compatible_event", "Core models import"),
        
        # Complex import chain test
        ("""
import agentic_reliability_framework as arf
from agentic_reliability_framework import HealingIntent
from agentic_reliability_framework import OSSMCPClient
from agentic_reliability_framework.arf_core import constants
from agentic_reliability_framework.arf_core.models import healing_intent
print(f'All imports successful: OSS={arf.OSS_EDITION}')
        """, "Complex import chain"),
    ]
    
    for code, desc in import_tests:
        success = run_python_test(code, desc)
        test_results.append(("Import Stability", desc, success))
    
    # ================== SCRIPT VERIFICATION ==================
    print("\n📜 SCRIPT VERIFICATION")
    
    script_tests = [
        ("python scripts/oss_boundary_check.py", "OSS boundary checker runs"),
        ("python scripts/verify_circular_fix.py --quick", "Circular import check (quick)"),
        ("python Test/verify_import_fix.py", "Import verification test"),
    ]
    
    for cmd, desc in script_tests:
        success = run_command(cmd, desc)
        test_results.append(("Scripts", desc, success))
    
    # ================== PRE-COMMIT HOOK TEST ==================
    print("\n🔧 PRE-COMMIT HOOK TEST")
    
    # Test that pre-commit config is valid
    success = run_command("pre-commit --version", "Pre-commit installed")
    test_results.append(("Pre-commit", "Pre-commit installed", success))
    
    if success:
        # Try to run hooks on a sample file
        sample_content = """
# Test file for pre-commit hooks
def test_function():
    return "Hello World"
"""
        
        # Create a test file
        test_file = Path("test_precommit.py")
        test_file.write_text(sample_content)
        
        # Run pre-commit on it
        success = run_command(f"pre-commit run --files {test_file}", "Pre-commit hooks run")
        test_results.append(("Pre-commit", "Hooks execute", success))
        
        # Clean up
        test_file.unlink()
    
    # ================== SUMMARY ==================
    print("\n" + "=" * 80)
    print("📊 VERIFICATION RESULTS SUMMARY")
    print("=" * 80)
    
    # Organize results by category
    categories = {}
    for category, description, success in test_results:
        if category not in categories:
            categories[category] = {"passed": 0, "total": 0}
        categories[category]["passed"] += 1 if success else 0
        categories[category]["total"] += 1
    
    # Print category summaries
    for category in sorted(categories.keys()):
        passed = categories[category]["passed"]
        total = categories[category]["total"]
        percentage = (passed / total) * 100 if total > 0 else 0
        status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
        
        print(f"\n{category:20} {status} {passed:2}/{total:2} ({percentage:.0f}%)")
        
        # Show failed tests in this category
        if passed < total:
            print("  Failed:")
            for cat, desc, success in test_results:
                if cat == category and not success:
                    print(f"    • {desc}")
    
    # Overall summary
    total_passed = sum(cat["passed"] for cat in categories.values())
    total_tests = sum(cat["total"] for cat in categories.values())
    overall_percentage = (total_passed / total_tests) * 100
    
    print("\n" + "=" * 80)
    print(f"🏁 OVERALL: {total_passed}/{total_tests} tests passed ({overall_percentage:.0f}%)")
    
    if total_passed == total_tests:
        print("\n🎉 🎉 🎉 ALL OSS VERIFICATION TESTS PASSED!")
        print("=" * 80)
        print("✅ OSS boundary issues are FIXED")
        print("✅ Circular imports are RESOLVED")
        print("✅ Project hygiene is ESTABLISHED")
        print("✅ Import structure is STABLE")
        print("✅ Package is READY FOR PRODUCTION")
        print("\n📦 Next steps:")
        print("   1. Run full test suite: python -m pytest Test/ -v")
        print("   2. Install pre-commit hooks: pre-commit install")
        print("   3. Package for release: python -m build")
        print("   4. Upload to PyPI: twine upload dist/*")
        print("=" * 80)
        return 0
    else:
        failures = total_tests - total_passed
        print(f"\n⚠️  {failures} TEST(S) FAILED")
        print("=" * 80)
        print("❌ OSS verification incomplete")
        print("❌ Fix remaining issues before release")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
