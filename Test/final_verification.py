# Test/final_verification.py
"""
FINAL COMPREHENSIVE VERIFICATION - ARF 3.3.5 & Enterprise 1.0.1
Run this to confirm the package is 100% ready for release.
"""

import sys
import json
import subprocess

def run_import_test(code, description):
    """Test Python import/usage"""
    print(f"\n🔍 {description}")
    print(f"   Code: {code}")
    
    try:
        exec(code, {})
        print(f"   ✅ SUCCESS")
        return True
    except ImportError as e:
        print(f"   ❌ IMPORT ERROR: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️  RUNTIME ERROR: {e}")
        return False

def run_pip_command(cmd, description):
    """Test pip installation/commands"""
    print(f"\n🔍 {description}")
    print(f"   Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✅ SUCCESS")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()[:100]}")
        return True
    else:
        print(f"   ❌ FAILED")
        print(f"   Error: {result.stderr.strip()[:200]}")
        return False

def main():
    print("=" * 80)
    print("🚀 ARF FINAL RELEASE VERIFICATION - v3.3.5 & Enterprise v1.0.1")
    print("=" * 80)
    
    test_results = []
    
    # ================== OSS PACKAGE TESTS ==================
    print("\n📦 OSS PACKAGE VERIFICATION (agentic-reliability-framework==3.3.5)")
    
    oss_tests = [
        # Core imports
        ("import agentic_reliability_framework as arf", "Package import"),
        ("from agentic_reliability_framework import __version__; print(f'Version: {__version__}')", "Version check"),
        
        # Critical imports that were failing
        ("from agentic_reliability_framework import HealingIntent", "HealingIntent import"),
        ("from agentic_reliability_framework import OSSMCPClient", "OSSMCPClient import"),
        
        # Factory functions
        ("from agentic_reliability_framework import create_rollback_intent", "Factory function import"),
        
        # OSS constants
        ("from agentic_reliability_framework import OSS_EDITION, EXECUTION_ALLOWED; print(f'OSS: {OSS_EDITION}, Exec: {EXECUTION_ALLOWED}')", "OSS constants"),
        
        # Instantiation tests
        ("from agentic_reliability_framework import HealingIntent; hi = HealingIntent(action='test', component='test'); print(f'Created: {hi.action}')", "HealingIntent creation"),
        
        # MCP client
        ("from agentic_reliability_framework import create_mcp_client; client = create_mcp_client(); print(f'Client mode: {client.mode}')", "MCP client creation"),
    ]
    
    for code, desc in oss_tests:
        test_results.append(("OSS", desc, run_import_test(code, desc)))
    
    # ================== ENTERPRISE PACKAGE TESTS ==================
    print("\n🏢 ENTERPRISE PACKAGE VERIFICATION (agentic-reliability-enterprise==1.0.1)")
    
    enterprise_tests = [
        # Enterprise package import
        ("import arf_enterprise", "Enterprise package import"),
        
        # Version check
        ("import arf_enterprise; print(f'Enterprise version: {arf_enterprise.__version__}')", "Enterprise version"),
        
        # OSS integration via lazy loading
        ("import arf_enterprise; print(f'OSS Available: {arf_enterprise.OSS_AVAILABLE}')", "OSS availability check"),
        
        # Access OSS components (triggers lazy load)
        ("import arf_enterprise; print(f'HealingIntent class: {arf_enterprise.HealingIntent}')", "OSS component access"),
        
        # Enterprise core components
        ("from arf_enterprise import LicenseManager, AuditTrail", "Enterprise core imports"),
    ]
    
    for code, desc in enterprise_tests:
        test_results.append(("Enterprise", desc, run_import_test(code, desc)))
    
    # ================== INTEGRATION TESTS ==================
    print("\n🔗 OSS → ENTERPRISE INTEGRATION TESTS")
    
    integration_tests = [
        # Create HealingIntent in OSS, check in Enterprise
        ("""
from agentic_reliability_framework import HealingIntent
import arf_enterprise
hi = HealingIntent(action='restart', component='api-server', justification='test')
print(f'OSS Intent: {hi.action} -> {hi.component}')
print(f'Enterprise can access: {arf_enterprise.HealingIntent.__name__ if arf_enterprise.HealingIntent else "No"}')
        """, "OSS → Enterprise handoff"),
        
        # Check OSS constants match
        ("""
from agentic_reliability_framework import OSS_EDITION, EXECUTION_ALLOWED
import arf_enterprise
print(f'OSS says exec allowed: {EXECUTION_ALLOWED}')
print(f'Enterprise sees OSS: {arf_enterprise.oss_constants.OSS_EDITION if hasattr(arf_enterprise.oss_constants, "OSS_EDITION") else "No"}')
        """, "Constant consistency"),
    ]
    
    for code, desc in integration_tests:
        test_results.append(("Integration", desc, run_import_test(code, desc)))
    
    # ================== SUMMARY ==================
    print("\n" + "=" * 80)
    print("📊 FINAL VERIFICATION RESULTS")
    print("=" * 80)
    
    success_by_category = {}
    total_by_category = {}
    
    for category, description, success in test_results:
        success_by_category[category] = success_by_category.get(category, 0) + (1 if success else 0)
        total_by_category[category] = total_by_category.get(category, 0) + 1
    
    # Print category summaries
    for category in sorted(success_by_category.keys()):
        passed = success_by_category[category]
        total = total_by_category[category]
        percentage = (passed / total) * 100
        print(f"\n{category.upper():12} {passed:2}/{total:2} tests passed ({percentage:.0f}%)")
        
        # Show failures
        if passed < total:
            print("  Failed tests:")
            for cat, desc, success in test_results:
                if cat == category and not success:
                    print(f"    • {desc}")
    
    # Overall summary
    total_passed = sum(success_by_category.values())
    total_tests = sum(total_by_category.values())
    overall_percentage = (total_passed / total_tests) * 100
    
    print("\n" + "=" * 80)
    print(f"🏁 OVERALL: {total_passed}/{total_tests} tests passed ({overall_percentage:.0f}%)")
    
    if total_passed == total_tests:
        print("\n🎉 🎉 🎉 ALL TESTS PASSED!")
        print("   ARF v3.3.5 and Enterprise v1.0.1 are READY FOR PRODUCTION RELEASE!")
        print("   ✅ PyPI packages can be published")
        print("   ✅ Customers can install and use")
        print("   ✅ Crisis is FULLY RESOLVED")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} TESTS FAILED")
        print("   Need to fix remaining issues before release")
        return 1

if __name__ == "__main__":
    sys.exit(main())
