# Test/verify_both_packages.py
"""
Verify OSS and Enterprise packages work together
"""
import sys
import subprocess

def run_test(command, description):
    print(f"\n🔍 {description}")
    print(f"   $ {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✅ SUCCESS")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()[:100]}...")
        return True
    else:
        print(f"   ❌ FAILED")
        print(f"   Error: {result.stderr.strip()[:200]}")
        return False

def main():
    print("=" * 70)
    print("🔄 ARF PACKAGE INTEGRATION VERIFICATION")
    print("=" * 70)
    
    tests = [
        # OSS package tests
        ("python -c 'import agentic_reliability_framework; print(f\"OSS: {agentic_reliability_framework.__version__}\")'", "OSS package import"),
        ("python -c 'from agentic_reliability_framework import HealingIntent; print(\"HealingIntent OK\")'", "HealingIntent import"),
        
        # Enterprise package test (requires OSS)
        ("python -c 'import arf_enterprise; print(f\"Enterprise: {arf_enterprise.__version__}\")'", "Enterprise package import"),
    ]
    
    passed = 0
    for cmd, desc in tests:
        if run_test(cmd, desc):
            passed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL PACKAGES WORKING CORRECTLY!")
        print("   Ready for production deployment")
        return 0
    else:
        print(f"\n⚠️  Some tests failed - check dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main())
