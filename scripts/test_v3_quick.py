#!/usr/bin/env python3
"""
Quick test for V3 validation scripts
"""

import sys
from pathlib import Path

def test_imports():
    """Test if we can import required modules."""
    print("🧪 Testing imports...")
    
    try:
        # Try to import OSS constants
        from agentic_reliability_framework.arf_core.constants import (
            OSS_EDITION, OSS_VERSION, OSS_CONSTANTS_HASH
        )
        print(f"✅ OSS constants imported: {OSS_EDITION} v{OSS_VERSION}")
        print(f"✅ Constants hash: {OSS_CONSTANTS_HASH[:8]}...")
        return True
    except ImportError as e:
        print(f"❌ Failed to import OSS constants: {e}")
        return False

def test_script_exists(script_name):
    """Test if a script exists."""
    path = Path(__file__).parent / script_name
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {script_name}: {'Exists' if exists else 'Missing'}")
    return exists

def test_certification_files():
    """Test if certification files exist."""
    print("\n📄 Checking certification files...")
    
    cert_path = Path("V3_COMPLIANCE_CERTIFICATION.json")
    issues_path = Path("V3_COMPLIANCE_ISSUES.json")
    
    if cert_path.exists():
        print(f"✅ V3_COMPLIANCE_CERTIFICATION.json FOUND")
        try:
            import json
            with open(cert_path, 'r') as f:
                data = json.load(f)
            print(f"   Version: {data.get('version', 'Unknown')}")
            print(f"   Status: {data.get('overall_status', 'Unknown')}")
            return True
        except Exception as e:
            print(f"   Error reading: {e}")
            return False
    elif issues_path.exists():
        print(f"⚠️  V3_COMPLIANCE_ISSUES.json FOUND")
        return False
    else:
        print(f"📝 No V3 certification files found")
        return False

def main():
    """Run quick tests."""
    print("=" * 60)
    print("🚀 QUICK V3 VALIDATION TEST")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test script existence
    print("\n📁 Checking validation scripts...")
    scripts = [
        "oss_boundary_check.py",
        "enhanced_v3_boundary_check.py",
        "v3_boundary_integration.py", 
        "run_v3_validation.py",
        "check_v3_status.py",
        "analyze_v3_validation_results.py",
    ]
    
    scripts_ok = all(test_script_exists(script) for script in scripts)
    
    # Test certification files
    cert_ok = test_certification_files()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if imports_ok:
        print("✅ OSS imports work correctly")
    else:
        print("❌ OSS import issues")
    
    if scripts_ok:
        print("✅ All V3 validation scripts exist")
    else:
        print("❌ Some V3 validation scripts missing")
    
    if cert_ok:
        print("✅ V3 certification exists")
    else:
        print("❌ No valid V3 certification found")
    
    print("\n🚀 To run V3 validation:")
    print("   1. python scripts/check_v3_status.py")
    print("   2. python scripts/run_v3_validation.py --fast")
    print("   3. python scripts/run_v3_validation.py --certify")
    
    # Return success if imports work and scripts exist
    return 0 if imports_ok and scripts_ok else 1

if __name__ == "__main__":
    sys.exit(main())
