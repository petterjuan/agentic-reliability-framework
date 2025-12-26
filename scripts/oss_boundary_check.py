#!/usr/bin/env python3
"""
OSS Boundary Check - Debug version
"""

import os
import sys
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="OSS Boundary Check")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()

def check_oss_file_structure(debug=False):
    """Check that OSS package structure exists"""
    print("📁 Checking OSS file structure...")
    
    arf_core_path = Path("agentic_reliability_framework/arf_core")
    
    if debug:
        print(f"   Debug: Looking for {arf_core_path}")
    
    if not arf_core_path.exists():
        print(f"❌ arf_core directory not found at: {arf_core_path}")
        return False
    
    print(f"✅ arf_core directory exists")
    
    if debug:
        print("   Debug: Listing contents:")
        for item in arf_core_path.iterdir():
            print(f"     - {item.name}")
    
    # Check for critical files
    critical_files = [
        arf_core_path / "__init__.py",
        arf_core_path / "constants.py",
    ]
    
    missing_files = []
    for file_path in critical_files:
        if debug:
            print(f"   Debug: Checking {file_path}")
        
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"❌ Missing critical files:")
        for missing in missing_files:
            print(f"  - {missing}")
        return False
    
    print("✅ Critical OSS files exist")
    return True

def check_constants_file(debug=False):
    """Check OSS constants file"""
    print("\n📋 Checking OSS constants...")
    
    constants_file = Path("agentic_reliability_framework/arf_core/constants.py")
    
    if not constants_file.exists():
        print("❌ constants.py not found")
        return False
    
    try:
        with open(constants_file, 'r') as f:
            content = f.read()
        
        if debug:
            print(f"   Debug: File size: {len(content)} bytes")
            print(f"   Debug: First 3 lines:")
            lines = content.split('\n')[:3]
            for i, line in enumerate(lines):
                print(f"     {i+1}: {line}")
        
        # Check for required constants
        required = [
            "MAX_INCIDENT_HISTORY",
            "MCP_MODES_ALLOWED",
            "EXECUTION_ALLOWED",
            "GRAPH_STORAGE",
        ]
        
        missing = []
        for const in required:
            if const not in content:
                missing.append(const)
        
        if missing:
            print(f"❌ Missing constants: {', '.join(missing)}")
            return False
        
        print("✅ All required constants found")
        
        # Check for OSS-only values
        if "advisory" not in content.lower():
            print("⚠️  constants.py doesn't mention 'advisory' mode")
            # Don't fail for this
        
        if "in_memory" not in content.lower():
            print("⚠️  constants.py doesn't mention 'in_memory' storage")
            # Don't fail for this
        
        # Check for forbidden modes
        if "APPROVAL" in content or "AUTONOMOUS" in content:
            print("❌ constants.py contains non-advisory modes")
            return False
        
        print("✅ OSS constants are correct")
        return True
        
    except Exception as e:
        print(f"❌ Error reading constants.py: {e}")
        return False

def check_no_enterprise_imports(debug=False):
    """Check that OSS code doesn't import Enterprise modules"""
    print("\n🔍 Checking for Enterprise imports...")
    
    oss_dir = Path("agentic_reliability_framework/arf_core")
    if not oss_dir.exists():
        print("⚠️  arf_core directory not found")
        return True  # Already checked above
    
    # Absolute no-nos
    critical_patterns = [
        "EnterpriseMCPServer",
        "LicenseManager", 
        "license_key",
        "validate_license",
        "ARF-ENT-",  # License key pattern
    ]
    
    violations = []
    
    for py_file in oss_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                line_num = i + 1
                for pattern in critical_patterns:
                    if pattern in line:
                        # Skip comments
                        stripped = line.strip()
                        if stripped.startswith('#'):
                            continue
                        
                        violations.append(f"{py_file}:{line_num}: {pattern}")
                        break
                        
        except Exception as e:
            if debug:
                print(f"   Debug: Error reading {py_file}: {e}")
            continue
    
    if violations:
        print("❌ Critical Enterprise patterns found:")
        for violation in violations:
            print(f"  - {violation}")
        return False
    else:
        print("✅ No critical Enterprise patterns found")
        return True

def main():
    """Main boundary check"""
    args = parse_args()
    
    print("🔐 OSS BOUNDARY CHECK")
    print("=" * 50)
    
    if args.debug:
        print("🔧 Debug mode enabled")
    
    # Run checks
    results = []
    
    results.append(("File Structure", check_oss_file_structure(args.debug)))
    results.append(("Constants", check_constants_file(args.debug)))
    results.append(("Enterprise Imports", check_no_enterprise_imports(args.debug)))
    
    print("\n" + "=" * 50)
    print("📊 RESULTS:")
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All OSS boundary checks passed!")
        return 0
    else:
        print("\n🚨 OSS boundary violations detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())
