#!/usr/bin/env python3
"""
OSS Boundary Check - No external dependencies
Checks that OSS code is pure (no Enterprise imports) and structure is correct
"""

import os
import sys
import re
from pathlib import Path

def check_oss_constants():
    """Check that OSS constants are properly defined"""
    print("🔍 Checking OSS constants...")
    
    # Look for arf_core directory
    arf_core_path = Path("agentic_reliability_framework/arf_core")
    
    if not arf_core_path.exists():
        print(f"❌ arf_core directory not found at: {arf_core_path}")
        return False
    
    # Look for constants.py
    constants_path = arf_core_path / "constants.py"
    if not constants_path.exists():
        print(f"❌ constants.py not found at: {constants_path}")
        return False
    
    try:
        with open(constants_path, 'r') as f:
            content = f.read()
        
        # Check for required constants
        required_constants = [
            "MAX_INCIDENT_HISTORY",
            "MCP_MODES_ALLOWED", 
            "EXECUTION_ALLOWED",
            "GRAPH_STORAGE",
            "MAX_INCIDENT_NODES",
            "MAX_OUTCOME_NODES",
            "FAISS_INDEX_TYPE",
        ]
        
        missing = []
        for const in required_constants:
            if const not in content:
                missing.append(const)
        
        if missing:
            print(f"❌ Missing constants: {', '.join(missing)}")
            return False
        
        # Check values (simple string checks)
        checks = [
            ("MAX_INCIDENT_HISTORY", ["1000", "1_000"]),
            ("MCP_MODES_ALLOWED", ["advisory"]),
            ("EXECUTION_ALLOWED", ["False"]),
            ("GRAPH_STORAGE", ["in_memory"]),
            ("FAISS_INDEX_TYPE", ["IndexFlatL2"]),
        ]
        
        for const_name, expected_values in checks:
            # Find the line with the constant
            lines = content.split('\n')
            found = False
            for line in lines:
                if f"{const_name}:" in line or f"{const_name} =" in line:
                    found = True
                    # Check if any expected value is in the line
                    if not any(exp in line for exp in expected_values):
                        print(f"❌ {const_name} has unexpected value: {line.strip()}")
                        return False
                    break
            
            if not found:
                print(f"⚠️  Could not find value for {const_name}")
        
        print("✅ OSS constants are properly defined")
        return True
        
    except Exception as e:
        print(f"❌ Error checking constants: {e}")
        return False

def check_no_enterprise_imports():
    """Check that OSS code doesn't import Enterprise modules"""
    print("\n🔍 Checking for Enterprise imports...")
    
    oss_dirs = [
        Path("agentic_reliability_framework/arf_core"),
    ]
    
    forbidden_patterns = [
        "arf_enterprise",
        "EnterpriseMCPServer", 
        "LicenseManager",
        "license_key",
        "validate_license",
        "audit_trail",
    ]
    
    violations = []
    
    for oss_dir in oss_dirs:
        if not oss_dir.exists():
            print(f"⚠️  OSS directory not found: {oss_dir}")
            continue
            
        for py_file in oss_dir.rglob("*.py"):
            # Skip test files and cache
            if "test_" in py_file.name or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check each line for forbidden patterns
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    line_num = i + 1
                    for pattern in forbidden_patterns:
                        if pattern in line:
                            # Skip if it's in a comment or string
                            stripped = line.strip()
                            if (stripped.startswith('#') or 
                                stripped.startswith('"""') or 
                                stripped.startswith("'''")):
                                continue
                            
                            # Check if it's in a string literal
                            if '"' + pattern in line or "'" + pattern in line:
                                continue
                            
                            violations.append(f"{py_file}:{line_num}: Contains '{pattern}'")
                            break
                                
            except Exception as e:
                print(f"⚠️  Error reading {py_file}: {e}")
    
    if violations:
        print("❌ Enterprise patterns found in OSS code:")
        for violation in violations[:5]:  # Show first 5
            print(f"  - {violation}")
        if len(violations) > 5:
            print(f"  ... and {len(violations) - 5} more violations")
        return False
    else:
        print("✅ No Enterprise imports found in OSS code")
        return True

def check_oss_file_structure():
    """Check that OSS package structure is correct"""
    print("\n🔍 Checking OSS file structure...")
    
    required_files = [
        Path("agentic_reliability_framework/arf_core/__init__.py"),
        Path("agentic_reliability_framework/arf_core/constants.py"),
        Path("agentic_reliability_framework/arf_core/models/healing_intent.py"),
        Path("agentic_reliability_framework/arf_core/engine/oss_mcp_client.py"),
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print("❌ Missing required OSS files:")
        for missing in missing_files:
            print(f"  - {missing}")
        return False
    else:
        print("✅ All required OSS files exist")
        return True

def check_oss_vs_enterprise_separation():
    """Check that OSS and Enterprise code is properly separated"""
    print("\n🔍 Checking OSS/Enterprise separation...")
    
    # Check that arf_core doesn't have execution code
    arf_core_path = Path("agentic_reliability_framework/arf_core")
    
    if not arf_core_path.exists():
        print("⚠️  arf_core directory not found")
        return True  # Not a violation, just missing
    
    execution_patterns = [
        "await.*execute",
        "autonomous",
        "approval.*execute",
        "def execute",
    ]
    
    violations = []
    
    for py_file in arf_core_path.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read().lower()  # Case-insensitive
            
            for pattern in execution_patterns:
                # Simple check - if the pattern exists (not foolproof but good enough)
                if re.search(pattern, content):
                    violations.append(f"{py_file}: Contains execution pattern '{pattern}'")
                    break
                        
        except Exception as e:
            print(f"⚠️  Error reading {py_file}: {e}")
    
    if violations:
        print("❌ Execution patterns found in OSS code:")
        for violation in violations[:3]:
            print(f"  - {violation}")
        print("\n💡 OSS code should only create HealingIntent, not execute actions")
        return False
    else:
        print("✅ OSS code is properly separated (analysis only)")
        return True

def main():
    """Main boundary check"""
    print("🔐 OSS BOUNDARY CHECK")
    print("=" * 50)
    
    # Run all checks
    results = []
    
    results.append(("File Structure", check_oss_file_structure()))
    results.append(("Constants", check_oss_constants()))
    results.append(("Enterprise Imports", check_no_enterprise_imports()))
    results.append(("OSS/Enterprise Separation", check_oss_vs_enterprise_separation()))
    
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
