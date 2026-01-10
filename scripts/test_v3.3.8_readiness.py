#!/usr/bin/env python3
"""
Test V3.3.8 release readiness
"""
import json
import sys
from pathlib import Path
import subprocess

def check_version():
    """Check if version is 3.3.8"""
    print("🔍 Checking version...")
    
    # Check pyproject.toml
    with open("pyproject.toml", "r") as f:
        if 'version = "3.3.8"' in f.read():
            print("✅ pyproject.toml: version = 3.3.8")
        else:
            print("❌ pyproject.toml: version is NOT 3.3.8")
            return False
    
    # Check __version__.py
    version_file = Path("agentic_reliability_framework/__version__.py")
    if version_file.exists():
        content = version_file.read_text()
        if '__version__ = "3.3.8"' in content:
            print("✅ __version__.py: __version__ = 3.3.8")
            return True
        else:
            print("❌ __version__.py: version is NOT 3.3.8")
            return False
    else:
        print("❌ __version__.py not found")
        return False

def check_workflows():
    """Check release workflows exist"""
    print("\n🔍 Checking workflows...")
    
    workflows = [
        ".github/workflows/v3_milestone_sequence.yml",
        ".github/workflows/v3_release_automation.yml",
        ".github/workflows/pypi-publish-v3.3.7.yml",  # This will need renaming
    ]
    
    all_exist = True
    for workflow in workflows:
        if Path(workflow).exists():
            print(f"✅ {workflow}")
        else:
            print(f"❌ {workflow} - MISSING")
            all_exist = False
    
    return all_exist

def check_scripts():
    """Check critical scripts"""
    print("\n🔍 Checking scripts...")
    
    scripts = [
        "scripts/smart_v3_validator.py",
        "scripts/review_v3_artifacts.py",
        "scripts/oss_boundary_check.py",
    ]
    
    all_exist = True
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} - MISSING")
            all_exist = False
    
    return all_exist

def run_smart_validator():
    """Test the smart validator"""
    print("\n🔍 Testing smart_v3_validator.py...")
    try:
        result = subprocess.run(
            ["python", "scripts/smart_v3_validator.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ smart_v3_validator.py runs successfully")
            print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
            return True
        else:
            print(f"❌ smart_v3_validator.py failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error running validator: {e}")
        return False

def main():
    print("🚀 V3.3.8 Release Readiness Test")
    print("=" * 60)
    
    checks = {
        "version": check_version(),
        "workflows": check_workflows(),
        "scripts": check_scripts(),
        "validator": run_smart_validator(),
    }
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name.replace('_', ' ').title()}")
    
    if all_passed:
        print("\n🎉 READY FOR V3.3.8 RELEASE!")
        print("\nNext steps:")
        print("1. Commit all changes")
        print("2. Create tag: v3.3.8")
        print("3. Push tag to trigger automation")
        print("4. Monitor GitHub Actions")
        return 0
    else:
        print("\n⚠️ NOT READY - Fix issues above")
        print("\nPriority fixes:")
        if not checks["version"]:
            print("  • Update version to 3.3.8 in pyproject.toml and __version__.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
