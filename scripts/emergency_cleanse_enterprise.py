# scripts/emergency_cleanse_enterprise.py
# CORRECTED VERSION - Add this file through GitHub UI
import os
import sys
import shutil
from pathlib import Path

def remove_enterprise_from_oss():
    """Remove all enterprise code from OSS repository"""
    repo_root = Path.cwd()
    
    # CRITICAL: Remove enterprise directory from OSS
    enterprise_dir = repo_root / "agentic_reliability_framework" / "enterprise"
    if enterprise_dir.exists():
        print(f"🚨 CRITICAL: Removing enterprise directory from OSS: {enterprise_dir}")
        shutil.rmtree(enterprise_dir)
    
    # Scan for enterprise imports in OSS files
    oss_dirs = [
        "agentic_reliability_framework/engine",
        "agentic_reliability_framework/memory",
        "agentic_reliability_framework"
    ]
    
    enterprise_patterns = [
        "from enterprise.",
        "import enterprise.",
        "license_key",
        "LicenseManager",
        "EnterpriseMCPServer",
        "audit_trail",
        "learning_engine"
    ]
    
    violations = []
    for dir_path in oss_dirs:
        dir_full = repo_root / dir_path
        if dir_full.exists():
            for py_file in dir_full.rglob("*.py"):
                with open(py_file, 'r') as f:
                    content = f.read()
                    for pattern in enterprise_patterns:
                        if pattern in content:
                            violations.append(f"{py_file}: Contains {pattern}")
    
    if violations:
        print("⚠️  Enterprise imports found in OSS files:")
        for v in violations:
            print(f"  {v}")
    
    return len(violations) == 0

if __name__ == "__main__":
    success = remove_enterprise_from_oss()
    sys.exit(0 if success else 1)
