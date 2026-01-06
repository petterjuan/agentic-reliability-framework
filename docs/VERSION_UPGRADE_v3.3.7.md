# File: docs/VERSION_UPGRADE_v3.3.7.md

# 🚀 ARF v3.3.7 - OSS Boundary Compliance Release

## 🎯 Executive Summary
v3.3.7 includes **all surgical fixes** from GitHub v3.3.6, now available on PyPI. This release ensures 100% OSS boundary compliance and import stability.

## 🔧 What's New in v3.3.7

### Fixed from v3.3.6 (PyPI version):
1. ✅ **OSS Boundary Compliance**: `license_key` patterns completely removed
2. ✅ **Import Stability**: Circular import issues resolved
3. ✅ **Version Consistency**: All files show 3.3.7
4. ✅ **Project Hygiene**: Pre-commit hooks, verification scripts added

### Verification Status:
- ✅ **156+ CI Runs**: All tests passing (#147-156)
- ✅ **Package Verification**: Test Built Package #1-2 passing
- ✅ **OSS Compliance**: No enterprise patterns detected

## 📦 Installation

### New Installation:
```bash
pip install agentic-reliability-framework==3.3.7
```
Upgrade from v3.3.6 (PyPI):

```bash
pip install --upgrade agentic-reliability-framework
Upgrade from GitHub v3.3.6:
bash
# If you installed from GitHub, switch to PyPI
pip uninstall agentic-reliability-framework
pip install agentic-reliability-framework==3.3.7
```

🔍 Verification

```python
import agentic_reliability_framework as arf

print(f"Version: {arf.__version__}")  # Should be 3.3.7

# Verify OSS compliance
from agentic_reliability_framework import OSS_EDITION, EXECUTION_ALLOWED
print(f"OSS Edition: {OSS_EDITION}, Execution Allowed: {EXECUTION_ALLOWED}")

# Test imports
from agentic_reliability_framework import HealingIntent, OSSMCPClient
print("✅ All imports working")
```

📁 Files Updated for v3.3.7
---------------------------

1.  pyproject.toml - Version updated to 3.3.7
    
2.  agentic\_reliability\_framework/arf\_core/\_\_init\_\_.py - Version updated to 3.3.7
    
3.  All OSS boundary fixes from GitHub v3.3.6 included
    

🚨 Important Notes
------------------

### For v3.3.6 (PyPI) Users:

*   **v3.3.6 on PyPI** remains available but has OSS boundary issues
    
*   **Upgrade to v3.3.7** recommended for all users
    
*   No breaking changes - pure fixes
    

### For GitHub v3.3.6 Users:

*   Your version already has the fixes
    
*   **v3.3.7 on PyPI** is identical to GitHub v3.3.6
    
*   Switch to PyPI for official distribution
    

🧪 Test Results
---------------

Test SuiteStatusRunsOSS Boundary Tests✅ PASSING#147-156OSS Tests✅ PASSING#779-783OSS Comprehensive Tests✅ PASSING#91-97Test Built Package✅ PASSING#1-2

📞 Support
----------

*   **Issues**: [GitHub Issues](https://github.com/petterjuan/agentic-reliability-framework/issues)
    
*   **Documentation**: [ARF Docs](https://github.com/petterjuan/agentic-reliability-framework/tree/main/docs)
    
*   **PyPI**: [v3.3.7](https://pypi.org/project/agentic-reliability-framework/3.3.7/)
