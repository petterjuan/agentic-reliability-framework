# Release v3.3.4 - Stable Release

## 🎯 What's Changed
**Critical Stability Fixes:**
- ✅ **FIXED**: Circular import dependencies in `__init__.py` files
- ✅ **FIXED**: OSS/Enterprise boundary violations (removed license keys from OSS)
- ✅ **FIXED**: CI/CD pipeline now passing all tests
- ✅ **FIXED**: Package installation and import issues

**Architecture Improvements:**
- 🔧 **IMPROVED**: Direct imports for OSS components (no lazy loading)
- 🔧 **IMPROVED**: Proper relative imports in `simple_mcp_client.py`
- 🔧 **IMPROVED**: Updated test expectations for OSS edition
- 🔧 **IMPROVED**: Verification scripts for circular import detection

**Dependencies:**
- 📦 **Python 3.10+** required (matches CI/CD testing)
- 📦 **All dependencies updated** to latest stable versions

## 🚀 Quick Start
```bash
pip install agentic-reliability-framework==3.3.4
```
```python
import agentic_reliability_framework as arf
from agentic_reliability_framework import HealingIntent, OSSMCPClient

print(f"✅ ARF v{arf.__version__} - Stable and Ready!")
```
