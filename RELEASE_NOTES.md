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
🧪 Test Status
--------------

*   ✅ OSS Boundary Tests: PASSING
    
*   ✅ Circular Import Verification: PASSING
    
*   ✅ Basic Functionality Tests: PASSING
    
*   ✅ CI/CD Pipeline: ALL GREEN
    

📁 File Structure
-----------------

text

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   agentic_reliability_framework/  ├── __init__.py              # Fixed: Direct imports, no circular deps  ├── arf_core/__init__.py     # Fixed: Property-based dynamic loading  ├── arf_core/constants.py    # Fixed: No Enterprise code  └── arf_core/engine/simple_mcp_client.py  # Fixed: Correct relative imports   `

🔒 OSS Purity
-------------

*   ✅ **100% Apache 2.0** compliant
    
*   ✅ **No Enterprise code** in OSS edition
    
*   ✅ **Advisory-only** execution mode
    
*   ✅ **Clear upgrade path** to Enterprise
    

🐛 Known Issues Resolved
------------------------

*   #CI-001: Circular imports causing RecursionError - **FIXED**
    
*   #CI-002: OSS boundary violations - **FIXED**
    
*   #CI-003: Package installation failures - **FIXED**
    
*   #CI-004: Test suite failures - **FIXED**
    

📞 Support
----------

*   GitHub Issues: [https://github.com/petterjuan/agentic-reliability-framework/issues](https://github.com/petterjuan/agentic-reliability-framework/issues)
    
*   OSS Documentation: [https://docs.arf.dev/oss]([https://docs.arf.dev/oss](https://github.com/petterjuan/agentic-reliability-framework/tree/main/docs))
    
*   Enterprise Upgrade: [https://arf.dev/enterprise](https://arf.dev/enterprise)
    

🙏 Acknowledgments
------------------

Thanks to all contributors and testers who helped identify and fix these critical stability issues.
