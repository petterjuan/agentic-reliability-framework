# scripts/diagnose_ci_issues.py
import sys
import os
import traceback

def diagnose_imports():
    """Diagnose import issues in CI environment"""
    print("🔍 DIAGNOSING IMPORT ISSUES")
    print("=" * 50)
    
    # 1. Check Python path
    print("\n1. Python Path:")
    for i, path in enumerate(sys.path[:10]):  # First 10 entries
        print(f"   {i}: {path}")
    
    # 2. Check current directory
    print(f"\n2. Current Directory: {os.getcwd()}")
    
    # 3. Try importing arf_core
    print("\n3. Testing arf_core imports:")
    try:
        from agentic_reliability_framework.arf_core import HealingIntent
        print("   ✅ agentic_reliability_framework.arf_core.HealingIntent")
    except ImportError as e:
        print(f"   ❌ Failed: {e}")
    
    # 4. Try relative import
    print("\n4. Testing relative import:")
    try:
        # This is what mcp_server.py tries
        from ..arf_core import HealingIntent
        print("   ✅ ..arf_core.HealingIntent")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 5. Try direct import
    print("\n5. Testing direct import:")
    try:
        import arf_core
        print(f"   ✅ arf_core module: {arf_core}")
    except ImportError as e:
        print(f"   ❌ Failed: {e}")
    
    # 6. Check if arf_core exists
    print("\n6. Checking arf_core directory:")
    arf_core_path = os.path.join(os.getcwd(), "agentic_reliability_framework", "arf_core")
    if os.path.exists(arf_core_path):
        print(f"   ✅ Exists: {arf_core_path}")
        print(f"   Contents: {os.listdir(arf_core_path)}")
    else:
        print(f"   ❌ Not found: {arf_core_path}")

def diagnose_mcp_server():
    """Diagnose mcp_server.py specific issues"""
    print("\n🔍 DIAGNOSING MCP SERVER")
    print("=" * 50)
    
    mcp_server_path = os.path.join(
        os.getcwd(), 
        "agentic_reliability_framework", 
        "engine", 
        "mcp_server.py"
    )
    
    if not os.path.exists(mcp_server_path):
        print(f"❌ mcp_server.py not found at: {mcp_server_path}")
        return
    
    print(f"Found mcp_server.py at: {mcp_server_path}")
    
    # Read and check imports
    with open(mcp_server_path, 'r') as f:
        content = f.read()
    
    # Check for problematic imports
    import_lines = [line for line in content.split('\n') if 'import' in line and 'arf_core' in line]
    
    print("\nImport lines containing 'arf_core':")
    for line in import_lines:
        print(f"  {line.strip()}")

if __name__ == "__main__":
    diagnose_imports()
    diagnose_mcp_server()
