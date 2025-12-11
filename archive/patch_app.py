#!/usr/bin/env python3
"""
Patch app.py to use lazy initialization
"""

import re
import sys

def patch_app_file():
    with open('arf/app.py', 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    patched_lines = []
    changes_made = []
    
    # Track if we've added the import
    lazy_import_added = False
    
    for i, line in enumerate(lines, 1):
        current_line = line
        
        # 1. Comment out the heavy initialization
        if 'enhanced_engine = EnhancedReliabilityEngine()' in line and not line.strip().startswith('#'):
            current_line = f'# {line}  # LAZY: Replaced with get_engine()'
            changes_made.append(f'Line {i}: Commented out EnhancedReliabilityEngine instantiation')
        
        # 2. Add import for lazy_init near other imports
        if not lazy_import_added and 'from .config import config' in line:
            # Add after the config imports
            patched_lines.append(current_line)
            patched_lines.append('from .lazy_init import get_engine, get_agents, get_faiss_index, enhanced_engine')
            lazy_import_added = True
            changes_made.append(f'Line {i+1}: Added lazy_init imports')
            continue
        
        # 3. Replace direct engine usage with get_engine() calls
        # We'll do this after the basic patch
        
        patched_lines.append(current_line)
    
    # Now create a second pass to replace engine usage
    final_lines = []
    for line in patched_lines:
        # Replace direct enhanced_engine usage with get_engine() calls
        # But not in comments or strings
        if 'enhanced_engine.' in line and not line.strip().startswith('#'):
            # Simple replacement for method calls
            new_line = line.replace('enhanced_engine.', 'get_engine().')
            if new_line != line:
                changes_made.append(f'Replaced enhanced_engine. with get_engine().')
            final_lines.append(new_line)
        else:
            final_lines.append(line)
    
    # Write the patched file
    with open('arf/app.py.patched', 'w') as f:
        f.write('\n'.join(final_lines))
    
    return changes_made

def create_backward_compatibility_shim():
    """
    Create a shim for any direct references to enhanced_engine variable
    """
    shim_code = '''
# LAZY INITIALIZATION BACKWARD COMPATIBILITY
# This ensures any direct references to 'enhanced_engine' variable still work
import sys

class EngineProxy:
    """Proxy that lazily loads the real engine"""
    def __getattr__(self, name):
        # Lazy load the engine when any attribute is accessed
        from .lazy_init import get_engine
        engine = get_engine()
        return getattr(engine, name)
    
    def __call__(self, *args, **kwargs):
        # Handle if someone tries to call enhanced_engine directly
        from .lazy_init import get_engine
        engine = get_engine()
        return engine(*args, **kwargs)

# Create proxy instance
enhanced_engine = EngineProxy()
'''
    
    with open('arf/engine_proxy.py', 'w') as f:
        f.write(shim_code)
    
    return "Created engine_proxy.py for backward compatibility"

if __name__ == '__main__':
    print("Patching app.py for lazy initialization...")
    
    # Create backup
    import shutil
    shutil.copy2('arf/app.py', 'arf/app.py.backup_pre_lazy')
    print("✅ Created backup: arf/app.py.backup_pre_lazy")
    
    # Patch the file
    changes = patch_app_file()
    
    # Create compatibility shim
    shim_msg = create_backward_compatibility_shim()
    
    print(f"\n✅ Patch complete with {len(changes)} changes:")
    for change in changes:
        print(f"  • {change}")
    print(f"  • {shim_msg}")
    
    print("\n📋 Next steps:")
    print("  1. Review the patch: diff arf/app.py arf/app.py.patched | head -50")
    print("  2. Apply the patch: mv arf/app.py.patched arf/app.py")
    print("  3. Test: python -c 'import agentic_reliability_framework as arf; print(\"Import successful\")'")
    print("  4. Test CLI: arf --version")
