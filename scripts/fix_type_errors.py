"""
Fix specific mypy type errors in the codebase
"""

import re
from pathlib import Path
from typing import Dict, Any, List
import ast


def fix_app_py_dict_typing(content: str) -> str:
    """Fix dict typing errors in app.py"""
    # Lines 1147-1167 are causing dict-item errors
    # The issue is that UI components are typed as Dict[str, Any] but have complex dict values
    
    # Find and fix the problematic dict
    ui_component_pattern = r'ui_components = \{'
    if ui_component_pattern in content:
        # Extract the dict section
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this is the problematic dict
            if line.strip().startswith('"Markdown": {'):
                # This is the problematic section
                # Add proper type annotation
                fixed_lines.append('ui_components: Dict[str, Dict[str, Any]] = {')
                # Skip the original line
                i += 1
                continue
                
            fixed_lines.append(line)
            i += 1
        
        content = '\n'.join(fixed_lines)
    
    return content


def fix_missing_return_types(content: str, filepath: Path) -> str:
    """Add missing return type annotations"""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check for function definitions without return types
        if line.strip().startswith('def ') and '->' not in line:
            # Skip __post_init__ methods (they should be handled separately)
            if '__post_init__' in line:
                # Add -> None
                if line.endswith(':'):
                    line = line[:-1] + ' -> None:'
                else:
                    line = line + ' -> None'
            # Skip lines with decorators for now
            elif '@' not in line:
                # Add -> None for simple functions
                if line.endswith(':'):
                    line = line[:-1] + ' -> None:'
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_unreachable_code(content: str) -> str:
    """Fix unreachable code errors"""
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Remove 'if False:' blocks
        if 'if False:' in line or 'if 0:' in line:
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(f"{' ' * indent}# REMOVED unreachable code: {line.strip()}")
        else:
            fixed_lines.append(line)
        
        i += 1
    
    return '\n'.join(fixed_lines)


def process_file(filepath: Path) -> bool:
    """Process a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply fixes based on file type
        if 'app.py' in str(filepath):
            content = fix_app_py_dict_typing(content)
        
        content = fix_missing_return_types(content, filepath)
        content = fix_unreachable_code(content)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"⚠️  Error processing {filepath}: {e}")
        return False


def main():
    """Main function"""
    files_to_fix = [
        "agentic_reliability_framework/arf_core/models/healing_intent.py",
        "agentic_reliability_framework/arf_core/config/oss_config.py",
        "agentic_reliability_framework/engine/mcp_client.py",
        "agentic_reliability_framework/engine/mcp_factory.py",
        "agentic_reliability_framework/engine/mcp_server.py",
        "agentic_reliability_framework/memory/rag_graph.py",
        "agentic_reliability_framework/app.py",
        "agentic_reliability_framework/arf_core/engine/oss_mcp_client.py",
    ]
    
    print("🔧 Fixing mypy errors...")
    fixed_count = 0
    
    for filepath_str in files_to_fix:
        filepath = Path(filepath_str)
        if filepath.exists():
            if process_file(filepath):
                fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} files")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
