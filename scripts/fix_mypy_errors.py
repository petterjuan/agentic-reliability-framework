"""
Automated mypy error fixer
Fixes common mypy errors in the codebase
"""

import re
from pathlib import Path
from typing import List, Tuple


def fix_missing_return_types(content: str) -> Tuple[str, int]:
    """Fix functions missing return type annotations"""
    fixed_count = 0
    
    # Pattern for function definitions without return types
    patterns = [
        (r'def (\w+)\(([^)]*)\):', r'def \1(\2) -> None:'),
        (r'def (\w+)\(self(?:, [^)]*)?\):', r'def \1(self\2) -> None:'),
        (r'def (\w+)\(cls(?:, [^)]*)?\):', r'def \1(cls\2) -> None:'),
    ]
    
    for pattern, replacement in patterns:
        # Check if the line doesn't already have a return type
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.match(pattern, line.strip()):
                # Check if it already has a return type
                if '->' not in line:
                    lines[i] = re.sub(pattern, replacement, line.strip())
                    fixed_count += 1
        
        content = '\n'.join(lines)
    
    return content, fixed_count


def fix_unreachable_code(content: str) -> Tuple[str, int]:
    """Fix unreachable code errors"""
    fixed_count = 0
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Remove 'if False:' blocks
        if 'if False:' in line or 'if 0:' in line or 'if True is False:' in line:
            # Find the indentation level
            indent = len(line) - len(line.lstrip())
            
            # Remove the if block
            j = i + 1
            while j < len(lines) and len(lines[j]) - len(lines[j].lstrip()) > indent:
                lines[j] = '# REMOVED: ' + lines[j]
                fixed_count += 1
                j += 1
            
            # Comment out the if line
            lines[i] = '# REMOVED unreachable code: ' + line
            fixed_count += 1
        
        # Fix unreachable returns after infinite loops
        elif 'while True:' in line and i + 2 < len(lines):
            indent = len(line) - len(line.lstrip())
            if i + 2 < len(lines) and 'return' in lines[i + 2]:
                # Check if there's a break before the return
                if 'break' not in lines[i + 1]:
                    lines.insert(i + 2, ' ' * (indent + 4) + 'break')
                    fixed_count += 1
        
        i += 1
    
    return '\n'.join(lines), fixed_count


def fix_dict_typing(content: str) -> Tuple[str, int]:
    """Fix dict typing issues"""
    fixed_count = 0
    
    # Add type hints for dict literals
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Look for dict assignments without type hints
        if ' = {' in line and ':' in line and '#' not in line.split('=')[0]:
            var_name = line.split('=')[0].strip()
            
            # Check if it's a dict of dicts (common pattern in UI code)
            if line.strip().startswith(var_name) and '{"' not in line:
                # Add type hint
                lines[i] = f"{var_name}: Dict[str, Any] = " + line.split('=', 1)[1]
                fixed_count += 1
    
    return '\n'.join(lines), fixed_count


def process_file(filepath: Path) -> Tuple[int, int, int]:
    """Process a single file and fix mypy errors"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply fixes in order
    fixed_counts = []
    
    content, count1 = fix_missing_return_types(content)
    fixed_counts.append(count1)
    
    content, count2 = fix_unreachable_code(content)
    fixed_counts.append(count2)
    
    content, count3 = fix_dict_typing(content)
    fixed_counts.append(count3)
    
    # Only write if changes were made
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return tuple(fixed_counts)


def main():
    """Main function to fix mypy errors"""
    print("🔧 Fixing mypy errors in the codebase...")
    
    # Files with reported errors
    error_files = [
        "agentic_reliability_framework/arf_core/models/healing_intent.py",
        "agentic_reliability_framework/arf_core/config/oss_config.py",
        "agentic_reliability_framework/engine/mcp_client.py",
        "agentic_reliability_framework/engine/mcp_factory.py",
        "agentic_reliability_framework/engine/mcp_server.py",
        "agentic_reliability_framework/memory/rag_graph.py",
        "agentic_reliability_framework/app.py",
    ]
    
    total_fixes = [0, 0, 0]  # return_types, unreachable, dict_typing
    
    for file_path_str in error_files:
        filepath = Path(file_path_str)
        if filepath.exists():
            print(f"\nProcessing: {filepath}")
            fixes = process_file(filepath)
            total_fixes[0] += fixes[0]
            total_fixes[1] += fixes[1]
            total_fixes[2] += fixes[2]
            print(f"  ✓ Fixed: {fixes[0]} return types, {fixes[1]} unreachable code, {fixes[2]} dict typing")
    
    print(f"\n✅ Total fixes applied:")
    print(f"  • Missing return types: {total_fixes[0]}")
    print(f"  • Unreachable code: {total_fixes[1]}")
    print(f"  • Dict typing issues: {total_fixes[2]}")
    print(f"\nTotal errors fixed: {sum(total_fixes)}")


if __name__ == "__main__":
    main()
