#!/usr/bin/env python3
"""
Direct Violation Check - Checks specific known problematic files
"""

from pathlib import Path
import sys

# Known problematic patterns and their fixes
PATTERNS_TO_CHECK = {
    'require_admin(': {
        'description': 'Admin requirement (must use require_operator in OSS)',
        'fix': 'require_operator(',
        'files_to_check': [
            'agentic_reliability_framework/engine/mcp_server.py',
            'agentic_reliability_framework/engine/mcp_factory.py',
            'agentic_reliability_framework/engine/engine_factory.py',
            'agentic_reliability_framework/cli.py',
        ]
    },
    'license_key =': {
        'description': 'License key assignment (Enterprise only)',
        'fix': '# license_key =  # REMOVED: Enterprise-only',
        'files_to_check': [
            'agentic_reliability_framework/config.py',
            'oss/constants.py',
            'agentic_reliability_framework/arf_core/constants.py',
        ]
    },
    'learning_enabled = True': {
        'description': 'Learning enabled (must be False in OSS)',
        'fix': 'learning_enabled = False  # OSS: Always False',
        'files_to_check': [
            'agentic_reliability_framework/config.py',
            'agentic_reliability_framework/engine/mcp_server.py',
        ]
    },
    'beta_testing_enabled = True': {
        'description': 'Beta testing enabled (must be False in OSS)',
        'fix': 'beta_testing_enabled = False  # OSS: Always False',
        'files_to_check': [
            'agentic_reliability_framework/config.py',
        ]
    },
    'rollout_percentage =': {
        'description': 'Rollout percentage (must be 0 in OSS)',
        'fix': 'rollout_percentage = 0  # OSS: Always 0',
        'files_to_check': [
            'agentic_reliability_framework/config.py',
        ]
    },
    'MCPMode.APPROVAL': {
        'description': 'MCP approval mode (Enterprise only)',
        'fix': '# MCPMode.APPROVAL  # REMOVED: Enterprise-only',
        'files_to_check': [
            'agentic_reliability_framework/engine/mcp_server.py',
            'agentic_reliability_framework/engine/mcp_factory.py',
        ]
    },
    'MCPMode.AUTONOMOUS': {
        'description': 'MCP autonomous mode (Enterprise only)',
        'fix': '# MCPMode.AUTONOMOUS  # REMOVED: Enterprise-only',
        'files_to_check': [
            'agentic_reliability_framework/engine/mcp_server.py',
            'agentic_reliability_framework/engine/mcp_factory.py',
        ]
    },
}

def check_file_for_pattern(file_path: Path, pattern: str, description: str) -> list:
    """Check a file for a specific pattern."""
    violations = []
    
    if not file_path.exists():
        return violations
    
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            if pattern in line:
                # Check if it's not commented out
                stripped = line.strip()
                if not stripped.startswith('#'):
                    violations.append({
                        'line': line_num,
                        'code': line.strip(),
                        'description': description
                    })
    
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
    
    return violations

def main():
    """Check all known problematic patterns."""
    print("=" * 70)
    print("🔍 DIRECT CHECK FOR KNOWN V3 VIOLATIONS")
    print("=" * 70)
    
    all_violations = []
    
    for pattern, info in PATTERNS_TO_CHECK.items():
        print(f"\n🔎 Checking pattern: {pattern}")
        print(f"   Description: {info['description']}")
        
        for file_path_str in info['files_to_check']:
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                print(f"   ⚠️  File not found: {file_path}")
                continue
            
            violations = check_file_for_pattern(file_path, pattern, info['description'])
            
            if violations:
                print(f"   ❌ {file_path}: {len(violations)} violations")
                for violation in violations[:2]:  # Show first 2
                    print(f"      Line {violation['line']}: {violation['code'][:60]}...")
                all_violations.extend(violations)
            else:
                print(f"   ✅ {file_path}: No violations")
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    if all_violations:
        print(f"\n🚨 Found {len(all_violations)} total violations")
        
        # Group by file
        violations_by_file = {}
        for violation in all_violations:
            # Extract file from pattern (a bit hacky)
            for pattern, info in PATTERNS_TO_CHECK.items():
                if pattern in violation.get('code', ''):
                    for file_path_str in info['files_to_check']:
                        violations_by_file[file_path_str] = violations_by_file.get(file_path_str, 0) + 1
                    break
        
        print("\n📋 Violations by file:")
        for file_path, count in violations_by_file.items():
            print(f"   • {file_path}: {count}")
        
        print("\n🔧 QUICK FIXES:")
        print("Run this command to see exact fixes:")
        print("python scripts/find_exact_v3_violations.py")
        
        return 1
    else:
        print("\n✅ No known violations found!")
        print("\nThe violations might be different patterns.")
        print("Run comprehensive check:")
        print("python scripts/identify_v3_violations.py")
        return 0

if __name__ == "__main__":
    sys.exit(main())
