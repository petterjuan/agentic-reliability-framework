#!/usr/bin/env python3
"""
Show V3 Violations Only - Shows violations without fixing anything
"""

import re
from pathlib import Path
import sys

def scan_file_for_violations(file_path: Path):
    """Scan a file for V3 violations and return them."""
    violations = []
    
    if not file_path.exists():
        return violations
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # V3 boundary violation patterns
        patterns = [
            (r'license_key\s*=', 'License key assignment (Enterprise only)'),
            (r'require_admin\(', 'Admin requirement (use require_operator in OSS)'),
            (r'autonomous.*execute', 'Autonomous execution (Enterprise only)'),
            (r'learning_enabled\s*=\s*(True|\d+)', 'Learning enabled (must be False in OSS)'),
            (r'beta_testing_enabled\s*=\s*(True|\d+)', 'Beta testing enabled (must be False in OSS)'),
            (r'rollout_percentage\s*=\s*[1-9]\d*', 'Rollout percentage > 0 (must be 0 in OSS)'),
            (r'MCPMode\.APPROVAL', 'MCP approval mode (Enterprise only)'),
            (r'MCPMode\.AUTONOMOUS', 'MCP autonomous mode (Enterprise only)'),
            (r'mcp_mode\s*=\s*["\'](approval|autonomous)["\']', 'Non-advisory MCP mode (advisory only in OSS)'),
            (r'audit_trail\s*=', 'Audit trail (Enterprise only)'),
            (r'audit_log\s*=', 'Audit log (Enterprise only)'),
            (r'graph_storage\s*=\s*["\'](neo4j|postgres|redis)["\']', 'Persistent storage (in-memory only in OSS)'),
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if it's commented out
                    stripped = line.strip()
                    if not stripped.startswith('#'):
                        # Check if it's in a string literal (crude check)
                        if '"' not in line or '"' not in line[line.find(pattern):]:
                            violations.append({
                                'file': str(file_path),
                                'line': line_num,
                                'description': description,
                                'code': line.strip()[:80]
                            })
    
    except Exception as e:
        violations.append({
            'file': str(file_path),
            'error': f"Error scanning file: {e}"
        })
    
    return violations

def main():
    """Show all V3 violations without fixing anything."""
    print("=" * 70)
    print("🔍 SHOWING V3 VIOLATIONS ONLY (NO FIXES APPLIED)")
    print("=" * 70)
    print("\nThis script scans for V3 boundary violations and shows them.")
    print("It does NOT fix anything - just shows what needs to be fixed.")
    print()
    
    # Files to check (based on validation output and common locations)
    files_to_check = [
        Path("oss/constants.py"),
        Path("agentic_reliability_framework/config.py"),
        Path("agentic_reliability_framework/arf_core/constants.py"),
        Path("agentic_reliability_framework/engine/mcp_server.py"),
        Path("agentic_reliability_framework/engine/mcp_factory.py"),
        Path("agentic_reliability_framework/engine/engine_factory.py"),
        Path("agentic_reliability_framework/cli.py"),
        Path("agentic_reliability_framework/app.py"),
        Path("agentic_reliability_framework/engine/mcp_client.py"),
        Path("agentic_reliability_framework/engine/oss_mcp_client_wrapper.py"),
    ]
    
    all_violations = []
    
    print("📋 Checking files for V3 violations:")
    print("-" * 70)
    
    for file_path in files_to_check:
        print(f"\n📄 {file_path}: ", end="")
        
        if not file_path.exists():
            print("File not found")
            continue
        
        violations = scan_file_for_violations(file_path)
        
        if violations:
            print(f"❌ {len(violations)} violations")
            for violation in violations:
                if 'error' in violation:
                    print(f"   Error: {violation['error']}")
                else:
                    print(f"   Line {violation['line']}: {violation['description']}")
                    print(f"     Code: {violation['code']}")
            all_violations.extend(violations)
        else:
            print("✅ No violations")
    
    # Also scan all Python files in the OSS directory
    print("\n" + "=" * 70)
    print("🔎 Scanning all OSS Python files for violations...")
    print("=" * 70)
    
    oss_files = list(Path("agentic_reliability_framework").glob("**/*.py"))
    other_violations = []
    
    for file_path in oss_files:
        # Skip test files
        if 'test' in str(file_path).lower():
            continue
        
        # Skip files we already checked
        if any(file_path.samefile(f) for f in files_to_check if f.exists()):
            continue
        
        violations = scan_file_for_violations(file_path)
        if violations:
            rel_path = file_path.relative_to(Path.cwd())
            print(f"\n📄 {rel_path}: {len(violations)} violations")
            for violation in violations[:2]:  # Show first 2
                print(f"   Line {violation['line']}: {violation['description']}")
            other_violations.extend(violations)
    
    all_violations.extend(other_violations)
    
    print("\n" + "=" * 70)
    print("📊 VIOLATION SUMMARY")
    print("=" * 70)
    
    if all_violations:
        print(f"\n🚨 TOTAL VIOLATIONS FOUND: {len(all_violations)}")
        
        # Group by file
        violations_by_file = {}
        for violation in all_violations:
            if 'error' not in violation:
                file = violation['file']
                violations_by_file[file] = violations_by_file.get(file, 0) + 1
        
        print("\n📋 Violations by file:")
        for file, count in violations_by_file.items():
            print(f"   • {Path(file).name}: {count} violations")
        
        print("\n🔍 Violation types:")
        violation_types = {}
        for violation in all_violations:
            if 'error' not in violation:
                desc = violation['description'].split(':')[0]
                violation_types[desc] = violation_types.get(desc, 0) + 1
        
        for desc, count in violation_types.items():
            print(f"   • {desc}: {count}")
        
        print("\n🎯 NEXT STEPS (Review these violations):")
        print("1. Open each file above and look at the violations")
        print("2. Decide how to fix each violation:")
        print("   - Replace require_admin() with require_operator()")
        print("   - Set learning_enabled = False")
        print("   - Set beta_testing_enabled = False")
        print("   - Set rollout_percentage = 0")
        print("   - Change mcp_mode to 'advisory'")
        print("   - Comment out or remove Enterprise-only code")
        print("3. After reviewing, you can run fix scripts")
        
        # Save report
        report_path = Path("V3_VIOLATIONS_REVIEW_REPORT.txt")
        with open(report_path, 'w') as f:
            f.write("V3 VIOLATIONS REVIEW REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total violations found: {len(all_violations)}\n\n")
            
            f.write("Files with violations:\n")
            for file, count in violations_by_file.items():
                f.write(f"  {Path(file).name}: {count} violations\n")
            
            f.write("\nDetailed violations:\n")
            for violation in all_violations:
                if 'error' not in violation:
                    f.write(f"\n{Path(violation['file']).name}: Line {violation['line']}\n")
                    f.write(f"  Type: {violation['description']}\n")
                    f.write(f"  Code: {violation['code']}\n")
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        return 1
    else:
        print("\n✅ NO V3 VIOLATIONS FOUND!")
        print("\nThe validation failures might be false positives.")
        print("Run the full validation to check:")
        print("  python scripts/enhanced_v3_boundary_check.py")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
