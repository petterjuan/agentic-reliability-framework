#!/usr/bin/env python3
"""
Find Exact V3 Violations - Runs enhanced_v3_boundary_check.py and extracts the exact violations
"""

import subprocess
import sys
from pathlib import Path
import json

def run_enhanced_check():
    """Run enhanced_v3_boundary_check.py and capture output."""
    print("🔍 Running enhanced V3 boundary check to find exact violations...")
    print("=" * 70)
    
    script_path = Path(__file__).parent / "enhanced_v3_boundary_check.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        return result.stdout, result.stderr, result.returncode
        
    except subprocess.TimeoutExpired:
        return None, "Timeout expired", 1
    except Exception as e:
        return None, str(e), 1

def extract_violations_from_output(output: str):
    """Extract and parse violations from the output."""
    violations = []
    
    if not output:
        return violations
    
    lines = output.split('\n')
    
    # Look for the violation section
    in_violation_section = False
    current_file = None
    
    for line in lines:
        # Look for OSS Repository Violations section
        if "OSS Repository Violations" in line:
            in_violation_section = True
            continue
        
        # Look for file paths in violations
        if in_violation_section:
            # File paths typically start with bullet points
            if "• " in line and ":" in line:
                # Extract file path and violation
                parts = line.split(":", 1)
                if len(parts) == 2:
                    file_part = parts[0].strip("• ").strip()
                    violation_part = parts[1].strip()
                    
                    # Try to find the actual file
                    file_path = find_file(file_part)
                    if file_path:
                        violations.append({
                            'file': str(file_path),
                            'violation': violation_part,
                            'raw_line': line.strip()
                        })
    
    return violations

def find_file(file_part: str) -> Path:
    """Find the actual file from a partial path."""
    possible_paths = [
        Path(file_part),
        Path("agentic_reliability_framework") / file_part,
        Path("scripts") / file_part,
        Path("oss") / file_part,
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Try to find by glob pattern
    if "/" in file_part or "." in file_part:
        parts = file_part.split("/")
        if len(parts) > 1:
            # Try to find file with similar name
            search_pattern = f"**/{parts[-1]}"
            matches = list(Path(".").glob(search_pattern))
            if matches:
                return matches[0]
    
    return None

def show_file_content(file_path: Path, violation_desc: str):
    """Show relevant content from a file with violations."""
    if not file_path or not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n📄 FILE: {file_path}")
    print(f"   Violation: {violation_desc}")
    print("-" * 60)
    
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        # Show the file with line numbers
        for i, line in enumerate(lines, 1):
            # Look for patterns related to the violation
            if any(pattern in violation_desc.lower() for pattern in ['license', 'admin', 'autonomous', 'rollout', 'audit', 'mcp']):
                # Show lines that might contain the violation
                violation_keywords = ['license', 'admin', 'autonomous', 'rollout', 'audit', 'mcp', 'learning', 'beta']
                if any(keyword in line.lower() for keyword in violation_keywords):
                    print(f"{i:4}: {line.rstrip()}")
            else:
                # Show all lines for small files
                if len(lines) < 100:
                    print(f"{i:4}: {line.rstrip()}")
    
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    """Main function to find exact violations."""
    print("=" * 70)
    print("🔎 FINDING EXACT V3 BOUNDARY VIOLATIONS")
    print("=" * 70)
    print("\nThis script runs the validation and shows EXACTLY which files")
    print("and lines are causing the 12 violations.")
    print()
    
    # Run the enhanced check
    stdout, stderr, returncode = run_enhanced_check()
    
    if not stdout:
        print("❌ Failed to run validation:")
        print(stderr)
        return 1
    
    # Extract violations
    violations = extract_violations_from_output(stdout)
    
    if not violations:
        print("⚠️  Could not extract violations from output.")
        print("\nFull output:")
        print(stdout[:2000])
        return 1
    
    print(f"\n🚨 Found {len(violations)} violations mentioned in validation:")
    
    # Show each violation with file content
    for i, violation in enumerate(violations, 1):
        print(f"\n{i}. {violation['raw_line']}")
        
        file_path = Path(violation['file']) if violation['file'] else None
        if file_path and file_path.exists():
            show_file_content(file_path, violation['violation'])
        else:
            print(f"   File: {violation['file']}")
            print(f"   Could not locate file.")
    
    # Also show the original output for context
    print("\n" + "=" * 70)
    print("📋 ORIGINAL VALIDATION OUTPUT (Summary)")
    print("=" * 70)
    
    # Extract and show just the violation part
    lines = stdout.split('\n')
    showing = False
    for line in lines:
        if "OSS Repository Violations" in line:
            showing = True
        if showing:
            if line.strip() and not line.startswith("  ..."):
                print(line)
            if "OSS Constants:" in line:
                break
    
    # Generate actionable report
    print("\n" + "=" * 70)
    print("🎯 ACTION PLAN")
    print("=" * 70)
    
    if violations:
        print("\n1. For each file above, open it and:")
        print("   - Look for the violation patterns mentioned")
        print("   - Fix by:")
        print("     • Replacing require_admin() with require_operator()")
        print("     • Setting learning_enabled = False")
        print("     • Setting beta_testing_enabled = False")
        print("     • Setting rollout_percentage = 0")
        print("     • Changing mcp_mode to 'advisory'")
        print("     • Commenting out or removing Enterprise-only code")
        
        print("\n2. After fixing, run:")
        print("   python scripts/run_v3_validation.py --fast")
        
        print("\n3. If fixes work, get certification:")
        print("   python scripts/run_v3_validation.py --certify")
        
        # Save violations to file
        report_path = Path("EXACT_V3_VIOLATIONS.txt")
        with open(report_path, 'w') as f:
            f.write("EXACT V3 VIOLATIONS FOUND\n")
            f.write("=" * 50 + "\n\n")
            for violation in violations:
                f.write(f"{violation['raw_line']}\n")
                if violation['file']:
                    f.write(f"File: {violation['file']}\n")
                f.write(f"Description: {violation['violation']}\n")
                f.write("-" * 40 + "\n")
        
        print(f"\n📄 Detailed report saved to: {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
