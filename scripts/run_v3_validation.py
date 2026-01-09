#!/usr/bin/env python3
"""
V3 Validation Runner - Single command to run all V3 boundary checks

Usage:
    python run_v3_validation.py [--fast] [--certify] [--output=report.json]
"""

import argparse
import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime


def run_script(script_path: Path, name: str) -> dict:
    """Run a script and return results."""
    print(f"   Running: {name}...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        passed = result.returncode == 0
        
        if passed:
            print(f"   ✅ {name} PASSED")
            # Show success message if present
            lines = result.stdout.split('\n')
            for line in lines:
                if "🎉" in line or "✅" in line or "PASSED" in line:
                    print(f"     {line.strip()}")
        else:
            print(f"   ❌ {name} FAILED")
            # Show error
            if result.stderr:
                print(f"     Error: {result.stderr[:200]}...")
            elif result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "❌" in line or "FAILED" in line or "ERROR" in line:
                        print(f"     {line.strip()}")
        
        return {
            "name": name,
            "script": script_path.name,
            "passed": passed,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        
    except subprocess.TimeoutExpired:
        print(f"   ⏰ {name} TIMEOUT")
        return {
            "name": name,
            "script": script_path.name,
            "passed": False,
            "error": "Timeout after 120 seconds",
        }
    except Exception as e:
        print(f"   💥 {name} ERROR: {e}")
        return {
            "name": name,
            "script": script_path.name,
            "passed": False,
            "error": str(e),
        }


def generate_certification(results: list) -> dict:
    """Generate V3 compliance certification."""
    all_passed = all(r.get("passed", False) for r in results)
    
    certification = {
        "version": "V3.0",
        "timestamp": datetime.utcnow().isoformat(),
        "validation_suite": "ARF V3 Boundary Validation",
        "overall_status": "VERIFIED" if all_passed else "FAILED",
        "components_validated": [
            "OSS/Enterprise Split",
            "Execution Ladder Boundaries",
            "Rollback API Boundaries",
            "License Manager Boundaries",
            "Cross-Repository Dependencies",
        ],
        "test_results": results,
        "compliance_levels": {
            "v3_0_advisory_intelligence": "VERIFIED" if all_passed else "FAILED",
            "v3_1_execution_governance": "PENDING",
            "v3_2_risk_bounded_autonomy": "PENDING",
            "v3_3_outcome_learning": "PENDING",
        },
        "boundary_verification": {
            "require_operator_vs_require_admin": "VERIFIED" if all_passed else "FAILED",
            "oss_execution_prevention": "VERIFIED" if all_passed else "FAILED",
            "enterprise_license_enforcement": "VERIFIED" if all_passed else "FAILED",
            "rollback_mandatory_analysis": "VERIFIED" if all_passed else "FAILED",
            "novel_execution_protocol": "VERIFIED" if all_passed else "FAILED",
        },
    }
    
    return certification


def main():
    parser = argparse.ArgumentParser(description="Run V3 boundary validation")
    parser.add_argument("--fast", action="store_true", 
                       help="Run only essential checks")
    parser.add_argument("--certify", action="store_true",
                       help="Generate V3 compliance certification")
    parser.add_argument("--output", type=str,
                       help="Output report file path")
    
    args = parser.parse_args()
    
    print("🚀 ARF V3 Boundary Validation Suite")
    print("=" * 60)
    
    # Determine which scripts to run
    if args.fast:
        print("⚡ Fast mode - running essential checks only")
        scripts = [
            ("oss_boundary_check.py", "OSS Boundary Check"),
            ("enhanced_v3_boundary_check.py", "Enhanced V3 Boundary Check"),
        ]
    else:
        print("🔍 Comprehensive mode - running all checks")
        scripts = [
            ("oss_boundary_check.py", "OSS Boundary Check"),
            ("enhanced_v3_boundary_check.py", "Enhanced V3 Boundary Check"),
            ("v3_boundary_integration.py", "V3 Boundary Integration"),
        ]
    
    if args.certify:
        print("🏆 Certification mode - generating V3 compliance certification")
    
    # Run all scripts
    print("\n" + "=" * 60)
    print("🧪 RUNNING VALIDATION SCRIPTS")
    print("=" * 60)
    
    results = []
    all_passed = True
    
    for script_file, script_name in scripts:
        script_path = Path(__file__).parent / script_file
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_file}")
            results.append({
                "name": script_name,
                "script": script_file,
                "passed": False,
                "error": f"Script not found: {script_file}"
            })
            all_passed = False
            continue
        
        result = run_script(script_path, script_name)
        results.append(result)
        
        if not result.get("passed", False):
            all_passed = False
    
    # Generate certification if requested
    if args.certify:
        print("\n" + "=" * 60)
        print("🏆 GENERATING V3 CERTIFICATION")
        print("=" * 60)
        
        certification = generate_certification(results)
        
        if all_passed:
            cert_path = Path("V3_COMPLIANCE_CERTIFICATION.json")
            with open(cert_path, 'w') as f:
                json.dump(certification, f, indent=2)
            print(f"✅ V3 Certification saved to: {cert_path}")
        else:
            issues_path = Path("V3_COMPLIANCE_ISSUES.json")
            with open(issues_path, 'w') as f:
                json.dump(certification, f, indent=2)
            print(f"⚠️  V3 Compliance Issues saved to: {issues_path}")
    
    # Generate output report if requested
    if args.output:
        output_path = Path(args.output)
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "args": vars(args),
            "results": results,
            "all_passed": all_passed,
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved to: {output_path}")
    
    # Final report
    print("\n" + "=" * 60)
    print("📊 FINAL VALIDATION REPORT")
    print("=" * 60)
    
    passed_count = sum(1 for r in results if r.get("passed", False))
    total_count = len(results)
    
    print(f"\nTests Run: {total_count}")
    print(f"Tests Passed: {passed_count}")
    print(f"Tests Failed: {total_count - passed_count}")
    
    if all_passed:
        print("\n🎉 ALL V3 VALIDATIONS PASSED!")
        print("\nThe system is V3 compliant with:")
        print("  • Mechanical OSS/Enterprise boundaries")
        print("  • Advisory-only execution in OSS")
        print("  • Proper license enforcement")
        print("  • Mandatory rollback analysis")
        
        if args.certify:
            print("\n✅ V3.0 ADVISORY INTELLIGENCE LOCK-IN VERIFIED")
            print("\nReady for V3.0 OSS package release!")
        
        sys.exit(0)
    else:
        print("\n🚨 V3 VALIDATION FAILURES DETECTED")
        print("\nFailed tests:")
        for result in results:
            if not result.get("passed", False):
                print(f"  • {result['name']}")
                if result.get("error"):
                    print(f"    Error: {result['error']}")
        
        print("\n🔧 Next steps:")
        print("  1. Run failed scripts individually for detailed output:")
        for result in results:
            if not result.get("passed", False):
                print(f"     python scripts/{result['script']}")
        print("  2. Fix identified boundary violations")
        print("  3. Re-run validation: python scripts/run_v3_validation.py")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
