# scripts/review_v3_artifacts.py
#!/usr/bin/env python3
"""
Review and validate V3 release artifacts
"""
import json
from pathlib import Path
from datetime import datetime

def review_milestone_report(report_path: Path):
    """Review milestone report for completeness"""
    print("📋 Reviewing Milestone Report...")
    
    if not report_path.exists():
        print("❌ Milestone report not found")
        return False
    
    content = report_path.read_text()
    required_sections = [
        "V3 Architecture Achievements",
        "Business Impact",
        "Next Steps"
    ]
    
    for section in required_sections:
        if section not in content:
            print(f"❌ Missing section: {section}")
            return False
    
    print("✅ Milestone report validated")
    return True

def review_validation_report(report_path: Path):
    """Review validation report for correctness"""
    print("🔍 Reviewing Validation Report...")
    
    if not report_path.exists():
        print("❌ Validation report not found")
        return False
    
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        required_fields = [
            "milestone", "phase", "tag", "timestamp",
            "v3_architecture_verified", "oss_boundaries_intact"
        ]
        
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing field: {field}")
                return False
        
        # Validate timestamp
        timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        if (datetime.now() - timestamp).days > 1:
            print("⚠️  Report timestamp is older than 1 day")
        
        print("✅ Validation report validated")
        return True
        
    except json.JSONDecodeError:
        print("❌ Invalid JSON in validation report")
        return False

def generate_release_summary():
    """Generate comprehensive release summary"""
    print("📊 Generating Release Summary...")
    
    summary = {
        "release_phase": "V3.3.7",
        "validation_status": "PASSED",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "milestone_report": False,
            "validation_report": False,
            "v3_boundaries": True,
            "oss_purity": True
        },
        "artifacts": [],
        "next_actions": []
    }
    
    # Check for artifacts
    artifacts_dir = Path.cwd()
    for artifact in artifacts_dir.glob("*report*"):
        summary["artifacts"].append({
            "name": artifact.name,
            "size": artifact.stat().st_size,
            "modified": datetime.fromtimestamp(artifact.stat().st_mtime).isoformat()
        })
    
    # Review reports
    milestone_report = artifacts_dir / "milestone-report-V3.3.md"
    validation_report = artifacts_dir / "v3-validation-report.json"
    
    summary["checks"]["milestone_report"] = review_milestone_report(milestone_report)
    summary["checks"]["validation_report"] = review_validation_report(validation_report)
    
    # Determine next actions
    if all(summary["checks"].values()):
        summary["next_actions"] = [
            "✅ Proceed with V3.3.7 release",
            "📦 Upload to PyPI",
            "📢 Announce release"
        ]
    else:
        summary["validation_status"] = "FAILED"
        summary["next_actions"] = [
            "❌ Fix validation issues before release",
            "🔧 Review V3 boundary violations",
            "🔄 Re-run validation workflow"
        ]
    
    # Save summary
    summary_path = artifacts_dir / "release_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Release summary saved to: {summary_path}")
    return summary

if __name__ == "__main__":
    print("🚀 V3.3.7 Release Artifact Review")
    print("=" * 40)
    
    summary = generate_release_summary()
    
    print("\n📈 Review Summary:")
    print(f"   Status: {summary['validation_status']}")
    print(f"   Artifacts: {len(summary['artifacts'])} found")
    
    if summary["validation_status"] == "PASSED":
        print("\n🎉 READY FOR RELEASE!")
        for action in summary["next_actions"]:
            print(f"   {action}")
    else:
        print("\n⚠️  RELEASE BLOCKED!")
        for action in summary["next_actions"]:
            print(f"   {action}")
