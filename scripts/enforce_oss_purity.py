"""
Build-time enforcement of OSS purity
Apache 2.0 Licensed

Copyright 2025 Juan Petter

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
from pathlib import Path


def main():
    """Simple OSS boundary checker - no AST, just pattern matching"""
    print("🔍 OSS Boundary Check")
    
    violations = []
    
    # Critical Enterprise patterns that MUST NOT be in OSS
    forbidden_patterns = [
        # Config fields (removed from OSS)
        "config.learning_enabled",
        "config.rollout_percentage", 
        "config.beta_testing_enabled",
        
        # MCP modes (Enterprise only)
        "MCPMode.APPROVAL",
        "MCPMode.AUTONOMOUS",
        
        # License/Enterprise references
        "license_key",
        "validate_license",
        "EnterpriseMCPServer",
        "audit_trail",
    ]
    
    # Check only critical files
    critical_files = [
        Path("agentic_reliability_framework/config.py"),
        Path("agentic_reliability_framework/engine/engine_factory.py"),
        Path("agentic_reliability_framework/engine/mcp_server.py"),
        Path("agentic_reliability_framework/engine/mcp_factory.py"),
        Path("agentic_reliability_framework/app.py"),
        Path("agentic_reliability_framework/cli.py"),
    ]
    
    for filepath in critical_files:
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue
            
        try:
            content = filepath.read_text()
            
            # Check each forbidden pattern
            for pattern in forbidden_patterns:
                if pattern in content:
                    # Find line number for better error messages
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        if pattern in line:
                            violations.append(
                                f"{filepath}:{line_num}: {pattern}"
                            )
                            break
            
        except Exception as e:
            print(f"⚠️  Error reading {filepath}: {e}")
            continue
    
    # Report results
    if violations:
        print("\n❌ OSS BOUNDARY VIOLATIONS DETECTED:")
        for violation in violations:
            print(f"  {violation}")
        print("\n🚫 Build failed: Enterprise code detected in OSS repository")
        print("\n💡 Fix these violations:")
        print("   - Replace config.learning_enabled with False")
        print("   - Replace config.rollout_percentage with 0")
        print("   - Remove MCPMode.APPROVAL and MCPMode.AUTONOMOUS")
        print("   - Remove license validation code")
        sys.exit(1)
    else:
        print("\n✅ All critical files are OSS-compliant")
        print("🎉 Ready for OSS package extraction!")
        sys.exit(0)


if __name__ == "__main__":
    main()
