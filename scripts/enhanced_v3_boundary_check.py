#!/usr/bin/env python3
"""
Enhanced V3 Boundary Validator
Build-time verification of OSS/Enterprise split with deep integration to constants

Apache 2.0 Licensed
Copyright 2025 Juan Petter
"""

import ast
import re
import json
import hashlib
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Any
import sys
import importlib.util

# Try to import OSS constants for validation
try:
    from agentic_reliability_framework.arf_core.constants import (
        OSSBoundaryError,
        MCP_MODES_ALLOWED,
        EXECUTION_ALLOWED,
        GRAPH_STORAGE,
        FAISS_INDEX_TYPE,
        MAX_INCIDENT_NODES,
        MAX_OUTCOME_NODES,
        OSS_EDITION,
        OSS_CONSTANTS_HASH,
        validate_oss_config,
        get_oss_capabilities,
        check_oss_compliance,
    )
    OSS_CONSTANTS_AVAILABLE = True
except ImportError:
    OSS_CONSTANTS_AVAILABLE = False
    print("⚠️  OSS constants not available - running in basic mode")


class V3BoundaryValidator:
    """Comprehensive V3 boundary validator with deep integration."""
    
    def __init__(self):
        self.oss_repo = Path(__file__).parent.parent
        self.enterprise_repo = self.oss_repo.parent / "agentic-reliability-enterprise"
        self.api_repo = self.oss_repo.parent / "arf-api-repository"
        
        # Load existing boundary patterns from enforce_oss_purity.py
        self._load_existing_patterns()
        
        # V3-specific boundaries from constants
        self.v3_boundaries = self._define_v3_boundaries()
        
        # Results storage
        self.results = {
            "oss": {"violations": [], "warnings": []},
            "enterprise": {"violations": [], "warnings": []},
            "api": {"violations": [], "warnings": []},
            "cross_deps": [],
            "constants_validation": {},
        }
    
    def _load_existing_patterns(self):
        """Load patterns from existing enforce_oss_purity.py"""
        # These patterns come from enforce_oss_purity.py
        self.existing_prohibited = {
            "audit_trail": "Audit trail variable assignment (Enterprise only)",
            "audit_log": "Audit log variable assignment (Enterprise only)",
            "license_key": "License key variable assignment (Enterprise only)",
            "learning_enabled": "Learning enabled flag (Enterprise only)",
            "rollout_percentage": "Rollout percentage assignment (Enterprise only)",
            "beta_testing_enabled": "Beta testing enabled flag (Enterprise only)",
            "class EnterpriseMCPServer": "Enterprise MCPServer class definition (Enterprise only)",
            "validate_license": "License validation function call (Enterprise only)",
            "MCPMode.APPROVAL": "APPROVAL mode usage in code (Enterprise only)",
            "MCPMode.AUTONOMOUS": "AUTONOMOUS mode usage in code (Enterprise only)",
        }
        
        self.existing_required = {
            "require_operator": "Operator confirmation required for any advisory action",
            "advisory_only": "Must mark advisory-only endpoints",
            "non_binding": "Confidence scores must be non-binding",
        }
    
    def _define_v3_boundaries(self) -> Dict[str, Dict[str, Any]]:
        """Define V3 architectural boundaries."""
        boundaries = {
            # === EXECUTION BOUNDARIES ===
            "require_operator": {
                "description": "OSS-only confirmation requirement",
                "allowed_in": ["oss"],
                "prohibited_in": ["enterprise", "api"],
                "pattern": r"require_operator\s*\(",
                "v3_component": "Execution Ladder",
            },
            "require_admin": {
                "description": "Enterprise-only admin approval",
                "allowed_in": ["enterprise", "api"],
                "prohibited_in": ["oss"],
                "pattern": r"require_admin\s*\(",
                "v3_component": "Execution Ladder",
            },
            "rollback_execute": {
                "description": "Rollback execution endpoint",
                "allowed_in": ["enterprise", "api"],
                "prohibited_in": ["oss"],
                "pattern": r"execute_rollback|rollback_execute",
                "v3_component": "Rollback API",
            },
            "novel_execution": {
                "description": "Novel action execution protocol",
                "allowed_in": ["enterprise"],
                "prohibited_in": ["oss", "api"],
                "pattern": r"novel_execution|validate_novel_execution",
                "v3_component": "Execution Governance",
            },
            
            # === LICENSE BOUNDARIES ===
            "license_validation": {
                "description": "License validation logic",
                "allowed_in": ["enterprise"],
                "prohibited_in": ["oss", "api"],
                "pattern": r"validate_license|has_enterprise_license",
                "v3_component": "License Manager",
            },
            "autonomous_execution": {
                "description": "Autonomous execution capability",
                "allowed_in": ["enterprise"],
                "prohibited_in": ["oss", "api"],
                "pattern": r"autonomous.*execute|AUTONOMOUS.*MODE",
                "v3_component": "Execution Governance",
            },
            
            # === MEMORY BOUNDARIES ===
            "neo4j_storage": {
                "description": "Neo4j graph storage (Enterprise only)",
                "allowed_in": ["enterprise", "api"],
                "prohibited_in": ["oss"],
                "pattern": r"neo4j|Neo4j|GraphDatabase",
                "v3_component": "Memory System",
            },
            "advanced_faiss": {
                "description": "Advanced FAISS indices (Enterprise only)",
                "allowed_in": ["enterprise"],
                "prohibited_in": ["oss"],
                "pattern": r"IndexIVF|IndexHNSW|IndexPQ|IndexScalarQuantizer",
                "v3_component": "Memory System",
            },
            
            # === FEATURE BOUNDARIES ===
            "learning_engine": {
                "description": "Learning engine (Enterprise only)",
                "allowed_in": ["enterprise"],
                "prohibited_in": ["oss"],
                "pattern": r"learning_engine|LearningEngine",
                "v3_component": "Outcome Learning Loop",
            },
            "audit_export": {
                "description": "Audit trail export (Enterprise only)",
                "allowed_in": ["enterprise", "api"],
                "prohibited_in": ["oss"],
                "pattern": r"export_audit|audit_export",
                "v3_component": "Audit System",
            },
            
            # === MCP BOUNDARIES ===
            "mcp_approval": {
                "description": "MCP approval mode (Enterprise only)",
                "allowed_in": ["enterprise"],
                "prohibited_in": ["oss"],
                "pattern": r"MCPMode\.APPROVAL|mcp_mode.*approval",
                "v3_component": "MCP Server",
            },
            "mcp_autonomous": {
                "description": "MCP autonomous mode (Enterprise only)",
                "allowed_in": ["enterprise"],
                "prohibited_in": ["oss"],
                "pattern": r"MCPMode\.AUTONOMOUS|mcp_mode.*autonomous",
                "v3_component": "MCP Server",
            },
        }
        
        return boundaries
    
    def analyze_file(self, file_path: Path, repo_type: str) -> Tuple[List[str], List[str]]:
        """
        Analyze a single file for boundary violations.
        
        Returns:
            Tuple of (violations, warnings)
        """
        violations = []
        warnings = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Skip test files for certain checks
            is_test_file = "test" in str(file_path).lower()
            
            # Check all V3 boundaries
            for boundary_name, boundary in self.v3_boundaries.items():
                if repo_type in boundary.get("prohibited_in", []):
                    # This boundary is prohibited here
                    if re.search(boundary["pattern"], content, re.IGNORECASE):
                        if not self._is_in_string_literal(content, boundary["pattern"]):
                            if not (is_test_file and "test" in boundary_name.lower()):
                                violations.append(
                                    f"{boundary_name}: {boundary['description']} "
                                    f"(V3: {boundary['v3_component']})"
                                )
            
            # Check existing patterns from enforce_oss_purity.py
            for pattern, description in self.existing_prohibited.items():
                if repo_type == "oss":  # These are prohibited in OSS
                    if pattern in content:
                        if not self._is_in_string_literal_simple(content, pattern):
                            violations.append(f"Existing pattern: {description}")
            
            # Check for missing required patterns in OSS
            if repo_type == "oss" and "engine" in str(file_path) or "mcp" in str(file_path):
                for pattern, description in self.existing_required.items():
                    if pattern not in content:
                        warnings.append(f"Missing required pattern: {description}")
            
            # Check OSS constants compliance
            if OSS_CONSTANTS_AVAILABLE and repo_type == "oss":
                self._check_oss_constants_compliance(file_path, content, violations)
            
        except Exception as e:
            warnings.append(f"Error analyzing file: {e}")
        
        return violations, warnings
    
    def _check_oss_constants_compliance(self, file_path: Path, content: str, violations: List[str]):
        """Check compliance with OSS constants."""
        # Check for hardcoded values that exceed OSS limits
        hardcoded_checks = [
            (r"max_events.*?=\s*(\d+)", MAX_INCIDENT_NODES, "MAX_INCIDENT_NODES"),
            (r"max_nodes.*?=\s*(\d+)", MAX_OUTCOME_NODES, "MAX_OUTCOME_NODES"),
            (r"rag_max.*?=\s*(\d+)", MAX_INCIDENT_NODES, "RAG node limit"),
        ]
        
        for pattern, limit, limit_name in hardcoded_checks:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    value = int(match)
                    if value > limit:
                        violations.append(
                            f"Hardcoded value {value} exceeds OSS {limit_name} limit of {limit}"
                        )
                except ValueError:
                    pass
        
        # Check for MCP mode violations
        if "mcp_mode" in content.lower() and "advisory" not in content.lower():
            # Look for non-advisory MCP modes
            non_advisory_modes = ["approval", "autonomous", "execute", "action"]
            for mode in non_advisory_modes:
                if mode in content.lower():
                    violations.append(
                        f"Non-advisory MCP mode detected: '{mode}'. "
                        f"OSS only supports 'advisory' mode."
                    )
        
        # Check for execution attempts
        if "execute(" in content and "recommend" not in content:
            # Check if it's a function definition or call
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "execute(" in line and "def " not in line and "#" not in line.lstrip():
                    violations.append(
                        f"Potential execution attempt at line {i+1}. "
                        f"OSS only supports advisory recommendations."
                    )
    
    def _is_in_string_literal(self, content: str, pattern: str) -> bool:
        """
        Check if pattern appears inside string literals using AST.
        
        Args:
            content: Python file content
            pattern: Pattern to check
            
        Returns:
            True if pattern only appears in string literals
        """
        try:
            tree = ast.parse(content)
            
            # Find all string literals
            string_literals = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    string_literals.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))
            
            # Find pattern occurrences
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    start_pos = match.start()
                    
                    # Check if this position is inside any string literal
                    in_string = False
                    for sl_start_line, sl_start_col, sl_end_line, sl_end_col in string_literals:
                        if line_num == sl_start_line == sl_end_line:
                            # Same line string literal
                            if sl_start_col <= start_pos <= sl_end_col:
                                in_string = True
                                break
                    
                    if not in_string:
                        return False
            
            return True  # All occurrences are in string literals
            
        except SyntaxError:
            # Fallback to simple check if AST parsing fails
            return self._is_in_string_literal_simple(content, pattern)
    
    def _is_in_string_literal_simple(self, content: str, pattern: str) -> bool:
        """Simple check for string literals (fallback)."""
        lines = content.split('\n')
        for line in lines:
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            # Check if pattern is in quotes
            if pattern in line:
                # Simple quote detection
                before = line.split(pattern)[0]
                after = line.split(pattern)[1]
                
                # Count quotes before and after
                quotes_before = before.count('"') + before.count("'")
                quotes_after = after.count('"') + after.count("'")
                
                # If odd number of quotes before and after, it's inside string
                if quotes_before % 2 == 1 and quotes_after % 2 == 1:
                    return True
        
        return False
    
    def validate_repository(self, repo_path: Path, repo_type: str) -> Dict[str, Any]:
        """Validate a single repository."""
        if not repo_path.exists():
            return {"violations": [], "warnings": [f"Repository not found: {repo_path}"]}
        
        all_violations = []
        all_warnings = []
        
        print(f"🔍 Analyzing {repo_type.upper()} repository: {repo_path.name}")
        
        # Find all Python files
        python_files = list(repo_path.glob("**/*.py"))
        print(f"  Found {len(python_files)} Python files")
        
        for i, py_file in enumerate(python_files):
            # Skip __pycache__ and test files for some checks
            if "__pycache__" in str(py_file):
                continue
            
            # Show progress
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(python_files)} files...")
            
            violations, warnings = self.analyze_file(py_file, repo_type)
            
            if violations or warnings:
                rel_path = py_file.relative_to(repo_path)
                for violation in violations:
                    all_violations.append(f"{rel_path}: {violation}")
                for warning in warnings:
                    all_warnings.append(f"{rel_path}: {warning}")
        
        return {
            "violations": all_violations,
            "warnings": all_warnings,
            "files_analyzed": len(python_files),
        }
    
    def check_cross_dependencies(self) -> List[str]:
        """Check for improper cross-repository dependencies."""
        violations = []
        
        # Check OSS doesn't import Enterprise
        for py_file in self.oss_repo.glob("**/*.py"):
            if "__pycache__" in str(py_file) or "test" in str(py_file).lower():
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Check for Enterprise imports
                enterprise_indicators = [
                    "import arf_enterprise",
                    "from arf_enterprise",
                    "agentic-reliability-enterprise",
                ]
                
                for indicator in enterprise_indicators:
                    if indicator in content:
                        # Check if it's in a comment or string
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if indicator in line:
                                stripped = line.strip()
                                if stripped.startswith('import') or stripped.startswith('from'):
                                    if not (stripped.startswith('#') or 
                                           (stripped.startswith('"""') or stripped.startswith("'''"))):
                                        rel_path = py_file.relative_to(self.oss_repo)
                                        violations.append(f"{rel_path}:{line_num}: {line.strip()}")
                                        break
                                
            except Exception as e:
                print(f"⚠️  Error checking {py_file}: {e}")
        
        return violations
    
    def validate_with_oss_constants(self) -> Dict[str, Any]:
        """Validate using OSS constants if available."""
        if not OSS_CONSTANTS_AVAILABLE:
            return {"available": False, "message": "OSS constants not imported"}
        
        results = {
            "available": True,
            "constants_hash": OSS_CONSTANTS_HASH,
            "edition": OSS_EDITION,
            "compliance_check": check_oss_compliance(),
            "capabilities": get_oss_capabilities(),
        }
        
        # Validate a sample configuration
        sample_config = {
            "mcp_mode": "advisory",
            "execution_allowed": False,
            "max_events_stored": 1000,
            "graph_storage": "in_memory",
            "faiss_index_type": "IndexFlatL2",
        }
        
        try:
            validate_oss_config(sample_config)
            results["sample_validation"] = "PASSED"
        except OSSBoundaryError as e:
            results["sample_validation"] = f"FAILED: {str(e)[:200]}"
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        print("\n" + "=" * 60)
        print("🧪 V3 ARCHITECTURAL BOUNDARY VALIDATION")
        print("=" * 60)
        
        # Validate OSS repository
        print("\n📦 1. OSS Repository Validation")
        oss_results = self.validate_repository(self.oss_repo, "oss")
        self.results["oss"] = oss_results
        
        # Validate Enterprise repository  
        print("\n💼 2. Enterprise Repository Validation")
        enterprise_results = self.validate_repository(self.enterprise_repo, "enterprise")
        self.results["enterprise"] = enterprise_results
        
        # Validate API repository
        print("\n🌐 3. API Repository Validation")
        api_results = self.validate_repository(self.api_repo, "api")
        self.results["api"] = api_results
        
        # Check cross dependencies
        print("\n🔗 4. Cross-Repository Dependency Check")
        cross_deps = self.check_cross_dependencies()
        self.results["cross_deps"] = cross_deps
        
        # Validate with OSS constants
        print("\n⚙️  5. OSS Constants Integration Check")
        constants_results = self.validate_with_oss_constants()
        self.results["constants_validation"] = constants_results
        
        # Generate summary
        total_violations = (
            len(oss_results["violations"]) +
            len(enterprise_results["violations"]) +
            len(api_results["violations"]) +
            len(cross_deps)
        )
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"\n🚨 Critical Violations: {total_violations}")
        print(f"⚠️   Warnings: {len(oss_results['warnings']) + len(enterprise_results['warnings']) + len(api_results['warnings'])}")
        
        if oss_results["violations"]:
            print(f"\n❌ OSS Repository Violations ({len(oss_results['violations'])}):")
            for violation in oss_results["violations"][:3]:  # Show first 3
                print(f"  • {violation}")
            if len(oss_results["violations"]) > 3:
                print(f"  ... and {len(oss_results['violations']) - 3} more")
        
        if cross_deps:
            print(f"\n❌ Cross-Repository Dependencies ({len(cross_deps)}):")
            for dep in cross_deps[:3]:
                print(f"  • {dep}")
            if len(cross_deps) > 3:
                print(f"  ... and {len(cross_deps) - 3} more")
        
        if OSS_CONSTANTS_AVAILABLE:
            print(f"\n✅ OSS Constants: {constants_results['constants_hash']}")
            print(f"✅ Edition: {constants_results['edition']}")
            print(f"✅ Compliance Check: {'PASS' if constants_results['compliance_check'] else 'FAIL'}")
        
        return {
            "total_violations": total_violations,
            "oss_violations": len(oss_results["violations"]),
            "enterprise_issues": len(enterprise_results["violations"]),
            "api_issues": len(api_results["violations"]),
            "cross_deps": len(cross_deps),
            "constants_available": OSS_CONSTANTS_AVAILABLE,
            "passed": total_violations == 0,
        }


def main():
    """Main validation entry point."""
    validator = V3BoundaryValidator()
    
    try:
        report = validator.generate_report()
        
        print("\n" + "=" * 60)
        if report["passed"]:
            print("🎉 V3 BOUNDARIES VERIFIED SUCCESSFULLY!")
            print("\n✅ All architectural boundaries are properly enforced.")
            print("✅ No cross-repository dependency violations.")
            
            if report["constants_available"]:
                print("✅ OSS constants integration verified.")
            
            print("\n📋 READY FOR:")
            print("  1. OSS package release (V3.0 - Advisory Intelligence Lock-In)")
            print("  2. Enterprise deployment (V3.1 - Execution Governance)")
            print("  3. API service deployment")
            
            # Generate a verification hash
            verification_data = {
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "oss_violations": report["oss_violations"],
                "cross_deps": report["cross_deps"],
                "passed": report["passed"],
            }
            verification_hash = hashlib.sha256(
                json.dumps(verification_data, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            print(f"\n🔐 Verification Hash: {verification_hash}")
            print("   (Include this hash in release notes)")
            
            sys.exit(0)
        else:
            print("🚨 V3 BOUNDARY VIOLATIONS DETECTED")
            print("\n🔧 REQUIRED ACTIONS:")
            print("  1. Fix OSS boundary violations first")
            print("  2. Remove Enterprise code from OSS repository")
            print("  3. Ensure proper use of require_operator vs require_admin")
            print("  4. Run enforce_oss_purity.py for detailed cleanup")
            print("\n💡 TIPS:")
            print("  • Use the existing enforce_oss_purity.py script")
            print("  • Check the specific file violations above")
            print("  • Verify OSS constants are properly imported")
            
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
