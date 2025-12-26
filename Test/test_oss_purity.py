"""
OSS Purity Tests - Ensure no Enterprise code in OSS
"""

import pytest
import ast
import importlib
from pathlib import Path


class TestOSSPurity:
    """Tests to ensure OSS codebase purity"""
    
    def test_no_enterprise_imports(self):
        """Test that OSS code doesn't import Enterprise modules"""
        oss_dirs = ["agentic_reliability_framework/engine", 
                   "agentic_reliability_framework/memory",
                   "agentic_reliability_framework/config"]
        
        forbidden_imports = [
            "arf_enterprise",
            "enterprise",
            "license_key",
            "validate_license",
        ]
        
        violations = []
        
        for dir_path in oss_dirs:
            dir_path = Path(dir_path)
            if dir_path.exists():
                for py_file in dir_path.rglob("*.py"):
                    try:
                        content = py_file.read_text()
                        for forbidden in forbidden_imports:
                            if f"import {forbidden}" in content or f"from {forbidden}" in content:
                                violations.append(f"{py_file}: imports {forbidden}")
                    except Exception:
                        continue
        
        assert len(violations) == 0, f"Enterprise imports found:\n" + "\n".join(violations)
    
    def test_no_execution_methods_in_tools(self):
        """Test that OSS tools don't have execute methods"""
        from agentic_reliability_framework.engine.mcp_server import (
            RollbackTool, RestartContainerTool, ScaleOutTool
        )
        
        tools = [RollbackTool(), RestartContainerTool(), ScaleOutTool()]
        
        for tool in tools:
            # OSS tools should only have validate, not execute
            assert hasattr(tool, 'validate'), f"{tool.__class__.__name__} missing validate method"
            assert not hasattr(tool, 'execute'), f"{tool.__class__.__name__} has execute method (OSS violation)"
    
    def test_healing_intent_available(self):
        """Test that HealingIntent model is available"""
        try:
            from arf_core.models.healing_intent import HealingIntent
            assert HealingIntent is not None
        except ImportError:
            pytest.fail("HealingIntent model not available")
    
    def test_oss_client_advisory_only(self):
        """Test that OSS MCP client only supports advisory mode"""
        try:
            from arf_core.engine.oss_mcp_client import OSSMCPClient
            client = OSSMCPClient()
            assert client.mode == "advisory"
            assert not hasattr(client, 'execute_tool'), "OSS client should not have execute_tool"
        except ImportError:
            pytest.skip("OSSMCPClient not available")
