"""
OSS MCP Server Tests - Pure advisory mode
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


@pytest.fixture
def oss_server():
    """Create OSS-only MCP server"""
    from agentic_reliability_framework.engine.mcp_server import MCPServer
    return MCPServer()


@pytest.mark.asyncio
class TestMCPServerOSS:
    """Tests for OSS-only MCP server"""
    
    async def test_server_advisory_only(self, oss_server):
        """Test that server only supports advisory mode"""
        assert oss_server.mode.value == "advisory"
        assert oss_server.enforce_oss_purity() is True
    
    async def test_rejects_non_advisory_requests(self, oss_server):
        """Test that server rejects non-advisory modes"""
        # Try to send autonomous request
        request_dict = {
            "tool": "restart_container",
            "component": "api-service",
            "mode": "autonomous",  # ❌ Should be rejected
            "justification": "Test"
        }
        
        response = await oss_server.execute_tool(request_dict)
        
        # Should reject or override to advisory
        assert not response.executed
        assert "advisory" in response.message.lower() or response.status == "rejected"
    
    async def test_tools_advisory_only(self, oss_server):
        """Test that all tools are advisory only"""
        for tool_name, tool in oss_server.registered_tools.items():
            # OSS tools should only validate, not execute
            assert hasattr(tool, 'validate')
            assert not hasattr(tool, 'execute'), f"Tool {tool_name} has execute method"
            
            # Tool info should indicate OSS edition
            info = tool.get_tool_info()
            assert info["oss_edition"] is True
            assert info["can_execute"] is False
            assert info["requires_enterprise"] is True
    
    async def test_stats_show_oss_limits(self, oss_server):
        """Test that server stats show OSS limitations"""
        stats = oss_server.get_server_stats()
        
        assert stats["edition"] == "oss"
        assert stats["execution_allowed"] is False
        assert stats["oss_restricted"] is True
        assert stats["oss_limits"]["execution_allowed"] is False
