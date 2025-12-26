"""
OSS MCP Client Tests - Advisory only
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def oss_client():
    """Create OSS MCP client for testing"""
    try:
        from arf_core.engine.oss_mcp_client import OSSMCPClient
        return OSSMCPClient()
    except ImportError:
        pytest.skip("OSSMCPClient not available")


@pytest.mark.asyncio
class TestOSSMCPClient:
    """Tests for OSS-only MCP client"""
    
    async def test_client_advisory_only(self, oss_client):
        """Test that client only supports advisory mode"""
        assert oss_client.mode == "advisory"
        
        # Should not have execution methods
        assert not hasattr(oss_client, 'execute')
        assert not hasattr(oss_client, 'execute_tool')
        
        # Should have advisory methods
        assert hasattr(oss_client, 'analyze_and_recommend')
    
    async def test_creates_healing_intent(self, oss_client):
        """Test that client creates HealingIntent"""
        # Mock the analysis method
        oss_client._generate_justification = AsyncMock(
            return_value="Based on analysis, recommend restart"
        )
        oss_client._calculate_confidence = Mock(return_value=0.85)
        oss_client._find_similar_incidents = AsyncMock(return_value=[])
        
        # Test analysis
        intent = await oss_client.analyze_and_recommend(
            tool_name="restart_container",
            component="api-service",
            parameters={"force": True},
            context={"incident_id": "inc_123"}
        )
        
        # Verify HealingIntent
        assert intent.action == "restart_container"
        assert intent.component == "api-service"
        assert intent.requires_enterprise is True
    
    def test_client_info_includes_oss_limits(self, oss_client):
        """Test that client info includes OSS limits"""
        info = oss_client.get_client_info()
        
        assert info["edition"] == "oss"
        assert info["execution_allowed"] is False
        assert "advisory" in info["modes"]
        assert "autonomous" not in info["modes"]
        assert "approval" not in info["modes"]
