"""
Simple OSS MCP Client - No config validation triggers
Minimal implementation that avoids circular imports
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime


class OSSMCPClient:
    """
    Simple OSS MCP Client - Advisory mode only
    
    Minimal implementation that avoids importing the main config module.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.mode = "advisory"
        self.config = config or {}
        
    async def execute_tool(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """OSS advisory analysis - never executes"""
        # Import HealingIntent directly to avoid circular imports
        from ...arf_core.models.healing_intent import HealingIntent
        
        intent = HealingIntent(
            action=request_dict.get("tool", ""),
            component=request_dict.get("component", ""),
            parameters=request_dict.get("parameters", {}),
            justification=request_dict.get("justification", ""),
            confidence=0.85,
            incident_id=request_dict.get("metadata", {}).get("incident_id", ""),
            detected_at=datetime.now().timestamp()
        )
        
        return {
            "request_id": request_dict.get("request_id", "oss-request"),
            "status": "completed",
            "message": f"Advisory: Would execute {intent.action} on {intent.component}",
            "executed": False,
            "result": {
                "mode": "advisory",
                "healing_intent": intent.to_enterprise_request(),
                "requires_enterprise": True,
                "upgrade_url": "https://arf.dev/enterprise",
                "enterprise_features": [
                    "autonomous_execution",
                    "approval_workflows",
                    "learning_engine",
                    "persistent_storage",
                    "audit_trails",
                    "compliance_reporting"
                ]
            }
        }
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get OSS client statistics and capabilities"""
        return {
            "mode": self.mode,
            "oss_edition": True,
            "can_execute": False,
            "can_advise": True,
            "registered_tools": 6,
            "enterprise_upgrade_available": True,
            "enterprise_upgrade_url": "https://arf.dev/enterprise"
        }


def create_mcp_client(config: Optional[Dict[str, Any]] = None) -> OSSMCPClient:
    """
    Factory function for creating OSS MCP client
    """
    return OSSMCPClient(config)


__all__ = ["OSSMCPClient", "create_mcp_client"]
