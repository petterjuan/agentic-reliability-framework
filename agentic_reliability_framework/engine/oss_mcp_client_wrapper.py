"""
OSS MCP Client Wrapper - Avoids circular dependencies
Imports from OSS package if available, provides fallback otherwise
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Initialize flags
HEALING_INTENT_AVAILABLE = False
OSSMCPClient = None
create_oss_mcp_client = None
HealingIntent = None

try:
    # Try to import from installed OSS package
    # This assumes arf_core is installed via pip
    import arf_core
    
    # Import OSS components
    from arf_core.engine.oss_mcp_client import (
        OSSMCPClient as OSSClient,
        create_oss_mcp_client as create_client,
        OSSMCPResponse
    )
    
    # Import HealingIntent
    from arf_core.models.healing_intent import HealingIntent as HI
    
    # Assign to module-level variables
    OSSMCPClient = OSSClient
    create_oss_mcp_client = create_client
    HealingIntent = HI
    HEALING_INTENT_AVAILABLE = True
    
    logger.info("✅ Successfully imported OSS components from arf_core package")
    
except ImportError as e:
    logger.warning(f"arf_core package not available: {e}")
    
    # Fallback implementation for development/testing
    class FallbackOSSMCPClient:
        """Fallback OSS client when arf_core is not installed"""
        def __init__(self):
            self.mode = "advisory"
            self._metrics = {"requests": 0}
        
        async def analyze_and_recommend(self, tool_name, component, parameters, context=None):
            from dataclasses import dataclass
            from datetime import datetime
            
            @dataclass
            class FallbackHealingIntent:
                action: str
                component: str
                parameters: Dict[str, Any]
                justification: str = ""
                confidence: float = 0.85
                incident_id: str = ""
                detected_at: Any = None
                
                def to_enterprise_request(self):
                    return {
                        "action": self.action,
                        "component": self.component,
                        "parameters": self.parameters,
                        "justification": self.justification,
                        "requires_enterprise": True,
                    }
            
            return FallbackHealingIntent(
                action=tool_name,
                component=component,
                parameters=parameters,
                justification=f"Fallback analysis for {tool_name} on {component}",
                confidence=0.75
            )
        
        def get_client_info(self):
            return {
                "status": "fallback_mode",
                "metrics": self._metrics,
                "cache_size": 0
            }
    
    # Assign fallback
    OSSMCPClient = FallbackOSSMCPClient
    
    def create_oss_mcp_client():
        return FallbackOSSMCPClient()
    
    create_oss_mcp_client = create_oss_mcp_client
    
    # Define fallback HealingIntent
    from dataclasses import dataclass
    
    @dataclass
    class FallbackHealingIntent:
        action: str
        component: str
        parameters: Dict[str, Any]
        justification: str = ""
        confidence: float = 0.85
        incident_id: str = ""
        
        def to_enterprise_request(self):
            return {
                "action": self.action,
                "component": self.component,
                "parameters": self.parameters,
                "justification": self.justification,
                "requires_enterprise": True,
            }
    
    HealingIntent = FallbackHealingIntent

__all__ = [
    "OSSMCPClient",
    "create_oss_mcp_client",
    "HealingIntent",
    "HEALING_INTENT_AVAILABLE",
]
