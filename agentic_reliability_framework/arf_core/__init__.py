"""
ARF Core Module - OSS Edition
Production-grade multi-agent AI for reliability monitoring
OSS Edition: Advisory mode only, Apache 2.0 Licensed
"""

from typing import Any

# Public API exports
__all__ = [
    # OSS Models
    "HealingIntent",
    "HealingIntentSerializer",
    
    # OSS Engine
    "OSSMCPClient",
    "create_mcp_client",
    
    # OSS Constants
    "MAX_INCIDENT_HISTORY",
    "MCP_MODES_ALLOWED",
    "EXECUTION_ALLOWED",
    "GRAPH_STORAGE",
    "validate_oss_constants",
    "get_oss_capabilities",
    "OSSBoundaryError",
    
    # OSS Config
    "OSSConfig",
    "load_oss_config_from_env",
]

# Lazy loading configuration - USE MINIMAL IMPORTS TO AVOID CIRCULAR ISSUES
_LAZY_IMPORTS = {
    # Models - import directly to avoid config validation
    "HealingIntent": ("agentic_reliability_framework.arf_core.models.healing_intent", "HealingIntent"),
    "HealingIntentSerializer": ("agentic_reliability_framework.arf_core.models.healing_intent", "HealingIntentSerializer"),
    
    # Engine - use SIMPLE import to avoid triggering config validation
    "OSSMCPClient": ("agentic_reliability_framework.arf_core.engine.simple_mcp_client", "OSSMCPClient"),
    "create_mcp_client": ("agentic_reliability_framework.arf_core.engine.simple_mcp_client", "create_mcp_client"),
    
    # Constants - safe to import
    "MAX_INCIDENT_HISTORY": ("agentic_reliability_framework.arf_core.constants", "MAX_INCIDENT_HISTORY"),
    "MCP_MODES_ALLOWED": ("agentic_reliability_framework.arf_core.constants", "MCP_MODES_ALLOWED"),
    "EXECUTION_ALLOWED": ("agentic_reliability_framework.arf_core.constants", "EXECUTION_ALLOWED"),
    "GRAPH_STORAGE": ("agentic_reliability_framework.arf_core.constants", "GRAPH_STORAGE"),
    "validate_oss_constants": ("agentic_reliability_framework.arf_core.constants", "validate_oss_constants"),
    "get_oss_capabilities": ("agentic_reliability_framework.arf_core.constants", "get_oss_capabilities"),
    "OSSBoundaryError": ("agentic_reliability_framework.arf_core.constants", "OSSBoundaryError"),
    
    # Config - safe to import
    "OSSConfig": ("agentic_reliability_framework.arf_core.config.oss_config", "OSSConfig"),
    "load_oss_config_from_env": ("agentic_reliability_framework.arf_core.config.oss_config", "load_oss_config_from_env"),
}


def __getattr__(name: str) -> Any:
    """
    Lazy load modules on demand.
    
    This allows us to import heavy modules only when they're actually used.
    """
    if name in _LAZY_IMPORTS:
        import importlib
        
        module_path, attr_name = _LAZY_IMPORTS[name]
        
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        except ImportError as e:
            # Special handling for OSSMCPClient - create minimal version
            if name in ["OSSMCPClient", "create_mcp_client"]:
                print(f"⚠️  Creating minimal {name} (original import failed: {e})")
                return _create_minimal_mcp_client(name)
            raise AttributeError(
                f"Module {__name__!r} has no attribute {name!r}. "
                f"Failed to import from {module_path}: {e}"
            ) from e
    
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")


def _create_minimal_mcp_client(name: str) -> Any:
    """Create a minimal OSSMCPClient if import fails"""
    if name == "OSSMCPClient":
        # Create minimal class that doesn't trigger config validation
        class MinimalOSSMCPClient:
            def __init__(self, config=None):
                self.mode = "advisory"
                self.config = config or {}
            
            async def execute_tool(self, request_dict):
                from agentic_reliability_framework.arf_core.models.healing_intent import HealingIntent
                from datetime import datetime
                
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
                        "upgrade_url": "https://arf.dev/enterprise"
                    }
                }
            
            def get_client_stats(self):
                return {
                    "mode": self.mode,
                    "oss_edition": True,
                    "can_execute": False,
                    "can_advise": True,
                    "registered_tools": 6,
                    "enterprise_upgrade_available": True
                }
        
        return MinimalOSSMCPClient
    
    elif name == "create_mcp_client":
        def create_minimal_mcp_client(config=None):
            from agentic_reliability_framework.arf_core import OSSMCPClient
            return OSSMCPClient(config)
        return create_minimal_mcp_client
    
    return None


def __dir__() -> list[str]:
    """
    Return list of available attributes for tab completion.
    
    Combines direct attributes with lazy-loadable names.
    """
    import sys
    
    # Get current module attributes
    current_module = sys.modules[__name__]
    names = set(current_module.__dict__.keys())
    
    # Add all lazy-loadable names
    names.update(_LAZY_IMPORTS.keys())
    
    # Filter out private attributes
    public_names = sorted(name for name in names if not name.startswith("_"))
    
    return public_names


# Module metadata
OSS_EDITION = True
OSS_LICENSE = "Apache 2.0"
OSS_VERSION = "2.0.2"
ENTERPRISE_UPGRADE_URL = "https://arf.dev/enterprise"
