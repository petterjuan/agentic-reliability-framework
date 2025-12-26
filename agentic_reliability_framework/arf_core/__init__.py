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
    
    # OSS Config
    "OSSConfig",
    "load_oss_config_from_env",
]

# Lazy loading configuration
_LAZY_IMPORTS = {
    # Models
    "HealingIntent": ("agentic_reliability_framework.arf_core.models.healing_intent", "HealingIntent"),
    "HealingIntentSerializer": ("agentic_reliability_framework.arf_core.models.healing_intent", "HealingIntentSerializer"),
    
    # Engine - FIXED PATH: points to correct location
    "OSSMCPClient": ("agentic_reliability_framework.engine.oss_mcp_client", "OSSMCPClient"),
    "create_mcp_client": ("agentic_reliability_framework.engine.oss_mcp_client", "create_mcp_client"),
    
    # Constants
    "MAX_INCIDENT_HISTORY": ("agentic_reliability_framework.arf_core.constants", "MAX_INCIDENT_HISTORY"),
    "MCP_MODES_ALLOWED": ("agentic_reliability_framework.arf_core.constants", "MCP_MODES_ALLOWED"),
    "EXECUTION_ALLOWED": ("agentic_reliability_framework.arf_core.constants", "EXECUTION_ALLOWED"),
    "GRAPH_STORAGE": ("agentic_reliability_framework.arf_core.constants", "GRAPH_STORAGE"),
    "validate_oss_constants": ("agentic_reliability_framework.arf_core.constants", "validate_oss_constants"),
    "get_oss_capabilities": ("agentic_reliability_framework.arf_core.constants", "get_oss_capabilities"),
    
    # Config
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
            raise AttributeError(
                f"Module {__name__!r} has no attribute {name!r}. "
                f"Failed to import from {module_path}: {e}"
            ) from e
    
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")


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
