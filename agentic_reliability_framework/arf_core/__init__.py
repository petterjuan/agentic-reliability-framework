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
    "HealingIntent": (".models.healing_intent", "HealingIntent"),
    "HealingIntentSerializer": (".models.healing_intent", "HealingIntentSerializer"),
    
    # Engine
    "OSSMCPClient": (".engine.mcp_client", "OSSMCPClient"),
    "create_mcp_client": (".engine.mcp_client", "create_mcp_client"),
    
    # Constants
    "MAX_INCIDENT_HISTORY": (".constants", "MAX_INCIDENT_HISTORY"),
    "MCP_MODES_ALLOWED": (".constants", "MCP_MODES_ALLOWED"),
    "EXECUTION_ALLOWED": (".constants", "EXECUTION_ALLOWED"),
    "GRAPH_STORAGE": (".constants", "GRAPH_STORAGE"),
    "validate_oss_constants": (".constants", "validate_oss_constants"),
    "get_oss_capabilities": (".constants", "get_oss_capabilities"),
    
    # Config
    "OSSConfig": (".config.oss_config", "OSSConfig"),
    "load_oss_config_from_env": (".config.oss_config", "load_oss_config_from_env"),
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
            module = importlib.import_module(module_path, __package__)
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
