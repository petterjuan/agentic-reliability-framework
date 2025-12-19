"""
MCP Server Factory - Detects OSS vs Enterprise and returns appropriate implementation
Maintains 100% backward compatibility while enabling clean separation
"""

import os
import logging
import importlib.util
from typing import Dict, Any, Optional, Union, Type, cast, overload, TYPE_CHECKING

from .mcp_server import MCPServer, MCPMode
from .mcp_client import OSSMCPClient, create_mcp_client

# Type checking only imports to avoid circular dependencies
if TYPE_CHECKING:
    from ..enterprise.mcp_server import EnterpriseMCPServer
    from ..oss.healing_intent import HealingIntent

logger = logging.getLogger(__name__)

# Type alias for factory returns
MCPInstance = Union["MCPServer", "OSSMCPClient"]


def detect_edition() -> str:
    """
    Detect whether to use OSS or Enterprise edition
    
    Detection logic:
    1. Check ARF_TIER environment variable
    2. Check ARF_LICENSE_KEY for Enterprise/Trial patterns
    3. Check for Enterprise module availability
    4. Default to OSS
    
    Returns:
        "oss" or "enterprise"
    """
    # 1. Check environment variable
    tier = os.getenv("ARF_TIER", "").lower()
    if tier == "enterprise":
        logger.debug("Edition detected: enterprise (via ARF_TIER)")
        return "enterprise"
    
    # 2. Check license key
    license_key = os.getenv("ARF_LICENSE_KEY", "")
    if license_key.startswith("ARF-ENT-") or license_key.startswith("ARF-TRIAL-"):
        logger.debug(f"Edition detected: enterprise (via license key: {license_key[:12]}...)")
        return "enterprise"
    
    # 3. Check if Enterprise module is available
    try:
        enterprise_spec = importlib.util.find_spec("agentic_reliability_framework.enterprise.mcp_server")
        if enterprise_spec is not None:
            logger.debug("Edition detected: enterprise (Enterprise module available)")
            return "enterprise"
    except Exception:
        pass
    
    # 4. Default to OSS
    logger.debug("Edition detected: oss (default)")
    return "oss"


def get_edition_info() -> Dict[str, Any]:
    """
    Get detailed edition information
    
    Returns:
        Dictionary with edition details
    """
    edition = detect_edition()
    license_key = os.getenv("ARF_LICENSE_KEY", "")
    tier = os.getenv("ARF_TIER", "oss")
    
    info = {
        "edition": edition,
        "tier": tier,
        "license_key_present": bool(license_key),
        "license_key_type": "none",
    }
    
    if license_key:
        if license_key.startswith("ARF-TRIAL-"):
            info["license_key_type"] = "trial"
        elif license_key.startswith("ARF-ENT-"):
            info["license_key_type"] = "enterprise"
        else:
            info["license_key_type"] = "unknown"
    
    # Add OSS capabilities if OSS
    if edition == "oss":
        try:
            from ..oss.constants import get_oss_capabilities
            info["capabilities"] = get_oss_capabilities()
        except ImportError:
            info["capabilities"] = {"edition": "oss", "license": "Apache 2.0"}
    
    return info


@overload
def create_mcp_server(
    mode: Optional[Union[str, MCPMode]] = None,
    config: Optional[Dict[str, Any]] = None,
    force_edition: None = None
) -> MCPInstance: ...

@overload
def create_mcp_server(
    mode: Optional[Union[str, MCPMode]] = None,
    config: Optional[Dict[str, Any]] = None,
    force_edition: Literal["oss"] = "oss"
) -> "OSSMCPClient": ...

@overload
def create_mcp_server(
    mode: Optional[Union[str, MCPMode]] = None,
    config: Optional[Dict[str, Any]] = None,
    force_edition: Literal["enterprise"] = "enterprise"
) -> "MCPServer": ...

def create_mcp_server(
    mode: Optional[Union[str, MCPMode]] = None,
    config: Optional[Dict[str, Any]] = None,
    force_edition: Optional[str] = None
) -> MCPInstance:
    """
    Factory function that creates appropriate MCP server based on edition
    
    This is the primary entry point for MCP server creation.
    It maintains backward compatibility while enabling clean separation.
    
    Args:
        mode: MCP mode (advisory, approval, autonomous)
            If None, uses default from config or edition
        config: Configuration dictionary
            If None, uses default configuration
        force_edition: Force specific edition ("oss" or "enterprise")
            Useful for testing
    
    Returns:
        MCPServer for Enterprise, OSSMCPClient for OSS
    
    Raises:
        ValueError: If mode is invalid for edition
        ImportError: If Enterprise edition requested but not available
    """
    # Determine edition
    if force_edition:
        edition = force_edition.lower()
        if edition not in ("oss", "enterprise"):
            raise ValueError(f"Invalid edition: {edition}. Must be 'oss' or 'enterprise'")
    else:
        edition = detect_edition()
    
    logger.info(f"Creating MCP server for {edition} edition")
    
    # Convert mode string to enum if needed
    mcp_mode: Optional[MCPMode] = None
    if mode:
        if isinstance(mode, str):
            try:
                mcp_mode = MCPMode(mode)
            except ValueError:
                raise ValueError(f"Invalid MCP mode: {mode}. Must be one of: {[m.value for m in MCPMode]}")
        elif isinstance(mode, MCPMode):
            mcp_mode = mode
        else:
            # This should never happen due to type annotations
            raise TypeError(f"Invalid mode type: {type(mode)}")
    
    # OSS Edition - always returns OSSMCPClient
    if edition == "oss":
        logger.info("📦 Using OSS MCP Client (advisory mode only)")
        
        # Validate mode for OSS
        if mcp_mode and mcp_mode != MCPMode.ADVISORY:
            logger.warning(
                f"OSS only supports advisory mode. "
                f"Requested mode '{mcp_mode.value}' will be ignored."
            )
        
        # Create OSS client
        client = create_mcp_client(config)
        
        # Log OSS capabilities
        try:
            from ..oss.constants import get_oss_capabilities
            capabilities = get_oss_capabilities()
            logger.info(f"OSS Capabilities: {capabilities['execution']}")
        except ImportError:
            logger.info("OSS Capabilities: advisory mode only")
        
        return client  # Type: OSSMCPClient
    
    # Enterprise Edition - returns MCPServer (or EnterpriseMCPServer)
    elif edition == "enterprise":
        logger.info("🚀 Using Enterprise MCP Server (all modes available)")
        
        try:
            # Check if Enterprise module exists
            enterprise_spec = importlib.util.find_spec("agentic_reliability_framework.enterprise.mcp_server")
            if enterprise_spec is None:
                raise ImportError("Enterprise module not found")
            
            # Import Enterprise MCPServer
            from ..enterprise.mcp_server import create_enterprise_mcp_server
            
            # Create Enterprise server
            server = create_enterprise_mcp_server(mode=mcp_mode, config=config)
            
            # Log Enterprise features
            stats = server.get_server_stats()
            if "enterprise" in stats:
                logger.info(f"Enterprise Features: {stats['enterprise']}")
            
            return server  # Type: MCPServer (or subclass)
            
        except ImportError as e:
            logger.error(f"Enterprise features not available: {e}")
            logger.warning("Falling back to OSS edition")
            
            # Fall back to OSS
            oss_client = create_mcp_client(config)
            # We need to cast to MCPInstance since we're returning from enterprise branch
            return cast(MCPInstance, oss_client)
    
    else:
        # This should never happen due to validation above
        raise ValueError(f"Unknown edition: {edition}")


def get_mcp_server_class() -> Union[Type["MCPServer"], Type["OSSMCPClient"]]:
    """
    Get the appropriate MCP server class based on edition
    
    Useful for type annotations and testing
    
    Returns:
        MCPServer class for Enterprise, OSSMCPClient class for OSS
    """
    edition = detect_edition()
    
    if edition == "enterprise":
        try:
            enterprise_spec = importlib.util.find_spec("agentic_reliability_framework.enterprise.mcp_server")
            if enterprise_spec is not None:
                from ..enterprise.mcp_server import EnterpriseMCPServer
                return EnterpriseMCPServer
        except ImportError:
            pass
        
        # Fall back to OSS if Enterprise not available
        return OSSMCPClient
    else:
        return OSSMCPClient


def create_healing_intent_from_request(request_dict: Dict[str, Any]) -> "HealingIntent":
    """
    Create HealingIntent from request (OSS only feature)
    
    Args:
        request_dict: MCP request dictionary
        
    Returns:
        HealingIntent object
        
    Raises:
        ImportError: If OSS features not available
    """
    try:
        from ..oss.healing_intent import HealingIntent
        return HealingIntent.from_mcp_request(request_dict)
    except ImportError as e:
        raise ImportError(
            "HealingIntent feature requires OSS module. "
            "Make sure arf-core is properly installed."
        ) from e


# Backward compatibility aliases
def get_mcp_server(*args: Any, **kwargs: Any) -> MCPInstance:
    """Backward compatibility alias for create_mcp_server"""
    return create_mcp_server(*args, **kwargs)


# Export
__all__ = [
    "detect_edition",
    "get_edition_info",
    "create_mcp_server",
    "get_mcp_server_class",
    "create_healing_intent_from_request",
    "get_mcp_server",  # Backward compatibility
]
