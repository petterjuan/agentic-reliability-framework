"""
Engine factory for creating v2/v3 engines without circular imports
"""

import logging
from typing import Dict, Any, Optional
from contextlib import suppress

from ..config import config
from .interfaces import ReliabilityEngineProtocol

logger = logging.getLogger(__name__)


def create_engine() -> ReliabilityEngineProtocol:
    """
    Factory function to create appropriate reliability engine
    
    Returns:
        v2 or v3 engine based on configuration
    """
    # Check if v3 features should be enabled
    should_use_v3 = (
        config.rag_enabled or 
        config.mcp_enabled or 
        config.learning_enabled or 
        config.rollout_percentage > 0
    )
    
    if should_use_v3:
        with suppress(ImportError, Exception):
            # Try to create v3 engine
            engine = _create_v3_engine()
            if engine is not None:
                return engine
    
    # Fallback to v2 engine
    return _create_v2_engine()


def _create_v2_engine() -> ReliabilityEngineProtocol:
    """Create v2 reliability engine (base V3 engine)"""
    with suppress(ImportError):
        from .reliability import V3ReliabilityEngine
        logger.info("Creating base V3 reliability engine (v2 functionality)")
        engine = V3ReliabilityEngine()
        # Cast to protocol
        from typing import cast
        return cast(ReliabilityEngineProtocol, engine)
    
    # If V3ReliabilityEngine is not available, raise error
    raise ImportError("V3ReliabilityEngine not available")


def _create_v3_engine() -> ReliabilityEngineProtocol:
    """Create enhanced v3 reliability engine with RAG and MCP"""
    try:
        # Import lazily to avoid circular dependencies
        from ..lazy import get_rag_graph, get_mcp_server
        from .v3_reliability import V3ReliabilityEngine as EnhancedV3Engine
        
        rag_graph = get_rag_graph()
        mcp_server = get_mcp_server()
        
        # Check if we have the required v3 components
        if not (rag_graph and mcp_server):
            logger.warning("v3 components not available, falling back to base engine")
            return _create_v2_engine()
        
        logger.info("Creating enhanced V3 reliability engine with RAG and MCP")
        engine = EnhancedV3Engine(rag_graph=rag_graph, mcp_server=mcp_server)
        # Cast to protocol
        from typing import cast
        return cast(ReliabilityEngineProtocol, engine)
        
    except ImportError as e:
        logger.warning(f"Enhanced v3 engine not available: {e}")
        return _create_v2_engine()


class EngineFactory:
    """Engine factory with metadata"""
    
    @staticmethod
    def get_engine() -> ReliabilityEngineProtocol:
        """Get engine instance (singleton pattern)"""
        return create_engine()
    
    @staticmethod
    def get_engine_info() -> Dict[str, Any]:
        """Get engine information"""
        # Use available functions from lazy.py
        from ..lazy import (
            is_engine_loaded,
            is_rag_loaded,
            is_mcp_loaded,
            get_rag_graph,
            get_mcp_server
        )
        
        # Check actual types loaded
        rag_type = "unknown"
        mcp_type = "unknown"
        
        if is_rag_loaded():
            rag_instance = get_rag_graph()
            rag_type = type(rag_instance).__name__ if rag_instance else "None"
            
        if is_mcp_loaded():
            mcp_instance = get_mcp_server()
            mcp_type = type(mcp_instance).__name__ if mcp_instance else "None"
        
        # Determine engine type
        engine_type = "v3_enhanced" if (is_rag_loaded() and is_mcp_loaded()) else "v3_base"
        
        return {
            "engine_loaded": is_engine_loaded(),
            "rag_loaded": is_rag_loaded(),
            "rag_type": rag_type,
            "mcp_loaded": is_mcp_loaded(),
            "mcp_type": mcp_type,
            "v3_features_enabled": {
                "rag": config.rag_enabled,
                "mcp": config.mcp_enabled,
                "learning": config.learning_enabled,
                "rollout": config.rollout_percentage,
            },
            "engine_type": engine_type,
        }
