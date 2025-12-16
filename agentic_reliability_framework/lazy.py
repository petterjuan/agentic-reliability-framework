"""
Lazy loading module for ARF components.
Prevents circular imports and enables graceful degradation.
"""

import logging
from typing import Optional, Any

from .config import config

logger = logging.getLogger(__name__)

# Global instances with lazy initialization
_ENGINE: Any = None
_AGENTS: Any = None
_FAISS_INDEX: Any = None
_BUSINESS_METRICS: Any = None
_RAG_GRAPH: Any = None
_MCP_SERVER: Any = None

# === Helper functions ===
def _safe_import(module_name: str, class_name: str) -> Optional[Any]:
    """Safely import a class with error handling"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        logger.error(f"Failed to import {class_name} from {module_name}: {e}")
        return None
    except AttributeError as e:
        logger.error(f"Class {class_name} not found in {module_name}: {e}")
        return None


# === Lazy getters ===
def get_engine() -> Any:
    """Get or create the main reliability engine"""
    global _ENGINE
    
    if _ENGINE is None:
        try:
            from .engine.engine_factory import create_engine
            _ENGINE = create_engine()
            logger.info("Lazy-loaded reliability engine")
        except Exception as e:
            logger.error(f"Failed to create engine: {e}", exc_info=True)
            # Return a minimal fallback engine
            from .engine.reliability import V3ReliabilityEngine
            _ENGINE = V3ReliabilityEngine()
    
    return _ENGINE


def get_agents() -> Any:
    """Get or create agent orchestration manager"""
    global _AGENTS
    
    if _AGENTS is None:
        try:
            from .app import OrchestrationManager
            _AGENTS = OrchestrationManager()
            logger.info("Lazy-loaded agent orchestration manager")
        except Exception as e:
            logger.error(f"Failed to create agents: {e}", exc_info=True)
            # Return minimal fallback
            _AGENTS = object()  # Placeholder
    
    return _AGENTS


def get_faiss_index() -> Any:
    """Get or create FAISS index"""
    global _FAISS_INDEX
    
    if _FAISS_INDEX is None:
        try:
            from .memory.faiss_index import create_faiss_index
            # FIX: Added missing dim parameter - using config or default
            dim = getattr(config, 'vector_dim', 384)  # Use config or default
            _FAISS_INDEX = create_faiss_index(dim=dim)
            logger.info(f"Lazy-loaded FAISS index with dim={dim}")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}", exc_info=True)
            # Return minimal fallback
            from .memory.faiss_index import ProductionFAISSIndex
            _FAISS_INDEX = ProductionFAISSIndex(dim=384)
    
    return _FAISS_INDEX


def get_business_metrics() -> Any:
    """Get or create business metrics tracker"""
    global _BUSINESS_METRICS
    
    if _BUSINESS_METRICS is None:
        try:
            from .engine.business import BusinessMetricsTracker
            _BUSINESS_METRICS = BusinessMetricsTracker()
            logger.info("Lazy-loaded business metrics tracker")
        except Exception as e:
            logger.error(f"Failed to create business metrics: {e}", exc_info=True)
            # Return minimal fallback
            _BUSINESS_METRICS = object()  # Placeholder
    
    return _BUSINESS_METRICS


def get_rag_graph() -> Any:
    """Get or create RAG graph memory"""
    global _RAG_GRAPH
    
    if _RAG_GRAPH is None:
        try:
            faiss_index = get_faiss_index()
            from .memory.rag_graph import RAGGraphMemory
            _RAG_GRAPH = RAGGraphMemory(faiss_index)
            logger.info("Lazy-loaded RAG graph memory")
        except Exception as e:
            logger.error(f"Failed to create RAG graph: {e}", exc_info=True)
            # Return minimal fallback
            _RAG_GRAPH = None  # Explicit None for disabled state
    
    return _RAG_GRAPH


def get_mcp_server() -> Any:
    """Get or create MCP server"""
    global _MCP_SERVER
    
    if _MCP_SERVER is None:
        try:
            from .engine.mcp_server import MCPServer
            _MCP_SERVER = MCPServer()
            logger.info("Lazy-loaded MCP server")
        except Exception as e:
            logger.error(f"Failed to create MCP server: {e}", exc_info=True)
            # Return minimal fallback
            _MCP_SERVER = None  # Explicit None for disabled state
    
    return _MCP_SERVER


def get_reliability_engine() -> Any:
    """Alias for get_engine for backward compatibility"""
    return get_engine()


def get_enhanced_reliability_engine() -> Any:
    """Get enhanced reliability engine (v3)"""
    try:
        from .engine.v3_reliability import create_v3_engine
        rag_graph = get_rag_graph()
        mcp_server = get_mcp_server()
        
        if rag_graph and mcp_server:
            engine = create_v3_engine(rag_graph=rag_graph, mcp_server=mcp_server)
            if engine:
                logger.info("Created enhanced V3 reliability engine")
                return engine
            else:
                logger.warning("Failed to create V3 engine, falling back to V2")
                return get_engine()
        else:
            logger.warning("Missing RAG or MCP for V3 engine, falling back to V2")
            return get_engine()
            
    except Exception as e:
        logger.error(f"Failed to create enhanced engine: {e}", exc_info=True)
        return get_engine()


# === Reset functions for testing ===
def reset_all() -> None:
    """Reset all lazy-loaded instances (for testing)"""
    global _ENGINE, _AGENTS, _FAISS_INDEX, _BUSINESS_METRICS, _RAG_GRAPH, _MCP_SERVER
    
    _ENGINE = None
    _AGENTS = None
    _FAISS_INDEX = None
    _BUSINESS_METRICS = None
    _RAG_GRAPH = None
    _MCP_SERVER = None
    
    logger.info("Reset all lazy-loaded instances")


def is_engine_loaded() -> bool:
    """Check if engine is loaded"""
    return _ENGINE is not None


def is_faiss_loaded() -> bool:
    """Check if FAISS is loaded"""
    return _FAISS_INDEX is not None


def is_rag_loaded() -> bool:
    """Check if RAG is loaded"""
    return _RAG_GRAPH is not None


def is_mcp_loaded() -> bool:
    """Check if MCP is loaded"""
    return _MCP_SERVER is not None
