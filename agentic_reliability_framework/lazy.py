"""
Simple lazy loading for ARF - No circular dependencies!
Pythonic improvements with type safety
"""

import threading
import logging
from typing import Callable, Optional, Dict, Any, TypeVar, Generic, cast
from contextlib import suppress
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LazyLoader(Generic[T]):
    """Thread-safe lazy loader with better typing support"""
    
    def __init__(self, loader_func: Callable[[], Optional[T]]) -> None:
        self._loader_func = loader_func
        self._lock = threading.RLock()
        self._instance: Optional[T] = None
    
    def __call__(self) -> Optional[T]:
        """Get or create instance (double-checked locking)"""
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._loader_func()
        return self._instance
    
    def reset(self) -> None:
        """Reset instance (for testing)"""
        with self._lock:
            self._instance = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if instance is loaded"""
        return self._instance is not None


# ========== GLOBAL INSTANCES WITH CONCRETE TYPES ==========

# These instances use concrete types for type safety in v3 engine
_rag_graph_instance: Optional['RAGGraphMemory'] = None
_mcp_server_instance: Optional['MCPServer'] = None
_engine_instance: Optional['ReliabilityEngineProtocol'] = None


# ========== TYPE-SAFE LOADER FUNCTIONS ==========

def _load_rag_graph() -> Optional['RAGGraphMemory']:
    """Create RAGGraphMemory for v3 features with graceful fallback"""
    global _rag_graph_instance
    
    if _rag_graph_instance is not None:
        return _rag_graph_instance
    
    with suppress(ImportError, Exception):
        from .memory.rag_graph import RAGGraphMemory
        from .config import config
        
        # Get FAISS index first (could return None)
        faiss_index = _load_faiss_index_safe()
        
        if faiss_index and config.rag_enabled:
            # Create RAG graph with the FAISS index
            _rag_graph_instance = RAGGraphMemory(faiss_index)
            logger.info("Initialized RAGGraphMemory for v3 features")
            return _rag_graph_instance
        else:
            logger.debug("RAG disabled or FAISS index not available")
    
    return None


def _load_mcp_server() -> Optional['MCPServer']:
    """Create MCP Server for v3 features with graceful fallback"""
    global _mcp_server_instance
    
    if _mcp_server_instance is not None:
        return _mcp_server_instance
    
    with suppress(ImportError, Exception):
        from .engine.mcp_server import MCPServer, MCPMode
        from .config import config
        
        if config.mcp_enabled:
            mcp_mode = MCPMode(config.mcp_mode)
            _mcp_server_instance = MCPServer(mode=mcp_mode)
            logger.info(f"Initialized MCPServer in {mcp_mode.value} mode")
            return _mcp_server_instance
        else:
            logger.debug("MCP disabled")
    
    return None


def _load_engine() -> Optional['ReliabilityEngineProtocol']:
    """Load the reliability engine without circular imports"""
    global _engine_instance
    
    if _engine_instance is not None:
        return _engine_instance
    
    with suppress(ImportError, Exception):
        # Use a factory pattern to avoid direct import
        from .engine.engine_factory import create_engine
        logger.info("Loading reliability engine via factory")
        _engine_instance = create_engine()
        return _engine_instance
    
    # Fallback to direct import only if factory fails
    with suppress(ImportError, Exception):
        from .engine.reliability import EnhancedReliabilityEngine
        logger.info("Loading EnhancedReliabilityEngine directly")
        _engine_instance = EnhancedReliabilityEngine()
        return cast(Optional['ReliabilityEngineProtocol'], _engine_instance)
    
    logger.error("Failed to load reliability engine")
    return None


def _load_agents() -> Optional['OrchestrationManager']:
    """Load agent orchestrator"""
    with suppress(ImportError, Exception):
        from .app import OrchestrationManager
        logger.info("Loading OrchestrationManager")
        return OrchestrationManager()
    
    logger.error("Failed to load agents")
    return None


def _load_faiss_index_safe() -> Optional['ProductionFAISSIndex']:
    """Load FAISS index with safe error handling"""
    with suppress(ImportError, Exception):
        from .memory.faiss_index import ProductionFAISSIndex, create_faiss_index
        logger.info("Loading FAISS index")
        index = create_faiss_index()
        # Ensure we return the proper type
        if isinstance(index, ProductionFAISSIndex):
            return index
        else:
            logger.warning(f"FAISS index has unexpected type: {type(index)}")
            return cast(ProductionFAISSIndex, index)  # Cast if needed
    
    logger.warning("FAISS index not available")
    return None


def _load_business_metrics() -> Optional['BusinessMetricsTracker']:
    """Load business metrics tracker"""
    with suppress(ImportError, Exception):
        from .engine.business import BusinessMetricsTracker
        logger.info("Loading BusinessMetricsTracker")
        return BusinessMetricsTracker()
    
    logger.warning("Business metrics tracker not available")
    return None


# ========== CREATE LAZY LOADERS WITH EXPLICIT TYPES ==========

# Note: We need to specify the type for each loader
rag_graph_loader: LazyLoader[Optional['RAGGraphMemory']] = LazyLoader(_load_rag_graph)
mcp_server_loader: LazyLoader[Optional['MCPServer']] = LazyLoader(_load_mcp_server)
engine_loader: LazyLoader[Optional['ReliabilityEngineProtocol']] = LazyLoader(_load_engine)
agents_loader: LazyLoader[Optional['OrchestrationManager']] = LazyLoader(_load_agents)
faiss_index_loader: LazyLoader[Optional['ProductionFAISSIndex']] = LazyLoader(_load_faiss_index_safe)
business_metrics_loader: LazyLoader[Optional['BusinessMetricsTracker']] = LazyLoader(_load_business_metrics)


# ========== PUBLIC API WITH CONCRETE TYPES ==========

def get_rag_graph() -> Optional['RAGGraphMemory']:
    """
    Get or create RAGGraphMemory (v3 feature)
    
    Returns:
        RAGGraphMemory instance or None if not available
    """
    return rag_graph_loader()


def get_mcp_server() -> Optional['MCPServer']:
    """
    Get or create MCPServer (v3 feature)
    
    Returns:
        MCPServer instance or None if not available
    """
    return mcp_server_loader()


def get_engine() -> Optional['ReliabilityEngineProtocol']:
    """
    Get or create reliability engine
    
    Returns:
        Reliability engine instance or None if not available
    """
    return engine_loader()


def get_agents() -> Optional['OrchestrationManager']:
    """
    Get or create agent orchestrator
    
    Returns:
        OrchestrationManager instance or None if not available
    """
    return agents_loader()


def get_faiss_index() -> Optional['ProductionFAISSIndex']:
    """
    Get or create FAISS index
    
    Returns:
        ProductionFAISSIndex instance or None if not available
    """
    return faiss_index_loader()


def get_business_metrics() -> Optional['BusinessMetricsTracker']:
    """
    Get or create business metrics tracker
    
    Returns:
        BusinessMetricsTracker instance or None if not available
    """
    return business_metrics_loader()


def enhanced_engine() -> Optional['ReliabilityEngineProtocol']:
    """
    Get enhanced reliability engine (alias for get_engine)
    
    Returns:
        EnhancedReliabilityEngine instance or None if not available
    """
    return get_engine()


def reset_all_instances() -> None:
    """Reset all lazy-loaded instances (for testing)"""
    global _rag_graph_instance, _mcp_server_instance, _engine_instance
    
    rag_graph_loader.reset()
    mcp_server_loader.reset()
    engine_loader.reset()
    agents_loader.reset()
    faiss_index_loader.reset()
    business_metrics_loader.reset()
    
    _rag_graph_instance = None
    _mcp_server_instance = None
    _engine_instance = None
    
    logger.info("Reset all lazy-loaded instances")


def get_v3_status() -> Dict[str, Any]:
    """Get v3 feature status"""
    from .config import config
    
    return {
        "rag_available": rag_graph_loader.is_loaded,
        "rag_instance_type": type(_rag_graph_instance).__name__ if _rag_graph_instance else "None",
        "mcp_available": mcp_server_loader.is_loaded,
        "mcp_instance_type": type(_mcp_server_instance).__name__ if _mcp_server_instance else "None",
        "engine_available": engine_loader.is_loaded,
        "rag_enabled": config.rag_enabled,
        "mcp_enabled": config.mcp_enabled,
        "learning_enabled": config.learning_enabled,
        "rollout_percentage": config.rollout_percentage,
        "beta_testing": config.beta_testing_enabled,
    }


# ========== TYPE-SAFE CONVENIENCE METHODS ==========

def ensure_rag_graph() -> 'RAGGraphMemory':
    """Get RAG graph or raise exception if not available"""
    rag = get_rag_graph()
    if rag is None:
        raise RuntimeError("RAGGraphMemory not available. Check if RAG is enabled in config.")
    return rag


def ensure_mcp_server() -> 'MCPServer':
    """Get MCP server or raise exception if not available"""
    mcp = get_mcp_server()
    if mcp is None:
        raise RuntimeError("MCPServer not available. Check if MCP is enabled in config.")
    return mcp


def ensure_engine() -> 'ReliabilityEngineProtocol':
    """Get engine or raise exception if not available"""
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Reliability engine not available.")
    return engine
