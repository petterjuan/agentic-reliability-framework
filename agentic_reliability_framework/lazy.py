"""
Simple lazy loading for ARF - No circular dependencies!
"""

import threading
import logging
from typing import Callable, Optional, Any, Dict

logger = logging.getLogger(__name__)


class LazyLoader:
    """Simple thread-safe lazy loader"""
    def __init__(self, loader_func: Callable[[], Any]) -> None:
        self._loader_func = loader_func
        self._lock = threading.RLock()
        self._instance: Optional[Any] = None
    
    def __call__(self) -> Any:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._loader_func()
        return self._instance
    
    def reset(self) -> None:
        """Reset instance (for testing)"""
        with self._lock:
            self._instance = None


# ========== MODULE-LEVEL IMPORTS ==========

def _load_rag_graph() -> Optional[Any]:
    """Create RAGGraphMemory for v3 features with graceful fallback"""
    try:
        from .memory.rag_graph import RAGGraphMemory
        from ..config import config
        from . import get_faiss_index
        
        # Get FAISS index first
        faiss_index = get_faiss_index()
        
        if faiss_index and config.rag_enabled:
            # Create RAG graph with the FAISS index
            rag_graph = RAGGraphMemory(faiss_index)
            logger.info("Initialized RAGGraphMemory for v3 features")
            return rag_graph
        else:
            logger.debug("RAG disabled or FAISS index not available")
            return None
            
    except ImportError as e:
        logger.warning(f"RAGGraphMemory not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading RAGGraphMemory: {e}", exc_info=True)
        return None

def _load_mcp_server() -> Optional[Any]:
    """Create MCP Server for v3 features with graceful fallback"""
    try:
        from .mcp_server import MCPServer, MCPMode
        from ..config import config
        
        if config.mcp_enabled:
            mcp_mode = MCPMode(config.mcp_mode)
            mcp_server = MCPServer(mode=mcp_mode)
            logger.info(f"Initialized MCPServer in {mcp_mode.value} mode")
            return mcp_server
        else:
            logger.debug("MCP disabled")
            return None
            
    except ImportError as e:
        logger.warning(f"MCPServer not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading MCPServer: {e}", exc_info=True)
        return None


# ========== CREATE LAZY LOADERS ==========

rag_graph_loader = LazyLoader(_load_rag_graph)
mcp_server_loader = LazyLoader(_load_mcp_server)


# ========== PUBLIC API ==========

def get_rag_graph() -> Optional[Any]:
    """
    Get or create RAGGraphMemory (v3 feature)
    
    Returns:
        RAGGraphMemory instance or None if not available
    """
    return rag_graph_loader()

def get_mcp_server() -> Optional[Any]:
    """
    Get or create MCPServer (v3 feature)
    
    Returns:
        MCPServer instance or None if not available
    """
    return mcp_server_loader()

def get_v3_status() -> Dict[str, Any]:
    """Get v3 feature status"""
    from ..config import config
    from .engine_factory import EngineFactory
    
    return {
        "engine_info": EngineFactory.get_engine_info(),
        "rag_available": get_rag_graph() is not None,
        "mcp_available": get_mcp_server() is not None,
        "rag_enabled": config.rag_enabled,
        "mcp_enabled": config.mcp_enabled,
        "learning_enabled": config.learning_enabled,
        "rollout_percentage": config.rollout_percentage,
    }

# ========== ADD THESE TO lazy.py ==========

# These functions should already exist elsewhere, export them here
def get_engine():
    """Get the reliability engine"""
    from .app import get_engine as _get_engine
    return _get_engine()

def get_agents():
    """Get the agent orchestrator"""
    from .app import get_agents as _get_agents
    return _get_agents()

def get_faiss_index():
    """Get the FAISS index instance"""
    from .app import get_faiss_index as _get_faiss_index
    return _get_faiss_index()

def get_business_metrics():
    """Get business metrics tracker"""
    from .app import get_business_metrics as _get_business_metrics
    return _get_business_metrics()

def enhanced_engine():
    """Get enhanced reliability engine"""
    return get_engine()
