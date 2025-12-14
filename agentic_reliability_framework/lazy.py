"""
Simple lazy loading for ARF - No circular dependencies!

UPDATED: Now uses modular imports from memory/ and other packages
"""

import threading
import sys  # ADDED: Fix for F821 error
import logging  # MOVED: From line 200 to here to fix E402
from typing import Callable, Optional, Any, Dict
import os

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

def _load_engine() -> Any:
    """Import and create EnhancedReliabilityEngine"""
    # Import here to avoid circular dependency
    from .engine.reliability import EnhancedReliabilityEngine
    return EnhancedReliabilityEngine()

def _load_agents() -> Dict[str, Any]:
    """Create all agent instances using modular imports"""
    from .agents.detective import AnomalyDetectionAgent
    from .agents.diagnostician import RootCauseAgent
    from .agents.predictive import PredictiveAgent
    from .agents.orchestrator import OrchestrationManager
    from .engine.predictive import SimplePredictiveEngine
    
    predictive_engine = SimplePredictiveEngine()
    detective = AnomalyDetectionAgent()
    diagnostician = RootCauseAgent()
    predictive = PredictiveAgent(predictive_engine)
    
    return {
        'detective': detective,
        'diagnostician': diagnostician,
        'predictive': predictive,
        'manager': OrchestrationManager(detective, diagnostician, predictive),
        'predictive_engine': predictive_engine
    }

def _load_faiss_index() -> Any:
    """Load or create FAISS index using memory module"""
    import json
    import faiss
    from .config import config
    from .memory.faiss_index import ProductionFAISSIndex
    from sentence_transformers import SentenceTransformer
    
    # FIXED: Use correct attribute names from config.py
    index_file = config.index_file
    texts_file = config.incident_texts_file
    
    # Load or create index
    if os.path.exists(index_file):
        try:
            index = faiss.read_index(index_file)
            incident_texts = []
            if os.path.exists(texts_file):
                with open(texts_file, 'r') as f:
                    incident_texts = json.load(f)
            logger.debug(f"Loaded existing FAISS index with {len(incident_texts)} vectors")
        except Exception as e:
            logger.warning(f"Error loading FAISS index, creating new: {e}")
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            dimension = model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatIP(dimension)
            incident_texts = []
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dimension)
        incident_texts = []
        logger.info(f"Created new FAISS index with dimension {dimension}")
    
    return ProductionFAISSIndex(index, incident_texts)

def _load_business_metrics() -> Any:
    """Create BusinessMetricsTracker"""
    from .engine.business import BusinessMetricsTracker
    return BusinessMetricsTracker()

def _load_rag_graph() -> Any:
    """Create RAGGraphMemory for v3 features"""
    try:
        from .memory.rag_graph import RAGGraphMemory
        from .memory.faiss_index import ProductionFAISSIndex  # noqa: F401
        
        # Get FAISS index first
        faiss_index = get_faiss_index()
        
        # Create RAG graph with the FAISS index
        rag_graph = RAGGraphMemory(faiss_index)
        logger.info("Initialized RAGGraphMemory for v3 features")
        return rag_graph
    except ImportError as e:
        logger.warning(f"RAGGraphMemory not available, falling back to basic FAISS: {e}")
        return None  # Return None for graceful degradation
    except Exception as e:
        logger.error(f"Error loading RAGGraphMemory: {e}", exc_info=True)
        return None

def _load_policy_engine() -> Any:
    """Create PolicyEngine"""
    from .healing_policies import PolicyEngine
    return PolicyEngine()

def _load_event_store() -> Any:
    """Create ThreadSafeEventStore"""
    from .engine.reliability import ThreadSafeEventStore
    return ThreadSafeEventStore()


# ========== CREATE LAZY LOADERS ==========

engine_loader = LazyLoader(_load_engine)
agents_loader = LazyLoader(_load_agents)
faiss_loader = LazyLoader(_load_faiss_index)
business_metrics_loader = LazyLoader(_load_business_metrics)
rag_graph_loader = LazyLoader(_load_rag_graph)
policy_engine_loader = LazyLoader(_load_policy_engine)
event_store_loader = LazyLoader(_load_event_store)


# ========== PUBLIC API ==========

def get_engine() -> Any:
    """Get or create EnhancedReliabilityEngine"""
    return engine_loader()

def get_agents() -> Dict[str, Any]:
    """Get or create agent instances"""
    return agents_loader()

def get_faiss_index() -> Any:
    """Get or create FAISS index"""
    return faiss_loader()

def get_business_metrics() -> Any:
    """Get or create BusinessMetricsTracker"""
    return business_metrics_loader()

def get_rag_graph() -> Optional[Any]:
    """
    Get or create RAGGraphMemory (v3 feature)
    
    Returns:
        RAGGraphMemory instance or None if not available
    """
    return rag_graph_loader()

def get_policy_engine() -> Any:
    """Get or create PolicyEngine"""
    return policy_engine_loader()

def get_event_store() -> Any:
    """Get or create ThreadSafeEventStore"""
    return event_store_loader()

def enhanced_engine() -> Any:
    """Alias for get_engine()"""
    return get_engine()

def get_rag_enabled() -> bool:
    """Check if RAG features are available and enabled"""
    rag_graph = get_rag_graph()
    return rag_graph is not None and hasattr(rag_graph, 'is_enabled') and rag_graph.is_enabled()

def reset_all() -> None:
    """Reset all loaders (for testing)"""
    engine_loader.reset()
    agents_loader.reset()
    faiss_loader.reset()
    business_metrics_loader.reset()
    rag_graph_loader.reset()
    policy_engine_loader.reset()
    event_store_loader.reset()
    logger.info("Reset all lazy loaders")

# ========== BACKWARD COMPATIBILITY ==========
def _create_legacy_compatibility_layer() -> None:
    """
    Create backward compatibility for existing code
    that might import from app directly
    """
    try:
        # Import key classes that might be referenced elsewhere
        from .engine.reliability import (
            EnhancedReliabilityEngine,
            ThreadSafeEventStore
        )
        from .engine.predictive import SimplePredictiveEngine
        from .engine.anomaly import AdvancedAnomalyDetector
        from .engine.business import BusinessImpactCalculator, BusinessMetricsTracker
        
        # Make them available at module level for backward compatibility
        globals().update({
            'EnhancedReliabilityEngine': EnhancedReliabilityEngine,
            'ThreadSafeEventStore': ThreadSafeEventStore,
            'SimplePredictiveEngine': SimplePredictiveEngine,
            'AdvancedAnomalyDetector': AdvancedAnomalyDetector,
            'BusinessImpactCalculator': BusinessImpactCalculator,
            'BusinessMetricsTracker': BusinessMetricsTracker,
        })
        
        logger.debug("Created backward compatibility layer")
    except ImportError as e:
        logger.warning(f"Could not create full backward compatibility: {e}")

# Initialize backward compatibility on module load
_create_legacy_compatibility_layer()


# ========== DIAGNOSTICS ==========
def get_diagnostics() -> Dict[str, Any]:
    """Get diagnostics about lazy loader state"""
    return {
        'engine_loaded': engine_loader._instance is not None,
        'agents_loaded': agents_loader._instance is not None,
        'faiss_loaded': faiss_loader._instance is not None,
        'business_metrics_loaded': business_metrics_loader._instance is not None,
        'rag_graph_loaded': rag_graph_loader._instance is not None,
        'policy_engine_loaded': policy_engine_loader._instance is not None,
        'event_store_loaded': event_store_loader._instance is not None,
        'rag_available': get_rag_graph() is not None,
        'module_path': __file__,
        'python_version': sys.version,  # NOW WORKS WITH sys IMPORT
    }
