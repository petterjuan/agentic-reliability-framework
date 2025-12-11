"""
Lazy initialization for ARF heavy components - PROPER VERSION
Handles ProductionFAISSIndex with index and incident_texts parameters
"""

import threading
import logging
import os
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Lazy loader storage
_engine_instance = None
_agents_instance = None
_faiss_index_instance = None
_business_metrics_instance = None
_lock = threading.RLock()

def get_engine():
    """
    Lazy loader for EnhancedReliabilityEngine
    Only initializes when actually needed
    """
    global _engine_instance
    
    if _engine_instance is None:
        with _lock:
            if _engine_instance is None:
                logger.info("Lazy initializing EnhancedReliabilityEngine...")
                # Import here to avoid circular imports
                from .app import EnhancedReliabilityEngine
                _engine_instance = EnhancedReliabilityEngine()
                logger.info("EnhancedReliabilityEngine initialized")
    
    return _engine_instance

def get_agents() -> Dict[str, Any]:
    """
    Lazy loader for agent instances
    """
    global _agents_instance
    
    if _agents_instance is None:
        with _lock:
            if _agents_instance is None:
                logger.info("Lazy initializing agents...")
                from .app import (
                    AnomalyDetectionAgent, 
                    RootCauseAgent, 
                    PredictiveAgent,
                    OrchestrationManager,
                    SimplePredictiveEngine
                )
                
                # Need SimplePredictiveEngine for PredictiveAgent
                predictive_engine = SimplePredictiveEngine()
                
                # Create agents
                detective = AnomalyDetectionAgent()
                diagnostician = RootCauseAgent()
                predictive = PredictiveAgent(predictive_engine)  # FIXED: Added engine parameter
                
                # Create orchestration manager
                manager = OrchestrationManager(
                    detective=detective,
                    diagnostician=diagnostician,
                    predictive=predictive
                )
                
                _agents_instance = {
                    'detective': detective,
                    'diagnostician': diagnostician,
                    'predictive': predictive,
                    'manager': manager,
                    'predictive_engine': predictive_engine
                }
                logger.info("Agents initialized")
    
    return _agents_instance

def get_faiss_index():
    """
    Lazy loader for FAISS index
    Loads the index and incident_texts from files when needed
    """
    global _faiss_index_instance
    
    if _faiss_index_instance is None:
        with _lock:
            if _faiss_index_instance is None:
                logger.info("Lazy loading FAISS index...")
                from .app import ProductionFAISSIndex
                from .config import config
                
                # Load index and incident_texts from files (same as original code)
                index = None
                incident_texts = {}
                
                try:
                    import faiss
                    if os.path.exists(config.index_file):
                        logger.info(f"Loading existing FAISS index from {config.index_file}")
                        index = faiss.read_index(config.index_file)
                        
                        # Load incident texts
                        if os.path.exists(config.incident_texts_file):
                            with open(config.incident_texts_file, 'r') as f:
                                incident_texts = json.load(f)
                            logger.info(f"Loaded {len(incident_texts)} incident texts")
                        else:
                            logger.warning(f"Incident texts file not found: {config.incident_texts_file}")
                    else:
                        logger.info("No existing FAISS index found, will create empty one")
                        
                        # We need the model to create embeddings dimension
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                        dimension = model.get_sentence_embedding_dimension()
                        
                        # Create empty index
                        index = faiss.IndexFlatIP(dimension)
                        
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {e}")
                    # Create empty index as fallback
                    import faiss
                    index = faiss.IndexFlatIP(384)  # Default dimension
                
                # Create the ProductionFAISSIndex with loaded data
                _faiss_index_instance = ProductionFAISSIndex(index, incident_texts)
                logger.info("FAISS index loaded")
    
    return _faiss_index_instance

def get_business_metrics():
    """
    Lazy loader for BusinessMetricsTracker
    """
    global _business_metrics_instance
    
    if _business_metrics_instance is None:
        with _lock:
            if _business_metrics_instance is None:
                logger.info("Lazy creating BusinessMetricsTracker...")
                from .app import BusinessMetricsTracker
                _business_metrics_instance = BusinessMetricsTracker()
                logger.info("BusinessMetricsTracker created")
    
    return _business_metrics_instance

def reset_all():
    """
    Reset all lazy instances (for testing)
    """
    global _engine_instance, _agents_instance, _faiss_index_instance, _business_metrics_instance
    with _lock:
        _engine_instance = None
        _agents_instance = None
        _faiss_index_instance = None
        _business_metrics_instance = None
    logger.debug("All lazy instances reset")

# Backward compatibility alias
def enhanced_engine():
    """Alias for get_engine() for backward compatibility"""
    return get_engine()

# Initialize function for CLI that needs everything
def initialize_all():
    """
    Force initialization of all components
    Useful for CLI commands that need everything ready
    """
    logger.info("Initializing all ARF components...")
    get_engine()
    get_agents()
    get_faiss_index()
    get_business_metrics()
    logger.info("All components initialized")
