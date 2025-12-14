"""
Constants for memory module
"""

class MemoryConstants:
    """Memory-specific constants"""
    
    # FAISS
    FAISS_BATCH_SIZE = 10
    FAISS_SAVE_INTERVAL_SECONDS = 30
    VECTOR_DIM = 384
    
    # RAG Graph
    MAX_INCIDENT_NODES = 1000
    MAX_OUTCOME_NODES = 5000
    GRAPH_CACHE_SIZE = 100
    SIMILARITY_THRESHOLD = 0.3
