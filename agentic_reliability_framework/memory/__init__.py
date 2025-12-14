"""
Memory module for vector storage and RAG graph functionality
Updated for v3
"""

from .faiss_index import ProductionFAISSIndex
from .enhanced_faiss import EnhancedFAISSIndex  # NEW
from .rag_graph import RAGGraphMemory
from .models import (
    IncidentNode, OutcomeNode, GraphEdge,
    SimilarityResult, NodeType, EdgeType
)
from .constants import MemoryConstants

__all__ = [
    'ProductionFAISSIndex',
    'EnhancedFAISSIndex',  # NEW
    'RAGGraphMemory',
    'IncidentNode',
    'OutcomeNode', 
    'GraphEdge',
    'SimilarityResult',
    'NodeType',
    'EdgeType',
    'MemoryConstants'
]
