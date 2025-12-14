"""
Complete RAG Graph Memory implementation for ARF v3

Phase 1: RAG Graph Foundation (2-3 weeks)
Goal: Make memory useful - retrieval before execution
"""

import numpy as np
import threading
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import OrderedDict
from dataclasses import asdict

from .faiss_index import ProductionFAISSIndex
from .enhanced_faiss import EnhancedFAISSIndex
from .models import (
    IncidentNode, OutcomeNode, GraphEdge, 
    SimilarityResult, EdgeType, NodeType
)
from .constants import MemoryConstants
from ..models import ReliabilityEvent
from ..config import config

logger = logging.getLogger(__name__)


class RAGGraphMemory:
    """
    Bridge between FAISS vectors and structured knowledge
    
    V3 Design Mandate 1: Memory → Decision influence (FAISS must be queried)
    
    Technical Decisions:
    - Start with in-memory graph (simple dicts) before database
    - FAISS must expose search - using EnhancedFAISSIndex
    - Incident IDs must be deterministic for idempotency
    """
    
    def __init__(self, faiss_index: ProductionFAISSIndex):
        """
        Initialize RAG Graph Memory
        
        Args:
            faiss_index: ProductionFAISSIndex instance
        """
        # Create enhanced FAISS index with search capability
        self.enhanced_faiss = EnhancedFAISSIndex(faiss_index)
        self.faiss = faiss_index
        
        # In-memory graph storage
        self.incident_nodes: Dict[str, IncidentNode] = {}
        self.outcome_nodes: Dict[str, OutcomeNode] = {}
        self.edges: List[GraphEdge] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "total_incidents_stored": 0,
            "total_outcomes_stored": 0,
            "total_edges_created": 0,
            "similarity_searches": 0,
            "cache_hits": 0,
            "last_search_time": None,
            "last_store_time": None,
        }
        
        # LRU cache for similarity results
        self._similarity_cache: OrderedDict[str, List[SimilarityResult]] = OrderedDict()
        self._max_cache_size = MemoryConstants.GRAPH_CACHE_SIZE
        
        # Embedding cache for performance
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._max_embedding_cache_size = 100
        
        logger.info(
            f"Initialized RAGGraphMemory for v3 features: "
            f"max_incidents={MemoryConstants.MAX_INCIDENT_NODES}, "
            f"max_outcomes={MemoryConstants.MAX_OUTCOME_NODES}, "
            f"cache_size={self._max_cache_size}"
        )
    
    def is_enabled(self) -> bool:
        """Check if RAG graph is enabled and ready"""
        return config.rag_enabled and (len(self.incident_nodes) > 0 or self.faiss.get_count() > 0)
    
    def _generate_incident_id(self, event: ReliabilityEvent) -> str:
        """
        Generate deterministic incident ID for idempotency
        
        Args:
            event: ReliabilityEvent to generate ID for
            
        Returns:
            Deterministic incident ID
        """
        # Create fingerprint from event data (excluding timestamp for idempotency)
        fingerprint_data = (
            f"{event.component}:"
            f"{event.service_mesh}:"
            f"{event.latency_p99:.2f}:"
            f"{event.error_rate:.4f}:"
            f"{event.throughput:.2f}"
        )
        
        # Use SHA-256 for security
        fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()
        
        # Return with prefix and truncation for readability
        return f"inc_{fingerprint[:16]}"
    
    def _generate_outcome_id(self, incident_id: str, actions_hash: str) -> str:
        """Generate outcome ID"""
        data = f"{incident_id}:{actions_hash}:{datetime.now().isoformat()}"
        return f"out_{hashlib.sha256(data.encode()).hexdigest()[:16]}"
    
    def _generate_edge_id(self, source_id: str, target_id: str, edge_type: EdgeType) -> str:
        """Generate edge ID"""
        data = f"{source_id}:{target_id}:{edge_type.value}:{datetime.now().isoformat()}"
        return f"edge_{hashlib.sha256(data.encode()).hexdigest()[:16]}"
    
    def _embed_incident(self, event: ReliabilityEvent, analysis: Dict[str, Any]) -> np.ndarray:
        """
        Create embedding vector from incident data
        
        Args:
            event: ReliabilityEvent
            analysis: Agent analysis results
            
        Returns:
            Embedding vector
        """
        cache_key = f"{event.fingerprint}:{hash(str(analysis))}"
        
        # Check cache first
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            # Create comprehensive embedding from event and analysis
            features = []
            
            # 1. Basic metrics (normalized)
            features.append(event.latency_p99 / 1000.0)  # Normalize to seconds
            features.append(event.error_rate)  # Already 0-1
            features.append(event.throughput / 10000.0)  # Normalize
            
            # 2. Resource utilization (if available)
            if event.cpu_util:
                features.append(event.cpu_util)
            else:
                features.append(0.0)
                
            if event.memory_util:
                features.append(event.memory_util)
            else:
                features.append(0.0)
            
            # 3. Severity encoding
            severity_map = {
                "low": 0.1,
                "medium": 0.3,
                "high": 0.7,
                "critical": 1.0
            }
            features.append(severity_map.get(event.severity.value, 0.1))
            
            # 4. Component hash (for component similarity)
            component_hash = int(hashlib.md5(event.component.encode()).hexdigest()[:8], 16) / 2**32
            features.append(component_hash)
            
            # 5. Analysis confidence (if available)
            if analysis and 'incident_summary' in analysis:
                confidence = analysis['incident_summary'].get('anomaly_confidence', 0.5)
                features.append(confidence)
            else:
                features.append(0.5)
            
            # Pad or truncate to target dimension
            target_dim = MemoryConstants.VECTOR_DIM
            if len(features) < target_dim:
                # Pad with zeros
                features = features + [0.0] * (target_dim - len(features))
            else:
                # Truncate
                features = features[:target_dim]
            
            embedding = np.array(features, dtype=np.float32)
            
            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Cache for performance
            self._embedding_cache[cache_key] = embedding
            
            # Manage cache size
            if len(self._embedding_cache) > self._max_embedding_cache_size:
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            # Return zero vector as fallback
            return np.zeros(MemoryConstants.VECTOR_DIM, dtype=np.float32)
    
    def store_incident(self, event: ReliabilityEvent, analysis: Dict[str, Any]) -> str:
        """
        Convert event+analysis to IncidentNode and store in graph
        
        V3 Feature: Store incidents with embeddings for similarity search
        
        Args:
            event: ReliabilityEvent to store
            analysis: Agent analysis results
            
        Returns:
            incident_id: Generated incident ID
        """
        if not config.rag_enabled:
            logger.debug("RAG disabled, skipping incident storage")
            return ""
        
        incident_id = self._generate_incident_id(event)
        
        with self._lock:
            # Check if already exists
            if incident_id in self.incident_nodes:
                logger.debug(f"Incident {incident_id} already exists, updating")
                # Update existing node
                node = self.incident_nodes[incident_id]
                node.agent_analysis = analysis
                node.metadata["last_updated"] = datetime.now().isoformat()
                return incident_id
            
            # Create embedding
            embedding = self._embed_incident(event, analysis)
            
            # Store in FAISS
            faiss_index_id = None
            try:
                # Add to FAISS index
                self.faiss.add_async(embedding.reshape(1, -1), f"{event.component} {event.latency_p99} {event.error_rate}")
                
                # Create text description for FAISS
                text_description = (
                    f"{event.component} "
                    f"{event.latency_p99:.1f} "
                    f"{event.error_rate:.4f} "
                    f"{event.throughput:.0f} "
                    f"{analysis.get('incident_summary', {}).get('severity', 'unknown')}"
                )
                
                # Use existing FAISS add method
                if hasattr(self.faiss, 'add_text'):
                    self.faiss.add_text(text_description, embedding.tolist())
                
            except Exception as e:
                logger.error(f"Error storing in FAISS: {e}", exc_info=True)
            
            # Create IncidentNode
            node = IncidentNode(
                incident_id=incident_id,
                component=event.component,
                severity=event.severity.value,
                timestamp=event.timestamp.isoformat(),
                metrics={
                    "latency_ms": event.latency_p99,
                    "error_rate": event.error_rate,
                    "throughput": event.throughput,
                    "cpu_util": event.cpu_util if event.cpu_util else 0.0,
                    "memory_util": event.memory_util if event.memory_util else 0.0
                },
                agent_analysis=analysis,
                embedding_id=faiss_index_id,
                faiss_index=faiss_index_id,
                metadata={
                    "revenue_impact": event.revenue_impact,
                    "user_impact": event.user_impact,
                    "upstream_deps": event.upstream_deps,
                    "downstream_deps": event.downstream_deps,
                    "service_mesh": event.service_mesh,
                    "fingerprint": event.fingerprint,
                    "created_at": datetime.now().isoformat(),
                    "embedding_dim": MemoryConstants.VECTOR_DIM
                }
            )
            
            # Store in memory
            self.incident_nodes[incident_id] = node
            self._stats["total_incidents_stored"] += 1
            self._stats["last_store_time"] = datetime.now().isoformat()
            
            # Enforce memory limits
            if len(self.incident_nodes) > MemoryConstants.MAX_INCIDENT_NODES:
                # Remove oldest incident (by timestamp)
                oldest_id = min(
                    self.incident_nodes.keys(),
                    key=lambda x: self.incident_nodes[x].metadata.get("created_at", "")
                )
                del self.incident_nodes[oldest_id]
                logger.debug(f"Evicted oldest incident {oldest_id} from RAG cache")
            
            logger.info(
                f"Stored incident {incident_id} in RAG graph: {event.component}, "
                f"severity={event.severity.value}, "
                f"latency={event.latency_p99:.0f}ms, "
                f"errors={event.error_rate*100:.1f}%"
            )
            
            return incident_id
    
    def find_similar(self, query_event: ReliabilityEvent, k: int = 5) -> List[IncidentNode]:
        """
        Semantic search + graph expansion
        
        V3 Core Feature: Retrieve similar incidents before making decisions
        
        Args:
            query_event: Event to find similar incidents for
            k: Number of similar incidents to return
            
        Returns:
            List of similar IncidentNodes with expanded outcomes
        """
        if not config.rag_enabled:
            logger.debug("RAG disabled, returning empty similar incidents")
            return []
        
        # Check circuit breaker for RAG timeout
        if self._is_rag_circuit_broken():
            logger.warning("RAG circuit breaker triggered, bypassing similarity search")
            return []
        
        cache_key = f"{query_event.fingerprint}:{k}"
        
        # Check cache first
        cached_results = self._get_cached_similarity(cache_key)
        if cached_results is not None:
            self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit for {cache_key}, returning {len(cached_results)} incidents")
            return cached_results
        
        try:
            # Start timing for circuit breaker
            import time
            start_time = time.time()
            
            # 1. FAISS similarity search
            query_embedding = self._embed_incident(query_event, {})
            
            # Perform search with timeout protection
            distances, indices = self.enhanced_faiss.search(query_embedding, k * 2)  # Get extra for filtering
            
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > config.safety_guardrails["rag_timeout_ms"]:
                logger.warning(f"RAG search took {elapsed_ms:.0f}ms (> {config.safety_guardrails['rag_timeout_ms']}ms)")
            
            # 2. Load incident nodes from FAISS results
            similar_incidents = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx == -1:  # FAISS returns -1 for no match
                    continue
                
                # Find incident with matching FAISS index or similar metrics
                found_node = None
                for node in self.incident_nodes.values():
                    if node.faiss_index == idx:
                        found_node = node
                        break
                
                # If not found by index, try to find by similarity in our graph
                if not found_node:
                    # This is a fallback - in production you'd have proper mapping
                    found_node = self._find_node_by_similarity(query_event, idx)
                
                if found_node:
                    # Calculate similarity score
                    similarity_score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0
                    found_node.metadata["similarity_score"] = similarity_score
                    found_node.metadata["search_distance"] = float(distance)
                    found_node.metadata["search_rank"] = i + 1
                    
                    similar_incidents.append(found_node)
                    
                    # Stop if we have enough
                    if len(similar_incidents) >= k:
                        break
            
            # 3.
