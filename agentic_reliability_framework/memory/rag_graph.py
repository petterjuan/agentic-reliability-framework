"""
Enhanced RAG Graph Memory for ARF v3
Production-ready with safety features and proper type hints
"""

import numpy as np
import threading
import logging
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, TypedDict, cast
from datetime import datetime
from collections import OrderedDict
from dataclasses import asdict
from contextlib import contextmanager

from .faiss_index import ProductionFAISSIndex
from .enhanced_faiss import EnhancedFAISSIndex
from .models import (
    IncidentNode, OutcomeNode, GraphEdge, 
    SimilarityResult, EdgeType
)
from .constants import MemoryConstants
from ..models import ReliabilityEvent
from ..config import config

logger = logging.getLogger(__name__)


# Type definitions for better type safety
class EffectivenessStats(TypedDict):
    """Type for action effectiveness statistics"""
    action: str
    total_uses: int
    successful_uses: int
    success_rate: float
    avg_resolution_time_minutes: float
    resolution_time_std: float
    component_filter: Optional[str]
    data_points: int


class GraphStats(TypedDict):
    """Type for RAG graph statistics"""
    incident_nodes: int
    outcome_nodes: int
    edges: int
    similarity_cache_size: int
    embedding_cache_size: int
    cache_hit_rate: float
    incidents_with_outcomes: int
    avg_outcomes_per_incident: float
    component_distribution: Dict[str, int]
    stats: Dict[str, Any]
    memory_limits: Dict[str, Any]
    v3_enabled: bool
    is_operational: bool
    circuit_breaker: Dict[str, Any]


class RAGGraphMemory:
    """
    Enhanced RAG Graph Memory with safety features
    
    Key improvements:
    1. Real circuit breaker implementation
    2. Proper FAISS ↔ RAG node identity mapping
    3. Thread-safe operations with context managers
    4. Better type hints and error handling
    5. Performance monitoring
    """
    
    def __init__(self, faiss_index: ProductionFAISSIndex):
        """
        Initialize enhanced RAG Graph Memory
        
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
        
        # Statistics with better organization
        self._stats: Dict[str, Any] = {
            "total_incidents_stored": 0,
            "total_outcomes_stored": 0,
            "total_edges_created": 0,
            "similarity_searches": 0,
            "cache_hits": 0,
            "failed_searches": 0,
            "last_search_time": None,
            "last_store_time": None,
        }
        
        # Circuit breaker state
        self._rag_failures = 0
        self._rag_disabled_until = 0.0
        self._rag_last_failure_time = 0.0
        
        # LRU cache for similarity results
        self._similarity_cache: OrderedDict[str, List[SimilarityResult]] = OrderedDict()
        self._max_cache_size = MemoryConstants.GRAPH_CACHE_SIZE
        
        # Embedding cache for performance
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._max_embedding_cache_size = 100
        
        # FAISS index to incident ID mapping (critical fix)
        self._faiss_to_incident: Dict[int, str] = {}
        
        logger.info(
            f"Initialized enhanced RAGGraphMemory: "
            f"max_incidents={MemoryConstants.MAX_INCIDENT_NODES}, "
            f"max_outcomes={MemoryConstants.MAX_OUTCOME_NODES}, "
            f"cache_size={self._max_cache_size}"
        )
    
    @contextmanager
    def _transaction(self):
        """Context manager for thread-safe operations"""
        with self._lock:
            yield
    
    def is_enabled(self) -> bool:
        """Check if RAG graph is enabled and ready"""
        return config.rag_enabled and (
            len(self.incident_nodes) > 0 or 
            self.faiss.get_count() > 0
        )
    
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
        Create embedding vector from incident data with explicit field handling
        
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
            features: List[float] = []
            
            # 1. Basic metrics (normalized with explicit handling)
            features.append(float(event.latency_p99) / 1000.0)  # Normalize to seconds
            features.append(float(event.error_rate))  # Already 0-1
            features.append(float(event.throughput) / 10000.0)  # Normalize
            
            # 2. Resource utilization (explicit None handling)
            cpu_val = event.cpu_util if event.cpu_util is not None else 0.0
            mem_val = event.memory_util if event.memory_util is not None else 0.0
            features.append(float(cpu_val))
            features.append(float(mem_val))
            
            # 3. Severity encoding (explicit mapping)
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
            confidence = 0.5  # Default
            if analysis and 'incident_summary' in analysis:
                confidence = analysis['incident_summary'].get('anomaly_confidence', 0.5)
            features.append(float(confidence))
            
            # Pad or truncate to target dimension
            target_dim = MemoryConstants.VECTOR_DIM
            current_len = len(features)
            
            if current_len < target_dim:
                # Pad with zeros
                features.extend([0.0] * (target_dim - current_len))
            else:
                # Truncate
                features = features[:target_dim]
            
            embedding = np.array(features, dtype=np.float32)
            
            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Cache for performance
            with self._transaction():
                self._embedding_cache[cache_key] = embedding
                
                # Manage cache size (LRU eviction)
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
        Store incident with proper FAISS index binding
        
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
        
        with self._transaction():
            # Check if already exists
            if incident_id in self.incident_nodes:
                logger.debug(f"Incident {incident_id} already exists, updating")
                node = self.incident_nodes[incident_id]
                node.agent_analysis = analysis
                node.metadata["last_updated"] = datetime.now().isoformat()
                return incident_id
            
            # Create embedding
            embedding = self._embed_incident(event, analysis)
            
            # Store in FAISS and get index ID (CRITICAL FIX)
            faiss_index_id: Optional[int] = None
            try:
                # Create text description for FAISS
                text_description = (
                    f"{event.component} "
                    f"{event.latency_p99:.1f} "
                    f"{event.error_rate:.4f} "
                    f"{event.throughput:.0f} "
                    f"{analysis.get('incident_summary', {}).get('severity', 'unknown')}"
                )
                
                # Store in FAISS and get the actual index
                if hasattr(self.faiss, 'add_text'):
                    # add_text returns the index where the vector was stored
                    faiss_index_id = self.faiss.add_text(text_description, embedding.tolist())
                else:
                    # Fallback: add_async and track count
                    self.faiss.add_async(embedding.reshape(1, -1), text_description)
                    # Estimate index based on current count (pre-add)
                    faiss_index_id = self.faiss.get_count()
                
                # Store the mapping (CRITICAL)
                if faiss_index_id is not None:
                    self._faiss_to_incident[faiss_index_id] = incident_id
                
            except Exception as e:
                logger.error(f"Error storing in FAISS: {e}", exc_info=True)
                # Generate fallback ID
                faiss_index_id = len(self.incident_nodes)
            
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
                    "cpu_util": float(event.cpu_util) if event.cpu_util is not None else 0.0,
                    "memory_util": float(event.memory_util) if event.memory_util is not None else 0.0
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
            
            # Enforce memory limits with LRU eviction
            if len(self.incident_nodes) > MemoryConstants.MAX_INCIDENT_NODES:
                # Remove oldest incident by created_at timestamp
                oldest_id = min(
                    self.incident_nodes.keys(),
                    key=lambda x: self.incident_nodes[x].metadata.get("created_at", "")
                )
                
                # Clean up mapping and node
                oldest_node = self.incident_nodes[oldest_id]
                if oldest_node.faiss_index is not None:
                    self._faiss_to_incident.pop(oldest_node.faiss_index, None)
                del self.incident_nodes[oldest_id]
                
                logger.debug(f"Evicted oldest incident {oldest_id} from RAG cache")
            
            logger.info(
                f"Stored incident {incident_id} in RAG graph: {event.component}, "
                f"severity={event.severity.value}, "
                f"latency={event.latency_p99:.0f}ms, "
                f"errors={event.error_rate*100:.1f}%"
            )
            
            return incident_id
    
    def _is_rag_circuit_broken(self) -> bool:
        """
        Check if RAG circuit breaker is triggered
        
        Real implementation using config.safety_guardrails
        """
        current_time = time.time()
        
        # Check if we're in disabled period
        if current_time < self._rag_disabled_until:
            return True
        
        # Reset failures if timeout window passed
        guardrails = config.safety_guardrails
        failure_timeout = guardrails["circuit_breaker"]["timeout"]
        
        if current_time - self._rag_last_failure_time > failure_timeout:
            self._rag_failures = 0
        
        return False
    
    def _record_rag_failure(self):
        """
        Record RAG failure and trigger circuit breaker if needed
        
        Real implementation based on config thresholds
        """
        current_time = time.time()
        guardrails = config.safety_guardrails
        failure_threshold = guardrails["circuit_breaker"]["failures"]
        timeout_seconds = guardrails["circuit_breaker"]["timeout"]
        
        # Update failure count
        self._rag_failures += 1
        self._rag_last_failure_time = current_time
        
        # Check if we should trigger circuit breaker
        if self._rag_failures >= failure_threshold:
            self._rag_disabled_until = current_time + timeout_seconds
            logger.warning(
                f"RAG circuit breaker triggered after {self._rag_failures} failures. "
                f"Disabled for {timeout_seconds} seconds."
            )
    
    def reset_circuit_breaker(self):
        """Manually reset the RAG circuit breaker"""
        with self._transaction():
            self._rag_failures = 0
            self._rag_disabled_until = 0.0
            self._rag_last_failure_time = 0.0
            logger.info("RAG circuit breaker manually reset")
    
    def find_similar(self, query_event: ReliabilityEvent, k: int = 5) -> List[IncidentNode]:
        """
        Semantic search + graph expansion with safety features
        
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
        
        # Check circuit breaker (real implementation)
        if self._is_rag_circuit_broken():
            logger.warning("RAG circuit breaker triggered, bypassing similarity search")
            return []
        
        cache_key = f"{query_event.fingerprint}:{k}"
        
        # Check cache first
        cached_results = self._get_cached_similarity(cache_key)
        if cached_results is not None:
            with self._transaction():
                self._stats["cache_hits"] += 1
            logger.debug(f"Cache hit for {cache_key}, returning {len(cached_results)} incidents")
            return cached_results
        
        try:
            # Start timing for circuit breaker
            start_time = time.time()
            
            # 1. FAISS similarity search
            query_embedding = self._embed_incident(query_event, {})
            
            # Perform search with timeout protection
            distances, indices = self.enhanced_faiss.search(query_embedding, k * 2)  # Get extra for filtering
            
            # Check timeout against guardrails
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > config.safety_guardrails["rag_timeout_ms"]:
                logger.warning(f"RAG search took {elapsed_ms:.0f}ms (> {config.safety_guardrails['rag_timeout_ms']}ms)")
            
            # 2. Load incident nodes using proper FAISS mapping (CRITICAL FIX)
            similar_incidents: List[IncidentNode] = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx == -1:  # FAISS returns -1 for no match
                    continue
                
                # Use the mapping to find incident (FIXED)
                incident_id = self._faiss_to_incident.get(idx)
                if incident_id:
                    node = self.incident_nodes.get(incident_id)
                    if node:
                        found_node = node
                    else:
                        # Clean up stale mapping
                        self._faiss_to_incident.pop(idx, None)
                        found_node = None
                else:
                    found_node = None
                
                # Fallback: similarity search in our graph
                if not found_node:
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
            
            # 3. Graph expansion (get outcomes) - FIXED: Don't modify IncidentNode.outcomes
            expanded_incidents = []
            for incident in similar_incidents:
                # Get outcomes for this incident
                incident_outcomes = self._get_outcomes(incident.incident_id)
                
                # Calculate effectiveness metrics and store in metadata
                if incident_outcomes:
                    successful_outcomes = [o for o in incident_outcomes if o.success]
                    if successful_outcomes:
                        incident.metadata["success_rate"] = len(successful_outcomes) / len(incident_outcomes)
                        incident.metadata["avg_resolution_time"] = sum(
                            o.resolution_time_minutes for o in successful_outcomes
                        ) / len(successful_outcomes)
                    else:
                        incident.metadata["success_rate"] = 0.0
                        incident.metadata["avg_resolution_time"] = 0.0
                    
                    # Store outcomes count in metadata
                    incident.metadata["outcomes_count"] = len(incident_outcomes)
                    incident.metadata["successful_outcomes_count"] = len(successful_outcomes)
                else:
                    incident.metadata["success_rate"] = 0.0
                    incident.metadata["avg_resolution_time"] = 0.0
                    incident.metadata["outcomes_count"] = 0
                    incident.metadata["successful_outcomes_count"] = 0
                
                expanded_incidents.append(incident)
            
            # 4. Cache results
            self._cache_similarity(cache_key, expanded_incidents, distances[:len(expanded_incidents)])
            
            # Update statistics
            with self._transaction():
                self._stats["similarity_searches"] += 1
                self._stats["last_search_time"] = datetime.now().isoformat()
            
            logger.info(
                f"Found {len(expanded_incidents)} similar incidents for {query_event.component}, "
                f"cache_size={len(self._similarity_cache)}, "
                f"time={elapsed_ms:.0f}ms"
            )
            
            return expanded_incidents
            
        except Exception as e:
            logger.error(f"Error in find_similar: {e}", exc_info=True)
            
            # Update circuit breaker on failure
            with self._transaction():
                self._stats["failed_searches"] += 1
            self._record_rag_failure()
            
            return []  # Fail-safe: return empty list
    
    def _get_cached_similarity(self, cache_key: str) -> Optional[List[IncidentNode]]:
        """Get cached similarity results"""
        with self._transaction():
            if cache_key in self._similarity_cache:
                # Move to end (most recently used)
                self._similarity_cache.move_to_end(cache_key)
                
                # Convert SimilarityResult to IncidentNode
                results = self._similarity_cache[cache_key]
                return [result.incident_node for result in results]
            return None
    
    def _cache_similarity(self, cache_key: str, incidents: List[IncidentNode], distances: np.ndarray):
        """Cache similarity results"""
        with self._transaction():
            # Create SimilarityResult objects
            similarity_results = []
            for i, incident in enumerate(incidents):
                result = SimilarityResult(
                    incident_node=incident,
                    similarity_score=incident.metadata.get("similarity_score", 0.0),
                    raw_score=float(distances[i]) if i < len(distances) else 0.0,
                    faiss_index=incident.faiss_index or 0
                )
                similarity_results.append(result)
            
            # Store in cache
            self._similarity_cache[cache_key] = similarity_results
            self._similarity_cache.move_to_end(cache_key)
            
            # Evict oldest if cache full
            if len(self._similarity_cache) > self._max_cache_size:
                oldest_key, _ = self._similarity_cache.popitem(last=False)
                logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def _find_node_by_similarity(self, query_event: ReliabilityEvent, faiss_index: int) -> Optional[IncidentNode]:
        """Find node by similarity when direct mapping doesn't exist"""
        
        for node in self.incident_nodes.values():
            # Check component match
            if node.component != query_event.component:
                continue
            
            # Check latency similarity
            latency_diff = abs(node.metrics.get("latency_ms", 0) - query_event.latency_p99)
            if latency_diff > 100:  # 100ms threshold
                continue
            
            # Check error rate similarity
            error_diff = abs(node.metrics.get("error_rate", 0) - query_event.error_rate)
            if error_diff > 0.05:  # 5% threshold
                continue
            
            # Found a similar node
            return node
        
        return None
    
    def _get_outcomes(self, incident_id: str) -> List[OutcomeNode]:
        """Get outcomes for an incident"""
        outcomes = []
        for edge in self.edges:
            if (edge.source_id == incident_id and 
                edge.edge_type == EdgeType.RESOLVED_BY):
                outcome = self.outcome_nodes.get(edge.target_id)
                if outcome:
                    outcomes.append(outcome)
        return outcomes
    
    def store_outcome(self, incident_id: str, 
                     actions_taken: List[str],
                     success: bool,
                     resolution_time_minutes: float,
                     lessons_learned: Optional[List[str]] = None) -> str:
        """
        Store outcome for an incident with proper validation
        
        V3 Feature: Record outcomes for learning loop
        
        Args:
            incident_id: Incident ID
            actions_taken: List of actions taken
            success: Whether resolution was successful
            resolution_time_minutes: Time to resolve in minutes
            lessons_learned: Optional lessons learned
            
        Returns:
            outcome_id: Generated outcome ID
        """
        if not config.rag_enabled:
            return ""
        
        # Validate input
        if not incident_id or not actions_taken:
            logger.warning("Invalid outcome data: missing incident_id or actions_taken")
            return ""
        
        # Check if incident exists
        if incident_id not in self.incident_nodes:
            logger.warning(f"Cannot store outcome for non-existent incident: {incident_id}")
            return ""
        
        # Generate outcome ID
        sorted_actions = sorted(actions_taken)
        actions_hash = hashlib.md5(",".join(sorted_actions).encode()).hexdigest()[:8]
        outcome_id = self._generate_outcome_id(incident_id, actions_hash)
        
        with self._transaction():
            # Create OutcomeNode
            outcome = OutcomeNode(
                outcome_id=outcome_id,
                incident_id=incident_id,
                actions_taken=actions_taken,
                resolution_time_minutes=resolution_time_minutes,
                success=success,
                lessons_learned=lessons_learned or [],
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "actions_hash": actions_hash,
                    "resolution_minutes": resolution_time_minutes,
                    "success": success
                }
            )
            
            # Store outcome
            self.outcome_nodes[outcome_id] = outcome
            self._stats["total_outcomes_stored"] += 1
            
            # Create edge from incident to outcome
            edge = GraphEdge(
                edge_id=self._generate_edge_id(incident_id, outcome_id, EdgeType.RESOLVED_BY),
                source_id=incident_id,
                target_id=outcome_id,
                edge_type=EdgeType.RESOLVED_BY,
                weight=1.0,
                metadata={
                    "success": success,
                    "resolution_time": resolution_time_minutes,
                    "actions": actions_taken,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.edges.append(edge)
            self._stats["total_edges_created"] += 1
            
            # Enforce memory limits
            if len(self.outcome_nodes) > MemoryConstants.MAX_OUTCOME_NODES:
                # Remove oldest outcome
                oldest_id = min(
                    self.outcome_nodes.keys(),
                    key=lambda x: self.outcome_nodes[x].metadata.get("created_at", "")
                )
                del self.outcome_nodes[oldest_id]
                logger.debug(f"Evicted oldest outcome {oldest_id} from RAG cache")
            
            logger.info(
                f"Stored outcome {outcome_id} for incident {incident_id}: "
                f"success={success}, time={resolution_time_minutes:.1f}min, "
                f"actions={len(actions_taken)}"
            )
            
            return outcome_id
    
    def get_historical_effectiveness(self, action: str, component: Optional[str] = None) -> EffectivenessStats:
        """
        Get historical effectiveness of an action
        
        V3 Learning Feature: Used to inform policy decisions
        
        Args:
            action: Action to check effectiveness for
            component: Optional component filter
            
        Returns:
            Dictionary with effectiveness statistics
        """
        successful = 0
        total = 0
        resolution_times: List[float] = []
        
        for outcome in self.outcome_nodes.values():
            if action in outcome.actions_taken:
                # Apply component filter if specified
                if component:
                    incident = self.incident_nodes.get(outcome.incident_id)
                    if not incident or incident.component != component:
                        continue
                
                total += 1
                if outcome.success:
                    successful += 1
                    resolution_times.append(outcome.resolution_time_minutes)
        
        # Calculate statistics - FIXED: Handle numpy return types properly
        if resolution_times:
            mean_value = np.mean(resolution_times)
            # Check if it's a valid finite number
            if np.isfinite(mean_value):
                avg_resolution_time = float(mean_value)
            else:
                avg_resolution_time = 0.0
        else:
            avg_resolution_time = 0.0
        
        if resolution_times:
            std_value = np.std(resolution_times)
            if np.isfinite(std_value):
                resolution_std = float(std_value)
            else:
                resolution_std = 0.0
        else:
            resolution_std = 0.0
        
        return cast(EffectivenessStats, {
            "action": action,
            "total_uses": total,
            "successful_uses": successful,
            "success_rate": float(successful) / total if total > 0 else 0.0,
            "avg_resolution_time_minutes": avg_resolution_time,
            "resolution_time_std": resolution_std,
            "component_filter": component,
            "data_points": total
        })
    
    def _serialize_node(self, node: Any) -> Dict[str, Any]:
        """
        Serialize a node safely, converting numpy arrays to lists.
        
        Args:
            node: Any dataclass node
            
        Returns:
            Serializable dictionary
        """
        data = asdict(node)
        # Convert any numpy arrays to lists for JSON serialization
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, dict):
                # Recursively check nested dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        data[key][sub_key] = sub_value.tolist()
            # Handle nested lists that might contain numpy arrays
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, np.ndarray):
                        data[key][i] = item.tolist()
        
        # Explicitly cast to Dict[str, Any] for mypy
        return cast(Dict[str, Any], data)
    
    def get_most_effective_actions(self, component: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Get most effective actions for a component
        
        V3 Feature: Used to recommend actions based on historical success
        
        Args:
            component: Component to get actions for
            k: Number of actions to return
            
        Returns:
            List of actions sorted by effectiveness
        """
        action_stats: Dict[str, Dict[str, Any]] = {}
        
        # Collect all unique actions for this component
        for outcome in self.outcome_nodes.values():
            incident = self.incident_nodes.get(outcome.incident_id)
            if incident and incident.component == component:
                for action in outcome.actions_taken:
                    if action not in action_stats:
                        action_stats[action] = {
                            "total": 0,
                            "successful": 0,
                            "resolution_times": [],
                        }
                    
                    action_stats[action]["total"] += 1
                    if outcome.success:
                        action_stats[action]["successful"] += 1
                        action_stats[action]["resolution_times"].append(outcome.resolution_time_minutes)
        
        # Calculate effectiveness metrics
        effectiveness = []
        min_data_points = config.learning_min_data_points
        
        for action, stats in action_stats.items():
            if stats["total"] >= min_data_points:  # Only include if enough data
                success_rate = float(stats["successful"]) / stats["total"]
                
                # Calculate average resolution time
                resolution_times = stats["resolution_times"]
                if resolution_times:
                    mean_value = np.mean(resolution_times)
                    avg_time = float(mean_value) if np.isfinite(mean_value) else 0.0
                else:
                    avg_time = 0.0
                
                # Confidence based on data points (capped at 1.0)
                confidence = min(1.0, float(stats["total"]) / (min_data_points * 2))
                
                effectiveness.append({
                    "action": action,
                    "success_rate": success_rate,
                    "confidence": confidence,
                    "avg_resolution_time_minutes": avg_time,
                    "total_uses": stats["total"],
                    "successful_uses": stats["successful"],
                    "data_sufficiency": stats["total"] >= min_data_points
                })
        
        # Sort by success rate (descending), then by confidence (descending)
        effectiveness.sort(
            key=lambda x: (x["success_rate"], x["confidence"]), 
            reverse=True
        )
        
        return effectiveness[:k]
    
    def get_graph_stats(self) -> GraphStats:
        """Get comprehensive statistics about the RAG graph"""
        with self._transaction():
            # Calculate cache hit rate
            searches = self._stats["similarity_searches"]
            hits = self._stats["cache_hits"]
            cache_hit_rate = float(hits) / searches if searches > 0 else 0.0
            
            # Calculate average outcomes per incident
            incidents_with_outcomes = sum(
                1 for incident_id in self.incident_nodes
                if self._get_outcomes(incident_id)
            )
            
            incident_count = len(self.incident_nodes)
            avg_outcomes_per_incident = (
                float(len(self.outcome_nodes)) / incident_count 
                if incident_count > 0 else 0
            )
            
            # Get component distribution
            component_distribution: Dict[str, int] = {}
            for node in self.incident_nodes.values():
                component_distribution[node.component] = component_distribution.get(node.component, 0) + 1
            
            # Circuit breaker status
            current_time = time.time()
            circuit_breaker_status = {
                "failures": self._rag_failures,
                "disabled_until": self._rag_disabled_until,
                "is_active": current_time < self._rag_disabled_until,
                "seconds_until_reset": max(0, self._rag_disabled_until - current_time),
                "failure_threshold": config.safety_guardrails["circuit_breaker"]["failures"]
            }
            
            return cast(GraphStats, {
                "incident_nodes": incident_count,
                "outcome_nodes": len(self.outcome_nodes),
                "edges": len(self.edges),
                "similarity_cache_size": len(self._similarity_cache),
                "embedding_cache_size": len(self._embedding_cache),
                "cache_hit_rate": cache_hit_rate,
                "incidents_with_outcomes": incidents_with_outcomes,
                "avg_outcomes_per_incident": avg_outcomes_per_incident,
                "component_distribution": component_distribution,
                "stats": self._stats.copy(),
                "memory_limits": {
                    "max_incident_nodes": MemoryConstants.MAX_INCIDENT_NODES,
                    "max_outcome_nodes": MemoryConstants.MAX_OUTCOME_NODES,
                    "graph_cache_size": self._max_cache_size,
                    "embedding_cache_size": self._max_embedding_cache_size
                },
                "v3_enabled": config.rag_enabled,
                "is_operational": self.is_enabled(),
                "circuit_breaker": circuit_breaker_status
            })
    
    def clear_cache(self) -> None:
        """Clear similarity and embedding caches"""
        with self._transaction():
            self._similarity_cache.clear()
            self._embedding_cache.clear()
            logger.info("Cleared RAG graph caches")
    
    def export_graph(self, filepath: str) -> bool:
        """Export graph to JSON file with proper error handling"""
        try:
            with self._transaction():
                data: Dict[str, Any] = {
                    "version": "v3.0",
                    "export_timestamp": datetime.now().isoformat(),
                    "config": {
                        "rag_enabled": config.rag_enabled,
                        "max_incident_nodes": MemoryConstants.MAX_INCIDENT_NODES,
                        "max_outcome_nodes": MemoryConstants.MAX_OUTCOME_NODES,
                        "similarity_threshold": config.rag_similarity_threshold,
                    },
                    "incident_nodes": [self._serialize_node(node) for node in self.incident_nodes.values()],
                    "outcome_nodes": [self._serialize_node(node) for node in self.outcome_nodes.values()],
                    "edges": [self._serialize_node(edge) for edge in self.edges],
                    "stats": self.get_graph_stats(),
                    "faiss_mapping_size": len(self._faiss_to_incident)
                }
            
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported RAG graph to {filepath}: {len(data['incident_nodes'])} incidents")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting graph: {e}", exc_info=True)
            return False
    
    def cleanup_old_nodes(self, max_age_days: int = 30) -> int:
        """
        Clean up old incident and outcome nodes
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of nodes cleaned up
        """
        if not config.rag_enabled:
            return 0
        
        cutoff_timestamp = time.time() - (max_age_days * 24 * 3600)
        cleaned_count = 0
        
        with self._transaction():
            # Helper function to check if a date string is old
            def is_old(date_str: str) -> bool:
                try:
                    # Clean up date string
                    clean_date = date_str.replace('Z', '+00:00')
                    date_obj = datetime.fromisoformat(clean_date)
                    return date_obj.timestamp() < cutoff_timestamp
                except (ValueError, TypeError):
                    return True
            
            # Clean old incidents
            incidents_to_remove = []
            for incident_id, incident_node in self.incident_nodes.items():
                created_at = incident_node.metadata.get("created_at", "1970-01-01")
                if is_old(created_at):
                    incidents_to_remove.append(incident_id)
            
            for incident_id in incidents_to_remove:
                node = self.incident_nodes[incident_id]
                if node.faiss_index is not None:
                    self._faiss_to_incident.pop(node.faiss_index, None)
                del self.incident_nodes[incident_id]
                cleaned_count += 1
            
            # Clean old outcomes - FIXED: renamed variable to avoid type conflict
            outcomes_to_remove = []
            for outcome_id, outcome_node in self.outcome_nodes.items():
                created_at = outcome_node.metadata.get("created_at", "1970-01-01")
                if is_old(created_at):
                    outcomes_to_remove.append(outcome_id)
            
            for outcome_id in outcomes_to_remove:
                del self.outcome_nodes[outcome_id]
                cleaned_count += 1
            
            # Clean edges pointing to removed outcomes
            valid_edges = []
            for edge in self.edges:
                if (edge.source_id in self.incident_nodes and 
                    edge.target_id in self.outcome_nodes):
                    valid_edges.append(edge)
            self.edges = valid_edges
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old RAG graph nodes")
        
        return cleaned_count
