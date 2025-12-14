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
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import OrderedDict
from dataclasses import asdict

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
            
            # 3. Graph expansion (get outcomes)
            expanded_incidents = []
            for incident in similar_incidents:
                # Get outcomes for this incident
                incident.outcomes = self._get_outcomes(incident.incident_id)
                
                # Calculate effectiveness metrics
                if incident.outcomes:
                    successful_outcomes = [o for o in incident.outcomes if o.success]
                    incident.metadata["success_rate"] = len(successful_outcomes) / len(incident.outcomes)
                    if successful_outcomes:
                        incident.metadata["avg_resolution_time"] = sum(
                            o.resolution_time_minutes for o in successful_outcomes
                        ) / len(successful_outcomes)
                
                expanded_incidents.append(incident)
            
            # 4. Cache results
            self._cache_similarity(cache_key, expanded_incidents, distances[:len(expanded_incidents)])
            
            # Update statistics
            with self._lock:
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
            self._record_rag_failure()
            
            return []  # Fail-safe: return empty list
    
    def _get_cached_similarity(self, cache_key: str) -> Optional[List[IncidentNode]]:
        """Get cached similarity results"""
        with self._lock:
            if cache_key in self._similarity_cache:
                # Move to end (most recently used)
                self._similarity_cache.move_to_end(cache_key)
                
                # Convert SimilarityResult to IncidentNode
                results = self._similarity_cache[cache_key]
                return [result.incident_node for result in results]
            return None
    
    def _cache_similarity(self, cache_key: str, incidents: List[IncidentNode], distances: np.ndarray):
        """Cache similarity results"""
        with self._lock:
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
        # Simple implementation - in production you'd have better mapping
        for node in self.incident_nodes.values():
            if (node.component == query_event.component and
                abs(node.metrics.get("latency_ms", 0) - query_event.latency_p99) < 100 and
                abs(node.metrics.get("error_rate", 0) - query_event.error_rate) < 0.05):
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
    
    def _is_rag_circuit_broken(self) -> bool:
        """Check if RAG circuit breaker is triggered"""
        # Simple circuit breaker implementation
        # In production, track failures over time window
        return False  # Placeholder
    
    def _record_rag_failure(self):
        """Record RAG failure for circuit breaker"""
        # Placeholder for circuit breaker logic
        pass
    
    def store_outcome(self, incident_id: str, 
                     actions_taken: List[str],
                     success: bool,
                     resolution_time_minutes: float,
                     lessons_learned: Optional[List[str]] = None) -> str:
        """
        Store outcome for an incident
        
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
        
        # Check if incident exists
        if incident_id not in self.incident_nodes:
            logger.warning(f"Cannot store outcome for non-existent incident: {incident_id}")
            return ""
        
        # Generate outcome ID
        actions_hash = hashlib.md5(",".join(sorted(actions_taken)).encode()).hexdigest()[:8]
        outcome_id = self._generate_outcome_id(incident_id, actions_hash)
        
        with self._lock:
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
                    "actions_hash": actions_hash
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
                    "actions": actions_taken
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
                f"actions={actions_taken}"
            )
            
            return outcome_id
    
    def get_historical_effectiveness(self, action: str, component: str = None) -> Dict[str, Any]:
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
        resolution_times = []
        
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
        
        avg_resolution_time = np.mean(resolution_times) if resolution_times else 0.0
        
        return {
            "action": action,
            "total_uses": total,
            "successful_uses": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_resolution_time_minutes": avg_resolution_time,
            "resolution_time_std": np.std(resolution_times) if resolution_times else 0.0,
            "component_filter": component,
            "data_points": total
        }
    
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
        action_stats = {}
        
        # Collect all unique actions for this component
        for outcome in self.outcome_nodes.values():
            incident = self.incident_nodes.get(outcome.incident_id)
            if incident and incident.component == component:
                for action in outcome.actions_taken:
                    if action not in action_stats:
                        action_stats[action] = {
                            "total": 0,
                            "successful": 0,
                            "resolution_times": []
                        }
                    
                    action_stats[action]["total"] += 1
                    if outcome.success:
                        action_stats[action]["successful"] += 1
                        action_stats[action]["resolution_times"].append(outcome.resolution_time_minutes)
        
        # Calculate effectiveness metrics
        effectiveness = []
        for action, stats in action_stats.items():
            if stats["total"] >= config.learning_min_data_points:  # Only include if enough data
                success_rate = stats["successful"] / stats["total"]
                avg_time = np.mean(stats["resolution_times"]) if stats["resolution_times"] else 0.0
                
                effectiveness.append({
                    "action": action,
                    "success_rate": success_rate,
                    "confidence": min(1.0, stats["total"] / 20.0),  # Confidence based on data points
                    "avg_resolution_time_minutes": avg_time,
                    "total_uses": stats["total"],
                    "successful_uses": stats["successful"]
                })
        
        # Sort by success rate, then by confidence
        effectiveness.sort(
            key=lambda x: (x["success_rate"], x["confidence"]), 
            reverse=True
        )
        
        return effectiveness[:k]
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG graph"""
        with self._lock:
            # Calculate cache hit rate
            cache_hit_rate = (
                self._stats["cache_hits"] / self._stats["similarity_searches"] 
                if self._stats["similarity_searches"] > 0 else 0
            )
            
            # Calculate average outcomes per incident
            incidents_with_outcomes = sum(
                1 for incident_id in self.incident_nodes
                if self._get_outcomes(incident_id)
            )
            
            avg_outcomes_per_incident = (
                len(self.outcome_nodes) / len(self.incident_nodes) 
                if len(self.incident_nodes) > 0 else 0
            )
            
            # Get component distribution
            component_distribution = {}
            for node in self.incident_nodes.values():
                component_distribution[node.component] = component_distribution.get(node.component, 0) + 1
            
            return {
                "incident_nodes": len(self.incident_nodes),
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
                "is_operational": self.is_enabled()
            }
    
    def clear_cache(self) -> None:
        """Clear similarity and embedding caches"""
        with self._lock:
            self._similarity_cache.clear()
            self._embedding_cache.clear()
            logger.info("Cleared RAG graph caches")
    
    def export_graph(self, filepath: str) -> bool:
        """Export graph to JSON file"""
        try:
            with self._lock:
                data = {
                    "version": "v3.0",
                    "export_timestamp": datetime.now().isoformat(),
                    "config": {
                        "rag_enabled": config.rag_enabled,
                        "max_incident_nodes": MemoryConstants.MAX_INCIDENT_NODES,
                        "max_outcome_nodes": MemoryConstants.MAX_OUTCOME_NODES,
                    },
                    "incident_nodes": [asdict(node) for node in self.incident_nodes.values()],
                    "outcome_nodes": [asdict(node) for node in self.outcome_nodes.values()],
                    "edges": [asdict(edge) for edge in self.edges],
                    "stats": self.get_graph_stats()
                }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported RAG graph to {filepath}: {len(data['incident_nodes'])} incidents")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting graph: {e}", exc_info=True)
            return False
