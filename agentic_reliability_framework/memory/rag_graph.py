"""
RAG Graph Memory: Bridge between FAISS vectors and structured knowledge

Phase 1: RAG Graph Foundation (2-3 weeks)
Goal: Make memory useful - retrieval before execution
"""

import numpy as np
import threading
import logging
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import OrderedDict

from .faiss_index import ProductionFAISSIndex
from .models import (
    IncidentNode, OutcomeNode, GraphEdge, 
    SimilarityResult, EdgeType
)
from .constants import MemoryConstants
from ..models import ReliabilityEvent

logger = logging.getLogger(__name__)


class RAGGraphMemory:
    """
    Bridge between FAISS vectors and structured knowledge
    
    Technical Decisions:
    - Start with in-memory graph (simple dicts) before database
    - FAISS must expose search - already implemented in ProductionFAISSIndex
    - Incident IDs must be deterministic for idempotency
    """
    
    def __init__(self, faiss_index: ProductionFAISSIndex):
        """
        Initialize RAG Graph Memory
        
        Args:
            faiss_index: ProductionFAISSIndex instance with search capability
        """
        self.faiss = faiss_index
        self.incident_nodes: Dict[str, IncidentNode] = {}  # In-memory first
        self.outcome_nodes: Dict[str, OutcomeNode] = {}
        self.edges: List[GraphEdge] = []
        self._lock = threading.RLock()
        self._stats = {
            "total_incidents": 0,
            "total_outcomes": 0,
            "total_edges": 0,
            "similarity_searches": 0,
            "cache_hits": 0,
            "last_search_time": None
        }
        
        # LRU cache for similarity results
        self._similarity_cache: OrderedDict[str, List[SimilarityResult]] = OrderedDict()
        self._max_cache_size = MemoryConstants.GRAPH_CACHE_SIZE
        
        logger.info(
            f"Initialized RAGGraphMemory with FAISS index, "
            f"max_cache={self._max_cache_size}, "
            f"similarity_threshold={MemoryConstants.SIMILARITY_THRESHOLD}"
        )
    
    def is_enabled(self) -> bool:
        """Check if RAG graph is enabled and ready"""
        return len(self.incident_nodes) > 0 or self.faiss.get_count() > 0
    
    def _generate_incident_id(self, event: ReliabilityEvent) -> str:
        """
        Generate deterministic incident ID for idempotency
        
        Args:
            event: ReliabilityEvent to generate ID for
            
        Returns:
            Deterministic incident ID
        """
        # Create fingerprint from event data
        fingerprint_data = f"{event.component}:{event.latency_p99}:{event.error_rate}:{event.timestamp.isoformat()}"
        fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()
        
        # Return with prefix for readability
        return f"inc_{fingerprint[:16]}"
    
    def _generate_outcome_id(self, incident_id: str, action_hash: str) -> str:
        """Generate outcome ID"""
        data = f"{incident_id}:{action_hash}:{datetime.now().isoformat()}"
        return f"out_{hashlib.sha256(data.encode()).hexdigest()[:16]}"
    
    def _generate_edge_id(self, source_id: str, target_id: str, edge_type: EdgeType) -> str:
        """Generate edge ID"""
        data = f"{source_id}:{target_id}:{edge_type.value}:{datetime.now().isoformat()}"
        return f"edge_{hashlib.sha256(data.encode()).hexdigest()[:16]}"
    
    def _embed_incident(self, event: ReliabilityEvent, analysis: Dict[str, Any]) -> np.ndarray:
        """
        Create embedding for incident
        
        Note: This uses the existing FAISS encoding pipeline
        """
        # Create text representation for embedding
        text = (
            f"{event.component} {event.latency_p99} {event.error_rate} "
            f"severity:{event.severity.value} "
            f"analysis:{json.dumps(analysis)[:200]}"
        )
        
        # Use FAISS's existing encoder pool (non-blocking)
        # This is already implemented in ProductionFAISSIndex.find_similar_incidents
        # We'll reuse that infrastructure
        return None  # Placeholder - actual embedding done in FAISS
    
    def store_incident(self, event: ReliabilityEvent, analysis: Dict[str, Any]) -> str:
        """
        Convert event+analysis to IncidentNode and store in graph
        
        Args:
            event: ReliabilityEvent to store
            analysis: Agent analysis results
            
        Returns:
            incident_id: Generated incident ID
        """
        incident_id = self._generate_incident_id(event)
        
        with self._lock:
            # Check if already exists
            if incident_id in self.incident_nodes:
                logger.debug(f"Incident {incident_id} already exists, skipping")
                return incident_id
            
            # Create embedding and store in FAISS
            # Note: FAISS storage happens asynchronously via add_async
            # We'll store metadata immediately, FAISS vector will be added later
            
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
                metadata={
                    "revenue_impact": event.revenue_impact,
                    "user_impact": event.user_impact,
                    "upstream_deps": event.upstream_deps,
                    "downstream_deps": event.downstream_deps,
                    "service_mesh": event.service_mesh
                }
            )
            
            # Store in memory
            self.incident_nodes[incident_id] = node
            self._stats["total_incidents"] += 1
            
            # Create text for FAISS storage (async)
            faiss_text = (
                f"{event.component} {event.latency_p99} {event.error_rate} "
                f"{json.dumps(analysis)[:500]}"
            )
            
            # Note: FAISS vector will be added asynchronously via existing pipeline
            # The embedding_id will be updated later when FAISS returns it
            
            logger.info(
                f"Stored incident {incident_id} in RAG graph: {event.component}, "
                f"severity={event.severity.value}, metrics={node.metrics}"
            )
            
            # Evict oldest if at capacity
            if len(self.incident_nodes) > MemoryConstants.MAX_INCIDENT_NODES:
                oldest_id = next(iter(self.incident_nodes))
                del self.incident_nodes[oldest_id]
                logger.debug(f"Evicted oldest incident {oldest_id} from cache")
            
            return incident_id
    
    def store_outcome(self, outcome_node: OutcomeNode) -> str:
        """
        Store outcome node and connect to incident
        
        Args:
            outcome_node: OutcomeNode to store
            
        Returns:
            outcome_id: Generated outcome ID
        """
        with self._lock:
            # Check if incident exists
            if outcome_node.incident_id not in self.incident_nodes:
                logger.warning(
                    f"Cannot store outcome for non-existent incident: {outcome_node.incident_id}"
                )
                return outcome_node.outcome_id
            
            # Store outcome
            self.outcome_nodes[outcome_node.outcome_id] = outcome_node
            self._stats["total_outcomes"] += 1
            
            # Create edge from incident to outcome
            edge = GraphEdge(
                edge_id=self._generate_edge_id(
                    outcome_node.incident_id,
                    outcome_node.outcome_id,
                    EdgeType.RESOLVED_BY
                ),
                source_id=outcome_node.incident_id,
                target_id=outcome_node.outcome_id,
                edge_type=EdgeType.RESOLVED_BY,
                weight=1.0,
                metadata={
                    "success": outcome_node.success,
                    "resolution_time": outcome_node.resolution_time_minutes
                }
            )
            
            self.edges.append(edge)
            self._stats["total_edges"] += 1
            
            # Evict oldest if at capacity
            if len(self.outcome_nodes) > MemoryConstants.MAX_OUTCOME_NODES:
                oldest_id = next(iter(self.outcome_nodes))
                del self.outcome_nodes[oldest_id]
                logger.debug(f"Evicted oldest outcome {oldest_id} from cache")
            
            logger.info(
                f"Stored outcome {outcome_node.outcome_id} for incident {outcome_node.incident_id}: "
                f"success={outcome_node.success}, time={outcome_node.resolution_time_minutes}min"
            )
            
            return outcome_node.outcome_id
    
    def find_similar(self, query_event: ReliabilityEvent, k: int = 5) -> List[IncidentNode]:
        """
        Semantic search + graph expansion
        
        Args:
            query_event: Event to find similar incidents for
            k: Number of similar incidents to return
            
        Returns:
            List of similar IncidentNodes with expanded outcomes
        """
        cache_key = f"{query_event.component}:{query_event.latency_p99}:{query_event.error_rate}"
        
        with self._lock:
            # Check cache first
            if cache_key in self._similarity_cache:
                self._stats["cache_hits"] += 1
                self._similarity_cache.move_to_end(cache_key)  # Mark as recently used
                cached_results = self._similarity_cache[cache_key]
                
                # Convert SimilarityResult to IncidentNode
                incidents = [result.incident_node for result in cached_results[:k]]
                logger.debug(f"Cache hit for {cache_key}, returning {len(incidents)} incidents")
                return incidents
            
        try:
            # 1. FAISS similarity search
            query_text = (
                f"{query_event.component} latency {query_event.latency_p99}ms "
                f"error {query_event.error_rate:.3f}"
            )
            
            # Use existing FAISS search
            similar_incidents = []
            
            # Try async search first
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                faiss_results = loop.run_until_complete(
                    self.faiss.find_similar_incidents(
                        query_text, 
                        k=k * 2,  # Get more for filtering
                        min_similarity=MemoryConstants.SIMILARITY_THRESHOLD
                    )
                )
            except (RuntimeError, ImportError):
                # Fallback to sync search
                faiss_results = self.faiss.find_similar_by_metrics(
                    query_event.component,
                    query_event.latency_p99,
                    query_event.error_rate,
                    k=k * 2
                )
            
            # 2. Load incident nodes from FAISS results
            for result in faiss_results:
                # Look for incident by matching metrics
                for node in self.incident_nodes.values():
                    if (abs(node.metrics["latency_ms"] - result.get("latency", 0)) < 10 and
                        abs(node.metrics["error_rate"] - result.get("error_rate", 0)) < 0.01 and
                        node.component == result.get("component")):
                        
                        # Update FAISS index if not set
                        if node.faiss_index is None and "faiss_index" in result:
                            node.faiss_index = result["faiss_index"]
                        
                        similar_incidents.append(node)
                        break
            
            # 3. Graph expansion (get outcomes)
            expanded_incidents = []
            for incident in similar_incidents[:k]:  # Limit to k
                # Get outcomes for this incident
                incident.outcomes = self._get_outcomes(incident.incident_id)
                expanded_incidents.append(incident)
            
            # 4. Update cache
            with self._lock:
                similarity_results = [
                    SimilarityResult(
                        incident_node=incident,
                        similarity_score=0.8,  # Placeholder - would come from FAISS
                        raw_score=0.9,  # Placeholder
                        faiss_index=incident.faiss_index or 0
                    )
                    for incident in expanded_incidents
                ]
                
                self._similarity_cache[cache_key] = similarity_results
                self._similarity_cache.move_to_end(cache_key)
                
                # Evict oldest if cache full
                if len(self._similarity_cache) > self._max_cache_size:
                    oldest_key, _ = self._similarity_cache.popitem(last=False)
                    logger.debug(f"Evicted cache entry: {oldest_key}")
                
                self._stats["similarity_searches"] += 1
                self._stats["last_search_time"] = datetime.now().isoformat()
            
            logger.info(
                f"Found {len(expanded_incidents)} similar incidents for {query_event.component}, "
                f"cache_size={len(self._similarity_cache)}"
            )
            
            return expanded_incidents
            
        except Exception as e:
            logger.error(f"Error in find_similar: {e}", exc_info=True)
            return []
    
    def _get_outcomes(self, incident_id: str) -> List[OutcomeNode]:
        """
        Get outcomes for an incident
        
        Args:
            incident_id: Incident ID to get outcomes for
            
        Returns:
            List of OutcomeNodes for this incident
        """
        outcomes = []
        for edge in self.edges:
            if (edge.source_id == incident_id and 
                edge.edge_type == EdgeType.RESOLVED_BY):
                outcome = self.outcome_nodes.get(edge.target_id)
                if outcome:
                    outcomes.append(outcome)
        return outcomes
    
    def get_historical_effectiveness(self, action: str, component: str = None) -> Dict[str, Any]:
        """
        Get historical effectiveness of an action
        
        Args:
            action: Action to check effectiveness for
            component: Optional component filter
            
        Returns:
            Dictionary with effectiveness statistics
        """
        successful = 0
        total = 0
        avg_resolution_time = 0.0
        
        for outcome in self.outcome_nodes.values():
            if action in outcome.actions_taken:
                if component and self.incident_nodes.get(outcome.incident_id):
                    incident = self.incident_nodes[outcome.incident_id]
                    if incident.component != component:
                        continue
                
                total += 1
                if outcome.success:
                    successful += 1
                    avg_resolution_time += outcome.resolution_time_minutes
        
        if successful > 0:
            avg_resolution_time /= successful
        
        return {
            "action": action,
            "total_uses": total,
            "successful_uses": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_resolution_time_minutes": avg_resolution_time,
            "component_filter": component
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG graph"""
        with self._lock:
            return {
                "incident_nodes": len(self.incident_nodes),
                "outcome_nodes": len(self.outcome_nodes),
                "edges": len(self.edges),
                "similarity_cache_size": len(self._similarity_cache),
                "cache_hit_rate": (
                    self._stats["cache_hits"] / self._stats["similarity_searches"] 
                    if self._stats["similarity_searches"] > 0 else 0
                ),
                "stats": self._stats,
                "max_incident_nodes": MemoryConstants.MAX_INCIDENT_NODES,
                "max_outcome_nodes": MemoryConstants.MAX_OUTCOME_NODES,
                "graph_cache_size": self._max_cache_size,
                "is_enabled": self.is_enabled()
            }
    
    def clear_cache(self) -> None:
        """Clear similarity cache"""
        with self._lock:
            self._similarity_cache.clear()
            logger.info("Cleared RAG graph similarity cache")
    
    def export_graph(self, filepath: str) -> bool:
        """
        Export graph to JSON file
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                "incident_nodes": [node.to_dict() for node in self.incident_nodes.values()],
                "outcome_nodes": [node.to_dict() for node in self.outcome_nodes.values()],
                "edges": [edge.to_dict() for edge in self.edges],
                "export_timestamp": datetime.now().isoformat(),
                "stats": self.get_graph_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported RAG graph to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting graph: {e}", exc_info=True)
            return False
    
    def import_graph(self, filepath: str) -> bool:
        """
        Import graph from JSON file
        
        Args:
            filepath: Path to import file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                # Clear existing data
                self.incident_nodes.clear()
                self.outcome_nodes.clear()
                self.edges.clear()
                
                # Import nodes
                for node_data in data.get("incident_nodes", []):
                    node = IncidentNode.from_dict(node_data)
                    self.incident_nodes[node.incident_id] = node
                
                for node_data in data.get("outcome_nodes", []):
                    node = OutcomeNode.from_dict(node_data)
                    self.outcome_nodes[node.outcome_id] = node
                
                # Import edges
                for edge_data in data.get("edges", []):
                    edge = GraphEdge.from_dict(edge_data)
                    self.edges.append(edge)
                
                # Update stats
                self._stats["total_incidents"] = len(self.incident_nodes)
                self._stats["total_outcomes"] = len(self.outcome_nodes)
                self._stats["total_edges"] = len(self.edges)
            
            logger.info(f"Imported RAG graph from {filepath}: {len(self.incident_nodes)} incidents")
            return True
            
        except Exception as e:
            logger.error(f"Error importing graph: {e}", exc_info=True)
            return False
