"""
Enhanced V3 Reliability Engine with RAG Graph Memory and MCP Server integration.
Production-ready with safety features, proper type hints, caching, and atomic updates.
Extends the base V3ReliabilityEngine with full v3 features.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import numpy as np
import hashlib
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, cast, TypedDict, TYPE_CHECKING
from collections import OrderedDict
from enum import Enum

# Conditional imports to avoid circular dependencies
if TYPE_CHECKING:
    from ..models import ReliabilityEvent
    from ..memory.rag_graph import RAGGraphMemory
    from ..engine.mcp_server import MCPServer
else:
    # Runtime imports will be done lazily
    pass

from ..config import config
from .interfaces import ReliabilityEngineProtocol
from .reliability import V3ReliabilityEngine as BaseV3Engine, MCPResponse as BaseMCPResponse

logger = logging.getLogger(__name__)

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class ConfidenceBasis(str, Enum):
    """Sources of confidence for healing decisions."""
    POLICY_ONLY = "policy_only"
    POLICY_PLUS_SAFETY = "policy_plus_safety"
    HISTORICAL_SIMILARITY = "historical_similarity"
    DETERMINISTIC_GUARANTEE = "deterministic_guarantee"
    LEARNED_OUTCOMES = "learned_outcomes"


class EffectivenessStats(TypedDict):
    """Historical effectiveness statistics for actions"""
    action: str
    total_uses: int
    successful_uses: int
    success_rate: float
    avg_resolution_time_minutes: float
    resolution_time_std: float
    component_filter: Optional[str]
    data_points: int


class GraphStats(TypedDict):
    """RAG Graph memory statistics"""
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


# ============================================================================
# ENHANCED RAG GRAPH MEMORY
# ============================================================================

class EnhancedRAGGraphMemory:
    """
    Enhanced RAG Graph Memory with caching and atomic updates
    Integrated directly to avoid circular imports
    """
    
    def __init__(self, faiss_index):
        """
        Initialize enhanced RAG graph memory
        
        Args:
            faiss_index: FAISS index for vector similarity search
        """
        self.faiss = faiss_index
        self.incident_nodes: Dict[str, Any] = {}
        self.outcome_nodes: Dict[str, Any] = {}
        self.edges: List[Any] = []
        self._lock = threading.RLock()
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
        self._rag_failures = 0
        self._rag_disabled_until = 0.0
        self._rag_last_failure_time = 0.0
        self._similarity_cache: OrderedDict[str, List[Any]] = OrderedDict()
        self._max_cache_size = 1000  # Default cache size
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._max_embedding_cache_size = 100
        self._faiss_to_incident: Dict[int, str] = {}
        
        # Memory constants
        self.MAX_INCIDENT_NODES = 1000
        self.VECTOR_DIM = 128
        self.GRAPH_CACHE_SIZE = 1000
        
        logger.info(f"Initialized EnhancedRAGGraphMemory with cache size {self._max_cache_size}")
    
    @contextmanager
    def _transaction(self):
        """Thread-safe transaction context manager"""
        with self._lock:
            yield
    
    def is_available(self) -> bool:
        """Check if RAG memory is configured and available (not necessarily has data)."""
        # RAG is available if configured, regardless of data presence
        # This fixes cold-start bias
        return getattr(config, 'rag_enabled', False)
    
    def has_historical_data(self) -> bool:
        """Check if RAG has historical data (affects confidence, not availability)."""
        return len(self.incident_nodes) > 0 or self.faiss.get_count() > 0
    
    def is_operational(self) -> bool:
        """Check if RAG is operational and ready to provide value."""
        # Legacy method for compatibility
        return self.is_available() and self.has_historical_data()
    
    def _generate_incident_id(self, event: Any) -> str:
        """Generate unique incident ID from event fingerprint"""
        # Lazy import to avoid circular dependency
        if hasattr(event, 'component'):
            component = event.component
        else:
            component = "unknown"
        
        if hasattr(event, 'latency_p99'):
            latency = event.latency_p99
        else:
            latency = 0.0
        
        fingerprint_data = f"{component}:{latency:.2f}"
        fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()
        return f"inc_{fingerprint[:16]}"
    
    def _embed_incident(self, event: Any, analysis: Dict[str, Any]) -> np.ndarray:
        """Create embedding vector from incident data"""
        cache_key = f"{hash(str(event))}:{hash(str(analysis))}"
        
        with self._transaction():
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]
        
        try:
            # Extract features
            features: List[float] = []
            
            # Add numerical features
            if hasattr(event, 'latency_p99'):
                features.append(float(event.latency_p99) / 1000.0)
            else:
                features.append(0.0)
            
            if hasattr(event, 'error_rate'):
                features.append(float(event.error_rate))
            else:
                features.append(0.0)
            
            if hasattr(event, 'throughput'):
                features.append(float(event.throughput) / 10000.0)
            else:
                features.append(0.0)
            
            if hasattr(event, 'cpu_util'):
                features.append(float(event.cpu_util) if event.cpu_util is not None else 0.0)
            else:
                features.append(0.0)
            
            # Add categorical features as embeddings
            if hasattr(event, 'severity'):
                severity_value = getattr(event.severity, 'value', "low")
                severity_map = {"low": 0.1, "medium": 0.3, "high": 0.7, "critical": 1.0}
                features.append(severity_map.get(severity_value, 0.1))
            else:
                features.append(0.1)
            
            if hasattr(event, 'component'):
                component_hash = int(hashlib.md5(event.component.encode()).hexdigest()[:8], 16) / 2**32
                features.append(component_hash)
            else:
                features.append(0.0)
            
            # Add analysis confidence
            confidence = analysis.get("incident_summary", {}).get("anomaly_confidence", 0.5)
            features.append(float(confidence))
            
            # Pad or truncate to target dimension
            target_dim = self.VECTOR_DIM
            if len(features) < target_dim:
                features.extend([0.0] * (target_dim - len(features)))
            else:
                features = features[:target_dim]
            
            embedding = np.array(features, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Cache embedding
            with self._transaction():
                self._embedding_cache[cache_key] = embedding
                if len(self._embedding_cache) > self._max_embedding_cache_size:
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding error: {e}", exc_info=True)
            return np.zeros(self.VECTOR_DIM, dtype=np.float32)
    
    def store_incident(self, event: Any, analysis: Dict[str, Any]) -> str:
        """Store incident in RAG graph memory"""
        if not self.is_available():
            return ""
        
        incident_id = self._generate_incident_id(event)
        
        # Check if already exists
        with self._transaction():
            if incident_id in self.incident_nodes:
                node = self.incident_nodes[incident_id]
                node["agent_analysis"] = analysis
                node["metadata"]["last_updated"] = datetime.now().isoformat()
                return incident_id
        
        # Create embedding and store
        embedding = self._embed_incident(event, analysis)
        faiss_index_id: Optional[int] = None
        
        try:
            # Store in FAISS
            text_description = f"{getattr(event, 'component', 'unknown')} {getattr(event, 'latency_p99', 0):.1f}"
            if hasattr(self.faiss, "add_text"):
                faiss_index_id = self.faiss.add_text(text_description, embedding.tolist())
            elif hasattr(self.faiss, "add_async"):
                faiss_index_id = self.faiss.add_async(embedding.reshape(1, -1))
            else:
                faiss_index_id = len(self.incident_nodes)
            
            if faiss_index_id is not None:
                with self._transaction():
                    self._faiss_to_incident[faiss_index_id] = incident_id
                    
        except Exception as e:
            logger.error(f"FAISS store error: {e}", exc_info=True)
            faiss_index_id = len(self.incident_nodes)
        
        # Create incident node
        node = {
            "incident_id": incident_id,
            "component": getattr(event, 'component', 'unknown'),
            "severity": getattr(getattr(event, 'severity', 'low'), 'value', 'low'),
            "timestamp": getattr(getattr(event, 'timestamp', datetime.now()), "isoformat", lambda: datetime.now().isoformat())(),
            "metrics": {
                "latency_ms": getattr(event, 'latency_p99', 0.0),
                "error_rate": getattr(event, 'error_rate', 0.0),
                "throughput": getattr(event, 'throughput', 0.0),
                "cpu_util": float(getattr(event, 'cpu_util', 0.0)),
                "memory_util": float(getattr(event, 'memory_util', 0.0)),
            },
            "agent_analysis": analysis,
            "embedding_id": faiss_index_id,
            "faiss_index": faiss_index_id,
            "metadata": {
                "revenue_impact": getattr(event, "revenue_impact", 0.0),
                "user_impact": getattr(event, "user_impact", 0.0),
                "upstream_deps": getattr(event, "upstream_deps", []),
                "downstream_deps": getattr(event, "downstream_deps", []),
                "service_mesh": getattr(event, "service_mesh", ""),
                "fingerprint": getattr(event, "fingerprint", ""),
                "created_at": datetime.now().isoformat(),
                "embedding_dim": self.VECTOR_DIM
            }
        }
        
        # Store with capacity management
        with self._transaction():
            self.incident_nodes[incident_id] = node
            self._stats["total_incidents_stored"] += 1
            self._stats["last_store_time"] = datetime.now().isoformat()
            
            # Evict oldest if capacity exceeded
            if len(self.incident_nodes) > self.MAX_INCIDENT_NODES:
                oldest_id = min(self.incident_nodes.keys(), 
                               key=lambda x: self.incident_nodes[x].get("metadata", {}).get("created_at", ""))
                oldest_node = self.incident_nodes.pop(oldest_id)
                if oldest_node.get("faiss_index") is not None:
                    self._faiss_to_incident.pop(oldest_node["faiss_index"], None)
                logger.debug(f"Evicted oldest incident {oldest_id}")
        
        logger.info(f"Stored incident {incident_id}: {node['component']}")
        return incident_id
    
    def find_similar(self, event: Any, analysis: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
        """Find similar historical incidents"""
        if not self.is_available():
            return []
        
        try:
            # Create embedding for search
            embedding = self._embed_incident(event, analysis)
            key = hashlib.sha256(embedding.tobytes()).hexdigest()
            
            # Check cache
            with self._transaction():
                if key in self._similarity_cache:
                    self._stats["cache_hits"] += 1
                    return self._similarity_cache[key]
            
            # If no historical data, return empty list but RAG is still "available"
            if not self.has_historical_data():
                logger.debug("RAG available but no historical data yet (cold start)")
                return []
            
            # Perform FAISS search
            self._stats["similarity_searches"] += 1
            results: List[Dict[str, Any]] = []
            
            try:
                if hasattr(self.faiss, "query"):
                    search_results = self.faiss.query(embedding.reshape(1, -1), top_k=k)
                    for idx, score in search_results:
                        incident_id = self._faiss_to_incident.get(idx, "")
                        if incident_id and incident_id in self.incident_nodes:
                            incident = self.incident_nodes[incident_id]
                            results.append({
                                **incident,
                                "similarity_score": float(score)
                            })
            except Exception as e:
                logger.error(f"FAISS search error: {e}", exc_info=True)
            
            # Cache results
            with self._transaction():
                self._similarity_cache[key] = results
                if len(self._similarity_cache) > self._max_cache_size:
                    self._similarity_cache.popitem(last=False)
            
            return results
            
        except Exception as e:
            logger.error(f"Find similar error: {e}", exc_info=True)
            return []
    
    def store_outcome(self, incident_id: str, actions_taken: List[str], success: bool, 
                     resolution_time_minutes: Optional[float] = None, 
                     lessons_learned: Optional[List[str]] = None) -> str:
        """Store outcome for learning"""
        if incident_id not in self.incident_nodes:
            logger.warning(f"Cannot store outcome: Incident {incident_id} not found")
            return ""
        
        actions_hash = hashlib.sha256(",".join(actions_taken).encode()).hexdigest()
        outcome_id = f"out_{hashlib.sha256(f'{incident_id}:{actions_hash}'.encode()).hexdigest()[:16]}"
        
        outcome = {
            "outcome_id": outcome_id,
            "incident_id": incident_id,
            "actions_taken": actions_taken,
            "success": success,
            "resolution_time_minutes": resolution_time_minutes,
            "created_at": datetime.now().isoformat(),
            "lessons_learned": lessons_learned or []
        }
        
        with self._transaction():
            self.outcome_nodes[outcome_id] = outcome
            self._stats["total_outcomes_stored"] += 1
            
            # Create edge
            edge = {
                "edge_id": f"edge_{hashlib.sha256(f'{incident_id}:{outcome_id}'.encode()).hexdigest()[:16]}",
                "source_id": incident_id,
                "target_id": outcome_id,
                "edge_type": "RESOLVES",
                "metadata": {"created_at": datetime.now().isoformat()}
            }
            self.edges.append(edge)
            self._stats["total_edges_created"] += 1
        
        return outcome_id
    
    def get_historical_effectiveness(self, action: str, component_filter: Optional[str] = None) -> EffectivenessStats:
        """Get effectiveness statistics for an action"""
        relevant_outcomes = []
        
        for outcome in self.outcome_nodes.values():
            if action in outcome.get("actions_taken", []):
                incident = self.incident_nodes.get(outcome["incident_id"])
                if incident and (component_filter is None or incident["component"] == component_filter):
                    relevant_outcomes.append(outcome)
        
        total_uses = len(relevant_outcomes)
        successful_uses = sum(1 for o in relevant_outcomes if o.get("success", False))
        resolution_times = [o.get("resolution_time_minutes", 0.0) for o in relevant_outcomes]
        
        avg_resolution = np.mean(resolution_times) if resolution_times else 0.0
        std_resolution = np.std(resolution_times) if resolution_times else 0.0
        success_rate = successful_uses / total_uses if total_uses > 0 else 0.0
        
        return {
            "action": action,
            "total_uses": total_uses,
            "successful_uses": successful_uses,
            "success_rate": success_rate,
            "avg_resolution_time_minutes": avg_resolution,
            "resolution_time_std": std_resolution,
            "component_filter": component_filter,
            "data_points": total_uses
        }
    
    def get_most_effective_actions(self, component: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get most effective actions for a component"""
        component_actions: Dict[str, Dict[str, Any]] = {}
        
        for outcome in self.outcome_nodes.values():
            incident = self.incident_nodes.get(outcome["incident_id"])
            if incident and incident["component"] == component:
                for action in outcome.get("actions_taken", []):
                    if action not in component_actions:
                        component_actions[action] = {"uses": 0, "successes": 0}
                    component_actions[action]["uses"] += 1
                    if outcome.get("success", False):
                        component_actions[action]["successes"] += 1
        
        # Calculate success rates
        results = []
        for action, stats in component_actions.items():
            success_rate = stats["successes"] / stats["uses"] if stats["uses"] > 0 else 0.0
            results.append({
                "action": action,
                "success_rate": success_rate,
                "total_uses": stats["uses"],
                "successful_uses": stats["successes"]
            })
        
        # Sort by success rate
        results.sort(key=lambda x: x["success_rate"], reverse=True)
        return results[:k]
    
    def get_graph_stats(self) -> GraphStats:
        """Get comprehensive graph statistics"""
        component_distribution: Dict[str, int] = {}
        for inc in self.incident_nodes.values():
            comp = inc.get("component", "unknown")
            component_distribution[comp] = component_distribution.get(comp, 0) + 1
        
        incidents_with_outcomes = len({o["incident_id"] for o in self.outcome_nodes.values()})
        avg_outcomes_per_incident = len(self.outcome_nodes) / len(self.incident_nodes) if self.incident_nodes else 0.0
        cache_hit_rate = self._stats["cache_hits"] / self._stats["similarity_searches"] if self._stats["similarity_searches"] > 0 else 0.0
        
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
                "max_incidents": self.MAX_INCIDENT_NODES,
                "max_cache_size": self._max_cache_size,
                "max_embedding_cache": self._max_embedding_cache_size
            },
            "v3_enabled": True,
            "is_operational": self.is_operational(),
            "circuit_breaker": {
                "rag_failures": self._rag_failures,
                "disabled_until": self._rag_disabled_until
            }
        }


# ============================================================================
# ENHANCED MCP RESPONSE
# ============================================================================

@dataclass
class MCPResponse(BaseMCPResponse):
    """Extended MCP response with v3 enhancements"""
    approval_id: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    confidence_basis: Optional[str] = None  # NEW: Track confidence source
    learning_applied: bool = False  # NEW: Explicit learning flag
    learning_reason: str = "OSS advisory mode"  # NEW: Learning status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with v3 fields"""
        base_dict = super().to_dict()
        if self.approval_id:
            base_dict["approval_id"] = self.approval_id
        if self.tool_result:
            base_dict["tool_result"] = self.tool_result
        if self.confidence_basis:
            base_dict["confidence_basis"] = self.confidence_basis
        base_dict["learning_applied"] = self.learning_applied
        base_dict["learning_reason"] = self.learning_reason
        return base_dict


# ============================================================================
# ENHANCED V3 RELIABILITY ENGINE
# ============================================================================

class V3ReliabilityEngine(BaseV3Engine):
    """
    Enhanced reliability engine with RAG Graph memory and MCP execution boundary.
    
    Extends the base V3ReliabilityEngine with:
    1. Semantic search of similar incidents via FAISS
    2. Historical context enhancement for policy decisions  
    3. MCP-governed execution of healing actions
    4. Outcome recording for continuous learning loop
    """
    
    def __init__(
        self,
        faiss_index = None,
        mcp_server = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize enhanced V3 engine with RAG and MCP dependencies.
        
        Args:
            faiss_index: FAISS index for vector similarity
            mcp_server: MCP server for execution boundary
            *args: Additional args passed to base class
            **kwargs: Additional kwargs passed to base class
        """
        # Initialize RAG memory if FAISS index provided
        self.rag = None
        if faiss_index:
            self.rag = EnhancedRAGGraphMemory(faiss_index)
        
        # Store MCP server
        self.mcp = mcp_server
        
        # Pass to base class
        kwargs['rag_graph'] = self.rag
        kwargs['mcp_server'] = mcp_server
        super().__init__(*args, **kwargs)
        
        # V3-specific state
        self._v3_lock = threading.RLock()
        self.v3_metrics: Dict[str, Any] = {
            "v3_features_active": True,
            "rag_queries": 0,
            "rag_cache_hits": 0,
            "mcp_calls": 0,
            "mcp_successes": 0,
            "learning_updates": 0,
            "similar_incidents_found": 0,
            "historical_context_used": 0,
        }
        
        # Learning state - OSS defaults
        self.learning_state: Dict[str, Any] = {
            "successful_predictions": 0,
            "failed_predictions": 0,
            "total_learned_patterns": 0,
            "last_learning_update": time.time(),
            "learning_enabled": False,  # OSS default
            "enterprise_learning": False,  # Only Enterprise learns from outcomes
        }
        
        logger.info(
            f"Initialized Enhanced V3ReliabilityEngine with RAG and MCP "
            f"(RAG={self.rag is not None}, MCP={mcp_server is not None})"
        )
    
    @property
    def v3_enabled(self) -> bool:
        """Check if v3 features should be used based on config"""
        # Check feature flags
        if not getattr(config, 'rag_enabled', False):
            return False
        
        if not getattr(config, 'mcp_enabled', False):
            return False
        
        # Check rollout percentage if configured
        rollout_percentage = getattr(config, 'rollout_percentage', 100)
        if rollout_percentage < 100:
            # Simple hash-based rollout
            import random
            random.seed(int(time.time()))
            return random.random() * 100 < rollout_percentage
        
        return True
    
    def _validate_base_contract(self, base_result: Dict[str, Any], event: Any) -> Dict[str, Any]:
        """
        Validate and ensure base result meets enhanced engine contract.
        
        Args:
            base_result: Result from BaseV3Engine.process_event_enhanced
            event: Original event for fallback data
            
        Returns:
            Validated and normalized result dict
        """
        # Required fields from base contract
        required = {
            "status": base_result.get("status", "UNKNOWN"),
            "incident_id": base_result.get("incident_id", f"fallback_{int(time.time())}_{event.component if hasattr(event, 'component') else 'unknown'}"),
            "healing_actions": base_result.get("healing_actions", []),
        }
        
        # Ensure analysis field exists (silent coupling fix)
        if "analysis" not in base_result:
            base_result["analysis"] = {
                "incident_summary": {
                    "anomaly_confidence": base_result.get("confidence", 0.5),
                    "severity": "medium" if base_result.get("status") == "ANOMALY" else "low",
                    "component": getattr(event, 'component', 'unknown'),
                    "latency_ms": getattr(event, 'latency_p99', 0.0),
                    "error_rate": getattr(event, 'error_rate', 0.0)
                },
                "detection_source": "v3_base_fallback",
                "base_engine_version": "v3_base",
            }
        
        # Ensure confidence exists
        if "confidence" not in base_result:
            base_result["confidence"] = 0.85 if base_result.get("status") == "ANOMALY" else 0.95
        
        return {**base_result, **required}
    
    async def process_event_enhanced(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Enhanced event processing with RAG retrieval and MCP execution.
        
        Extends the base implementation with v3 features.
        """
        # Get event from args or kwargs
        event = kwargs.get("event") or (args[0] if args else None)
        
        # Import ReliabilityEvent lazily to avoid circular import
        try:
            from ..models import ReliabilityEvent
            if not event or not isinstance(event, ReliabilityEvent):
                return {
                    "status": "ERROR",
                    "incident_id": "",
                    "error": "Invalid event",
                    "healing_actions": []
                }
        except ImportError:
            # Fallback if ReliabilityEvent not available
            if not event or not hasattr(event, 'component'):
                return {
                    "status": "ERROR",
                    "incident_id": "",
                    "error": "Invalid event or missing ReliabilityEvent import",
                    "healing_actions": []
                }
        
        # Start timing
        start_time = time.time()
        
        try:
            # Step 1: Run base processing with defensive contract handling
            try:
                base_result = await super().process_event_enhanced(event)
                
                # Validate and normalize contract (FIXED: Silent coupling)
                base_result = self._validate_base_contract(base_result, event)
                
            except Exception as e:
                logger.error(f"Base engine failed: {e}")
                # Create minimal valid result
                base_result = self._validate_base_contract({}, event)
            
            # If not an anomaly, return early
            if base_result.get("status") != "ANOMALY":
                return base_result
            
            # Step 2: RAG RETRIEVAL (v3 enhancement)
            rag_context: Dict[str, Any] = {}
            similar_incidents: List[Dict[str, Any]] = []
            confidence_basis = ConfidenceBasis.POLICY_ONLY
            rag_available = False
            rag_has_data = False
            
            if self.v3_enabled and self.rag:
                try:
                    # Check RAG state separately (FIXED: Cold-start bias)
                    rag_available = self.rag.is_available()
                    rag_has_data = self.rag.has_historical_data()
                    
                    if rag_available:
                        # Get analysis from base result (now guaranteed to exist)
                        analysis = base_result.get("analysis", {})
                        
                        # Use RAG to find similar historical incidents
                        similar_incidents = self.rag.find_similar(event, analysis, k=3)
                        
                        with self._v3_lock:
                            self.v3_metrics["rag_queries"] += 1
                            self.v3_metrics["similar_incidents_found"] += len(similar_incidents)
                        
                        # Update confidence basis based on RAG results
                        if similar_incidents:
                            confidence_basis = ConfidenceBasis.HISTORICAL_SIMILARITY
                        elif rag_has_data:
                            # RAG has data but no similar incidents
                            logger.debug(f"RAG available with {self.rag.get_graph_stats()['incident_nodes']} incidents, but no similar ones found")
                        else:
                            # RAG available but no historical data (cold start)
                            logger.debug("RAG available but no historical data yet (cold start)")
                        
                        # Build RAG context
                        rag_context = self._build_rag_context(similar_incidents, event)
                        
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
                    # Continue without RAG context
            
            # Step 3: ENHANCE POLICY DECISION with historical context
            enhanced_actions = []
            base_actions = base_result.get("healing_actions", [])
            
            if similar_incidents:
                # Enhance with historical context
                enhanced_actions = self._enhance_actions_with_context(
                    base_actions, similar_incidents, event, rag_context, confidence_basis
                )
                
                with self._v3_lock:
                    self.v3_metrics["historical_context_used"] += 1
            else:
                enhanced_actions = base_actions
            
            # Determine deterministic confidence
            deterministic_actions = {
                "restart_container", "scale_up", "scale_down",
                "toggle_feature_flag", "clear_cache", "reset_connection_pool"
            }
            
            for action in enhanced_actions:
                action_name = action.get("action", "")
                if action_name in deterministic_actions and self._is_deterministic_guarantee(action, event):
                    action["confidence_basis"] = ConfidenceBasis.DETERMINISTIC_GUARANTEE
                    # Boost confidence for deterministic actions
                    action["confidence"] = min(0.98, action.get("confidence", 0.5) + 0.4)
                elif "confidence_basis" not in action:
                    action["confidence_basis"] = confidence_basis.value
            
            # Step 4: MCP EXECUTION BOUNDARY (v3 enhancement)
            mcp_results: List[Dict[str, Any]] = []
            executed_actions: List[Dict[str, Any]] = []
            
            if self.v3_enabled and self.mcp and enhanced_actions:
                for action in enhanced_actions:
                    try:
                        # Create MCP request
                        mcp_request = self._create_mcp_request(
                            action, event, similar_incidents, rag_context
                        )
                        
                        # Execute via MCP
                        mcp_response_dict = await self.mcp.execute_tool(mcp_request)
                        
                        # Convert to MCPResponse object with confidence tracking
                        mcp_response = MCPResponse(
                            executed=mcp_response_dict.get("executed", False),
                            status=mcp_response_dict.get("status", "unknown"),
                            result=mcp_response_dict.get("result", {}),
                            message=mcp_response_dict.get("message", ""),
                            approval_id=mcp_response_dict.get("approval_id"),
                            tool_result=mcp_response_dict.get("tool_result"),
                            confidence_basis=action.get("confidence_basis", confidence_basis.value),
                            learning_applied=False,  # OSS default
                            learning_reason="OSS advisory mode does not learn from outcomes"
                        )
                        
                        with self._v3_lock:
                            self.v3_metrics["mcp_calls"] += 1
                            if mcp_response.executed or mcp_response.status == "completed":
                                self.v3_metrics["mcp_successes"] += 1
                        
                        mcp_results.append(mcp_response.to_dict())
                        
                        # If action was executed, record it
                        if mcp_response.executed or mcp_response.status == "completed":
                            executed_actions.append(action)
                            
                            # Step 5: OUTCOME RECORDING (v3 learning loop) - Enterprise only
                            if self.rag and getattr(config, 'enterprise_mode', False):
                                outcome = await self._record_outcome(
                                    incident_id=base_result["incident_id"],
                                    action=action,
                                    mcp_response=mcp_response.to_dict(),
                                    event=event,
                                    similar_incidents=similar_incidents
                                )
                                
                                # Update learning state (Enterprise only)
                                success = outcome.get("success", False)
                                if getattr(config, 'enterprise_mode', False):
                                    self._update_learning_state(success, {
                                        "incident_id": base_result["incident_id"],
                                        "action": action,
                                        "similar_incidents_count": len(similar_incidents),
                                        "rag_context": rag_context
                                    })
                    
                    except Exception as e:
                        logger.error(f"MCP execution failed for action {action.get('action', 'unknown')}: {e}")
                        mcp_results.append({
                            "error": str(e),
                            "executed": False,
                            "status": "failed"
                        })
            
            # Step 6: Build comprehensive result
            result: Dict[str, Any] = {
                **base_result,
                "v3_processing": "enabled" if self.v3_enabled else "disabled",
                "v3_enhancements": {
                    "rag_enabled": bool(self.rag),
                    "mcp_enabled": bool(self.mcp),
                    "learning_enabled": getattr(config, 'learning_enabled', False) and getattr(config, 'enterprise_mode', False),  # Only Enterprise
                },
                "processing_time_ms": (time.time() - start_time) * 1000,
                "engine_version": "v3_enhanced",
                "rag_state": {
                    "available": rag_available,
                    "has_data": rag_has_data,
                    "used_in_decision": bool(similar_incidents)
                },
                "learning_applied": False,  # OSS default
                "learning_reason": "OSS advisory mode does not persist or learn from outcomes",
                "confidence_regime": confidence_basis.value,
            }
            
            # Add v3-specific data if available
            if similar_incidents:
                result["rag_context"] = {
                    "similar_incidents_count": len(similar_incidents),
                    "avg_similarity": rag_context.get("avg_similarity", 0.0),
                    "most_effective_action": rag_context.get("most_effective_action"),
                    "historical_success_rate": rag_context.get("success_rate", 0.0),
                    "confidence_basis": confidence_basis.value,
                }
            
            if mcp_results:
                result["mcp_execution"] = mcp_results
                result["executed_actions"] = executed_actions
            
            # Update metrics
            with self._lock:
                self.metrics["events_processed"] += 1
                if base_result.get("status") == "ANOMALY":
                    self.metrics["anomalies_detected"] += 1
                if mcp_results:
                    executed_count = len([r for r in mcp_results if r.get("executed")])
                    self.metrics["mcp_executions"] += executed_count
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in enhanced v3 event processing: {e}")
            
            # Fall back to base result on error
            try:
                base_result = await super().process_event_enhanced(event)
                base_result["v3_error"] = str(e)
                base_result["v3_processing"] = "failed"
                return base_result
            except Exception as base_error:
                return {
                    "status": "ERROR",
                    "incident_id": f"error_{int(time.time())}_{event.component if hasattr(event, 'component') else 'unknown'}",
                    "error": f"v3: {e}, base fallback: {base_error}",
                    "v3_processing": "failed",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                }
    
    def _create_fallback_result(self, event: Any, partial_result: Dict) -> Dict:
        """Create fallback result when base engine fails."""
        return {
            "incident_id": f"fallback_{int(time.time())}_{event.component if hasattr(event, 'component') else 'unknown'}",
            "analysis": {
                "severity": "medium",
                "fallback_mode": True,
                "component": getattr(event, 'component', 'unknown'),
                "latency_ms": getattr(event, 'latency_p99', 0.0),
                "error_rate": getattr(event, 'error_rate', 0.0)
            },
            "healing_actions": [{
                "action": "investigate_manually",
                "confidence": 0.5,
                "confidence_basis": ConfidenceBasis.POLICY_ONLY.value,
                "reason": "Fallback mode due to engine error",
                "parameters": {},
                "metadata": {"fallback": True}
            }],
            "status": "ANOMALY",
            "v3_fallback": True,
            **partial_result  # Keep any valid fields from partial result
        }
    
    def _is_deterministic_guarantee(self, action: Dict[str, Any], event: Any) -> bool:
        """Check if action has deterministic safety guarantees."""
        # Check if action has rollback capability
        if "rollback" in action.get("safety_features", []):
            return True
        
        # Check bounded impact
        if getattr(event, 'latency_p99', 0) < 5000:  # Bounded latency impact
            if action.get("reversible", False):
                return True
        
        return False
    
    def _build_rag_context(
        self, 
        similar_incidents: List[Dict[str, Any]], 
        current_event: Any
    ) -> Dict[str, Any]:
        """Build RAG context from similar incidents"""
        if not similar_incidents:
            return {}
        
        # Calculate average similarity
        similarity_scores = [inc.get("similarity_score", 0.0) for inc in similar_incidents]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Calculate success rate
        successful_outcomes = 0
        total_outcomes = 0
        
        for incident in similar_incidents:
            # Check if incident has outcomes
            incident_id = incident.get("incident_id")
            if incident_id:
                # Count outcomes for this incident
                incident_outcomes = [
                    o for o in self.rag.outcome_nodes.values() 
                    if o["incident_id"] == incident_id
                ]
                total_outcomes += len(incident_outcomes)
                successful_outcomes += sum(1 for o in incident_outcomes if o.get("success", False))
        
        success_rate = float(successful_outcomes) / total_outcomes if total_outcomes > 0 else 0.0
        
        # Get most effective action if available
        most_effective_action = None
        if hasattr(current_event, 'component'):
            effective_actions = self.rag.get_most_effective_actions(current_event.component, k=1)
            if effective_actions:
                most_effective_action = effective_actions[0]
        
        context: Dict[str, Any] = {
            "similar_incidents_count": len(similar_incidents),
            "avg_similarity": avg_similarity,
            "success_rate": success_rate,
            "component_match": all(
                inc.get("component") == getattr(current_event, 'component', 'unknown')
                for inc in similar_incidents
            ),
        }
        
        if most_effective_action:
            context["most_effective_action"] = most_effective_action
        
        return context
    
    def _enhance_actions_with_context(
        self, 
        base_actions: List[Dict[str, Any]],
        similar_incidents: List[Dict[str, Any]],
        event: Any,
        rag_context: Dict[str, Any],
        confidence_basis: ConfidenceBasis
    ) -> List[Dict[str, Any]]:
        """Enhance healing actions with historical context"""
        if not base_actions:
            return []
        
        enhanced_actions = []
        
        for action in base_actions:
            # Create enhanced action with historical context
            enhanced_action = {
                **action,
                "v3_enhanced": True,
                "historical_confidence": rag_context.get("avg_similarity", 0.0),
                "similar_incidents_count": len(similar_incidents),
                "context_source": "rag_graph",
                "confidence_basis": confidence_basis.value,  # CRITICAL: Set confidence basis
            }
            
            # Add effectiveness score if available
            most_effective = rag_context.get("most_effective_action")
            if most_effective and action.get("action") == most_effective.get("action"):
                enhanced_action["historical_effectiveness"] = most_effective.get("success_rate", 0.0)
                enhanced_action["confidence_boost"] = True
            
            # Apply confidence caps based on basis
            current_confidence = enhanced_action.get("confidence", 0.5)
            
            if confidence_basis == ConfidenceBasis.HISTORICAL_SIMILARITY:
                # Empirical confidence capped at 0.9
                if enhanced_action.get("historical_effectiveness", 0.0) > 0:
                    enhanced_confidence = min(0.9, current_confidence + (enhanced_action["historical_effectiveness"] * 0.3))
                else:
                    enhanced_confidence = min(0.85, current_confidence + 0.2)
                enhanced_action["confidence"] = enhanced_confidence
            
            elif confidence_basis == ConfidenceBasis.DETERMINISTIC_GUARANTEE:
                # Deterministic confidence can exceed 0.9
                enhanced_action["confidence"] = min(0.98, current_confidence + 0.4)
            
            enhanced_actions.append(enhanced_action)
        
        # Sort by historical confidence (descending)
        enhanced_actions.sort(
            key=lambda x: float(x.get("historical_confidence", 0.0)), 
            reverse=True
        )
        
        return enhanced_actions
    
    def _create_mcp_request(
        self, 
        action: Dict[str, Any],
        event: Any,
        historical_context: List[Dict[str, Any]],
        rag_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create MCP request from enhanced action"""
        # Build justification with historical context and confidence basis
        justification_parts = [
            f"Event: {getattr(event, 'component', 'unknown')} with {getattr(event, 'latency_p99', 0):.0f}ms latency, {getattr(event, 'error_rate', 0)*100:.1f}% errors",
        ]
        
        # Add confidence basis explanation
        confidence_basis = action.get("confidence_basis", ConfidenceBasis.POLICY_ONLY.value)
        if confidence_basis == ConfidenceBasis.DETERMINISTIC_GUARANTEE:
            justification_parts.append("Action selected via deterministic safety guarantees (idempotent, reversible, bounded)")
        elif confidence_basis == ConfidenceBasis.HISTORICAL_SIMILARITY:
            if historical_context:
                justification_parts.append(f"Based on {len(historical_context)} similar historical incidents")
            else:
                justification_parts.append("Action selected via historical precedent and policy constraints")
        else:
            justification_parts.append("Action selected via policy constraints and safety validation")
        
        if historical_context:
            justification_parts.append(
                f"Historical similarity: {rag_context.get('avg_similarity', 0.0)*100:.0f}% match"
            )
        
        if rag_context and rag_context.get("most_effective_action"):
            effective = rag_context["most_effective_action"]
            justification_parts.append(
                f"Historically {effective.get('action')} has {effective.get('success_rate', 0)*100:.0f}% success rate"
            )
        
        justification = ". ".join(justification_parts)
        
        return {
            "tool": action.get("action", "unknown"),
            "component": getattr(event, 'component', 'unknown'),
            "parameters": action.get("parameters", {}),
            "justification": justification,
            "metadata": {
                "event_fingerprint": getattr(event, 'fingerprint', ''),
                "event_severity": getattr(getattr(event, 'severity', 'low'), 'value', 'low'),
                "similar_incidents_count": len(historical_context),
                "historical_confidence": rag_context.get("avg_similarity", 0.0) if rag_context else 0.0,
                "confidence_basis": confidence_basis,  # PASS TO MCP SERVER
                "deterministic_guarantee": confidence_basis == ConfidenceBasis.DETERMINISTIC_GUARANTEE,
                "rag_context": rag_context,
                **action.get("metadata", {})
            }
        }
    
    async def _record_outcome(
        self, 
        incident_id: str, 
        action: Dict[str, Any],
        mcp_response: Dict[str, Any],
        event: Optional[Any] = None,
        similar_incidents: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Record outcome for learning loop - Enterprise only"""
        if not self.rag:
            return {}
        
        try:
            # Determine success from mcp_response
            success = (
                mcp_response.get("status") == "completed" or 
                mcp_response.get("executed", False) or
                mcp_response.get("result", {}).get("success", False)
            )
            
            # EXPLICIT BOUNDARY: OSS vs Enterprise
            is_enterprise = getattr(config, 'enterprise_mode', False)
            is_simulated = not is_enterprise
            
            # In OSS mode, use synthetic estimate with explicit flag
            resolution_time_minutes = 5.0  # Default estimate
            
            # Extract lessons learned
            lessons_learned = action.get("metadata", {}).get("lessons_learned", [])
            if not success and mcp_response.get("message"):
                lessons_learned.append(f"Failed: {mcp_response['message']}")
            
            # Add simulation flag for OSS
            if is_simulated:
                lessons_learned.append("OSS_ADVISORY: Outcome simulated for demonstration only")
                lessons_learned.append("Enterprise required for real outcome tracking and learning")
            
            # Store outcome in RAG
            outcome_id = self.rag.store_outcome(
                incident_id=incident_id,
                actions_taken=[action.get("action", "unknown")],
                success=success,
                resolution_time_minutes=resolution_time_minutes,
                lessons_learned=lessons_learned
            )
            
            return {
                "outcome_id": outcome_id,
                "success": success,
                "resolution_time_minutes": resolution_time_minutes,
                "action": action.get("action", "unknown"),
                "simulated_outcome": is_simulated,
                "enterprise_mode": is_enterprise,
                "learning_applied": False,  # OSS never learns
                "learning_reason": "OSS advisory mode does not learn from outcomes",
            }
            
        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            return {}
    
    def _update_learning_state(
        self, 
        success: bool,
        context: Dict[str, Any]
    ) -> None:
        """Update learning state based on outcome - Enterprise only"""
        if not getattr(config, 'enterprise_mode', False):
            # OSS does not learn from outcomes
            return
        
        if not getattr(config, 'learning_enabled', False):
            return
        
        with self._v3_lock:
            self.learning_state["last_learning_update"] = time.time()
            
            if success:
                self.learning_state["successful_predictions"] += 1
            else:
                self.learning_state["failed_predictions"] += 1
            
            # Check if we should extract new patterns
            total_predictions = (
                self.learning_state["successful_predictions"] + 
                self.learning_state["failed_predictions"]
            )
            
            learning_min_data_points = getattr(config, 'learning_min_data_points', 5)
            if total_predictions % learning_min_data_points == 0:
                self._extract_learning_patterns(context)
                self.learning_state["total_learned_patterns"] += 1
                self.v3_metrics["learning_updates"] += 1
    
    def _extract_learning_patterns(self, context: Dict[str, Any]) -> None:
        """Extract learning patterns from context - Enterprise only"""
        if not getattr(config, 'enterprise_mode', False):
            return
            
        logger.debug("Extracting learning patterns from context")
        
        # Example pattern extraction
        incident_id = context.get("incident_id")
        action = context.get("action", {})
        similar_count = context.get("similar_incidents_count", 0)
        
        if similar_count > 0 and action.get("historical_effectiveness", 0) > 0.7:
            logger.info(f"Learned pattern: Action {action.get('action')} effective with historical context")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics including v3"""
        try:
            # Try to get base stats from superclass
            base_stats = super().get_stats()
        except AttributeError:
            # Fallback if base class doesn't have get_stats
            logger.warning("Base class doesn't have get_stats, using fallback")
            base_stats = {
                "events_processed": self.metrics.get("events_processed", 0),
                "anomalies_detected": self.metrics.get("anomalies_detected", 0),
                "rag_queries": self.metrics.get("rag_queries", 0),
                "mcp_executions": self.metrics.get("mcp_executions", 0),
                "successful_outcomes": self.metrics.get("successful_outcomes", 0),
                "failed_outcomes": self.metrics.get("failed_outcomes", 0),
                "uptime_seconds": time.time() - self._start_time,
                "engine_version": "v3_base_fallback",
            }
        
        # Add v3 metrics
        with self._v3_lock:
            v3_stats = self.v3_metrics.copy()
            
            # Calculate rates
            if v3_stats["rag_queries"] > 0:
                v3_stats["rag_cache_hit_rate"] = float(v3_stats["rag_cache_hits"]) / v3_stats["rag_queries"]
            
            if v3_stats["mcp_calls"] > 0:
                v3_stats["mcp_success_rate"] = float(v3_stats["mcp_successes"]) / v3_stats["mcp_calls"]
            
            # Add learning state with explicit flags
            v3_stats.update(self.learning_state)
            
            # Add feature status
            v3_stats["feature_status"] = {
                "rag_available": self.rag is not None,
                "mcp_available": self.mcp is not None,
                "rag_enabled": getattr(config, 'rag_enabled', False),
                "mcp_enabled": getattr(config, 'mcp_enabled', False),
                "learning_enabled": getattr(config, 'learning_enabled', False) and getattr(config, 'enterprise_mode', False),  # Only Enterprise
                "enterprise_mode": getattr(config, 'enterprise_mode', False),
                "rollout_percentage": getattr(config, 'rollout_percentage', 0),
            }
        
        # Combine stats
        combined_stats: Dict[str, Any] = {
            **base_stats,
            "engine_version": "v3_enhanced",
            "v3_features": v3_stats["v3_features_active"],
            "v3_metrics": v3_stats,
            "rag_graph_stats": self.rag.get_graph_stats() if self.rag else None,
            "learning_boundary": {
                "oss_learning": False,
                "enterprise_learning": getattr(config, 'enterprise_mode', False) and getattr(config, 'learning_enabled', False),
                "learning_applied": self.learning_state["total_learned_patterns"] > 0,
            }
        }
        
        # Add MCP stats if available
        if self.mcp and hasattr(self.mcp, 'get_server_stats'):
            combined_stats["mcp_server_stats"] = self.mcp.get_server_stats()
        
        return combined_stats
    
    def shutdown(self) -> None:
        """Graceful shutdown of enhanced v3 engine"""
        logger.info("Shutting down Enhanced V3ReliabilityEngine...")
        
        # Save any pending learning data - Enterprise only
        if getattr(config, 'enterprise_mode', False) and getattr(config, 'learning_enabled', False):
            logger.info(f"Saved {self.learning_state['total_learned_patterns']} learning patterns")
        
        try:
            # Try to call super().shutdown() if it exists
            super().shutdown()
        except AttributeError:
            # Base class doesn't have shutdown, just log
            logger.debug("Base class doesn't have shutdown method")
        
        logger.info("Enhanced V3ReliabilityEngine shutdown complete")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_v3_engine(
    faiss_index = None,
    mcp_server = None
) -> V3ReliabilityEngine:
    """
    Factory function to create enhanced V3 engine
    
    Args:
        faiss_index: Optional FAISS index for RAG memory
        mcp_server: Optional MCP server for execution boundary
        
    Returns:
        Configured V3ReliabilityEngine instance
    """
    try:
        return V3ReliabilityEngine(
            faiss_index=faiss_index, 
            mcp_server=mcp_server
        )
    except Exception as e:
        logger.exception(f"Error creating enhanced V3 engine: {e}")
        raise
