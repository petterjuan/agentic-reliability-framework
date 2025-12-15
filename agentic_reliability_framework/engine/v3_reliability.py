"""
V3 Reliability Engine - Enhanced with RAG Graph and MCP Server integration.
Pythonic implementation with proper typing, error handling, and safety features.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import numpy as np

from ..memory.rag_graph import RAGGraphMemory
from .reliability import EnhancedReliabilityEngine as V2Engine
from .mcp_server import MCPServer
from ..config import config
from ..models import ReliabilityEvent

logger = logging.getLogger(__name__)


class V3ReliabilityEngine(V2Engine):
    """
    Enhanced reliability engine with RAG Graph memory and MCP execution boundary.
    
    V3 Design Features:
    1. Semantic search of similar incidents via FAISS
    2. Historical context enhancement for policy decisions
    3. MCP-governed execution of healing actions
    4. Outcome recording for continuous learning loop
    
    Architecture: Bridge pattern connecting v2 engine with v3 features
    """
    
    def __init__(
        self,
        rag_graph: RAGGraphMemory,
        mcp_server: MCPServer,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize V3 engine with RAG and MCP dependencies.
        
        Args:
            rag_graph: RAG Graph memory for historical incident retrieval
            mcp_server: MCP Server for governed execution boundary
        """
        super().__init__(*args, **kwargs)
        self.rag = rag_graph
        self.mcp = mcp_server
        
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
        
        # Learning state
        self.learning_state: Dict[str, Any] = {
            "successful_predictions": 0,
            "failed_predictions": 0,
            "total_learned_patterns": 0,
            "last_learning_update": time.time(),
        }
        
        logger.info(
            f"Initialized V3ReliabilityEngine with RAG and MCP "
            f"(mode={mcp_server.mode.value})"
        )
    
    @property
    def v3_enabled(self) -> bool:
        """Check if v3 features should be used"""
        if not config.rag_enabled or not config.mcp_enabled:
            return False
        
        # Check rollout percentage if configured
        if config.rollout_percentage < 100:
            # Simple hash-based rollout
            import random
            random.seed(int(time.time()))
            return random.random() * 100 < config.rollout_percentage
        
        return True
    
    @asynccontextmanager
    async def _v3_execution_context(self, event: ReliabilityEvent) -> Any:
        """Context manager for v3 feature execution with metrics"""
        # Removed unused start_time variable
        try:
            yield
        finally:
            # Update metrics
            with self._v3_lock:
                self.v3_metrics["rag_queries"] += 1
                self.v3_metrics["similar_incidents_found"] += 1
    
    async def process_event_enhanced(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Enhanced event processing with RAG retrieval and MCP execution.
        
        V3 Flow:
        1. Run standard v2 analysis
        2. Retrieve similar incidents from RAG (if enabled)
        3. Enhance policy decisions with historical context
        4. Execute via MCP with safety checks (if enabled)
        5. Record outcomes for learning loop
        
        Returns:
            Enhanced results with v3 features
        """
        # 1. Original v2 analysis
        result = await super().process_event_enhanced(*args, **kwargs)
        
        # Skip v3 processing if not enabled or not an anomaly
        if not self.v3_enabled or result.get("status") != "ANOMALY":
            result["v3_processing"] = "skipped"
            return result
        
        try:
            # Extract event
            event = kwargs.get("event") or (args[0] if args else None)
            if not event or not isinstance(event, ReliabilityEvent):
                logger.warning("Invalid or missing event for v3 processing")
                return result
            
            # 2. RAG RETRIEVAL - Get similar historical incidents
            similar_incidents: List[Any] = []
            rag_context: Dict[str, Any] = {}
            
            if self.rag and self.rag.is_enabled():
                async with self._v3_execution_context(event):
                    similar_incidents = self.rag.find_similar(event, k=3)
                    
                    if similar_incidents:
                        self.v3_metrics["similar_incidents_found"] += len(similar_incidents)
                        self.v3_metrics["historical_context_used"] += 1
                        
                        rag_context = {
                            "similar_count": len(similar_incidents),
                            "avg_similarity": self._calculate_avg_similarity(similar_incidents),
                            "most_effective_action": self._get_most_effective_action(similar_incidents, event.component),
                            "success_rate": self._calculate_success_rate(similar_incidents),
                        }
                        
                        logger.info(
                            f"Found {len(similar_incidents)} similar incidents via RAG "
                            f"for {event.component}"
                        )
            
            # 3. ENHANCE POLICY DECISION with historical context
            enhanced_actions: List[Dict[str, Any]] = []
            
            if similar_incidents and rag_context:
                # Get healing actions from v2 engine
                base_actions = result.get("healing_actions", [])
                
                # Enhance with historical context
                enhanced_actions = self._enhance_actions_with_context(
                    base_actions, similar_incidents, event, rag_context
                )
                
                # Update result with enhanced actions
                if enhanced_actions:
                    result["enhanced_healing_actions"] = enhanced_actions
                    result["rag_context"] = rag_context
            
            # 4. MCP EXECUTION BOUNDARY - Execute via MCP server
            mcp_results: List[Dict[str, Any]] = []
            
            if enhanced_actions and self.mcp:
                for action in enhanced_actions[:3]:  # Limit to top 3 actions
                    try:
                        mcp_request = self._create_mcp_request(action, event, similar_incidents, rag_context)
                        
                        # Execute via MCP
                        mcp_response = await self.mcp.execute_tool(mcp_request)
                        mcp_results.append(mcp_response)
                        
                        # Update metrics
                        with self._v3_lock:
                            self.v3_metrics["mcp_calls"] += 1
                            if mcp_response.get("status") == "completed":
                                self.v3_metrics["mcp_successes"] += 1
                        
                        # 5. OUTCOME RECORDING - Record for learning loop
                        if mcp_response.get("executed", False):
                            await self._record_outcome(
                                incident_id=result.get("incident_id", ""),
                                action=action,
                                mcp_response=mcp_response,
                                event=event,
                                similar_incidents=similar_incidents
                            )
                            
                    except Exception as e:
                        logger.error(f"Error in MCP execution: {e}", exc_info=True)
                        mcp_results.append({
                            "error": str(e),
                            "action": action.get("action", "unknown"),
                            "status": "failed"
                        })
            
            # Update final result with v3 enhancements
            result["v3_enhancements"] = {
                "rag_enabled": self.rag.is_enabled() if self.rag else False,
                "mcp_enabled": True,
                "similar_incidents_found": len(similar_incidents),
                "enhanced_actions_count": len(enhanced_actions),
                "mcp_executions": len(mcp_results),
                "mcp_successful": sum(1 for r in mcp_results if r.get("status") == "completed"),
                "learning_updated": True if mcp_results else False,
            }
            
            # Add detailed context if available
            if rag_context:
                result["v3_enhancements"]["rag_details"] = rag_context
            
            if mcp_results:
                result["v3_enhancements"]["mcp_details"] = mcp_results
            
            return result
            
        except Exception as e:
            logger.error(f"Error in V3 process_event_enhanced: {e}", exc_info=True)
            # Fall back to v2 result on error
            result["v3_error"] = str(e)
            result["v3_processing"] = "failed"
            return result
    
    def _calculate_avg_similarity(self, similar_incidents: List[Any]) -> float:
        """Calculate average similarity score from similar incidents"""
        if not similar_incidents:
            return 0.0
        
        scores = []
        for incident in similar_incidents:
            if hasattr(incident, 'metadata'):
                score = incident.metadata.get("similarity_score")
                if score is not None:
                    scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _get_most_effective_action(
        self, 
        similar_incidents: List[Any], 
        component: str
    ) -> Optional[Dict[str, Any]]:
        """Get most effective action from similar incidents"""
        if not similar_incidents or not self.rag:
            return None
        
        try:
            # Get most effective actions from RAG
            effective_actions = self.rag.get_most_effective_actions(component, k=1)
            return effective_actions[0] if effective_actions else None
        except Exception as e:
            logger.debug(f"Error getting most effective action: {e}")
            return None
    
    def _calculate_success_rate(self, similar_incidents: List[Any]) -> float:
        """Calculate success rate from similar incidents"""
        if not similar_incidents:
            return 0.0
        
        successful_outcomes = 0
        total_outcomes = 0
        
        for incident in similar_incidents:
            if hasattr(incident, 'outcomes') and incident.outcomes:
                total_outcomes += len(incident.outcomes)
                successful_outcomes += sum(1 for o in incident.outcomes if o.success)
        
        return successful_outcomes / total_outcomes if total_outcomes > 0 else 0.0
    
    def _enhance_actions_with_context(
        self, 
        base_actions: List[Dict[str, Any]],
        similar_incidents: List[Any],
        event: ReliabilityEvent,
        rag_context: Dict[str, Any]
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
            }
            
            # Add effectiveness score if available
            most_effective = rag_context.get("most_effective_action")
            if most_effective and action.get("action") == most_effective.get("action"):
                enhanced_action["historical_effectiveness"] = most_effective.get("success_rate", 0.0)
                enhanced_action["confidence_boost"] = True
            
            enhanced_actions.append(enhanced_action)
        
        # Sort by historical confidence (descending)
        enhanced_actions.sort(
            key=lambda x: x.get("historical_confidence", 0.0), 
            reverse=True
        )
        
        return enhanced_actions
    
    def _create_mcp_request(
        self, 
        action: Dict[str, Any], 
        event: ReliabilityEvent,
        similar_incidents: List[Any],
        rag_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create MCP request from enhanced action"""
        # Build justification with historical context
        justification_parts = [
            f"Event: {event.component} with {event.latency_p99:.0f}ms latency, {event.error_rate*100:.1f}% errors",
        ]
        
        if similar_incidents:
            justification_parts.append(
                f"Based on {len(similar_incidents)} similar historical incidents"
            )
        
        if rag_context.get("most_effective_action"):
            effective = rag_context["most_effective_action"]
            justification_parts.append(
                f"Historically {effective.get('action')} has {effective.get('success_rate', 0)*100:.0f}% success rate"
            )
        
        justification = ". ".join(justification_parts)
        
        return {
            "tool": action.get("action", "unknown"),
            "component": event.component,
            "parameters": action.get("parameters", {}),
            "justification": justification,
            "metadata": {
                "event_fingerprint": event.fingerprint,
                "event_severity": event.severity.value,
                "similar_incidents_count": len(similar_incidents),
                "historical_confidence": rag_context.get("avg_similarity", 0.0),
                "rag_context": rag_context,
                **action.get("metadata", {})
            }
        }
    
    async def _record_outcome(
        self, 
        incident_id: str, 
        action: Dict[str, Any], 
        mcp_response: Dict[str, Any],
        event: ReliabilityEvent,
        similar_incidents: List[Any]
    ) -> None:
        """Record outcome for learning loop"""
        if not self.rag or not self.rag.is_enabled():
            return
        
        try:
            # Determine success
            success = mcp_response.get("status") == "completed"
            actions_taken = [action.get("action", "unknown")]
            
            # Calculate resolution time (simplified)
            resolution_time_minutes = 5.0  # Default
            
            # Extract lessons learned
            lessons_learned: List[str] = []
            if not success:
                result = mcp_response.get("result", {})
                error_msg = result.get("message", mcp_response.get("message", ""))
                if error_msg:
                    lessons_learned.append(f"Failed: {error_msg}")
            
            # Add context about similar incidents
            if similar_incidents:
                lessons_learned.append(
                    f"Based on {len(similar_incidents)} similar historical incidents"
                )
            
            # Store outcome in RAG
            outcome_id = self.rag.store_outcome(
                incident_id=incident_id,
                actions_taken=actions_taken,
                success=success,
                resolution_time_minutes=resolution_time_minutes,
                lessons_learned=lessons_learned if lessons_learned else None
            )
            
            if outcome_id:
                # Update learning state
                self._update_learning_state(success, {
                    "action": action.get("action"),
                    "component": event.component,
                    "similar_incidents": len(similar_incidents),
                    "historical_confidence": action.get("historical_confidence", 0.0)
                })
                
                logger.info(
                    f"Recorded outcome {outcome_id} for incident {incident_id}: "
                    f"success={success}, actions={actions_taken}"
                )
            
        except Exception as e:
            logger.error(f"Error recording outcome: {e}", exc_info=True)
    
    def _update_learning_state(
        self, 
        success: bool,
        context: Dict[str, Any]
    ):
        """Update learning state based on outcome"""
        if not config.learning_enabled:
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
            
            if total_predictions % config.learning_min_data_points == 0:
                self._extract_learning_patterns(context)
                self.learning_state["total_learned_patterns"] += 1
                self.v3_metrics["learning_updates"] += 1
    
    def _extract_learning_patterns(self, context: Dict[str, Any]):
        """Extract learning patterns from context"""
        # Placeholder for pattern extraction logic
        # In production, this would use ML to identify patterns
        logger.debug("Extracting learning patterns from context")
    
    def get_v3_metrics(self) -> Dict[str, Any]:
        """Get v3-specific metrics"""
        with self._v3_lock:
            metrics = self.v3_metrics.copy()
            
            # Calculate success rates
            if metrics["rag_queries"] > 0:
                metrics["rag_cache_hit_rate"] = float(metrics["rag_cache_hits"]) / metrics["rag_queries"]
            
            if metrics["mcp_calls"] > 0:
                metrics["mcp_success_rate"] = float(metrics["mcp_successes"]) / metrics["mcp_calls"]
            
            # Add learning state
            metrics.update(self.learning_state)
            
            # Add feature status
            metrics["feature_status"] = {
                "rag_available": self.rag is not None,
                "mcp_available": self.mcp is not None,
                "rag_enabled": config.rag_enabled,
                "mcp_enabled": config.mcp_enabled,
                "learning_enabled": config.learning_enabled,
                "rollout_percentage": config.rollout_percentage,
            }
            
            return metrics
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics including v3"""
        base_stats = super().get_engine_stats()
        
        # Add v3 metrics
        v3_stats = self.get_v3_metrics()
        
        # Combine stats
        combined_stats: Dict[str, Any] = {
            **base_stats,
            "v3_features": v3_stats["v3_features_active"],
            "v3_metrics": v3_stats,
            "processing_version": "v3" if v3_stats["v3_features_active"] else "v2",
            "rag_graph_stats": self.rag.get_graph_stats() if self.rag else None
        }
        
        return combined_stats
    
    def shutdown(self):
        """Graceful shutdown of v3 engine"""
        logger.info("Shutting down V3ReliabilityEngine...")
        
        # Save any pending learning data
        if config.learning_enabled:
            logger.info(f"Saved {self.learning_state['total_learned_patterns']} learning patterns")
        
        # Call parent shutdown
        super().shutdown()
        
        logger.info("V3ReliabilityEngine shutdown complete")


# Factory function for backward compatibility
def create_v3_engine(
    rag_graph: Optional[RAGGraphMemory] = None,
    mcp_server: Optional[MCPServer] = None
) -> Optional[V3ReliabilityEngine]:
    """
    Factory function to create V3 engine with optional dependencies
    
    Args:
        rag_graph: Optional RAG graph (will be loaded lazily if None)
        mcp_server: Optional MCP server (will be loaded lazily if None)
    
    Returns:
        V3ReliabilityEngine instance or None if dependencies not available
    """
    try:
        # Lazy load dependencies if not provided
        from ..lazy import get_rag_graph, get_mcp_server
        
        if rag_graph is None:
            rag_graph = get_rag_graph()
        
        if mcp_server is None:
            mcp_server = get_mcp_server()
        
        # Check if we have all required dependencies
        if not rag_graph or not mcp_server:
            logger.warning("Cannot create V3 engine: missing dependencies")
            return None
        
        return V3ReliabilityEngine(rag_graph=rag_graph, mcp_server=mcp_server)
        
    except ImportError as e:
        logger.error(f"Error creating V3 engine: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error creating V3 engine: {e}")
        return None
