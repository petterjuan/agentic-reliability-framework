"""
V3 Reliability Engine with RAG integration and learning loop

Phase 3: Integration & Learning Loop (2 weeks)
Goal: Connect RAG → Policy → MCP → Outcome recording
"""

import asyncio
import threading
import logging
from typing import Dict, Any, List, Optional
import time

from ..models import ReliabilityEvent, HealingAction
from ..healing_policies import PolicyEngine
from ..config import config
from ..memory.rag_graph import RAGGraphMemory
from .reliability import EnhancedReliabilityEngine as V2Engine

logger = logging.getLogger(__name__)


class V3ReliabilityEngine(V2Engine):
    """
    Enhanced engine with learning capability
    
    V3 Design Mandates:
    1. Memory → Decision influence (FAISS must be queried)
    2. Explicit execution boundary (MCP server required)
    3. Explainable learning (RAG graph + audit trails)
    4. Clear OSS/Enterprise split (monetizable boundary)
    """
    
    def __init__(
        self,
        rag_graph: Optional[RAGGraphMemory] = None,
        mcp_server: Optional[Any] = None,  # Will be Phase 2
        **kwargs
    ):
        """
        Initialize v3 reliability engine with RAG support
        
        Args:
            rag_graph: RAGGraphMemory instance for v3 features
            mcp_server: MCP server for execution boundary (Phase 2)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)
        
        self.rag_graph = rag_graph
        self.mcp_server = mcp_server
        
        # Learning loop state
        self.learning_state = {
            "total_learned_patterns": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "last_learning_update": None,
            "active_learning": config.learning_enabled
        }
        
        # Performance metrics for v3
        self.v3_metrics = {
            "rag_queries": 0,
            "rag_cache_hits": 0,
            "rag_timeouts": 0,
            "mcp_calls": 0,
            "mcp_successes": 0,
            "mcp_failures": 0,
            "learning_updates": 0,
            "historical_recommendations_used": 0,
            "v3_features_active": False
        }
        
        self._v3_lock = threading.RLock()
        
        logger.info(
            f"Initialized V3ReliabilityEngine: "
            f"RAG={rag_graph is not None}, "
            f"MCP={mcp_server is not None}, "
            f"Learning={config.learning_enabled}"
        )
    
    async def process_event_enhanced(
        self,
        component: str,
        latency: float,
        error_rate: float,
        throughput: float = 1000,
        cpu_util: Optional[float] = None,
        memory_util: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process event with v3 enhancements
        
        Implementation of Phase 3 integration:
        1. Original v2 analysis
        2. RAG retrieval (if enabled)
        3. Enhanced policy decision with historical context
        4. MCP execution boundary (if enabled)
        5. Outcome recording (if actions executed)
        """
        # Check if v3 features should be used for this request
        if not self._should_use_v3_features():
            logger.debug("v3 features disabled for this request, using v2 engine")
            return await super().process_event_enhanced(
                component, latency, error_rate, throughput, cpu_util, memory_util
            )
        
        logger.info(
            f"Processing event with v3 features for {component}: "
            f"latency={latency}ms, error_rate={error_rate*100:.1f}%"
        )
        
        # 1. Original v2 analysis
        v2_result = await super().process_event_enhanced(
            component, latency, error_rate, throughput, cpu_util, memory_util
        )
        
        # If not an anomaly, return early (no need for RAG/MCP)
        if v2_result["status"] != "ANOMALY":
            return v2_result
        
        # Extract event from v2 result
        event = self._extract_event_from_result(v2_result)
        if not event:
            logger.warning("Could not extract event from v2 result")
            return v2_result
        
        # 2. RAG RETRIEVAL - NEW
        similar_incidents = []
        rag_context = {}
        
        if self.rag_graph and config.rag_enabled:
            similar_incidents = self.rag_graph.find_similar(event, k=3)
            
            with self._v3_lock:
                self.v3_metrics["rag_queries"] += 1
            
            if similar_incidents:
                rag_context = {
                    "similar_incidents_count": len(similar_incidents),
                    "similar_incidents": [
                        {
                            "incident_id": inc.incident_id,
                            "component": inc.component,
                            "severity": inc.severity,
                            "timestamp": inc.timestamp,
                            "similarity_score": inc.metadata.get("similarity_score", 0.0),
                            "has_outcomes": len(getattr(inc, 'outcomes', [])) > 0
                        }
                        for inc in similar_incidents
                    ],
                    "most_effective_actions": self._get_most_effective_actions(similar_incidents)
                }
                
                logger.info(f"RAG found {len(similar_incidents)} similar incidents")
        
        # 3. ENHANCE POLICY DECISION with historical context
        healing_actions = []
        
        if similar_incidents and config.learning_enabled:
            # Use learning-enhanced policy evaluation
            healing_actions = self._evaluate_policies_with_learning(
                event, 
                similar_incidents,
                v2_result.get("healing_actions", [])
            )
            
            with self._v3_lock:
                self.v3_metrics["historical_recommendations_used"] += 1
        else:
            # Use standard v2 policy evaluation
            healing_actions = v2_result.get("healing_actions", [])
        
        # Convert healing actions from strings to HealingAction enum if needed
        healing_action_objs = []
        for action_str in healing_actions:
            try:
                if isinstance(action_str, str):
                    action_obj = HealingAction(action_str)
                else:
                    action_obj = action_str
                healing_action_objs.append(action_obj)
            except (ValueError, AttributeError):
                logger.warning(f"Invalid healing action: {action_str}")
        
        # 4. MCP EXECUTION BOUNDARY - NEW
        mcp_results = []
        
        if self.mcp_server and config.mcp_enabled and healing_action_objs:
            for action in healing_action_objs:
                if action == HealingAction.NO_ACTION:
                    continue
                
                # Create MCP request
                mcp_request = self._create_mcp_request(
                    action=action,
                    event=event,
                    historical_context=similar_incidents,
                    v2_result=v2_result
                )
                
                try:
                    # Execute through MCP
                    mcp_response = await self.mcp_server.execute_tool(mcp_request)
                    mcp_results.append(mcp_response)
                    
                    with self._v3_lock:
                        self.v3_metrics["mcp_calls"] += 1
                        if getattr(mcp_response, 'executed', False):
                            self.v3_metrics["mcp_successes"] += 1
                        else:
                            self.v3_metrics["mcp_failures"] += 1
                    
                    # 5. OUTCOME RECORDING - NEW
                    if getattr(mcp_response, 'executed', False) and self.rag_graph:
                        # Record outcome for learning
                        outcome = await self._record_outcome(
                            incident_id=v2_result.get("incident_id", event.fingerprint),
                            action=action,
                            mcp_response=mcp_response,
                            event=event,
                            resolution_time_minutes=self._calculate_resolution_time(event, mcp_response)
                        )
                        
                        # Store in RAG
                        self.rag_graph.store_outcome(
                            incident_id=v2_result.get("incident_id", event.fingerprint),
                            actions_taken=[action.value],
                            success=getattr(mcp_response, 'success', True),
                            resolution_time_minutes=outcome.get("resolution_time_minutes", 5.0),
                            lessons_learned=outcome.get("lessons_learned", [])
                        )
                        
                        # Update learning state
                        self._update_learning_state(
                            action=action,
                            success=getattr(mcp_response, 'success', True),
                            context={
                                "event": event,
                                "similar_incidents": similar_incidents,
                                "mcp_response": mcp_response
                            }
                        )
                        
                except Exception as e:
                    logger.error(f"MCP execution error: {e}", exc_info=True)
                    mcp_results.append({
                        "error": str(e),
                        "action": action.value,
                        "executed": False
                    })
        
        # 6. Update result with v3 enhancements
        enhanced_result = v2_result.copy()
        
        # Add RAG context
        if rag_context:
            enhanced_result["rag_context"] = rag_context
        
        # Add MCP execution results
        if mcp_results:
            enhanced_result["mcp_execution"] = [
                {
                    "action": getattr(r, 'action', 'unknown'),
                    "status": getattr(r, 'status', 'error'),
                    "executed": getattr(r, 'executed', False),
                    "message": getattr(r, 'message', ''),
                    "timestamp": getattr(r, 'timestamp', '')
                }
                for r in mcp_results
            ]
            
            # Update healing actions based on MCP results
            executed_actions = [
                r.get("action") for r in enhanced_result["mcp_execution"]
                if r.get("executed", False)
            ]
            if executed_actions:
                enhanced_result["healing_actions"] = executed_actions
        
        # Add v3 metadata
        enhanced_result["v3_metadata"] = {
            "rag_enabled": config.rag_enabled and self.rag_graph is not None,
            "mcp_enabled": config.mcp_enabled and self.mcp_server is not None,
            "learning_enabled": config.learning_enabled,
            "similar_incidents_used": len(similar_incidents),
            "historical_effectiveness_used": len(rag_context.get("most_effective_actions", [])),
            "processing_version": "v3"
        }
        
        # Update v3 metrics
        with self._v3_lock:
            self.v3_metrics["v3_features_active"] = True
        
        logger.info(
            f"V3 processing complete: {len(similar_incidents)} similar incidents, "
            f"{len(mcp_results)} MCP executions, {len(healing_action_objs)} actions"
        )
        
        return enhanced_result
    
    def _should_use_v3_features(self) -> bool:
        """
        Determine if v3 features should be used for this request
        
        Implements gradual rollout and feature flagging
        """
        # Check feature flags
        if not config.rag_enabled and not config.mcp_enabled and not config.learning_enabled:
            return False
        
        # Check rollout percentage
        if config.rollout_percentage < 100:
            import random
            if random.randint(1, 100) > config.rollout_percentage:
                return False
        
        # Check if RAG is available
        if config.rag_enabled and not self.rag_graph:
            logger.warning("RAG enabled but RAG graph not available")
            return False
        
        # Check if MCP is available
        if config.mcp_enabled and not self.mcp_server:
            logger.warning("MCP enabled but MCP server not available")
            return False
        
        return True
    
    def _extract_event_from_result(self, result: Dict[str, Any]) -> Optional[ReliabilityEvent]:
        """Extract ReliabilityEvent from v2 result"""
        try:
            # This is a simplified extraction - adjust based on your actual data structure
            from datetime import datetime
            
            return ReliabilityEvent(
                component=result.get("component", "unknown"),
                latency_p99=result.get("latency_p99", 0.0),
                error_rate=result.get("error_rate", 0.0),
                throughput=result.get("throughput", 1000.0),
                cpu_util=result.get("cpu_util"),
                memory_util=result.get("memory_util"),
                severity=result.get("severity", "low"),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error extracting event: {e}")
            return None
    
    def _get_most_effective_actions(self, similar_incidents: List[Any]) -> List[Dict[str, Any]]:
        """Get most effective actions from similar incidents"""
        if not similar_incidents or not self.rag_graph:
            return []
        
        # Collect actions from successful outcomes
        action_effectiveness = {}
        
        for incident in similar_incidents:
            if hasattr(incident, 'outcomes') and incident.outcomes:
                for outcome in incident.outcomes:
                    if outcome.success:
                        for action in outcome.actions_taken:
                            if action not in action_effectiveness:
                                action_effectiveness[action] = {
                                    "success_count": 0,
                                    "total_count": 0,
                                    "resolution_times": []
                                }
                            
                            action_effectiveness[action]["success_count"] += 1
                            action_effectiveness[action]["total_count"] += 1
                            action_effectiveness[action]["resolution_times"].append(
                                outcome.resolution_time_minutes
                            )
        
        # Calculate effectiveness metrics
        effective_actions = []
        for action, stats in action_effectiveness.items():
            if stats["total_count"] > 0:
                success_rate = stats["success_count"] / stats["total_count"]
                avg_resolution_time = (
                    sum(stats["resolution_times"]) / len(stats["resolution_times"])
                    if stats["resolution_times"] else 0.0
                )
                
                # Only include if meets confidence threshold
                if success_rate >= config.learning_confidence_threshold:
                    effective_actions.append({
                        "action": action,
                        "success_rate": success_rate,
                        "avg_resolution_time_minutes": avg_resolution_time,
                        "data_points": stats["total_count"],
                        "confidence": min(1.0, stats["total_count"] / 10.0)
                    })
        
        # Sort by success rate, then by data points
        effective_actions.sort(
            key=lambda x: (x["success_rate"], x["data_points"]), 
            reverse=True
        )
        
        return effective_actions[:5]  # Return top 5
    
    def _evaluate_policies_with_learning(
        self,
        event: ReliabilityEvent,
        similar_incidents: List[Any],
        default_actions: List[str]
    ) -> List[str]:
        """
        Enhance policy decisions with learning from historical data
        
        Args:
            event: Current event
            similar_incidents: Similar historical incidents
            default_actions: Actions from standard policy evaluation
            
        Returns:
            Enhanced list of actions
        """
        if not similar_incidents:
            return default_actions
        
        # Get historical effectiveness
        effective_actions = self._get_most_effective_actions(similar_incidents)
        
        if not effective_actions:
            return default_actions
        
        # Filter default actions based on historical effectiveness
        enhanced_actions = []
        
        for action_str in default_actions:
            if action_str == "no_action":
                enhanced_actions.append(action_str)
                continue
            
            # Find historical effectiveness for this action
            historical_effectiveness = None
            for eff_action in effective_actions:
                if eff_action["action"] == action_str:
                    historical_effectiveness = eff_action
                    break
            
            if historical_effectiveness:
                # Check if action meets confidence threshold
                if (historical_effectiveness["success_rate"] >= config.learning_confidence_threshold and
                    historical_effectiveness["confidence"] >= 0.5):
                    enhanced_actions.append(action_str)
                    logger.debug(
                        f"Keeping action {action_str} based on historical "
                        f"effectiveness: {historical_effectiveness['success_rate']*100:.1f}% success"
                    )
                else:
                    logger.debug(
                        f"Skipping action {action_str}: low historical "
                        f"effectiveness ({historical_effectiveness['success_rate']*100:.1f}%)"
                    )
            else:
                # No historical data - use with caution
                logger.debug(f"No historical data for action {action_str}, using with caution")
                enhanced_actions.append(action_str)
        
        return enhanced_actions if enhanced_actions else default_actions
    
    def _create_mcp_request(
        self,
        action: HealingAction,
        event: ReliabilityEvent,
        historical_context: List[Any],
        v2_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create MCP request from action and context"""
        # Build justification from analysis
        justification_parts = []
        
        # Add severity
        severity = v2_result.get("severity", "unknown").upper()
        justification_parts.append(f"Severity: {severity}")
        
        # Add business impact if available
        if v2_result.get("business_impact"):
            impact = v2_result["business_impact"]
            justification_parts.append(
                f"Business Impact: ${impact.get('revenue_loss_estimate', 0):.2f} at risk"
            )
        
        # Add historical context
        if historical_context:
            similar_count = len(historical_context)
            successful_historical = sum(
                1 for inc in historical_context
                if hasattr(inc, 'outcomes') and inc.outcomes
                and any(o.success for o in inc.outcomes)
            )
            
            justification_parts.append(
                f"Historical Context: {successful_historical}/{similar_count} "
                f"similar incidents successfully resolved with this action"
            )
        
        # Add agent confidence
        if v2_result.get("multi_agent_analysis"):
            confidence = v2_result["multi_agent_analysis"].get(
                "incident_summary", {}
            ).get("anomaly_confidence", 0)
            justification_parts.append(f"Agent Confidence: {confidence*100:.1f}%")
        
        justification = "; ".join(justification_parts)
        
        return {
            "tool": action.value,
            "component": event.component,
            "parameters": {
                "latency": event.latency_p99,
                "error_rate": event.error_rate,
                "severity": severity,
                "event_timestamp": event.timestamp.isoformat(),
                "fingerprint": event.fingerprint
            },
            "justification": justification,
            "mode": config.mcp_mode,
            "metadata": {
                "historical_incidents_count": len(historical_context),
                "v2_result_summary": {
                    "status": v2_result.get("status"),
                    "business_impact": v2_result.get("business_impact", {}).get("severity_level")
                }
            }
        }
    
    async def _record_outcome(
        self,
        incident_id: str,
        action: HealingAction,
        mcp_response: Any,
        event: ReliabilityEvent,
        resolution_time_minutes: float
    ) -> Dict[str, Any]:
        """Record outcome of MCP execution"""
        success = getattr(mcp_response, 'success', False)
        result = getattr(mcp_response, 'result', {})
        
        # Extract lessons learned
        lessons_learned = []
        
        if success:
            lessons_learned.append(f"Successfully executed {action.value} on {event.component}")
            
            if result and "details" in result:
                lessons_learned.append(f"Execution details: {result['details']}")
        else:
            lessons_learned.append(f"Failed to execute {action.value} on {event.component}")
            
            error_msg = getattr(mcp_response, 'message', 'Unknown error')
            lessons_learned.append(f"Error: {error_msg}")
        
        # Add contextual lessons
        if event.latency_p99 > config.latency_critical:
            lessons_learned.append("High latency incidents may require different remediation strategies")
        
        if event.error_rate > config.error_rate_critical:
            lessons_learned.append("High error rates may indicate systemic issues")
        
        return {
            "incident_id": incident_id,
            "action": action.value,
            "success": success,
            "resolution_time_minutes": resolution_time_minutes,
            "lessons_learned": lessons_learned,
            "timestamp": time.time(),
            "mcp_response": {
                "status": getattr(mcp_response, 'status', 'unknown'),
                "message": getattr(mcp_response, 'message', '')
            }
        }
    
    def _calculate_resolution_time(self, event: ReliabilityEvent, mcp_response: Any) -> float:
        """Calculate resolution time in minutes"""
        # Simple implementation - in production, track actual time
        # between event detection and MCP response
        
        # Estimate based on action type
        action = getattr(mcp_response, 'action', 'unknown')
        
        time_estimates = {
            "restart_container": 2.5,
            "scale_out": 3.0,
            "traffic_shift": 1.5,
            "circuit_breaker": 0.5,
            "rollback": 5.0,
            "alert_team": 10.0,  # Human response time
        }
        
        return time_estimates.get(action, 2.0)
    
    def _update_learning_state(
        self,
        action: HealingAction,
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
                metrics["rag_cache_hit_rate"] = metrics["rag_cache_hits"] / metrics["rag_queries"]
            
            if metrics["mcp_calls"] > 0:
                metrics["mcp_success_rate"] = metrics["mcp_successes"] / metrics["mcp_calls"]
            
            # Add learning state
            metrics.update(self.learning_state)
            
            # Add feature status
            metrics["feature_status"] = {
                "rag_available": self.rag_graph is not None,
                "mcp_available": self.mcp_server is not None,
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
        combined_stats = {
            **base_stats,
            "v3_features": v3_stats["v3_features_active"],
            "v3_metrics": v3_stats,
            "processing_version": "v3" if v3_stats["v3_features_active"] else "v2",
            "rag_graph_stats": self.rag_graph.get_graph_stats() if self.rag_graph else None
        }
        
        return combined_stats
