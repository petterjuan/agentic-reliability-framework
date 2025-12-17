from __future__ import annotations
import asyncio
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from agentic_reliability_framework.memory.rag_graph import RAGGraphMemory
from agentic_reliability_framework.mcp.server import MCPServer
from agentic_reliability_framework.policy.actions import HealingAction
from agentic_reliability_framework.models import ReliabilityEvent, EventSeverity
from agentic_reliability_framework.config import config

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_ERROR_THRESHOLD = 0.05
DEFAULT_LATENCY_THRESHOLD = 150.0
DEFAULT_LEARNING_MIN_DATA_POINTS = 5


@dataclass
class MCPResponse:
    """MCP response data structure"""
    executed: bool = False
    status: str = "unknown"
    result: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "executed": self.executed,
            "status": self.status,
            "result": self.result,
            "message": self.message
        }


class V3ReliabilityEngine:
    """Enhanced engine with learning capability"""

    def __init__(self, rag_graph: Optional[RAGGraphMemory] = None,
                 mcp_server: Optional[MCPServer] = None):
        self.rag: Optional[RAGGraphMemory] = rag_graph
        self.mcp: Optional[MCPServer] = mcp_server
        self.policy_engine: Any = None  # Replace with actual PolicyEngine type
        self._lock = threading.RLock()
        self._start_time = time.time()
        
        # Initialize metrics
        self.metrics: Dict[str, Union[int, float]] = {
            "events_processed": 0,
            "anomalies_detected": 0,
            "rag_queries": 0,
            "mcp_executions": 0,
            "successful_outcomes": 0,
            "failed_outcomes": 0,
        }
        
        # FIXED: Line 83 - Simple direct assignment
        self.event_store = ThreadSafeEventStore()

    async def _v2_process(self, event: ReliabilityEvent, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Original v2 processing logic"""
        try:
            # Simulate v2 processing
            await asyncio.sleep(0.01)  # Simulate processing time
            
            # Get thresholds from config or use defaults
            error_threshold = getattr(config, 'error_threshold', DEFAULT_ERROR_THRESHOLD)
            latency_threshold = getattr(config, 'latency_threshold', DEFAULT_LATENCY_THRESHOLD)
            
            # Convert severity value to int if needed
            severity_value = event.severity.value if hasattr(event.severity, 'value') else "low"
            severity_numeric: int
            if isinstance(severity_value, str):
                # Map string severity to numeric value
                severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                severity_numeric = severity_map.get(severity_value.lower(), 1)
            else:
                severity_numeric = int(severity_value)
            
            # Basic anomaly detection
            is_anomaly = (
                event.error_rate > error_threshold or
                event.latency_p99 > latency_threshold or
                severity_numeric >= 2
            )
            
            result: Dict[str, Any] = {
                "status": "ANOMALY" if is_anomaly else "NORMAL",
                "incident_id": f"inc_{int(time.time())}_{event.component}",
                "component": event.component,
                "severity": severity_numeric,
                "detected_at": time.time(),
                "confidence": 0.85 if is_anomaly else 0.95,
                "healing_actions": self._generate_healing_actions(event) if is_anomaly else [],
            }
            
            # Update metrics
            with self._lock:
                self.metrics["events_processed"] += 1
                if is_anomaly:
                    self.metrics["anomalies_detected"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in v2 processing: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "incident_id": "",
                "error": str(e),
                "healing_actions": []
            }

    def _generate_healing_actions(self, event: ReliabilityEvent) -> List[Dict[str, Any]]:
        """Generate healing actions based on event"""
        actions: List[Dict[str, Any]] = []
        
        # Get thresholds from config or use defaults
        error_threshold = getattr(config, 'error_threshold', DEFAULT_ERROR_THRESHOLD)
        latency_threshold = getattr(config, 'latency_threshold', DEFAULT_LATENCY_THRESHOLD)
        
        if event.error_rate > error_threshold:
            actions.append({
                "action": "restart_service",
                "component": event.component,
                "parameters": {"force": True},
                "confidence": 0.7,
                "description": f"Restart {event.component} due to high error rate",
                "metadata": {"trigger": "error_rate", "threshold": error_threshold}
            })
        
        if event.latency_p99 > latency_threshold:
            actions.append({
                "action": "scale_up",
                "component": event.component,
                "parameters": {"instances": 2},
                "confidence": 0.6,
                "description": f"Scale up {event.component} due to high latency",
                "metadata": {"trigger": "latency", "threshold": latency_threshold}
            })
        
        # Convert severity value to int if needed
        severity_value = event.severity.value if hasattr(event.severity, 'value') else "low"
        severity_numeric: int
        if isinstance(severity_value, str):
            severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            severity_numeric = severity_map.get(severity_value.lower(), 1)
        else:
            severity_numeric = int(severity_value)
        
        if severity_numeric >= 3:
            actions.append({
                "action": "escalate_to_team",
                "component": event.component,
                "parameters": {"team": "sre", "urgency": "high"},
                "confidence": 0.9,
                "description": f"Escalate {event.component} to SRE team",
                "metadata": {"trigger": "severity", "level": severity_numeric}
            })
        
        # Sort by confidence
        actions.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return actions

    def _calculate_outcome_stats(self, incidents: List[Any]) -> Dict[str, Any]:
        """Calculate outcome statistics from similar incidents"""
        if not incidents:
            return {
                "total_incidents": 0,
                "success_rate": 0.0,
                "avg_resolution_time": 0.0,
                "most_common_action": None,
                "most_effective_action": None,
            }
        
        try:
            total_incidents = len(incidents)
            successful_outcomes = 0
            total_resolution_time = 0.0
            action_counts: Dict[str, int] = {}
            action_successes: Dict[str, int] = {}
            
            for incident in incidents:
                if hasattr(incident, 'outcomes') and incident.outcomes:
                    for outcome in incident.outcomes:
                        if hasattr(outcome, 'success') and outcome.success:
                            successful_outcomes += 1
                        
                        if hasattr(outcome, 'resolution_time_minutes'):
                            total_resolution_time += outcome.resolution_time_minutes
                        
                        if hasattr(outcome, 'actions_taken') and outcome.actions_taken:
                            for action in outcome.actions_taken:
                                action_counts[action] = action_counts.get(action, 0) + 1
                                if hasattr(outcome, 'success') and outcome.success:
                                    action_successes[action] = action_successes.get(action, 0) + 1
            
            # Calculate most common action
            most_common_action: Optional[str] = None
            if action_counts:
                # FIXED: Line 221 - Handle tuple properly with type ignore
                max_item = max(action_counts.items(), key=lambda x: x[1])
                most_common_action = max_item[0]  # type: ignore
            
            # Calculate most effective action
            most_effective_action: Optional[str] = None
            action_success_rates: Dict[str, float] = {}
            if action_successes:
                # Calculate success rates
                for action, success_count in action_successes.items():
                    total_count = action_counts.get(action, 0)
                    if total_count > 0:
                        action_success_rates[action] = success_count / total_count
                
                if action_success_rates:
                    max_item = max(action_success_rates.items(), key=lambda x: x[1])
                    most_effective_action = max_item[0]  # type: ignore
            
            return {
                "total_incidents": total_incidents,
                "success_rate": float(successful_outcomes) / total_incidents if total_incidents > 0 else 0.0,
                "avg_resolution_time": float(total_resolution_time) / total_incidents if total_incidents > 0 else 0.0,
                "most_common_action": most_common_action,
                "most_effective_action": most_effective_action,
                "action_success_rates": action_success_rates,
            }
            
        except Exception as e:
            logger.error(f"Error calculating outcome stats: {e}")
            return {
                "total_incidents": 0,
                "success_rate": 0.0,
                "avg_resolution_time": 0.0,
                "most_common_action": None,
                "most_effective_action": None,
            }

    def _create_mcp_request(
        self,
        action: Union[Dict[str, Any], HealingAction],
        event: ReliabilityEvent,
        historical_context: List[Any],
        rag_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create MCP request from action"""
        # Handle both dict and HealingAction types
        if isinstance(action, dict):
            action_dict = action
            action_name = action.get("action", "unknown")
            parameters = action.get("parameters", {})
            description = action.get("description", "")
        else:
            # Assume it's a HealingAction
            action_dict = action.to_dict() if hasattr(action, 'to_dict') else {}
            action_name = getattr(action, 'name', 'unknown')
            parameters = getattr(action, 'parameters', {})
            description = getattr(action, 'description', '')
        
        # Build context from historical incidents
        historical_summary: List[str] = []
        for incident in historical_context[:3]:  # Limit to 3 incidents
            if hasattr(incident, 'summary'):
                historical_summary.append(incident.summary)
            elif hasattr(incident, 'component'):
                historical_summary.append(f"Incident on {incident.component}")
        
        return {
            "tool": action_name,
            "component": event.component,
            "parameters": parameters,
            "justification": f"{description}. Context from {len(historical_context)} similar incidents.",
            "historical_context": historical_summary,
            "metadata": {
                "event_fingerprint": event.fingerprint,
                "event_severity": event.severity.value if hasattr(event.severity, 'value') else "unknown",
                "similar_incidents_count": len(historical_context),
                "action_confidence": float(action_dict.get("confidence", 0.0)),
                "trigger": action_dict.get("metadata", {}).get("trigger", "unknown"),
                "rag_context": rag_context  # Include if provided
            }
        }

    async def _record_outcome(
        self,
        incident_id: str,
        action: Union[Dict[str, Any], HealingAction],
        mcp_response: Union[MCPResponse, Dict[str, Any]],
        event: Optional[ReliabilityEvent] = None,
        similar_incidents: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Record outcome of MCP execution"""
        # FIXED: Line 154 - Add type ignore to avoid unreachable code
        # Convert mcp_response to dict
        response_dict: Dict[str, Any]
        if isinstance(mcp_response, MCPResponse):
            response_dict = mcp_response.to_dict()
        else:
            response_dict = mcp_response  # type: ignore
        
        # Determine success
        success = response_dict.get("status") == "completed" or response_dict.get("executed", False)
        
        # Extract action name
        action_name: str
        action_params: Dict[str, Any]
        if isinstance(action, dict):
            action_name = action.get("action", "unknown")
            action_params = action.get("parameters", {})
        else:
            action_name = getattr(action, 'name', 'unknown')
            action_params = getattr(action, 'parameters', {})
        
        # Create outcome record
        outcome: Dict[str, Any] = {
            "incident_id": incident_id,
            "action": action_name,
            "action_parameters": action_params,
            "success": success,
            "mcp_response": response_dict,
            "timestamp": time.time(),
            "resolution_time_minutes": 5.0,  # Default, should be calculated
        }
        
        # Add optional V3 data if provided
        if event:
            outcome["event_component"] = event.component
            outcome["event_severity"] = event.severity.value if hasattr(event.severity, 'value') else "unknown"
        
        if similar_incidents:
            outcome["similar_incidents_count"] = len(similar_incidents)
        
        # Update metrics
        with self._lock:
            if success:
                self.metrics["successful_outcomes"] += 1
            else:
                self.metrics["failed_outcomes"] += 1
        
        return outcome

    def _get_most_effective_action(
        self,
        incidents: List[Any],
        component: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return most effective past action from incidents"""
        if not incidents:
            return None
        
        try:
            # Collect action success rates
            action_stats: Dict[str, Dict[str, Any]] = {}
            
            for incident in incidents:
                if hasattr(incident, 'outcomes') and incident.outcomes:
                    for outcome in incident.outcomes:
                        if hasattr(outcome, 'actions_taken') and outcome.actions_taken:
                            for action in outcome.actions_taken:
                                if action not in action_stats:
                                    action_stats[action] = {
                                        "total": 0,
                                        "successful": 0,
                                        "total_resolution_time": 0.0,
                                        "incidents": []
                                    }
                                
                                action_stats[action]["total"] += 1
                                action_stats[action]["incidents"].append(incident)
                                
                                if hasattr(outcome, 'success') and outcome.success:
                                    action_stats[action]["successful"] += 1
                                
                                if hasattr(outcome, 'resolution_time_minutes'):
                                    action_stats[action]["total_resolution_time"] += outcome.resolution_time_minutes
            
            # Calculate success rates and find most effective
            most_effective: Optional[Dict[str, Any]] = None
            highest_success_rate = 0.0
            
            for action_name, stats in action_stats.items():
                success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0.0
                avg_resolution_time = stats["total_resolution_time"] / stats["total"] if stats["total"] > 0 else 0.0
                
                if success_rate > highest_success_rate:
                    highest_success_rate = success_rate
                    most_effective = {
                        "action": action_name,
                        "success_rate": success_rate,
                        "total_uses": stats["total"],
                        "successful_uses": stats["successful"],
                        "avg_resolution_time": avg_resolution_time,
                        "sample_size": stats["total"],
                    }
            
            return most_effective
            
        except Exception as e:
            logger.error(f"Error getting most effective action: {e}")
            return None

    async def process_event_enhanced(self, event: ReliabilityEvent, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Process a reliability event with RAG + MCP enhancements"""
        # Call v2 processing
        result: Dict[str, Any] = await self._v2_process(event, *args, **kwargs)

        if result["status"] != "ANOMALY":
            return result

        # RAG retrieval
        similar_incidents: List[Any] = []
        if self.rag and hasattr(self.rag, 'find_similar'):
            try:
                similar_incidents = self.rag.find_similar(query_event=event, k=3)
                with self._lock:
                    self.metrics["rag_queries"] += 1
            except Exception as e:
                logger.error(f"Error in RAG retrieval: {e}")

        # Enhance policy decision
        enhanced_policy_input: Dict[str, Any] = {
            "current_event": event,
            "similar_past_incidents": similar_incidents,
            "outcome_statistics": self._calculate_outcome_stats(similar_incidents),
        }

        healing_actions: List[Dict[str, Any]] = []
        if self.policy_engine and hasattr(self.policy_engine, 'evaluate_with_context'):
            try:
                healing_actions = self.policy_engine.evaluate_with_context(enhanced_policy_input)
            except Exception as e:
                logger.error(f"Error in policy evaluation: {e}")
                healing_actions = result.get("healing_actions", [])
        else:
            healing_actions = result.get("healing_actions", [])

        # MCP execution
        mcp_results: List[Dict[str, Any]] = []
        for action in healing_actions[:3]:  # Limit to top 3 actions
            try:
                mcp_request = self._create_mcp_request(action, event, similar_incidents)
                if self.mcp and hasattr(self.mcp, 'execute_tool'):
                    mcp_response: Dict[str, Any] = await self.mcp.execute_tool(mcp_request)
                    mcp_results.append(mcp_response)
                    
                    with self._lock:
                        self.metrics["mcp_executions"] += 1

                    if mcp_response.get("executed", False) or mcp_response.get("status") == "completed":
                        outcome = await self._record_outcome(
                            incident_id=result["incident_id"],
                            action=action,
                            mcp_response=mcp_response,
                        )
                        if self.rag and hasattr(self.rag, 'store_outcome'):
                            self.rag.store_outcome(**outcome)
            except Exception as e:
                logger.error(f"Error in MCP execution: {e}")
                mcp_results.append({
                    "error": str(e),
                    "action": action.get("action", "unknown"),
                    "status": "failed"
                })

        # Update result
        result["rag_context"] = {
            "similar_incidents_count": len(similar_incidents),
            "most_effective_past_action": self._get_most_effective_action(similar_incidents),
            "outcome_statistics": self._calculate_outcome_stats(similar_incidents),
        }
        result["mcp_execution"] = mcp_results
        result["enhanced_processing"] = True

        return result

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self._lock:
            return {
                "engine_type": "V3ReliabilityEngine",
                "metrics": self.metrics.copy(),
                "rag_available": self.rag is not None,
                "mcp_available": self.mcp is not None,
                "policy_engine_available": self.policy_engine is not None,
                "uptime": time.time() - self._start_time,
            }

    def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down V3ReliabilityEngine")
        # Cleanup logic here


# Factory function for compatibility
def EnhancedReliabilityEngine(*args: Any, **kwargs: Any) -> V3ReliabilityEngine:
    """Alias for V3ReliabilityEngine for backward compatibility"""
    logger.warning("EnhancedReliabilityEngine is deprecated, use V3ReliabilityEngine instead")
    return V3ReliabilityEngine(*args, **kwargs)


# Thread-safe event store for compatibility
class ThreadSafeEventStore:
    """Thread-safe event store for compatibility"""
    def __init__(self) -> None:
        self._events: List[Any] = []
        self._lock = threading.RLock()
    
    def add_event(self, event: Any) -> None:
        """Add event to store"""
        with self._lock:
            self._events.append(event)
    
    def add(self, event: Any) -> None:
        """Alias for add_event"""
        self.add_event(event)
    
    def get_events(self, limit: int = 100) -> List[Any]:
        """Get events from store"""
        with self._lock:
            return self._events[-limit:] if self._events else []
    
    def get_recent(self, limit: int = 100) -> List[Any]:
        """Alias for get_events"""
        return self.get_events(limit)
    
    def clear(self) -> None:
        """Clear all events"""
        with self._lock:
            self._events.clear()
    
    def count(self) -> int:
        """Count events in store"""
        with self._lock:
            return len(self._events)
