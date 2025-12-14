"""
Enhanced Reliability Engine with RAG Graph integration
Extracted from app.py for modularity
"""

import threading
import logging
import asyncio # noqa: F401
from typing import Dict, Any, Optional, List
# Removed: from datetime import datetime  # Unused - FIXED

from ..models import ReliabilityEvent, EventSeverity, HealingAction
from ..healing_policies import PolicyEngine
from ..config import config
from ..memory.rag_graph import RAGGraphMemory

logger = logging.getLogger(__name__)


class ThreadSafeEventStore:
    """Thread-safe storage for reliability events"""
    
    def __init__(self, max_size: int = 1000):
        self._events: List[ReliabilityEvent] = []
        self._lock = threading.RLock()
        self.max_size = max_size
        logger.info(f"Initialized ThreadSafeEventStore with max_size={max_size}")
    
    def add(self, event: ReliabilityEvent) -> None:
        """Add event to store"""
        with self._lock:
            self._events.append(event)
            # Enforce max size
            if len(self._events) > self.max_size:
                self._events = self._events[-self.max_size:]
            logger.debug(f"Added event for {event.component}: {event.severity.value}")
    
    def get_recent(self, n: int = 15) -> List[ReliabilityEvent]:
        """Get n most recent events"""
        with self._lock:
            return self._events[-n:] if self._events else []
    
    def get_all(self) -> List[ReliabilityEvent]:
        """Get all events"""
        with self._lock:
            return self._events.copy()
    
    def count(self) -> int:
        """Get total event count"""
        with self._lock:
            return len(self._events)


class EnhancedReliabilityEngine:
    """
    Main engine for processing reliability events with RAG integration
    
    Updated for v3: Integrates RAG Graph for historical context
    """
    
    def __init__(
        self,
        orchestrator=None,  # Will be injected
        policy_engine: Optional[PolicyEngine] = None,
        event_store: Optional[ThreadSafeEventStore] = None,
        anomaly_detector=None,  # Will be injected
        business_calculator=None,  # Will be injected
        rag_graph: Optional[RAGGraphMemory] = None
    ):
        """
        Initialize reliability engine with RAG support
        
        Args:
            rag_graph: Optional RAGGraphMemory instance for v3 features
        """
        # Import here to avoid circular dependencies
        from ..agents.orchestrator import OrchestrationManager
        from .anomaly import AdvancedAnomalyDetector
        from .business import BusinessImpactCalculator
        
        self.orchestrator = orchestrator or OrchestrationManager()
        self.policy_engine = policy_engine or PolicyEngine()
        self.event_store = event_store or ThreadSafeEventStore(max_size=config.max_events_stored)
        self.anomaly_detector = anomaly_detector or AdvancedAnomalyDetector()
        self.business_calculator = business_calculator or BusinessImpactCalculator()
        self.rag_graph = rag_graph
        
        # Performance metrics
        self.performance_metrics = {
            'total_incidents_processed': 0,
            'multi_agent_analyses': 0,
            'anomalies_detected': 0,
            'rag_queries': 0,
            'rag_cache_hits': 0,
            'rag_enabled': rag_graph is not None and config.rag_enabled
        }
        self._lock = threading.RLock()
        
        logger.info(
            f"Initialized EnhancedReliabilityEngine with RAG enabled={rag_graph is not None and config.rag_enabled}"
        )
    
    async def process_event_with_rag(
        self,
        component: str,
        latency: float,
        error_rate: float,
        throughput: float = 1000,
        cpu_util: Optional[float] = None,
        memory_util: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a reliability event with RAG-enhanced analysis
        
        Updated for v3: Uses RAG Graph for historical context if enabled
        """
        logger.info(
            f"Processing event for {component}: latency={latency}ms, "
            f"error_rate={error_rate*100:.1f}% (RAG enabled: {self.rag_graph is not None and config.rag_enabled})"
        )
        
        # Validate component ID
        if not component or not isinstance(component, str):
            return {'error': 'Invalid component ID', 'status': 'INVALID'}
        
        if len(component) > 255 or len(component) < 1:
            return {'error': 'Component ID must be 1-255 characters', 'status': 'INVALID'}
        
        # Create event
        try:
            event = ReliabilityEvent(
                component=component,
                latency_p99=latency,
                error_rate=error_rate,
                throughput=throughput,
                cpu_util=cpu_util,
                memory_util=memory_util,
                upstream_deps=["auth-service", "database"] if component == "api-service" else []
            )
        except Exception as e:
            logger.error(f"Event creation error: {e}", exc_info=True)
            return {'error': f'Invalid event data: {str(e)}', 'status': 'INVALID'}
        
        # === v3 ENHANCEMENT: RAG RETRIEVAL ===
        historical_context = None
        rag_insights = None
        
        if self.rag_graph and config.rag_enabled:
            try:
                # Find similar historical incidents
                similar_incidents = self.rag_graph.find_similar(event, k=3)
                
                historical_context = {
                    "similar_incidents_count": len(similar_incidents),
                    "similar_incidents": [
                        {
                            "component": incident.component,
                            "severity": incident.severity,
                            "timestamp": incident.timestamp,
                            "metrics": incident.metrics,
                            "has_outcomes": len(getattr(incident, 'outcomes', [])) > 0
                        }
                        for incident in similar_incidents
                    ]
                }
                
                # Generate RAG insights
                rag_insights = self._generate_rag_insights(similar_incidents)
                
                with self._lock:
                    self.performance_metrics['rag_queries'] += 1
                
                logger.info(f"RAG found {len(similar_incidents)} similar incidents for {component}")
                
                # Store current incident in RAG (after analysis)
                # We'll do this later when we have the agent analysis
                
            except Exception as e:
                logger.error(f"RAG retrieval error: {e}", exc_info=True)
                historical_context = {"error": str(e)}
                rag_insights = {"error": "RAG retrieval failed"}
        
        # Multi-agent analysis
        agent_analysis = await self.orchestrator.orchestrate_analysis(event)
        
        # Anomaly detection
        is_anomaly = self.anomaly_detector.detect_anomaly(event)
        
        # Determine severity based on agent confidence
        agent_confidence = 0.0
        if agent_analysis and 'incident_summary' in agent_analysis:
            agent_confidence = agent_analysis.get('incident_summary', {}).get('anomaly_confidence', 0)
        else:
            agent_confidence = 0.8 if is_anomaly else 0.1
        
        # Set event severity
        if agent_confidence > 0.8:
            severity = EventSeverity.CRITICAL
        elif agent_confidence > 0.6:
            severity = EventSeverity.HIGH
        elif agent_confidence > 0.4:
            severity = EventSeverity.MEDIUM
        else:
            severity = EventSeverity.LOW
        
        # Create mutable copy with updated severity
        event = event.model_copy(update={'severity': severity})
        
        # === v3 ENHANCEMENT: RAG-ENHANCED POLICY EVALUATION ===
        healing_actions = []
        if self.rag_graph and config.rag_enabled and rag_insights:
            # Use RAG insights to enhance policy evaluation
            healing_actions = self._evaluate_policies_with_rag(
                event, 
                rag_insights.get("historical_effectiveness", {})
            )
        else:
            # Original policy evaluation
            healing_actions = self.policy_engine.evaluate_policies(event)
        
        # Calculate business impact
        business_impact = None
        if is_anomaly:
            business_impact = self.business_calculator.calculate_impact(event)
        
        # === v3 ENHANCEMENT: STORE IN RAG GRAPH ===
        if self.rag_graph and config.rag_enabled and is_anomaly:
            try:
                # Store incident in RAG graph
                incident_id = self.rag_graph.store_incident(event, agent_analysis)
                
                logger.info(f"Stored incident {incident_id} in RAG graph")
                
            except Exception as e:
                logger.error(f"Error storing incident in RAG: {e}", exc_info=True)
        
        # Build comprehensive result
        result = {
            "timestamp": event.timestamp.isoformat(),
            "component": component,
            "latency_p99": latency,
            "error_rate": error_rate,
            "throughput": throughput,
            "status": "ANOMALY" if is_anomaly else "NORMAL",
            "multi_agent_analysis": agent_analysis,
            "healing_actions": [action.value for action in healing_actions],
            "business_impact": business_impact,
            "severity": event.severity.value,
            "similar_incidents_count": self.rag_graph.get_graph_stats()["incident_nodes"] if self.rag_graph and config.rag_enabled else 0,
            "processing_metadata": {
                "agents_used": agent_analysis.get('agent_metadata', {}).get('participating_agents', []),
                "analysis_confidence": agent_analysis.get('incident_summary', {}).get('anomaly_confidence', 0),
                "rag_enabled": self.rag_graph is not None and config.rag_enabled,
                "rag_insights": rag_insights,
                "historical_context": historical_context
            }
        }
        
        # Store event in history
        self.event_store.add(event)
        
        # Update performance metrics
        with self._lock:
            self.performance_metrics['total_incidents_processed'] += 1
            self.performance_metrics['multi_agent_analyses'] += 1
            if is_anomaly:
                self.performance_metrics['anomalies_detected'] += 1
        
        logger.info(
            f"Event processed: {result['status']} with {result['severity']} severity, "
            f"RAG insights: {rag_insights is not None}"
        )
        
        return result
    
    def _generate_rag_insights(self, similar_incidents: List) -> Dict[str, Any]:
        """
        Generate insights from similar historical incidents
        
        Args:
            similar_incidents: List of similar IncidentNodes
            
        Returns:
            Dictionary with RAG insights
        """
        if not similar_incidents:
            return {"message": "No similar historical incidents found"}
        
        insights = {
            "historical_effectiveness": {},
            "common_patterns": [],
            "recommendations": []
        }
        
        # Analyze historical effectiveness of actions
        if self.rag_graph:
            # Get all unique actions from outcomes
            all_actions = set()
            for incident in similar_incidents:
                if hasattr(incident, 'outcomes'):
                    for outcome in incident.outcomes:
                        all_actions.update(outcome.actions_taken)
            
            # Calculate effectiveness for each action
            for action in all_actions:
                effectiveness = self.rag_graph.get_historical_effectiveness(action)
                insights["historical_effectiveness"][action] = effectiveness
        
        # Identify common patterns
        severity_counts = {}
        component_counts = {}
        
        for incident in similar_incidents:
            severity = incident.severity
            component = incident.component
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            component_counts[component] = component_counts.get(component, 0) + 1
        
        if severity_counts:
            most_common_severity = max(severity_counts.items(), key=lambda x: x[1])
            insights["common_patterns"].append(
                f"Most common severity: {most_common_severity[0]} ({most_common_severity[1]} incidents)"
            )
        
        # Generate recommendations based on historical patterns
        if insights["historical_effectiveness"]:
            # Find most effective action
            effective_actions = [
                (action, data["success_rate"]) 
                for action, data in insights["historical_effectiveness"].items() 
                if data["total_uses"] > 0
            ]
            
            if effective_actions:
                most_effective = max(effective_actions, key=lambda x: x[1])
                insights["recommendations"].append(
                    f"Historically effective: {most_effective[0]} "
                    f"({most_effective[1]*100:.1f}% success rate)"
                )
        
        return insights
    
    def _evaluate_policies_with_rag(
        self, 
        event: ReliabilityEvent, 
        historical_effectiveness: Dict[str, Any]
    ) -> List[HealingAction]:
        """
        Evaluate policies with RAG-enhanced context
        
        Args:
            event: Current reliability event
            historical_effectiveness: Historical effectiveness data from RAG
            
        Returns:
            List of healing actions
        """
        # First, get standard policy evaluation
        standard_actions = self.policy_engine.evaluate_policies(event)
        
        if not historical_effectiveness:
            return standard_actions
        
        # Enhance actions based on historical effectiveness
        enhanced_actions = []
        
        for action in standard_actions:
            action_str = action.value
            
            # Check historical effectiveness
            if action_str in historical_effectiveness:
                effectiveness = historical_effectiveness[action_str]
                
                # If action has good historical success rate, keep it
                if effectiveness["success_rate"] >= 0.7:
                    enhanced_actions.append(action)
                    logger.debug(
                        f"Keeping action {action_str} based on historical "
                        f"success rate: {effectiveness['success_rate']*100:.1f}%"
                    )
                else:
                    # Consider replacing with more effective alternative
                    logger.debug(
                        f"Action {action_str} has low historical success rate: "
                        f"{effectiveness['success_rate']*100:.1f}%"
                    )
                    
                    # For now, we keep it but could implement smarter logic
                    enhanced_actions.append(action)
            else:
                # No historical data, use standard evaluation
                enhanced_actions.append(action)
        
        return enhanced_actions
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        with self._lock:
            stats = self.performance_metrics.copy()
            stats.update({
                'events_in_store': self.event_store.count(),
                'rag_available': self.rag_graph is not None,
                'rag_active': self.rag_graph is not None and config.rag_enabled
            })
            
            if self.rag_graph and config.rag_enabled:
                rag_stats = self.rag_graph.get_graph_stats()
                stats['rag_graph_stats'] = rag_stats
            
            return stats
    
    # Backward compatibility alias
    async def process_event_enhanced(self, *args, **kwargs):
        """Alias for backward compatibility"""
        return await self.process_event_with_rag(*args, **kwargs)
