"""
Enhanced Reliability Engine (v2)
Clean version without RAG/MCP integration for backward compatibility
"""

import threading
import logging
from typing import Dict, Any, Optional, List

from ..models import ReliabilityEvent, EventSeverity
from ..healing_policies import PolicyEngine
from ..config import config

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
    v2 Reliability Engine - No RAG/MCP integration
    
    This is the original v2 engine for backward compatibility
    when v3 features are disabled
    """
    
    def __init__(
        self,
        orchestrator=None,  # Will be injected
        policy_engine: Optional[PolicyEngine] = None,
        event_store: Optional[ThreadSafeEventStore] = None,
        anomaly_detector=None,  # Will be injected
        business_calculator=None,  # Will be injected
    ):
        """
        Initialize v2 reliability engine
        
        Args:
            orchestrator: OrchestrationManager instance
            policy_engine: PolicyEngine instance
            event_store: ThreadSafeEventStore instance
            anomaly_detector: AnomalyDetector instance
            business_calculator: BusinessImpactCalculator instance
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
        
        # Performance metrics
        self.performance_metrics = {
            'total_incidents_processed': 0,
            'multi_agent_analyses': 0,
            'anomalies_detected': 0,
        }
        self._lock = threading.RLock()
        
        logger.info("Initialized EnhancedReliabilityEngine (v2)")
    
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
        Process a reliability event (v2 implementation)
        
        No RAG retrieval, no MCP execution, no learning loop
        """
        logger.info(
            f"Processing event for {component}: latency={latency}ms, "
            f"error_rate={error_rate*100:.1f}%"
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
        
        # Original policy evaluation (no RAG enhancement)
        healing_actions = self.policy_engine.evaluate_policies(event)
        
        # Calculate business impact
        business_impact = None
        if is_anomaly:
            business_impact = self.business_calculator.calculate_impact(event)
        
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
            "processing_metadata": {
                "agents_used": agent_analysis.get('agent_metadata', {}).get('participating_agents', []),
                "analysis_confidence": agent_analysis.get('incident_summary', {}).get('anomaly_confidence', 0),
                "processing_version": "v2"
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
            f"Event processed: {result['status']} with {result['severity']} severity"
        )
        
        return result
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        with self._lock:
            stats: Dict[str, Any] = self.performance_metrics.copy()
            stats.update({
                'events_in_store': self.event_store.count(),
                'rag_available': False,
                'rag_active': False,
                'mcp_available': False,
                'mcp_active': False,
                'processing_version': 'v2'
            })
            return stats
