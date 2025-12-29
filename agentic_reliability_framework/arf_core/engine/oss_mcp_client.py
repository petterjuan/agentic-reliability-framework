"""
OSS MCP Client - Advisory Mode Only
Apache 2.0 Licensed - Enterprise execution requires commercial license

This is the OSS replacement for MCPServer that provides:
- Advisory mode only (no execution capability)
- HealingIntent creation for OSS→Enterprise handoff
- RAG graph integration for similarity-based recommendations
- Comprehensive safety checks and validation
- Clear upgrade path to Enterprise edition
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, Any, Optional, List, Union, Tuple, cast
from dataclasses import dataclass, field
from datetime import datetime

# FIXED: Use relative imports since we're inside arf_core
from ..models import ReliabilityEvent, EventSeverity, create_compatible_event

from ..constants import (
    OSS_EDITION,
    OSS_LICENSE,
    ENTERPRISE_UPGRADE_URL,
    MCP_MODES_ALLOWED,
    EXECUTION_ALLOWED,
    MAX_INCIDENT_NODES,
    MAX_OUTCOME_NODES,
    OSSBoundaryError,
    get_oss_capabilities,
    check_oss_compliance,
)

from ..models.healing_intent import (
    HealingIntent,
    HealingIntentSerializer,
    IntentSource,
    IntentStatus,
    create_rollback_intent,
    create_restart_intent,
    create_scale_out_intent,
    create_oss_advisory_intent,
)

from ..config.oss_config import oss_config

logger = logging.getLogger(__name__)


@dataclass
class OSSAnalysisResult:
    """Result of OSS advisory analysis"""
    healing_intent: HealingIntent
    confidence: float
    similar_incidents_count: int = 0
    rag_similarity_score: Optional[float] = None
    analysis_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    requires_enterprise: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "healing_intent": self.healing_intent.to_enterprise_request(),
            "confidence": self.confidence,
            "similar_incidents_count": self.similar_incidents_count,
            "rag_similarity_score": self.rag_similarity_score,
            "analysis_time_ms": self.analysis_time_ms,
            "warnings": self.warnings,
            "requires_enterprise": self.requires_enterprise,
            "is_oss_advisory": True,
            "execution_allowed": False,
            "upgrade_url": ENTERPRISE_UPGRADE_URL,
        }


@dataclass
class OSSMCPResponse:
    """OSS MCP response format (compatible with Enterprise MCPResponse)"""
    request_id: str
    status: str  # "completed", "rejected", "error"
    message: str
    executed: bool = False
    result: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Ensure OSS restrictions"""
        if self.executed:
            raise ValueError("OSS MCP responses cannot have executed=True")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "request_id": self.request_id,
            "status": self.status,
            "message": self.message,
            "executed": self.executed,
            "result": self.result,
            "timestamp": self.timestamp,
            "oss_edition": OSS_EDITION,
            "execution_allowed": False,
        }
    
    @classmethod
    def from_healing_intent(cls, intent: HealingIntent, request_id: str) -> "OSSMCPResponse":
        """Create response from HealingIntent"""
        return cls(
            request_id=request_id,
            status="completed",
            message=f"OSS Advisory: Recommended {intent.action} for {intent.component}",
            executed=False,
            result={
                "mode": "advisory",
                "would_execute": True,
                "confidence": intent.confidence,
                "healing_intent": intent.to_enterprise_request(),
                "requires_enterprise": True,
                "upgrade_url": ENTERPRISE_UPGRADE_URL,
                "enterprise_features": get_oss_capabilities()["enterprise_features"],
                "oss_analysis": {
                    "similar_incidents_count": len(intent.similar_incidents) if intent.similar_incidents else 0,
                    "rag_similarity_score": intent.rag_similarity_score,
                    "source": intent.source.value,
                    "is_oss_advisory": intent.is_oss_advisory,
                }
            }
        )
    
    @classmethod
    def error_response(cls, request_id: str, message: str) -> "OSSMCPResponse":
        """Create error response"""
        return cls(
            request_id=request_id,
            status="error",
            message=message,
            executed=False,
            result={
                "requires_enterprise": False,
                "error_type": "oss_validation_error",
                "oss_edition": OSS_EDITION,
            }
        )


class OSSMCPClient:
    """
    OSS MCP Client - Advisory Mode Only
    
    Key Features:
    1. Advisory-only analysis (no execution capability)
    2. HealingIntent creation for Enterprise handoff
    3. RAG graph integration for similarity-based recommendations
    4. Comprehensive safety validation
    5. Clear Enterprise upgrade path
    
    This replaces MCPServer.execute_tool() in OSS edition with advisory-only functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OSS MCP Client
        
        Args:
            config: Optional configuration override (will be validated against OSS limits)
        """
        # OSS EDITION: Always advisory mode
        self.mode = "advisory"
        self.oss_edition = OSS_EDITION
        self.oss_license = OSS_LICENSE
        
        # Configuration
        self._config = config or {}
        self._validate_oss_config()
        
        # Initialize tools (advisory-only)
        self.registered_tools = self._register_oss_tools()
        
        # Performance metrics
        self.metrics = {
            "requests_processed": 0,
            "healing_intents_created": 0,
            "rag_queries_performed": 0,
            "avg_analysis_time_ms": 0.0,
            "safety_checks_passed": 0,
            "safety_checks_failed": 0,
        }
        
        # Cache for similar incidents (in-memory only, OSS limit)
        self.similarity_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.max_cache_size = 100  # OSS limit
        
        # Safety guardrails from config
        self.safety_guardrails = oss_config.safety_guardrails
        
        logger.info(f"Initialized OSSMCPClient in {self.mode} mode")
        logger.info(f"OSS Edition: {OSS_EDITION} ({OSS_LICENSE})")
        logger.info(f"Enterprise Upgrade: {ENTERPRISE_UPGRADE_URL}")
        
        # Warn about OSS limitations
        self._warn_oss_limitations()
    
    def _validate_oss_config(self) -> None:
        """Validate configuration against OSS boundaries"""
        # Check OSS compliance
        if not check_oss_compliance():
            raise OSSBoundaryError(
                f"Environment not OSS compliant. "
                f"Check environment variables and dependencies."
            )
        
        # Ensure mode is advisory
        if self._config.get("mcp_mode", "advisory") != "advisory":
            logger.warning(
                f"Configuration requests non-advisory mode, forcing 'advisory'. "
                f"Enterprise required for other modes: {ENTERPRISE_UPGRADE_URL}"
            )
            self._config["mcp_mode"] = "advisory"
        
        # Ensure no execution capability
        if self._config.get("execution_allowed", False):
            logger.warning(
                f"Configuration requests execution capability, disabling. "
                f"Enterprise required for execution: {ENTERPRISE_UPGRADE_URL}"
            )
            self._config["execution_allowed"] = False
    
    def _warn_oss_limitations(self) -> None:
        """Log OSS edition limitations"""
        logger.warning(
            f"⚠️  OSS EDITION LIMITATIONS:\n"
            f"• Mode: Advisory only (no execution)\n"
            f"• Storage: In-memory only (max {MAX_INCIDENT_NODES} incidents)\n"
            f"• Learning: No learning engine\n"
            f"• Audit: No audit trails\n"
            f"• Upgrade to Enterprise for full features: {ENTERPRISE_UPGRADE_URL}"
        )
    
    def _register_oss_tools(self) -> Dict[str, Dict[str, Any]]:
        """Register OSS tools (analysis only)"""
        return {
            "rollback": {
                "name": "rollback",
                "description": "Analyze deployment rollback feasibility and impact",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "high",
                "parameters": {
                    "revision": {"type": "str", "required": True, "default": "previous"},
                    "force": {"type": "bool", "required": False, "default": False},
                },
                "oss_allowed": True,
                "requires_enterprise_for_execution": True,
            },
            "restart_container": {
                "name": "restart_container",
                "description": "Analyze container restart impact and timing",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "medium",
                "parameters": {
                    "container_id": {"type": "str", "required": False},
                    "grace_period": {"type": "int", "required": False, "default": 30},
                },
                "oss_allowed": True,
                "requires_enterprise_for_execution": True,
            },
            "scale_out": {
                "name": "scale_out",
                "description": "Analyze scaling feasibility and resource requirements",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "low",
                "parameters": {
                    "scale_factor": {"type": "int", "required": True, "default": 2, "min": 1, "max": 10},
                    "resource_profile": {"type": "str", "required": False, "default": "standard"},
                },
                "oss_allowed": True,
                "requires_enterprise_for_execution": True,
            },
            "circuit_breaker": {
                "name": "circuit_breaker",
                "description": "Analyze circuit breaker activation impact",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "medium",
                "parameters": {
                    "threshold": {"type": "float", "required": False, "default": 0.5},
                    "timeout": {"type": "int", "required": False, "default": 60},
                },
                "oss_allowed": True,
                "requires_enterprise_for_execution": True,
            },
            "traffic_shift": {
                "name": "traffic_shift",
                "description": "Analyze traffic shifting strategies",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "medium",
                "parameters": {
                    "percentage": {"type": "int", "required": True, "default": 50, "min": 1, "max": 100},
                    "target": {"type": "str", "required": True},
                },
                "oss_allowed": True,
                "requires_enterprise_for_execution": True,
            },
            "alert_team": {
                "name": "alert_team",
                "description": "Analyze when and how to alert human operators",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "low",
                "parameters": {
                    "severity": {"type": "str", "required": True, "default": "medium"},
                    "channels": {"type": "list", "required": False, "default": ["slack", "email"]},
                },
                "oss_allowed": True,
                "requires_enterprise_for_execution": True,
            },
        }
    
    async def analyze_and_recommend(
        self,
        tool_name: str,
        component: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        use_rag: bool = True,
    ) -> OSSAnalysisResult:
        """
        OSS: Analyze situation and create healing recommendation
        
        This is the primary OSS advisory method that:
        1. Validates the request
        2. Performs safety checks
        3. Optionally queries RAG for similar incidents
        4. Creates HealingIntent for Enterprise execution
        
        Args:
            tool_name: Name of the tool to analyze
            component: Target component
            parameters: Tool parameters
            context: Additional context (incident data, metrics, etc.)
            use_rag: Whether to use RAG graph for similarity search
        
        Returns:
            OSSAnalysisResult with HealingIntent and analysis metadata
        """
        start_time = time.time()
        self.metrics["requests_processed"] += 1
        
        try:
            # 1. Validate request
            validation = self._validate_request(tool_name, component, parameters, context)
            if not validation["valid"]:
                raise ValueError(f"Validation failed: {', '.join(validation['errors'])}")
            
            # 2. Perform safety checks
            safety_result = await self._perform_safety_checks(tool_name, component, parameters, context)
            if not safety_result["allowed"]:
                self.metrics["safety_checks_failed"] += 1
                raise ValueError(f"Safety check failed: {safety_result['reason']}")
            
            self.metrics["safety_checks_passed"] += 1
            
            # 3. Query RAG for similar incidents (if enabled and available)
            similar_incidents = []
            rag_similarity_score = None
            
            if use_rag and oss_config.get("rag_enabled", False):
                similar_incidents = await self._query_rag_for_similar_incidents(
                    component, parameters, context
                )
                if similar_incidents:
                    self.metrics["rag_queries_performed"] += 1
                    rag_similarity_score = self._calculate_rag_similarity_score(similar_incidents)
            
            # 4. Calculate confidence
            confidence = self._calculate_confidence(
                tool_name, component, parameters, similar_incidents, context
            )
            
            # 5. Create HealingIntent
            healing_intent = await self._create_healing_intent(
                tool_name=tool_name,
                component=component,
                parameters=parameters,
                context=context,
                similar_incidents=similar_incidents,
                rag_similarity_score=rag_similarity_score,
                confidence=confidence,
            )
            
            self.metrics["healing_intents_created"] += 1
            
            # 6. Calculate analysis time
            analysis_time_ms = (time.time() - start_time) * 1000
            
            # Update average analysis time
            total_requests = self.metrics["requests_processed"]
            current_avg = self.metrics["avg_analysis_time_ms"]
            self.metrics["avg_analysis_time_ms"] = (
                (current_avg * (total_requests - 1) + analysis_time_ms) / total_requests
            )
            
            logger.info(
                f"OSS Analysis: {tool_name} on {component} "
                f"(confidence: {confidence:.2f}, time: {analysis_time_ms:.1f}ms, "
                f"similar: {len(similar_incidents)})"
            )
            
            return OSSAnalysisResult(
                healing_intent=healing_intent,
                confidence=confidence,
                similar_incidents_count=len(similar_incidents),
                rag_similarity_score=rag_similarity_score,
                analysis_time_ms=analysis_time_ms,
                warnings=validation.get("warnings", []) + safety_result.get("warnings", []),
                requires_enterprise=True,
            )
            
        except Exception as e:
            logger.exception(f"Error in OSS analysis: {e}")
            analysis_time_ms = (time.time() - start_time) * 1000
            
            # Create fallback advisory intent
            fallback_intent = create_oss_advisory_intent(
                action=tool_name,
                component=component,
                parameters=parameters,
                justification=f"OSS analysis failed: {str(e)[:100]}",
                confidence=0.3,
                incident_id=context.get("incident_id", "") if context else "",
            )
            
            return OSSAnalysisResult(
                healing_intent=fallback_intent,
                confidence=0.3,
                analysis_time_ms=analysis_time_ms,
                warnings=[f"Analysis error: {str(e)[:100]}"],
                requires_enterprise=True,
            )
    
    def _validate_request(
        self,
        tool_name: str,
        component: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate MCP request"""
        errors = []
        warnings = []
        
        # Check if tool exists and is allowed in OSS
        if tool_name not in self.registered_tools:
            errors.append(f"Unknown tool: {tool_name}")
        else:
            tool_info = self.registered_tools[tool_name]
            if not tool_info.get("oss_allowed", True):
                errors.append(f"Tool {tool_name} requires Enterprise edition")
        
        # Check component
        if not component:
            errors.append("Component name is required")
        elif len(component) > 255:
            errors.append("Component name too long (max 255 characters)")
        
        # Check parameters against tool definition
        if tool_name in self.registered_tools:
            tool_params = self.registered_tools[tool_name].get("parameters", {})
            for param_name, param_def in tool_params.items():
                if param_def.get("required", False) and param_name not in parameters:
                    errors.append(f"Missing required parameter: {param_name}")
        
        # Check justification in context
        if context and "justification" in context:
            justification = context["justification"]
            if len(justification) < 10:
                warnings.append("Justification is brief (minimum 10 characters recommended)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    async def _perform_safety_checks(
        self,
        tool_name: str,
        component: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform safety checks for OSS advisory"""
        allowed = True
        reason = ""
        warnings = []
        
        # Check action blacklist
        action_blacklist = self.safety_guardrails.get("action_blacklist", [])
        if tool_name.upper() in [action.upper() for action in action_blacklist]:
            allowed = False
            reason = f"Tool {tool_name} is in safety blacklist"
        
        # Check blast radius
        affected_services = context.get("affected_services", [component]) if context else [component]
        max_blast_radius = self.safety_guardrails.get("max_blast_radius", 3)
        
        if len(affected_services) > max_blast_radius:
            warnings.append(
                f"Large blast radius: {len(affected_services)} services affected "
                f"(max: {max_blast_radius})"
            )
        
        # Check production environment warning
        if context and context.get("environment") == "production":
            warnings.append("Action requested in production environment - extra caution advised")
        
        return {
            "allowed": allowed,
            "reason": reason,
            "warnings": warnings,
        }
    
    async def _query_rag_for_similar_incidents(
        self,
        component: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Query RAG graph for similar historical incidents
        
        Returns empty list if RAG is not available or disabled.
        """
        # Check cache first
        cache_key = self._create_cache_key(component, parameters, context)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        try:
            # SAFE IMPORT: Try multiple import strategies
            rag_graph = None
            
            # Strategy 1: Try relative import first (works in development)
            try:
                from ...lazy import get_rag_graph
                rag_graph = get_rag_graph()
            except (ImportError, ValueError):
                # Strategy 2: Try absolute import (works when installed)
                try:
                    from agentic_reliability_framework.lazy import get_rag_graph
                    rag_graph = get_rag_graph()
                except ImportError:
                    logger.debug("RAG graph not available via absolute import")
            
            # Check if we got a RAG graph
            if rag_graph is None or not hasattr(rag_graph, 'is_enabled') or not rag_graph.is_enabled():
                return []
            
            # Create a compatibility event for RAG graph
            # The RAG graph expects a Pydantic ReliabilityEvent from models.py
            # We need to create a compatible event
            event = self._create_compatible_event_for_rag(component, context)
            
            if event is None:
                return []
            
            # Find similar incidents (limit to 5 for OSS)
            similar_nodes = rag_graph.find_similar(event, k=5)
            
            # Convert to dictionary format
            similar_incidents = []
            for node in similar_nodes:
                incident_dict = {
                    "incident_id": node.incident_id,
                    "component": node.component,
                    "severity": node.severity,
                    "similarity": node.metadata.get("similarity_score", 0.0),
                    "metrics": node.metrics,
                    "timestamp": node.timestamp,
                }
                
                # Get outcomes for this incident
                outcomes = rag_graph._get_outcomes(node.incident_id)
                if outcomes:
                    successful = [o for o in outcomes if o.success]
                    incident_dict["success"] = len(successful) > 0
                    incident_dict["success_rate"] = len(successful) / len(outcomes) if outcomes else 0.0
                    incident_dict["resolution_time_minutes"] = (
                        sum(o.resolution_time_minutes for o in successful) / len(successful) 
                        if successful else 0.0
                    )
                
                similar_incidents.append(incident_dict)
            
            # Cache results (with OSS size limit)
            self.similarity_cache[cache_key] = similar_incidents
            
            # Enforce cache size limit (LRU)
            if len(self.similarity_cache) > self.max_cache_size:
                oldest_key = next(iter(self.similarity_cache))
                del self.similarity_cache[oldest_key]
            
            return similar_incidents
            
        except Exception as e:
            logger.error(f"Error querying RAG: {e}", exc_info=True)
            return []
    
    def _create_compatible_event_for_rag(self, component: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a compatible event for the RAG graph
        
        The RAG graph expects a Pydantic ReliabilityEvent from models.py
        This method creates one that matches the expected structure
        """
        try:
            # Use the compatibility wrapper
            severity = EventSeverity.MEDIUM
            
            # Get severity from context if available
            if context and "severity" in context:
                severity_str = context["severity"]
                # Convert string to EventSeverity enum
                severity_map = {
                    "low": EventSeverity.LOW,
                    "medium": EventSeverity.MEDIUM,
                    "high": EventSeverity.HIGH,
                    "critical": EventSeverity.CRITICAL
                }
                if severity_str.lower() in severity_map:
                    severity = severity_map[severity_str.lower()]
            
            # Use the compatibility wrapper to create the event
            event = create_compatible_event(
                component=component,
                severity=severity,
                latency_p99=context.get("latency_p99", 100) if context else 100,
                error_rate=context.get("error_rate", 0.05) if context else 0.05,
                throughput=context.get("throughput", 1000) if context else 1000,
                cpu_util=context.get("cpu_util", 0.5) if context else 0.5,
                memory_util=context.get("memory_util", 0.5) if context else 0.5,
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error creating compatible event for RAG: {e}", exc_info=True)
            return None
    
    def _calculate_rag_similarity_score(self, similar_incidents: List[Dict[str, Any]]) -> float:
        """
        Calculate aggregate RAG similarity score
        
        Returns:
            Float between 0.0 and 1.0 representing average similarity
        """
        if not similar_incidents:
            return 0.0
        
        similarities: List[float] = []
        for inc in similar_incidents:
            similarity = inc.get("similarity")
            if isinstance(similarity, (int, float)):
                # Ensure similarity is within 0.0-1.0 range
                clamped_similarity = max(0.0, min(1.0, float(similarity)))
                similarities.append(clamped_similarity)
        
        if not similarities:
            return 0.0
        
        return sum(similarities) / len(similarities)
    
    def _calculate_confidence(
        self,
        tool_name: str,
        component: str,
        parameters: Dict[str, Any],
        similar_incidents: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for recommendation"""
        base_confidence = 0.85
        
        # Tool-specific adjustments
        tool_weights = {
            "restart_container": 1.0,
            "scale_out": 0.95,
            "circuit_breaker": 0.9,
            "traffic_shift": 0.85,
            "rollback": 0.8,
            "alert_team": 0.99,
        }
        
        base_confidence *= tool_weights.get(tool_name, 0.85)
        
        # Boost for historical context (RAG similarity)
        if similar_incidents:
            avg_similarity = self._calculate_rag_similarity_score(similar_incidents)
            # Boost confidence based on similarity (capped at 20% boost)
            similarity_boost = min(0.2, avg_similarity * 0.3)
            base_confidence *= (1.0 + similarity_boost)
            
            # Additional boost if historical incidents were successful
            success_rates = [inc.get("success_rate", 0.0) for inc in similar_incidents if "success_rate" in inc]
            if success_rates:
                avg_success_rate = sum(success_rates) / len(success_rates)
                success_boost = min(0.15, avg_success_rate * 0.2)
                base_confidence *= (1.0 + success_boost)
        
        # Context-based adjustments
        if context:
            # Higher confidence for critical incidents
            if context.get("severity") == "critical":
                base_confidence *= 1.1
            
            # Lower confidence for production environment (more caution)
            if context.get("environment") == "production":
                base_confidence *= 0.95
        
        # Cap at 1.0
        return min(base_confidence, 1.0)
    
    async def _create_healing_intent(
        self,
        tool_name: str,
        component: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        similar_incidents: List[Dict[str, Any]],
        rag_similarity_score: Optional[float],
        confidence: float,
    ) -> HealingIntent:
        """Create HealingIntent from analysis results"""
        
        # Generate justification
        justification = self._generate_justification(
            tool_name, component, parameters, similar_incidents, context
        )
        
        # Use appropriate factory method based on tool type
        if tool_name == "rollback":
            revision = parameters.get("revision", "previous")
            return create_rollback_intent(
                component=component,
                revision=revision,
                justification=justification,
                incident_id=context.get("incident_id", "") if context else "",
                similar_incidents=similar_incidents,
                rag_similarity_score=rag_similarity_score,
            )
        
        elif tool_name == "restart_container":
            container_id = parameters.get("container_id")
            return create_restart_intent(
                component=component,
                container_id=container_id,
                justification=justification,
                incident_id=context.get("incident_id", "") if context else "",
                similar_incidents=similar_incidents,
                rag_similarity_score=rag_similarity_score,
            )
        
        elif tool_name == "scale_out":
            scale_factor = parameters.get("scale_factor", 2)
            return create_scale_out_intent(
                component=component,
                scale_factor=scale_factor,
                justification=justification,
                incident_id=context.get("incident_id", "") if context else "",
                similar_incidents=similar_incidents,
                rag_similarity_score=rag_similarity_score,
            )
        
        else:
            # Generic intent for other tools
            return HealingIntent.from_analysis(
                action=tool_name,
                component=component,
                parameters=parameters,
                justification=justification,
                confidence=confidence,
                similar_incidents=similar_incidents,
                incident_id=context.get("incident_id", "") if context else "",
                source=IntentSource.RAG_SIMILARITY if similar_incidents else IntentSource.OSS_ANALYSIS,
                rag_similarity_score=rag_similarity_score,
            ).mark_as_oss_advisory()
    
    def _generate_justification(
        self,
        tool_name: str,
        component: str,
        parameters: Dict[str, Any],
        similar_incidents: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate justification for the healing intent"""
        
        if similar_incidents:
            similar_count = len(similar_incidents)
            
            # Calculate success rate from similar incidents
            success_rates = [inc.get("success_rate", 0.0) for inc in similar_incidents if "success_rate" in inc]
            if success_rates:
                avg_success_rate = sum(success_rates) / len(success_rates)
                success_text = f" with historical success rate of {avg_success_rate:.0%}"
            else:
                success_text = ""
            
            # Get average similarity
            avg_similarity = self._calculate_rag_similarity_score(similar_incidents)
            
            return (
                f"Based on {similar_count} similar historical incidents{success_text}. "
                f"Average similarity: {avg_similarity:.1%}. "
                f"Recommend {tool_name} for {component} with parameters {parameters}."
            )
        
        elif context and "justification" in context:
            justification = context["justification"]
            if isinstance(justification, str):
                return justification
            else:
                # Convert to string if it's not already
                return str(justification)
        
        else:
            return (
                f"Based on anomaly analysis, recommend {tool_name} "
                f"for {component} with parameters {parameters}."
            )
    
    def _create_cache_key(
        self,
        component: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Create cache key for similarity search
        
        Returns:
            String cache key based on component, parameters, and context
        """
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        context_str = json.dumps(context, sort_keys=True, default=str) if context else ""
        return f"{component}:{hash(param_str)}:{hash(context_str)}"
    
    async def execute_tool(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        OSS version of MCPServer.execute_tool() for backward compatibility
        
        Maintains same API signature but only provides advisory analysis.
        Creates HealingIntent for Enterprise execution.
        
        Args:
            request_dict: MCP request dictionary with keys:
                - tool: Tool name
                - component: Target component
                - parameters: Tool parameters
                - justification: Human-readable justification
                - metadata: Additional context
        
        Returns:
            Dictionary compatible with MCPResponse.to_dict()
        """
        # Extract request information
        tool_name = request_dict.get("tool", "")
        component = request_dict.get("component", "")
        parameters = request_dict.get("parameters", {})
        context = request_dict.get("metadata", {})
        
        # Add justification to context if provided
        justification = request_dict.get("justification", "")
        if justification:
            context["justification"] = justification
        
        # Perform OSS analysis
        analysis_result = await self.analyze_and_recommend(
            tool_name=tool_name,
            component=component,
            parameters=parameters,
            context=context,
            use_rag=oss_config.get("rag_enabled", False),
        )
        
        # Create OSS MCP response
        request_id = request_dict.get("request_id", str(uuid.uuid4()))
        response = OSSMCPResponse.from_healing_intent(
            analysis_result.healing_intent,
            request_id
        )
        
        # Add OSS analysis metadata
        if response.result is not None:  # FIXED: Check for None before accessing
            response.result["oss_analysis"] = {
                "analysis_time_ms": analysis_result.analysis_time_ms,
                "similar_incidents_found": analysis_result.similar_incidents_count,
                "rag_used": analysis_result.rag_similarity_score is not None,
                "confidence": analysis_result.confidence,
                "warnings": analysis_result.warnings,
            }
        
        return response.to_dict()
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get OSS MCP client information and capabilities"""
        capabilities = get_oss_capabilities()
        
        return {
            "mode": self.mode,
            "edition": self.oss_edition,
            "license": self.oss_license,
            "registered_tools": len(self.registered_tools),
            "metrics": self.metrics,
            "cache_size": len(self.similarity_cache),
            "oss_restricted": True,
            "capabilities": capabilities,
            "requires_enterprise_for_execution": True,
            "upgrade_url": ENTERPRISE_UPGRADE_URL,
            "config": {
                "mcp_mode": oss_config.get("mcp_mode", "advisory"),
                "mcp_enabled": oss_config.get("mcp_enabled", False),
                "rag_enabled": oss_config.get("rag_enabled", False),
                "execution_allowed": False,
            },
        }
    
    def get_tool_info(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about OSS tools"""
        if tool_name:
            tool = self.registered_tools.get(tool_name)
            if tool:
                return {
                    **tool,
                    "oss_edition": True,
                    "can_execute": False,
                    "requires_enterprise": True,
                    "upgrade_url": ENTERPRISE_UPGRADE_URL,
                }
            return {}
        
        return {
            name: {
                **info,
                "oss_edition": True,
                "can_execute": False,
                "requires_enterprise": True,
                "upgrade_url": ENTERPRISE_UPGRADE_URL,
            }
            for name, info in self.registered_tools.items()
        }
    
    def clear_cache(self) -> None:
        """Clear similarity cache"""
        self.similarity_cache.clear()
        logger.info("Cleared OSS similarity cache")
    
    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.metrics = {
            "requests_processed": 0,
            "healing_intents_created": 0,
            "rag_queries_performed": 0,
            "avg_analysis_time_ms": 0.0,
            "safety_checks_passed": 0,
            "safety_checks_failed": 0,
        }
        logger.info("Reset OSS MCP client metrics")


# Factory function for backward compatibility
def create_oss_mcp_client(config: Optional[Dict[str, Any]] = None) -> OSSMCPClient:
    """
    Factory function for creating OSS MCP client
    
    In OSS builds, returns OSSMCPClient (advisory only)
    In Enterprise builds, would return enhanced client
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OSSMCPClient instance
    """
    return OSSMCPClient(config)


# Export
__all__ = [
    "OSSMCPClient",
    "OSSMCPResponse",
    "OSSAnalysisResult",
    "create_oss_mcp_client",
]
