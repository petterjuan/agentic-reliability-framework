# agentic_reliability_framework/engine/mcp_client.py
"""
OSS MCP Client - Advisory mode only
Pythonic implementation that maintains same API as MCPServer
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List

from ..oss.constants import (
    MCP_MODES_ALLOWED,
    EXECUTION_ALLOWED,
    validate_oss_config,
    get_oss_capabilities,
    OSSBoundaryError,
    check_oss_compliance,
)
from ..oss.healing_intent import HealingIntent

logger = logging.getLogger(__name__)


class OSSMCPResponse:
    """OSS MCP response format (compatible with Enterprise MCPResponse)"""
    
    def __init__(
        self,
        request_id: str,
        status: str,
        message: str,
        executed: bool = False,
        result: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ):
        self.request_id = request_id
        self.status = status
        self.message = message
        self.executed = executed
        self.result = result
        self.timestamp = timestamp if timestamp is not None else time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "request_id": self.request_id,
            "status": self.status,
            "message": self.message,
            "executed": self.executed,
            "result": self.result,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_healing_intent(cls, intent: HealingIntent) -> "OSSMCPResponse":
        """Create response from HealingIntent"""
        return cls(
            request_id=intent.intent_id,
            status="completed",
            message=f"Advisory: Recommended {intent.action} for {intent.component}",
            executed=False,
            result={
                "mode": "advisory",
                "would_execute": True,
                "confidence": intent.confidence,
                "healing_intent": intent.to_enterprise_request(),
                "requires_enterprise": True,
                "upgrade_url": "https://arf.dev/enterprise",
                "enterprise_features": get_oss_capabilities()["enterprise_features"],
            }
        )


class OSSMCPClient:
    """
    OSS MCP Client - Advisory mode only
    
    Drop-in replacement for MCPServer in OSS edition.
    Same API as MCPServer.execute_tool() but with zero execution capability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OSS MCP Client
        
        Args:
            config: Configuration dictionary
        
        Raises:
            OSSBoundaryError: If config violates OSS boundaries
        """
        # Check if we should be in OSS mode
        if not check_oss_compliance():
            logger.warning(
                "Enterprise license detected but using OSS client. "
                "Consider using EnterpriseMCPServer for full features."
            )
        
        # Store configuration
        self.config = config or {}
        
        # Validate OSS boundaries
        validate_oss_config(self.config)
        
        # Hard-coded OSS mode
        self.mode = "advisory"
        
        # Initialize tools (analysis only)
        self.registered_tools = self._register_oss_tools()
        
        # Performance metrics
        self.metrics = {
            "requests_processed": 0,
            "healing_intents_created": 0,
            "avg_analysis_time_ms": 0.0,
        }
        
        # Cache for similar incidents
        self.similarity_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized OSSMCPClient in {self.mode} mode")
        logger.info(f"OSS Edition: {get_oss_capabilities()['edition']}")
    
    def _register_oss_tools(self) -> Dict[str, Dict[str, Any]]:
        """Register OSS tools (analysis only)"""
        return {
            "rollback": {
                "name": "rollback",
                "description": "Analyze deployment rollback feasibility and impact",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "high",
            },
            "restart_container": {
                "name": "restart_container", 
                "description": "Analyze container restart impact and timing",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "medium",
            },
            "scale_out": {
                "name": "scale_out",
                "description": "Analyze scaling feasibility and resource requirements",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "low",
            },
            "circuit_breaker": {
                "name": "circuit_breaker",
                "description": "Analyze circuit breaker activation impact",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "medium",
            },
            "traffic_shift": {
                "name": "traffic_shift",
                "description": "Analyze traffic shifting strategies",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "medium",
            },
            "alert_team": {
                "name": "alert_team",
                "description": "Analyze when and how to alert human operators",
                "can_execute": False,
                "analysis_only": True,
                "safety_level": "low",
            },
        }
    
    async def execute_tool(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        OSS version of MCPServer.execute_tool()
        
        Same API signature but only provides advisory analysis.
        """
        start_time = time.time()
        self.metrics["requests_processed"] += 1
        
        try:
            # Validate request
            validation = self._validate_request(request_dict)
            if not validation["valid"]:
                return OSSMCPResponse(
                    request_id=request_dict.get("request_id", str(uuid.uuid4())),
                    status="rejected",
                    message=f"Validation failed: {', '.join(validation['errors'])}",
                    executed=False,
                ).to_dict()
            
            # Perform OSS analysis
            analysis = await self._analyze_request(request_dict)
            
            # Create HealingIntent
            intent = HealingIntent.from_analysis(
                action=request_dict["tool"],
                component=request_dict["component"],
                parameters=request_dict.get("parameters", {}),
                justification=request_dict.get("justification", ""),
                confidence=analysis["confidence"],
                similar_incidents=analysis.get("similar_incidents"),
                reasoning_chain=analysis.get("reasoning_chain"),
                incident_id=request_dict.get("metadata", {}).get("incident_id", ""),
            )
            
            self.metrics["healing_intents_created"] += 1
            
            # Create OSS response
            response = OSSMCPResponse.from_healing_intent(intent)
            
            # Add OSS analysis context
            if response.result:
                similar_incidents = analysis.get("similar_incidents", [])
                response.result["oss_analysis"] = {
                    "analysis_time_ms": (time.time() - start_time) * 1000,
                    "similar_incidents_found": len(similar_incidents),
                    "rag_used": analysis.get("rag_used", False),
                    "cache_hit": analysis.get("cache_hit", False),
                }
            
            # Update metrics
            analysis_time = (time.time() - start_time) * 1000
            total_requests = self.metrics["requests_processed"]
            current_avg = self.metrics["avg_analysis_time_ms"]
            self.metrics["avg_analysis_time_ms"] = (
                (current_avg * (total_requests - 1) + analysis_time) / total_requests
            )
            
            logger.info(
                f"OSS Analysis: {intent.action} on {intent.component} "
                f"(confidence: {intent.confidence:.2f}, "
                f"time: {analysis_time:.1f}ms)"
            )
            
            return response.to_dict()
            
        except Exception as e:
            logger.exception(f"Error in OSS analysis: {e}")
            
            return OSSMCPResponse(
                request_id=request_dict.get("request_id", str(uuid.uuid4())),
                status="error",
                message=f"OSS analysis error: {str(e)}",
                executed=False,
            ).to_dict()
    
    def _validate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MCP request"""
        errors = []
        warnings = []
        
        # Check required fields
        if "tool" not in request:
            errors.append("Missing required field: tool")
        elif request["tool"] not in self.registered_tools:
            errors.append(f"Unknown tool: {request['tool']}")
        
        if "component" not in request:
            errors.append("Missing required field: component")
        
        # Check execution attempts
        if request.get("mode", "advisory") != "advisory":
            warnings.append(f"OSS only supports advisory mode. Got: {request.get('mode')}")
        
        # Check justification
        justification = request.get("justification", "")
        if len(justification) < 10:
            warnings.append("Justification is brief (minimum 10 characters recommended)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    async def _analyze_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform OSS analysis on request"""
        tool_name = request["tool"]
        component = request["component"]
        parameters = request.get("parameters", {})
        
        # Check cache first
        cache_key = self._create_cache_key(tool_name, component, parameters)
        if cache_key in self.similarity_cache:
            cached = self.similarity_cache[cache_key]
            return {**cached, "cache_hit": True}
        
        # Simulate RAG similarity search
        similar_incidents = await self._find_similar_incidents(
            component, parameters, request.get("metadata", {})
        )
        
        # Generate reasoning chain
        reasoning_chain = await self._generate_reasoning_chain(
            tool_name, component, parameters, similar_incidents
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            tool_name, component, parameters, similar_incidents
        )
        
        result = {
            "confidence": confidence,
            "similar_incidents": similar_incidents,
            "reasoning_chain": reasoning_chain,
            "rag_used": len(similar_incidents) > 0,
            "cache_hit": False,
        }
        
        # Cache result (LRU cache)
        self.similarity_cache[cache_key] = result
        if len(self.similarity_cache) > 100:
            self.similarity_cache.pop(next(iter(self.similarity_cache)))
        
        return result
    
    async def _find_similar_incidents(
        self, 
        component: str,
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find similar historical incidents using RAG"""
        # This would integrate with your existing RAG system
        await asyncio.sleep(0.01)  # Simulate RAG search
        
        # Return mock similar incidents
        return [
            {
                "incident_id": "inc_001",
                "component": component,
                "similarity": 0.85,
                "action_taken": "restart_container",
                "success": True,
                "resolution_time_minutes": 2.5,
            },
            {
                "incident_id": "inc_002", 
                "component": component,
                "similarity": 0.72,
                "action_taken": "scale_out",
                "success": True,
                "resolution_time_minutes": 5.0,
            }
        ]
    
    async def _generate_reasoning_chain(
        self,
        tool: str,
        component: str,
        parameters: Dict[str, Any],
        similar_incidents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate OSS reasoning chain"""
        chain = []
        
        chain.append({
            "step": 1,
            "type": "component_analysis",
            "description": f"Analyzing component: {component}",
            "details": {"component": component},
        })
        
        if similar_incidents:
            success_rate = sum(1 for inc in similar_incidents if inc.get("success")) / len(similar_incidents)
            chain.append({
                "step": 2,
                "type": "historical_context",
                "description": f"Found {len(similar_incidents)} similar incidents",
                "details": {
                    "similar_incidents_count": len(similar_incidents),
                    "historical_success_rate": success_rate,
                },
            })
        
        chain.append({
            "step": 3,
            "type": "action_recommendation",
            "description": f"Recommend {tool} for {component}",
            "details": {
                "tool": tool,
                "parameters": parameters,
                "justification": "Based on anomaly patterns and historical data",
            },
        })
        
        return chain
    
    def _calculate_confidence(
        self,
        tool: str,
        component: str,
        parameters: Dict[str, Any],
        similar_incidents: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for recommendation"""
        base_confidence = 0.85
        
        # Boost for historical context
        if similar_incidents:
            avg_similarity = sum(inc.get("similarity", 0.0) for inc in similar_incidents) / len(similar_incidents)
            base_confidence *= (0.9 + avg_similarity * 0.1)
        
        # Adjust for tool type
        tool_weights = {
            "restart_container": 1.0,
            "scale_out": 0.95,
            "circuit_breaker": 0.9,
            "traffic_shift": 0.85,
            "rollback": 0.8,
            "alert_team": 0.99,
        }
        
        base_confidence *= tool_weights.get(tool, 0.85)
        
        return min(base_confidence, 1.0)
    
    def _create_cache_key(
        self, 
        tool: str, 
        component: str, 
        parameters: Dict[str, Any]
    ) -> str:
        """Create cache key for similarity search"""
        import json
        param_str = json.dumps(parameters, sort_keys=True)
        return f"{tool}:{component}:{hash(param_str)}"
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get OSS MCP client statistics"""
        capabilities = get_oss_capabilities()
        
        return {
            "mode": self.mode,
            "registered_tools": len(self.registered_tools),
            "metrics": self.metrics,
            "cache_size": len(self.similarity_cache),
            "oss_edition": True,
            "capabilities": capabilities,
            "config": self.config,
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
                    "enterprise_required": True,
                }
            return {}
        
        return {
            name: {
                **info,
                "oss_edition": True,
                "can_execute": False,
                "enterprise_required": True,
            }
            for name, info in self.registered_tools.items()
        }
    
    def create_healing_intent(self, request_dict: Dict[str, Any]) -> HealingIntent:
        """Create HealingIntent directly from request"""
        # Use a synchronous wrapper to avoid nested event loops
        async def _async_create():
            return await self._analyze_request(request_dict)
        
        # Run the async function in a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use asyncio.create_task
                analysis_task = asyncio.create_task(_async_create())
                analysis = asyncio.run(asyncio.wait_for(analysis_task, timeout=10.0))
            else:
                # We're in a sync context, run the event loop
                analysis = asyncio.run(_async_create())
        except RuntimeError:
            # No event loop, create a new one
            analysis = asyncio.run(_async_create())
        
        return HealingIntent.from_analysis(
            action=request_dict["tool"],
            component=request_dict["component"],
            parameters=request_dict.get("parameters", {}),
            justification=request_dict.get("justification", ""),
            confidence=analysis["confidence"],
            similar_incidents=analysis.get("similar_incidents"),
            reasoning_chain=analysis.get("reasoning_chain"),
            incident_id=request_dict.get("metadata", {}).get("incident_id", ""),
        )


# Factory functions
def create_mcp_client(config: Optional[Dict[str, Any]] = None) -> OSSMCPClient:
    """
    Factory function for backward compatibility
    
    In OSS builds, returns OSSMCPClient
    In Enterprise builds, would return EnterpriseMCPServer
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OSSMCPClient instance
    """
    return OSSMCPClient(config)


def get_default_mcp_client() -> OSSMCPClient:
    """Get default MCP client with default configuration"""
    return OSSMCPClient()


__all__ = [
    "OSSMCPClient",
    "OSSMCPResponse",
    "create_mcp_client",
    "get_default_mcp_client",
]
