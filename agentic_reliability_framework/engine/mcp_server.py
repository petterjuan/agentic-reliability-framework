"""
Enhanced MCP Server for ARF v3 - OSS Edition (Pure Advisory)
Pythonic implementation with proper typing, error handling, and safety features
OSS EDITION: Advisory mode only, no execution capability
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Dict, Any, List, Optional, TypedDict, Protocol,
    AsyncGenerator, runtime_checkable, Union, Deque, DefaultDict, cast
)
from collections import defaultdict, deque

from ..config import config
from ..lazy import get_engine

logger = logging.getLogger(__name__)


# ========== TYPE DEFINITIONS ==========

class SafetyCheck(TypedDict):
    """Type for safety check results"""
    name: str
    passed: bool
    details: str


class ExecutionStats(TypedDict):
    """Type for execution statistics"""
    total: int
    successful: int
    failed: int
    average_duration_seconds: float
    last_execution: Optional[float]


class ToolMetadata(TypedDict, total=False):
    """Type for tool metadata"""
    name: str
    description: str
    version: str
    author: str
    supported_environments: List[str]
    safety_level: str
    timeout_seconds: int
    required_permissions: List[str]


# ========== ENUMS ==========

class MCPMode(str, Enum):
    """MCP execution modes - OSS ONLY supports advisory"""
    ADVISORY = "advisory"  # OSS default - no execution


class MCPRequestStatus(str, Enum):
    """MCP request status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# ========== DATA CLASSES ==========

@dataclass(frozen=True, slots=True)
class MCPRequest:
    """Immutable MCP request model"""
    request_id: str
    tool: str
    component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    mode: MCPMode = MCPMode.ADVISORY
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary"""
        return {
            "request_id": self.request_id,
            "tool": self.tool,
            "component": self.component,
            "parameters": self.parameters,
            "justification": self.justification,
            "mode": self.mode.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass(frozen=True, slots=True)
class MCPResponse:
    """Immutable MCP response model"""
    request_id: str
    status: MCPRequestStatus
    message: str
    executed: bool = False
    result: Optional[Dict[str, Any]] = None
    approval_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "message": self.message,
            "executed": self.executed,
            "result": self.result,
            "approval_id": self.approval_id,
            "timestamp": self.timestamp
        }


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Immutable context for tool execution (advisory only in OSS)"""
    component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: str = "production"
    metadata: Dict[str, Any] = field(default_factory=dict)
    safety_guardrails: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Immutable result of tool analysis (advisory only in OSS)"""
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def success_result(cls, message: str, **details: Any) -> "ToolResult":
        """Create a successful advisory result"""
        return cls(success=True, message=message, details=details)

    @classmethod
    def failure_result(cls, message: str, **details: Any) -> "ToolResult":
        """Create a failure advisory result"""
        return cls(success=False, message=message, details=details)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Immutable result of tool validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    safety_checks: Dict[str, SafetyCheck] = field(default_factory=dict)

    @classmethod
    def valid_result(cls, warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a valid result"""
        return cls(valid=True, warnings=warnings or [])

    @classmethod
    def invalid_result(cls, error: str, *additional_errors: str) -> "ValidationResult":
        """Create an invalid result"""
        return cls(valid=False, errors=[error, *additional_errors])


# ========== PROTOCOLS ==========

@runtime_checkable
class MCPTool(Protocol):
    """Protocol for MCP tools - advisory only in OSS"""

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata"""
        ...

    def validate(self, context: ToolContext) -> ValidationResult:
        """Validate the tool execution (advisory only)"""
        ...

    def get_tool_info(self) -> Dict[str, Any]:
        """Get comprehensive tool information"""
        ...


# ========== BASE TOOL CLASSES ==========

class BaseMCPTool:
    """Base class for MCP tools with common functionality"""

    def __init__(self, metadata: ToolMetadata):
        self._metadata = metadata

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata"""
        return self._metadata

    def get_tool_info(self) -> Dict[str, Any]:
        """Get comprehensive tool information"""
        return {
            **self.metadata,
            "class_name": self.__class__.__name__,
            "oss_edition": True,
            "can_execute": False,
            "requires_enterprise": True,
        }

    def _add_safety_check(
        self,
        validation: ValidationResult,
        name: str,
        passed: bool,
        details: str = ""
    ) -> ValidationResult:
        """Helper to add safety checks to validation result"""
        safety_checks = dict(validation.safety_checks)
        safety_checks[name] = SafetyCheck(
            name=name,
            passed=passed,
            details=details
        )

        return ValidationResult(
            valid=validation.valid,
            errors=validation.errors.copy(),
            warnings=validation.warnings.copy(),
            safety_checks=safety_checks
        )


class RollbackTool(BaseMCPTool):
    """K8s/ECS/VM rollback analysis - advisory only"""

    def __init__(self):
        super().__init__({
            "name": "rollback",
            "description": "Analyze deployment rollback feasibility (Advisory Only)",
            "supported_environments": ["kubernetes", "ecs", "vm"],
            "safety_level": "high",
            "timeout_seconds": 60,
            "required_permissions": ["deployment.write", "rollback.execute"]
        })

    def validate(self, context: ToolContext) -> ValidationResult:
        """Validate rollback with comprehensive safety checks"""
        validation = ValidationResult.valid_result()

        # Environment validation
        if context.environment not in self.metadata["supported_environments"]:
            return ValidationResult.invalid_result(
                f"Unsupported environment: {context.environment}"
            )

        # Safety checks
        safety_guardrails = context.safety_guardrails

        # Check: Production environment warning
        if context.metadata.get("environment", "production") == "production":
            validation = self._add_safety_check(
                validation, "production_environment", False,
                "Rollback in production carries higher risk"
            )
            validation.warnings.append("Rollback requested in production environment")
        else:
            validation = self._add_safety_check(
                validation, "production_environment", True
            )

        # Check: Healthy revision
        if not context.metadata.get("has_healthy_revision", False):
            return ValidationResult.invalid_result(
                "No healthy revision available for rollback"
            )
        validation = self._add_safety_check(
            validation, "has_healthy_revision", True
        )

        # Check: Blast radius
        affected_services = context.metadata.get("affected_services", [context.component])
        max_blast_radius = safety_guardrails.get("max_blast_radius", 3)

        if len(affected_services) > max_blast_radius:
            return ValidationResult.invalid_result(
                f"Blast radius too large: {len(affected_services)} services "
                f"(max: {max_blast_radius})"
            )
        validation = self._add_safety_check(
            validation, "blast_radius", True,
            f"Affects {len(affected_services)} service(s)"
        )

        # Check: Action blacklist
        if "ROLLBACK" in safety_guardrails.get("action_blacklist", []):
            return ValidationResult.invalid_result(
                "Rollback is in the safety blacklist"
            )
        validation = self._add_safety_check(
            validation, "not_blacklisted", True
        )

        # Check: Canary deployment (warning only)
        if not context.metadata.get("has_canary", False):
            validation = self._add_safety_check(
                validation, "has_canary", False,
                "No canary deployment detected"
            )
            validation.warnings.append("No canary deployment detected")
        else:
            validation = self._add_safety_check(
                validation, "has_canary", True
            )

        return validation


class RestartContainerTool(BaseMCPTool):
    """Container restart analysis - advisory only"""

    def __init__(self):
        super().__init__({
            "name": "restart_container",
            "description": "Analyze container restart impact (Advisory Only)",
            "supported_environments": ["kubernetes", "ecs", "docker"],
            "safety_level": "medium",
            "timeout_seconds": 30,
            "required_permissions": ["container.restart"]
        })

    def validate(self, context: ToolContext) -> ValidationResult:
        """Validate container restart"""
        validation = ValidationResult.valid_result()

        # Check restart count
        restart_count = context.metadata.get("restart_count", 0)
        if restart_count > 3:
            validation = self._add_safety_check(
                validation, "reasonable_restart_count", False,
                f"High restart count: {restart_count}"
            )
            validation.warnings.append(f"High restart count: {restart_count}")
        else:
            validation = self._add_safety_check(
                validation, "reasonable_restart_count", True
            )

        # Check container health
        if not context.metadata.get("container_healthy", True):
            validation.errors.append("Container is not healthy")
            validation = self._add_safety_check(
                validation, "container_healthy", False,
                "Container health check failed"
            )
        else:
            validation = self._add_safety_check(
                validation, "container_healthy", True
            )

        # Create new ValidationResult instead of modifying valid field
        return ValidationResult(
            valid=len(validation.errors) == 0,
            errors=validation.errors,
            warnings=validation.warnings,
            safety_checks=validation.safety_checks
        )


class ScaleOutTool(BaseMCPTool):
    """Scale out analysis - advisory only"""

    def __init__(self):
        super().__init__({
            "name": "scale_out",
            "description": "Analyze scale out feasibility (Advisory Only)",
            "supported_environments": ["kubernetes", "ecs"],
            "safety_level": "low",
            "timeout_seconds": 45,
            "required_permissions": ["deployment.scale"]
        })

    def validate(self, context: ToolContext) -> ValidationResult:
        """Validate scale out"""
        validation = ValidationResult.valid_result()
        scale_factor = context.parameters.get("scale_factor", 1)

        # Check scale factor
        if scale_factor > 10:
            validation.errors.append(f"Scale factor too high: {scale_factor} (max: 10)")
            validation = self._add_safety_check(
                validation, "reasonable_scale_factor", False
            )
        else:
            validation = self._add_safety_check(
                validation, "reasonable_scale_factor", True
            )

        # Check resource limits
        current_replicas = context.metadata.get("current_replicas", 1)
        max_replicas = context.metadata.get("max_replicas", 20)
        new_replicas = current_replicas * scale_factor

        if new_replicas > max_replicas:
            validation.errors.append(
                f"Scale would exceed max replicas: {new_replicas} > {max_replicas}"
            )
            validation = self._add_safety_check(
                validation, "within_resource_limits", False
            )
        else:
            validation = self._add_safety_check(
                validation, "within_resource_limits", True
            )

        # Create new ValidationResult instead of modifying valid field
        return ValidationResult(
            valid=len(validation.errors) == 0,
            errors=validation.errors,
            warnings=validation.warnings,
            safety_checks=validation.safety_checks
        )


# ========== FACTORY FUNCTIONS ==========

def create_circuit_breaker_tool() -> MCPTool:
    """Factory function for circuit breaker analysis tool"""

    class CircuitBreakerTool(BaseMCPTool):
        def __init__(self):
            super().__init__({
                "name": "circuit_breaker",
                "description": "Analyze circuit breaker feasibility (Advisory Only)",
                "supported_environments": ["all"],
                "safety_level": "low",
                "timeout_seconds": 10,
                "required_permissions": ["circuit_breaker.manage"]
            })

        def validate(self, context: ToolContext) -> ValidationResult:
            return ValidationResult.valid_result()

    return CircuitBreakerTool()


def create_traffic_shift_tool() -> MCPTool:
    """Factory function for traffic shift analysis tool"""

    class TrafficShiftTool(BaseMCPTool):
        def __init__(self):
            super().__init__({
                "name": "traffic_shift",
                "description": "Analyze traffic shifting strategies (Advisory Only)",
                "supported_environments": ["kubernetes", "ecs", "load_balancer"],
                "safety_level": "medium",
                "timeout_seconds": 30,
                "required_permissions": ["traffic.manage"]
            })

        def validate(self, context: ToolContext) -> ValidationResult:
            return ValidationResult.valid_result()

    return TrafficShiftTool()


def create_alert_tool() -> MCPTool:
    """Factory function for alert analysis tool"""

    class AlertTool(BaseMCPTool):
        def __init__(self):
            super().__init__({
                "name": "alert_team",
                "description": "Analyze alert requirements (Advisory Only)",
                "supported_environments": ["all"],
                "safety_level": "low",
                "timeout_seconds": 5,
                "required_permissions": ["alert.create"]
            })

        def validate(self, context: ToolContext) -> ValidationResult:
            return ValidationResult.valid_result()

    return AlertTool()


# ========== OSS INTEGRATION IMPORT ==========
# CORRECT: Import from arf_core package
try:
    # Use relative imports to avoid circular dependencies
    from ..arf_core import (
        HealingIntent,
        OSSMCPClient,
        create_mcp_client as create_oss_mcp_client
    )
    
    OSS_CLIENT_AVAILABLE = True
    HEALING_INTENT_AVAILABLE = True
    logger.info("✅ Successfully imported OSS components from arf_core")
    
except ImportError as e:
    logger.warning(f"Failed to import from arf_core: {e}. Creating fallbacks...")
    
    # Define minimal HealingIntent
    from dataclasses import dataclass
    from typing import Dict, Any
    
    @dataclass
    class HealingIntent:
        """Fallback HealingIntent for when arf_core is not available"""
        action: str
        component: str
        parameters: Dict[str, Any]
        justification: str = ""
        confidence: float = 0.85
        incident_id: str = ""
        
        def to_enterprise_request(self):
            return {
                "action": self.action,
                "component": self.component,
                "parameters": self.parameters,
                "justification": self.justification,
                "confidence": self.confidence,
                "incident_id": self.incident_id,
                "requires_enterprise": True,
                "oss_metadata": {"fallback": True}
            }
        
        def mark_as_oss_advisory(self):
            return self
        
        @classmethod
        def from_analysis(cls, action, component, parameters, justification, confidence, incident_id=""):
            return cls(
                action=action,
                component=component,
                parameters=parameters,
                justification=justification,
                confidence=confidence,
                incident_id=incident_id
            )
    
    # Define minimal OSSMCPClient
    class OSSMCPClient:
        def __init__(self):
            self.mode = "advisory"
        
        async def analyze_and_recommend(self, tool_name, component, parameters, context=None):
            return HealingIntent(
                action=tool_name,
                component=component,
                parameters=parameters,
                justification=f"Fallback analysis for {tool_name} on {component}",
                confidence=0.75,
                incident_id=context.get("incident_id", "") if context else ""
            )
    
    def create_oss_mcp_client():
        return OSSMCPClient()
    
    OSS_CLIENT_AVAILABLE = True
    HEALING_INTENT_AVAILABLE = True
    logger.info("✅ Created fallback OSS implementations")


# ========== MCP SERVER (OSS EDITION) ==========

class MCPServer:
    """
    Enhanced MCP Server - OSS Edition (Pure Advisory)
    Advisory mode only, no execution capability
    
    Features:
    - Thread-safe operations
    - Comprehensive error handling
    - Detailed metrics and monitoring
    - Extensible tool system
    - OSS Edition: Advisory mode only with HealingIntent integration
    
    Key OSS Limitations:
    - NO execution capability
    - NO autonomous mode
    - NO approval workflows
    - Creates HealingIntent for Enterprise handoff only
    """

    def __init__(self, mode: Optional[MCPMode] = None):
        """
        Initialize OSS MCP Server
        
        Args:
            mode: Ignored in OSS edition - always advisory mode
        """
        # OSS EDITION: Always advisory mode
        self.mode = MCPMode.ADVISORY
        
        logger.warning(
            "⚠️  OSS Edition - Advisory mode only\n"
            "• No execution capability\n"
            "• Analysis and recommendations only\n"
            "• Creates HealingIntent for Enterprise handoff\n"
            "• Upgrade to Enterprise for execution: https://arf.dev/enterprise"
        )
        
        # === OSS INTEGRATION ===
        self.oss_client = None
        if OSS_CLIENT_AVAILABLE:
            try:
                self.oss_client = create_oss_mcp_client()
                logger.info("Integrated with OSSMCPClient for HealingIntent creation")
            except Exception as e:
                logger.warning(f"Failed to initialize OSSMCPClient: {e}")
        
        # === EXISTING INITIALIZATION CODE ===
        self.registered_tools: Dict[str, MCPTool] = self._register_tools()
        self.safety_guardrails = config.safety_guardrails

        # State management
        self._cooldowns: Dict[str, float] = {}
        self._execution_history: Deque[Dict[str, Any]] = deque(maxlen=1000)

        # Metrics
        self._start_time = time.time()
        self._tool_stats: DefaultDict[str, ExecutionStats] = defaultdict(
            lambda: {"total": 0, "successful": 0, "failed": 0,
                     "average_duration_seconds": 0.0, "last_execution": None}
        )

        logger.info(f"Initialized OSS MCPServer (Advisory Only) - HealingIntent available: {HEALING_INTENT_AVAILABLE}")

    def _register_tools(self) -> Dict[str, MCPTool]:
        """Register all available tools (advisory only)"""
        tools: Dict[str, MCPTool] = {
            "rollback": RollbackTool(),
            "restart_container": RestartContainerTool(),
            "scale_out": ScaleOutTool(),
            "circuit_breaker": create_circuit_breaker_tool(),
            "traffic_shift": create_traffic_shift_tool(),
            "alert_team": create_alert_tool(),
        }

        logger.info(f"Registered {len(tools)} advisory tools: {list(tools.keys())}")
        return tools

    async def execute_tool(self, request_dict: Dict[str, Any]) -> MCPResponse:
        """
        Execute a tool with comprehensive safety checks
        
        OSS Edition: Advisory only, no execution
        Creates HealingIntent for Enterprise handoff
        """
        # OSS EDITION: Force advisory mode
        request_dict["mode"] = "advisory"
        
        # 1. Create and validate request
        request = self._create_request(request_dict)
        validation = self._validate_request(request)

        if not validation["valid"]:
            return self._create_error_response(
                request,
                MCPRequestStatus.REJECTED,
                f"Invalid request: {', '.join(validation['errors'])}"
            )

        # 2. Check permissions
        if not self._check_permissions(request):
            return self._create_error_response(
                request,
                MCPRequestStatus.REJECTED,
                "Permission denied"
            )

        # 3. Check cooldowns
        cooldown_check = self._check_cooldown(request.tool, request.component)
        if not cooldown_check["allowed"]:
            return self._create_error_response(
                request,
                MCPRequestStatus.REJECTED,
                f"In cooldown period: {cooldown_check['remaining']:.0f}s remaining"
            )

        # OSS EDITION: Only handle advisory mode
        try:
            return await self._handle_advisory_mode(request)
        except Exception as e:
            logger.exception(f"Error handling request {request.request_id}: {e}")
            return self._create_error_response(
                request,
                MCPRequestStatus.FAILED,
                f"OSS analysis error: {str(e)}"
            )

    def _create_request(self, request_dict: Dict[str, Any]) -> MCPRequest:
        """Create MCPRequest from dictionary with validation"""
        # OSS EDITION: Always advisory mode
        mode = MCPMode.ADVISORY

        return MCPRequest(
            request_id=request_dict.get("request_id", str(uuid.uuid4())),
            tool=request_dict["tool"],
            component=request_dict["component"],
            parameters=request_dict.get("parameters", {}),
            justification=request_dict.get("justification", ""),
            mode=mode,
            metadata=request_dict.get("metadata", {})
        )

    def _validate_request(self, request: MCPRequest) -> Dict[str, Any]:
        """Validate MCP request - clean and mypy-safe"""
        errors: List[str] = []
        warnings: List[str] = []
        
        # Check if tool exists
        if request.tool not in self.registered_tools:
            errors.append(f"Unknown tool: {request.tool}")
        
        # Check component
        if not request.component:
            errors.append("Component name is required")
        elif len(request.component) > 255:
            errors.append("Component name too long (max 255 characters)")
        
        # Check justification
        if len(request.justification) < 10:
            errors.append("Justification too short (min 10 characters)")
        
        # OSS-specific validation: Reject non-advisory modes
        if request.mode != MCPMode.ADVISORY:
            errors.append(f"OSS edition only supports advisory mode, got: {request.mode}")
        
        # Return immediately
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _check_permissions(self, request: MCPRequest) -> bool:
        """Check permissions for request"""
        # Check action blacklist
        action_blacklist = self.safety_guardrails.get("action_blacklist", [])
        if isinstance(action_blacklist, list):
            if request.tool.upper() in action_blacklist:
                logger.warning(f"Tool {request.tool} is in safety blacklist")
                return False

        return True

    def _check_cooldown(self, tool: str, component: str) -> Dict[str, Any]:
        """Check if tool is in cooldown period"""
        key = f"{component}:{tool}"
        current_time = time.time()

        if key in self._cooldowns:
            cooldown_until = self._cooldowns[key]
            remaining = cooldown_until - current_time

            if remaining > 0:
                return {
                    "allowed": False,
                    "remaining": remaining,
                    "cooldown_until": cooldown_until
                }

        # Clean up expired cooldowns
        self._cleanup_cooldowns()

        return {"allowed": True, "remaining": 0}

    def _cleanup_cooldowns(self) -> None:
        """Clean up expired cooldowns"""
        current_time = time.time()
        expired_keys = [
            k for k, v in self._cooldowns.items()
            if current_time > v
        ]
        for k in expired_keys:
            del self._cooldowns[k]

    def _create_error_response(
        self,
        request: MCPRequest,
        status: MCPRequestStatus,
        message: str
    ) -> MCPResponse:
        """Create an error response"""
        return MCPResponse(
            request_id=request.request_id,
            status=status,
            message=message,
            executed=False
        )

    async def _create_healing_intent(self, request: MCPRequest) -> HealingIntent:
        """
        Create a HealingIntent from MCP request
        
        Uses OSS analysis to create a complete HealingIntent
        that can be executed by Enterprise edition
        """
        # If HealingIntent is not available, create minimal version
        if not HEALING_INTENT_AVAILABLE or HealingIntent is None:
            logger.warning("HealingIntent not available, creating minimal version")
            return self._create_minimal_healing_intent(request)
        
        # Use OSS client for analysis if available
        if self.oss_client and hasattr(self.oss_client, 'analyze_and_recommend'):
            try:
                # Create tool context for validation
                context = ToolContext(
                    component=request.component,
                    parameters=request.parameters,
                    metadata=request.metadata,
                    safety_guardrails=self.safety_guardrails
                )
                
                # Validate the request
                tool = self.registered_tools.get(request.tool)
                if tool:
                    validation = tool.validate(context)
                    if not validation.valid:
                        raise ValueError(f"Validation failed: {', '.join(validation.errors)}")
                
                # Use OSS client for analysis
                # This returns a HealingIntent DIRECTLY (not a wrapper)
                healing_intent = await self.oss_client.analyze_and_recommend(
                    tool_name=request.tool,
                    component=request.component,
                    parameters=request.parameters,
                    context=request.metadata
                )
                
                # The OSS client returns a HealingIntent directly
                # Make sure it's marked as OSS advisory
                return healing_intent.mark_as_oss_advisory()
                    
            except Exception as e:
                logger.warning(f"OSSMCPClient analysis failed: {e}")
                # Fall back to basic HealingIntent
        
        # Fallback: Create basic HealingIntent using the factory method
        try:
            return HealingIntent.from_analysis(
                action=request.tool,
                component=request.component,
                parameters=request.parameters,
                justification=request.justification,
                confidence=0.85,
                incident_id=request.metadata.get("incident_id", ""),
            ).mark_as_oss_advisory()  # IMPORTANT: Mark as OSS advisory
        
        except Exception as e:
            logger.error(f"Failed to create HealingIntent: {e}")
            # Ultimate fallback
            return self._create_minimal_healing_intent(request)

    def _create_minimal_healing_intent(self, request: MCPRequest):
        """Create minimal healing intent when OSS package is not available"""
        from dataclasses import dataclass
        
        @dataclass
        class MinimalHealingIntent:
            action: str
            component: str
            parameters: Dict[str, Any]
            justification: str = ""
            confidence: float = 0.85
            incident_id: str = ""
            
            def to_enterprise_request(self):
                return {
                    "action": self.action,
                    "component": self.component,
                    "parameters": self.parameters,
                    "justification": self.justification,
                    "confidence": self.confidence,
                    "incident_id": self.incident_id,
                    "requires_enterprise": True,
                    "oss_metadata": {"fallback": True}
                }
        
        return MinimalHealingIntent(
            action=request.tool,
            component=request.component,
            parameters=request.parameters,
            justification=request.justification,
            incident_id=request.metadata.get("incident_id", ""),
            confidence=0.85,
        )

    async def _handle_advisory_mode(self, request: MCPRequest) -> MCPResponse:
        """
        Handle advisory mode by creating HealingIntent
        
        OSS Edition: Uses OSS analysis to create HealingIntent
        for Enterprise handoff. No execution capability.
        """
        start_time = time.time()
        
        try:
            # Create tool context for validation
            context = ToolContext(
                component=request.component,
                parameters=request.parameters,
                metadata=request.metadata,
                safety_guardrails=self.safety_guardrails
            )
            
            # Validate the request
            tool = self.registered_tools.get(request.tool)
            if not tool:
                return self._create_error_response(
                    request,
                    MCPRequestStatus.REJECTED,
                    f"Unknown tool: {request.tool}"
                )
            
            validation = tool.validate(context)
            if not validation.valid:
                return self._create_error_response(
                    request,
                    MCPRequestStatus.REJECTED,
                    f"Validation failed: {', '.join(validation.errors)}"
                )
            
            # Create HealingIntent for Enterprise handoff
            healing_intent = None
            if HEALING_INTENT_AVAILABLE:
                try:
                    healing_intent = await self._create_healing_intent(request)
                except Exception as e:
                    logger.warning(f"Failed to create HealingIntent: {e}")
                    # Continue without HealingIntent
            
            # Update metrics
            analysis_time = time.time() - start_time
            stats = self._tool_stats[request.tool]
            stats["total"] += 1
            stats["average_duration_seconds"] = (
                (stats["average_duration_seconds"] * (stats["total"] - 1) + analysis_time)
                / stats["total"]
            )
            stats["last_execution"] = time.time()
            
            # Record advisory execution
            self._execution_history.append({
                "request_id": request.request_id,
                "tool": request.tool,
                "component": request.component,
                "mode": request.mode.value,
                "status": "advisory_completed",
                "timestamp": time.time(),
                "analysis_time_seconds": analysis_time,
                "healing_intent_created": healing_intent is not None
            })
            
            # Return advisory response
            result_data = {
                "mode": "advisory",
                "executed": False,
                "justification": request.justification,
                "validation_passed": True,
                "safety_checks": validation.safety_checks,
                "warnings": validation.warnings,
                "requires_enterprise": True,
                "upgrade_url": "https://arf.dev/enterprise",
                "enterprise_features": [
                    "autonomous_execution",
                    "approval_workflows", 
                    "persistent_storage",
                    "learning_engine",
                    "audit_trails",
                    "compliance_reports"
                ],
                "analysis_time_seconds": analysis_time,
                "healing_intent_available": healing_intent is not None,
            }
            
            # Add HealingIntent data if available
            if healing_intent:
                result_data["healing_intent"] = healing_intent.to_enterprise_request()
                result_data["oss_analysis"] = {
                    "confidence": healing_intent.confidence,
                    "similar_incidents_count": len(healing_intent.similar_incidents or []),
                    "rag_used": healing_intent.rag_similarity_score is not None,
                }
            
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.COMPLETED,
                message=f"OSS Analysis: Recommend {request.tool} for {request.component}",
                executed=False,
                result=result_data
            )
            
        except Exception as e:
            logger.exception(f"Error in advisory analysis: {e}")
            return self._create_error_response(
                request,
                MCPRequestStatus.FAILED,
                f"Advisory analysis failed: {str(e)}"
            )

    def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive MCP server statistics"""
        engine = get_engine()
        
        # Get OSS client info if available
        oss_client_info = None
        if self.oss_client is not None:
            try:
                oss_client_info = self.oss_client.get_client_info()
            except Exception:
                pass

        # Ensure all values are JSON serializable and properly typed
        stats: Dict[str, Any] = {
            "mode": self.mode.value,
            "edition": "oss",
            "oss_restricted": True,
            "execution_allowed": False,
            "registered_tools": len(self.registered_tools),
            "active_cooldowns": len(self._cooldowns),
            "execution_history_count": len(self._execution_history),
            "tool_statistics": {k: dict(v) for k, v in self._tool_stats.items()},
            "uptime_seconds": float(time.time() - self._start_time),
            "engine_available": engine is not None,
            "engine_type": str(getattr(engine, "__class__.__name__", "unknown")) if engine else "None",
            "config": {
                "mcp_mode": str(config.mcp_mode),
                "mcp_enabled": bool(config.mcp_enabled),
                "mpc_cooldown_seconds": int(config.mpc_cooldown_seconds),
            },
            "oss_limits": {
                "max_incidents": 1000,
                "execution_allowed": False,
                "mode": "advisory",
                "healing_intent_support": HEALING_INTENT_AVAILABLE,
            },
            "oss_integration": {
                "using_oss_client": self.oss_client is not None,
                "healing_intent_support": HEALING_INTENT_AVAILABLE,
                "oss_client_available": OSS_CLIENT_AVAILABLE,
            }
        }
        
        # Add safety guardrails if available and properly formatted
        if hasattr(self.safety_guardrails, 'items'):
            stats["safety_guardrails"] = dict(self.safety_guardrails)
        else:
            # Handle case where safety_guardrails might be a string or other type
            stats["safety_guardrails"] = str(self.safety_guardrails)
        
        # Add OSS client metrics if available
        if oss_client_info:
            # Ensure metrics are properly typed
            metrics = oss_client_info.get("metrics", {})
            if isinstance(metrics, dict):
                stats["oss_client_metrics"] = dict(metrics)
                # Add rag_integration flag if available
                if metrics.get("rag_queries_performed", 0) > 0:
                    stats["oss_integration"]["rag_integration"] = True
            else:
                stats["oss_client_metrics"] = {}
            
            # Add cache size if available
            cache_size = oss_client_info.get("cache_size", 0)
            if isinstance(cache_size, (int, float)):
                stats["oss_client_cache_size"] = int(cache_size)
            else:
                stats["oss_client_cache_size"] = 0

        return stats

    def get_tool_info(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about tools"""
        # Try to use OSSMCPClient for tool info if available
        if self.oss_client is not None and hasattr(self.oss_client, 'get_tool_info'):
            try:
                return self.oss_client.get_tool_info(tool_name)
            except Exception:
                pass  # Fall back to existing implementation
        
        # Fallback to existing implementation
        if tool_name:
            tool = self.registered_tools.get(tool_name)
            if tool:
                info = tool.get_tool_info()
                # Add OSS edition info
                info["oss_edition"] = True
                info["can_execute"] = False
                info["requires_enterprise"] = True
                return info
            return {}
        
        # For all tools - FIXED: Explicitly type the result
        result: Dict[str, Any] = {}
        for name, tool in self.registered_tools.items():
            tool_info = tool.get_tool_info()
            result[name] = {
                **tool_info,
                "oss_edition": True,
                "can_execute": False,
                "requires_enterprise": True,
            }
        return result

    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history (advisory only)"""
        return list(self._execution_history)[-limit:]

    def reset_stats(self) -> None:
        """Reset server statistics"""
        self._tool_stats.clear()
        self._execution_history.clear()
        self._cooldowns.clear()
        logger.info("OSS MCP server statistics reset")

    def enforce_oss_purity(self) -> bool:
        """
        Validate that this server instance is pure OSS
        
        Returns:
            True if pure OSS, False if Enterprise features detected
        """
        purity_checks = []
        
        # Check mode is advisory only
        if self.mode != MCPMode.ADVISORY:
            purity_checks.append(f"Mode must be ADVISORY, got {self.mode}")
        
        # Check no execution capability
        if hasattr(self, '_execute_with_approval') or hasattr(self, '_execute_autonomous'):
            purity_checks.append("Execution methods should not exist in OSS")
        
        # Check tools are advisory only
        for tool_name, tool in self.registered_tools.items():
            if hasattr(tool, 'execute'):
                purity_checks.append(f"Tool {tool_name} has execute method (should be advisory only)")
        
        if purity_checks:
            logger.warning(f"OSS purity violations: {purity_checks}")
            return False
        
        logger.info("OSS purity check passed: No execution capabilities detected")
        return True


# Backward compatibility exports
__all__ = [
    "MCPServer",
    "MCPMode",
    "MCPRequest",
    "MCPResponse",
    "MCPRequestStatus",
    "ToolContext",
    "ToolResult",
    "ValidationResult",
    "MCPTool",
    "BaseMCPTool",
    "RollbackTool",
    "RestartContainerTool",
    "ScaleOutTool",
    "create_circuit_breaker_tool",
    "create_traffic_shift_tool",
    "create_alert_tool",
]

# For backward compatibility
create_mcp_server = MCPServer
