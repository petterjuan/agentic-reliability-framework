"""
Enhanced MCP Server for ARF v3 - OSS Edition
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
    """Immutable context for tool execution"""
    component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: str = "production"
    metadata: Dict[str, Any] = field(default_factory=dict)
    safety_guardrails: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Immutable result of tool execution"""
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def success_result(cls, message: str, **details: Any) -> "ToolResult":
        """Create a successful result"""
        return cls(success=True, message=message, details=details)

    @classmethod
    def failure_result(cls, message: str, **details: Any) -> "ToolResult":
        """Create a failure result"""
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
    """Protocol for MCP tools"""

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata"""
        ...

    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute the tool"""
        ...

    def validate(self, context: ToolContext) -> ValidationResult:
        """Validate the tool execution"""
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
        }

    def _add_safety_check(
        self,
        validation: ValidationResult,
        name: str,
        passed: bool,
        details: str = ""
    ) -> ValidationResult:
        """Helper to add safety checks to validation result"""
        # Create a copy of safety_checks dict and update it
        safety_checks = dict(validation.safety_checks)
        safety_checks[name] = SafetyCheck(
            name=name,
            passed=passed,
            details=details
        )

        # Create a new ValidationResult with updated safety_checks
        return ValidationResult(
            valid=validation.valid,
            errors=validation.errors.copy(),
            warnings=validation.warnings.copy(),
            safety_checks=safety_checks
        )


class RollbackTool(BaseMCPTool):
    """K8s/ECS/VM rollback adapter with enhanced safety"""

    def __init__(self):
        super().__init__({
            "name": "rollback",
            "description": "Rollback deployment to previous version",
            "supported_environments": ["kubernetes", "ecs", "vm"],
            "safety_level": "high",
            "timeout_seconds": 60,
            "required_permissions": ["deployment.write", "rollback.execute"]
        })

    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute rollback with proper error handling"""
        start_time = time.time()

        try:
            # Simulate different environment executions
            if context.environment == "kubernetes":
                result = await self._k8s_rollback(context)
            elif context.environment == "ecs":
                result = await self._ecs_rollback(context)
            elif context.environment == "vm":
                result = await self._vm_rollback(context)
            else:
                return ToolResult.failure_result(
                    f"Unsupported environment: {context.environment}",
                    supported_environments=self.metadata["supported_environments"]
                )

            # Update execution time
            return ToolResult(
                success=result.success,
                message=result.message,
                details=result.details,
                execution_time_seconds=time.time() - start_time,
                warnings=result.warnings
            )

        except asyncio.TimeoutError:
            return ToolResult.failure_result(
                f"Rollback timeout after {self.metadata['timeout_seconds']} seconds"
            )
        except Exception as e:
            logger.exception(f"Rollback execution error: {e}")
            return ToolResult.failure_result(
                f"Rollback failed: {str(e)}",
                error_type=type(e).__name__
            )

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

    async def _k8s_rollback(self, context: ToolContext) -> ToolResult:
        """Execute Kubernetes rollback"""
        await asyncio.sleep(1)  # Simulate API call
        return ToolResult.success_result(
            f"Successfully rolled back {context.component} in Kubernetes",
            action="k8s_rollback",
            component=context.component,
            namespace=context.metadata.get("namespace", "default"),
            deployment=context.metadata.get("deployment"),
            previous_revision=context.metadata.get("previous_revision"),
            new_revision=context.metadata.get("new_revision")
        )

    async def _ecs_rollback(self, context: ToolContext) -> ToolResult:
        """Execute ECS rollback"""
        await asyncio.sleep(1)
        return ToolResult.success_result(
            f"Successfully rolled back {context.component} in ECS",
            action="ecs_rollback",
            component=context.component,
            cluster=context.metadata.get("cluster"),
            service=context.metadata.get("service"),
            task_definition=context.metadata.get("task_definition")
        )

    async def _vm_rollback(self, context: ToolContext) -> ToolResult:
        """Execute VM rollback"""
        await asyncio.sleep(1)
        return ToolResult.success_result(
            f"Successfully rolled back {context.component} on VM",
            action="vm_rollback",
            component=context.component,
            host=context.metadata.get("host"),
            snapshot_id=context.metadata.get("snapshot_id")
        )


class RestartContainerTool(BaseMCPTool):
    """Container restart tool with safety limits"""

    def __init__(self):
        super().__init__({
            "name": "restart_container",
            "description": "Restart container instance",
            "supported_environments": ["kubernetes", "ecs", "docker"],
            "safety_level": "medium",
            "timeout_seconds": 30,
            "required_permissions": ["container.restart"]
        })

    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute container restart"""
        start_time = time.time()

        try:
            await asyncio.sleep(0.5)  # Simulate API call

            return ToolResult.success_result(
                f"Successfully restarted {context.component}",
                action="restart_container",
                component=context.component,
                environment=context.environment,
                container_id=context.metadata.get("container_id"),
                execution_time_seconds=time.time() - start_time
            )
        except Exception as e:
            logger.exception(f"Container restart error: {e}")
            return ToolResult.failure_result(
                f"Container restart failed: {str(e)}",
                execution_time_seconds=time.time() - start_time
            )

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
    """Scale out tool with resource limits"""

    def __init__(self):
        super().__init__({
            "name": "scale_out",
            "description": "Scale out service instances",
            "supported_environments": ["kubernetes", "ecs"],
            "safety_level": "low",
            "timeout_seconds": 45,
            "required_permissions": ["deployment.scale"]
        })

    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute scale out"""
        start_time = time.time()

        try:
            scale_factor = context.parameters.get("scale_factor", 2)
            await asyncio.sleep(1)  # Simulate API call

            return ToolResult.success_result(
                f"Successfully scaled {context.component} by factor {scale_factor}",
                action="scale_out",
                component=context.component,
                scale_factor=scale_factor,
                current_replicas=context.metadata.get("current_replicas"),
                new_replicas=context.metadata.get("new_replicas"),
                execution_time_seconds=time.time() - start_time
            )
        except Exception as e:
            logger.exception(f"Scale out error: {e}")
            return ToolResult.failure_result(
                f"Scale out failed: {str(e)}",
                execution_time_seconds=time.time() - start_time
            )

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
    """Factory function for circuit breaker tool"""

    class CircuitBreakerTool(BaseMCPTool):
        def __init__(self):
            super().__init__({
                "name": "circuit_breaker",
                "description": "Enable circuit breaker for service",
                "supported_environments": ["all"],
                "safety_level": "low",
                "timeout_seconds": 10,
                "required_permissions": ["circuit_breaker.manage"]
            })

        async def execute(self, context: ToolContext) -> ToolResult:
            await asyncio.sleep(0.1)
            return ToolResult.success_result(
                f"Circuit breaker enabled for {context.component}",
                action="circuit_breaker",
                component=context.component
            )

        def validate(self, context: ToolContext) -> ValidationResult:
            return ValidationResult.valid_result()

    return CircuitBreakerTool()


def create_traffic_shift_tool() -> MCPTool:
    """Factory function for traffic shift tool"""

    class TrafficShiftTool(BaseMCPTool):
        def __init__(self):
            super().__init__({
                "name": "traffic_shift",
                "description": "Shift traffic to canary or backup",
                "supported_environments": ["kubernetes", "ecs", "load_balancer"],
                "safety_level": "medium",
                "timeout_seconds": 30,
                "required_permissions": ["traffic.manage"]
            })

        async def execute(self, context: ToolContext) -> ToolResult:
            await asyncio.sleep(0.5)
            return ToolResult.success_result(
                f"Traffic shifted for {context.component}",
                action="traffic_shift",
                component=context.component
            )

        def validate(self, context: ToolContext) -> ValidationResult:
            return ValidationResult.valid_result()

    return TrafficShiftTool()


def create_alert_tool() -> MCPTool:
    """Factory function for alert tool"""

    class AlertTool(BaseMCPTool):
        def __init__(self):
            super().__init__({
                "name": "alert_team",
                "description": "Alert human team for intervention",
                "supported_environments": ["all"],
                "safety_level": "low",
                "timeout_seconds": 5,
                "required_permissions": ["alert.create"]
            })

        async def execute(self, context: ToolContext) -> ToolResult:
            await asyncio.sleep(0.1)
            return ToolResult.success_result(
                f"Alert sent for {context.component}",
                action="alert_team",
                component=context.component
            )

        def validate(self, context: ToolContext) -> ValidationResult:
            return ValidationResult.valid_result()

    return AlertTool()


# ========== OSS INTEGRATION IMPORT ==========
# Import OSSMCPClient for advisory mode delegation
try:
    from arf_core.engine.oss_mcp_client import create_oss_mcp_client
    OSS_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OSSMCPClient not available: {e}. Using fallback advisory mode.")
    OSS_CLIENT_AVAILABLE = False


# ========== MCP SERVER (OSS EDITION) ==========

class MCPServer:
    """
    Enhanced MCP Server - OSS Edition
    Advisory mode only, no execution capability
    
    Features:
    - Thread-safe operations
    - Comprehensive error handling
    - Detailed metrics and monitoring
    - Extensible tool system
    - OSS Edition: Advisory mode only with HealingIntent integration
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
        self._approval_requests: Dict[str, MCPRequest] = {}
        self._execution_history: Deque[Dict[str, Any]] = deque(maxlen=1000)

        # Metrics
        self._start_time = time.time()
        self._tool_stats: DefaultDict[str, ExecutionStats] = defaultdict(
            lambda: {"total": 0, "successful": 0, "failed": 0,
                     "average_duration_seconds": 0.0, "last_execution": None}
        )

        logger.info(f"Initialized OSS MCPServer in advisory mode")

    def _register_tools(self) -> Dict[str, MCPTool]:
        """Register all available tools"""
        tools: Dict[str, MCPTool] = {
            "rollback": RollbackTool(),
            "restart_container": RestartContainerTool(),
            "scale_out": ScaleOutTool(),
            "circuit_breaker": create_circuit_breaker_tool(),
            "traffic_shift": create_traffic_shift_tool(),
            "alert_team": create_alert_tool(),
        }

        logger.info(f"Registered {len(tools)} tools: {list(tools.keys())}")
        return tools

    @asynccontextmanager
    async def _execution_context(self, request: MCPRequest) -> AsyncGenerator[None, None]:
        """Context manager for tool execution with metrics"""
        start_time = time.time()
        try:
            yield
        finally:
            # Update execution time for stats
            execution_time = time.time() - start_time
            stats = self._tool_stats[request.tool]
            stats["total"] += 1
            stats["average_duration_seconds"] = (
                (stats["average_duration_seconds"] * (stats["total"] - 1) + execution_time)
                / stats["total"]
            )
            stats["last_execution"] = time.time()

    async def execute_tool(self, request_dict: Dict[str, Any]) -> MCPResponse:
        """
        Execute a tool with comprehensive safety checks
        
        OSS Edition: Advisory only, no execution
        """
        # OSS EDITION: Force advisory mode
        request_dict["mode"] = "advisory"
        
        # === EXISTING EXECUTE_TOOL CODE ===
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
                f"Internal server error: {str(e)}"
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

    async def _handle_advisory_mode(self, request: MCPRequest) -> MCPResponse:
        """
        Handle advisory mode by delegating to OSSMCPClient
        
        OSS Edition: Uses OSSMCPClient for proper HealingIntent creation
        and RAG integration. Maintains backward compatibility while
        ensuring OSS purity.
        """
        # Try to use OSSMCPClient if available
        if self.oss_client is not None:
            try:
                # Delegate to OSSMCPClient for proper OSS analysis
                request_dict = {
                    "tool": request.tool,
                    "component": request.component,
                    "parameters": request.parameters,
                    "justification": request.justification,
                    "metadata": request.metadata,
                    "request_id": request.request_id,
                }
                
                # Get OSS analysis result
                oss_response = await self.oss_client.execute_tool(request_dict)
                
                # Convert OSSMCPResponse to MCPResponse for backward compatibility
                return MCPResponse(
                    request_id=oss_response.get("request_id", request.request_id),
                    status=MCPRequestStatus.COMPLETED,
                    message=oss_response.get("message", f"OSS Analysis: Recommend {request.tool} for {request.component}"),
                    executed=False,
                    result=oss_response.get("result", {})
                )
                
            except Exception as e:
                logger.warning(f"OSSMCPClient failed, using fallback: {e}")
                # Continue to fallback implementation
        
        # Fallback to basic advisory response if OSSMCPClient is not available
        return MCPResponse(
            request_id=request.request_id,
            status=MCPRequestStatus.COMPLETED,
            message=f"Advisory: Would execute {request.tool} on {request.component}",
            executed=False,
            result={
                "mode": "advisory",
                "would_execute": True,
                "justification": request.justification,
                "validation": "Basic checks passed",
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
                "oss_fallback": True,  # Indicate this is a fallback response
                "healing_intent_available": False,  # No HealingIntent in fallback
            }
        )

    async def approve_request(
        self,
        approval_id: str,
        approved: bool = True,
        comment: str = ""
    ) -> MCPResponse:
        """
        Approve or reject a pending request
        
        OSS EDITION: Always returns advisory response
        """
        # OSS EDITION: No approval workflow
        return MCPResponse(
            request_id=approval_id,
            status=MCPRequestStatus.REJECTED,
            message="Approval workflow requires Enterprise edition",
            executed=False,
            result={
                "requires_enterprise": True,
                "upgrade_url": "https://arf.dev/enterprise"
            }
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

        stats = {
            "mode": self.mode.value,
            "edition": "oss",
            "oss_restricted": True,
            "registered_tools": len(self.registered_tools),
            "active_cooldowns": len(self._cooldowns),
            "pending_approvals": len(self._approval_requests),
            "execution_history_count": len(self._execution_history),
            "tool_statistics": dict(self._tool_stats),
            "uptime_seconds": time.time() - self._start_time,
            "safety_guardrails": self.safety_guardrails,
            "engine_available": engine is not None,
            "engine_type": getattr(engine, "__class__.__name__", "unknown") if engine else None,
            "config": {
                "mcp_mode": config.mcp_mode,
                "mcp_enabled": config.mcp_enabled,
                "mpc_cooldown_seconds": config.mpc_cooldown_seconds,
            },
            "oss_limits": {
                "max_incidents": 1000,
                "execution_allowed": False,
                "mode": "advisory"
            },
            "oss_integration": {
                "using_oss_client": self.oss_client is not None,
                "healing_intent_support": self.oss_client is not None,
                "rag_integration": oss_client_info.get("metrics", {}).get("rag_queries_performed", 0) > 0 if oss_client_info else False,
                "oss_client_available": OSS_CLIENT_AVAILABLE,
            }
        }
        
        # Add OSS client metrics if available
        if oss_client_info:
            stats["oss_client_metrics"] = oss_client_info.get("metrics", {})
            stats["oss_client_cache_size"] = oss_client_info.get("cache_size", 0)

        return stats

    def get_tool_info(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about tools"""
        # Try to use OSSMCPClient for tool info if available
        if self.oss_client is not None:
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

        return {
            name: {
                **tool.get_tool_info(),
                "oss_edition": True,
                "can_execute": False,
                "requires_enterprise": True,
            }
            for name, tool in self.registered_tools.items()
        }

    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return list(self._execution_history)[-limit:]

    def reset_stats(self) -> None:
        """Reset server statistics"""
        self._tool_stats.clear()
        self._execution_history.clear()
        self._cooldowns.clear()
        self._approval_requests.clear()
        logger.info("OSS MCP server statistics reset")


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
