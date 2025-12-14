"""
MCP Server Prototype for ARF v3

Phase 2: MCP Server Implementation (2-3 weeks)
Goal: Create explicit execution boundary
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import uuid

from ..config import config
from ..models import HealingAction

logger = logging.getLogger(__name__)


class MCPMode(Enum):
    """MCP execution modes"""
    ADVISORY = "advisory"  # OSS default - no execution
    APPROVAL = "approval"  # Human-in-loop
    AUTONOMOUS = "autonomous"  # Enterprise - with guardrails


class MCPRequestStatus(Enum):
    """MCP request status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class MCPRequest:
    """MCP request model"""
    request_id: str
    tool: str
    component: str
    parameters: Dict[str, Any]
    justification: str
    mode: MCPMode
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MCPResponse:
    """MCP response model"""
    request_id: str
    status: MCPRequestStatus
    message: str
    executed: bool = False
    result: Optional[Dict[str, Any]] = None
    approval_id: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ToolContext:
    """Context for tool execution"""
    component: str
    parameters: Dict[str, Any]
    environment: str = "production"
    metadata: Dict[str, Any] = None
    safety_guardrails: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.safety_guardrails is None:
            self.safety_guardrails = {}


@dataclass
class ToolResult:
    """Result of tool execution"""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time_seconds: float = 0.0
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class ValidationResult:
    """Result of tool validation"""
    valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    safety_checks: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.safety_checks is None:
            self.safety_checks = {}


class MCPTool(ABC):
    """
    Abstract tool interface
    
    V3 Design: All healing actions implement this interface
    """
    
    @abstractmethod
    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute the tool"""
        pass
    
    @abstractmethod
    def validate(self, context: ToolContext) -> ValidationResult:
        """Validate the tool execution"""
        pass
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": self.__class__.__name__,
            "description": getattr(self, "description", "No description"),
            "supported_environments": getattr(self, "supported_environments", []),
            "safety_level": getattr(self, "safety_level", "medium"),
            "timeout_seconds": getattr(self, "timeout_seconds", 30),
        }


class RollbackTool(MCPTool):
    """K8s/ECS/VM rollback adapter"""
    
    def __init__(self):
        self.description = "Rollback deployment to previous version"
        self.supported_environments = ["kubernetes", "ecs", "vm"]
        self.safety_level = "high"  # High risk - can cause downtime
        self.timeout_seconds = 60
    
    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute rollback"""
        start_time = time.time()
        
        try:
            if context.environment == "kubernetes":
                result = await self._k8s_rollback(context)
            elif context.environment == "ecs":
                result = await self._ecs_rollback(context)
            elif context.environment == "vm":
                result = await self._vm_rollback(context)
            else:
                return ToolResult(
                    success=False,
                    message=f"Unsupported environment: {context.environment}",
                    execution_time_seconds=time.time() - start_time
                )
            
            result.execution_time_seconds = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Rollback execution error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                message=f"Rollback failed: {str(e)}",
                execution_time_seconds=time.time() - start_time
            )
    
    def validate(self, context: ToolContext) -> ValidationResult:
        """Validate rollback"""
        errors = []
        warnings = []
        safety_checks = {}
        
        # Check: Is this production?
        if context.metadata.get("environment", "production") == "production":
            warnings.append("Rollback requested in production environment")
            safety_checks["production_environment"] = False
        else:
            safety_checks["production_environment"] = True
        
        # Check: Are there canaries?
        if not context.metadata.get("has_canary", False):
            warnings.append("No canary deployment detected")
            safety_checks["has_canary"] = False
        else:
            safety_checks["has_canary"] = True
        
        # Check: Is there a healthy revision?
        if not context.metadata.get("has_healthy_revision", False):
            errors.append("No healthy revision available for rollback")
            safety_checks["has_healthy_revision"] = False
        else:
            safety_checks["has_healthy_revision"] = True
        
        # Check blast radius
        affected_services = context.metadata.get("affected_services", [])
        if len(affected_services) > config.safety_guardrails["max_blast_radius"]:
            errors.append(
                f"Blast radius too large: {len(affected_services)} services "
                f"(max: {config.safety_guardrails['max_blast_radius']})"
            )
            safety_checks["blast_radius"] = False
        else:
            safety_checks["blast_radius"] = True
        
        # Check action blacklist
        if "ROLLBACK" in config.safety_guardrails["action_blacklist"]:
            errors.append("Rollback is in the safety blacklist")
            safety_checks["not_blacklisted"] = False
        else:
            safety_checks["not_blacklisted"] = True
        
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            safety_checks=safety_checks
        )
    
    async def _k8s_rollback(self, context: ToolContext) -> ToolResult:
        """Execute Kubernetes rollback"""
        # Placeholder - in production, implement actual k8s API calls
        await asyncio.sleep(1)  # Simulate API call
        
        return ToolResult(
            success=True,
            message=f"Successfully rolled back {context.component} in Kubernetes",
            details={
                "action": "k8s_rollback",
                "component": context.component,
                "namespace": context.metadata.get("namespace", "default"),
                "deployment": context.metadata.get("deployment", "unknown"),
                "previous_revision": context.metadata.get("previous_revision"),
                "new_revision": context.metadata.get("new_revision")
            }
        )
    
    async def _ecs_rollback(self, context: ToolContext) -> ToolResult:
        """Execute ECS rollback"""
        await asyncio.sleep(1)  # Simulate API call
        
        return ToolResult(
            success=True,
            message=f"Successfully rolled back {context.component} in ECS",
            details={
                "action": "ecs_rollback",
                "component": context.component,
                "cluster": context.metadata.get("cluster", "default"),
                "service": context.metadata.get("service", "unknown"),
                "task_definition": context.metadata.get("task_definition")
            }
        )
    
    async def _vm_rollback(self, context: ToolContext) -> ToolResult:
        """Execute VM rollback"""
        await asyncio.sleep(1)  # Simulate API call
        
        return ToolResult(
            success=True,
            message=f"Successfully rolled back {context.component} on VM",
            details={
                "action": "vm_rollback",
                "component": context.component,
                "host": context.metadata.get("host", "unknown"),
                "snapshot_id": context.metadata.get("snapshot_id")
            }
        )


class RestartContainerTool(MCPTool):
    """Container restart tool"""
    
    def __init__(self):
        self.description = "Restart container"
        self.supported_environments = ["kubernetes", "ecs", "docker"]
        self.safety_level = "medium"
        self.timeout_seconds = 30
    
    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute container restart"""
        start_time = time.time()
        
        try:
            await asyncio.sleep(0.5)  # Simulate API call
            
            return ToolResult(
                success=True,
                message=f"Successfully restarted {context.component}",
                details={
                    "action": "restart_container",
                    "component": context.component,
                    "environment": context.environment,
                    "container_id": context.metadata.get("container_id")
                },
                execution_time_seconds=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Container restart failed: {str(e)}",
                execution_time_seconds=time.time() - start_time
            )
    
    def validate(self, context: ToolContext) -> ValidationResult:
        """Validate container restart"""
        errors = []
        warnings = []
        safety_checks = {}
        
        # Check restart count
        restart_count = context.metadata.get("restart_count", 0)
        if restart_count > 3:
            warnings.append(f"High restart count: {restart_count}")
            safety_checks["reasonable_restart_count"] = False
        else:
            safety_checks["reasonable_restart_count"] = True
        
        # Check if container is healthy
        if not context.metadata.get("container_healthy", True):
            errors.append("Container is not healthy")
            safety_checks["container_healthy"] = False
        else:
            safety_checks["container_healthy"] = True
        
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            safety_checks=safety_checks
        )


class ScaleOutTool(MCPTool):
    """Scale out tool"""
    
    def __init__(self):
        self.description = "Scale out service"
        self.supported_environments = ["kubernetes", "ecs"]
        self.safety_level = "low"
        self.timeout_seconds = 45
    
    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute scale out"""
        start_time = time.time()
        
        try:
            await asyncio.sleep(1)  # Simulate API call
            
            scale_factor = context.parameters.get("scale_factor", 2)
            
            return ToolResult(
                success=True,
                message=f"Successfully scaled {context.component} by factor {scale_factor}",
                details={
                    "action": "scale_out",
                    "component": context.component,
                    "scale_factor": scale_factor,
                    "current_replicas": context.metadata.get("current_replicas"),
                    "new_replicas": context.metadata.get("new_replicas")
                },
                execution_time_seconds=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Scale out failed: {str(e)}",
                execution_time_seconds=time.time() - start_time
            )
    
    def validate(self, context: ToolContext) -> ValidationResult:
        """Validate scale out"""
        errors = []
        warnings = []
        safety_checks = {}
        
        # Check scale factor
        scale_factor = context.parameters.get("scale_factor", 1)
        if scale_factor > 10:
            errors.append(f"Scale factor too high: {scale_factor} (max: 10)")
            safety_checks["reasonable_scale_factor"] = False
        else:
            safety_checks["reasonable_scale_factor"] = True
        
        # Check resource limits
        current_replicas = context.metadata.get("current_replicas", 1)
        max_replicas = context.metadata.get("max_replicas", 20)
        
        new_replicas = current_replicas * scale_factor
        if new_replicas > max_replicas:
            errors.append(
                f"Scale would exceed max replicas: {new_replicas} > {max_replicas}"
            )
            safety_checks["within_resource_limits"] = False
        else:
            safety_checks["within_resource_limits"] = True
        
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            safety_checks=safety_checks
        )


class MCPServer:
    """
    Governed execution plane
    
    V3 Design Mandate 2: Explicit execution boundary (MCP server required)
    
    Stateless REST/gRPC service that wraps healing actions
    """
    
    def __init__(self, mode: MCPMode = None):
        """
        Initialize MCP Server
        
        Args:
            mode: Execution mode (defaults to config.mcp_mode)
        """
        self.mode = mode or MCPMode(config.mcp_mode)
        self.registered_tools = self._register_tools()
        self.cooldowns: Dict[str, float] = {}  # component:tool -> timestamp
        self.approval_requests: Dict[str, MCPRequest] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Safety guardrails
        self.safety_guardrails = config.safety_guardrails
        
        logger.info(f"Initialized MCPServer in {self.mode.value} mode")
    
    def _register_tools(self) -> Dict[str, MCPTool]:
        """Register all available tools"""
        tools = {
            "rollback": RollbackTool(),
            "restart_container": RestartContainerTool(),
            "scale_out": ScaleOutTool(),
            "circuit_breaker": self._create_circuit_breaker_tool(),
            "traffic_shift": self._create_traffic_shift_tool(),
            "alert_team": self._create_alert_tool(),
        }
        
        logger.info(f"Registered {len(tools)} tools")
        return tools
    
    def _create_circuit_breaker_tool(self) -> MCPTool:
        """Create circuit breaker tool"""
        tool = MCPTool()
        tool.description = "Enable circuit breaker for service"
        tool.supported_environments = ["all"]
        tool.safety_level = "low"
        tool.timeout_seconds = 10
        
        tool.execute = self._circuit_breaker_execute
        tool.validate = self._circuit_breaker_validate
        
        return tool
    
    def _create_traffic_shift_tool(self) -> MCPTool:
        """Create traffic shift tool"""
        tool = MCPTool()
        tool.description = "Shift traffic to canary or backup"
        tool.supported_environments = ["kubernetes", "ecs", "load_balancer"]
        tool.safety_level = "medium"
        tool.timeout_seconds = 30
        
        tool.execute = self._traffic_shift_execute
        tool.validate = self._traffic_shift_validate
        
        return tool
    
    def _create_alert_tool(self) -> MCPTool:
        """Create alert tool"""
        tool = MCPTool()
        tool.description = "Alert human team"
        tool.supported_environments = ["all"]
        tool.safety_level = "low"
        tool.timeout_seconds = 5
        
        tool.execute = self._alert_execute
        tool.validate = self._alert_validate
        
        return tool
    
    async def _circuit_breaker_execute(self, context: ToolContext) -> ToolResult:
        """Execute circuit breaker"""
        await asyncio.sleep(0.1)
        return ToolResult(
            success=True,
            message=f"Circuit breaker enabled for {context.component}",
            details={"action": "circuit_breaker", "component": context.component}
        )
    
    def _circuit_breaker_validate(self, context: ToolContext) -> ValidationResult:
        """Validate circuit breaker"""
        return ValidationResult(valid=True)
    
    async def _traffic_shift_execute(self, context: ToolContext) -> ToolResult:
        """Execute traffic shift"""
        await asyncio.sleep(0.5)
        return ToolResult(
            success=True,
            message=f"Traffic shifted for {context.component}",
            details={"action": "traffic_shift", "component": context.component}
        )
    
    def _traffic_shift_validate(self, context: ToolContext) -> ValidationResult:
        """Validate traffic shift"""
        return ValidationResult(valid=True)
    
    async def _alert_execute(self, context: ToolContext) -> ToolResult:
        """Execute alert"""
        await asyncio.sleep(0.1)
        return ToolResult(
            success=True,
            message=f"Alert sent for {context.component}",
            details={"action": "alert_team", "component": context.component}
        )
    
    def _alert_validate(self, context: ToolContext) -> ValidationResult:
        """Validate alert"""
        return ValidationResult(valid=True)
    
    async def execute_tool(self, request_dict: Dict[str, Any]) -> MCPResponse:
        """
        Single entry point for all tool execution
        
        Enforces: permissions, cooldowns, blast radius, safety checks
        
        Args:
            request_dict: MCP request as dictionary
            
        Returns:
            MCPResponse with execution result
        """
        # 1. Create request object
        request = self._create_request(request_dict)
        
        # 2. Validate request
        validation_result = self._validate_request(request)
        if not validation_result["valid"]:
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.REJECTED,
                message=f"Invalid request: {validation_result['errors']}",
                executed=False
            )
        
        # 3. Check permissions (placeholder - implement based on your auth system)
        if not self._check_permissions(request):
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.REJECTED,
                message="Permission denied",
                executed=False
            )
        
        # 4. Check cooldowns
        cooldown_check = self._check_cooldown(request.tool, request.component)
        if not cooldown_check["allowed"]:
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.REJECTED,
                message=f"In cooldown period: {cooldown_check['remaining']:.0f}s remaining",
                executed=False
            )
        
        # 5. Mode-specific handling
        if self.mode == MCPMode.ADVISORY:
            return self._handle_advisory_mode(request)
        
        elif self.mode == MCPMode.APPROVAL:
            return await self._handle_approval_mode(request)
        
        elif self.mode == MCPMode.AUTONOMOUS:
            return await self._handle_autonomous_mode(request)
        
        else:
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.REJECTED,
                message=f"Unknown mode: {self.mode}",
                executed=False
            )
    
    def _create_request(self, request_dict: Dict[str, Any]) -> MCPRequest:
        """Create MCPRequest from dictionary"""
        return MCPRequest(
            request_id=request_dict.get("request_id", str(uuid.uuid4())),
            tool=request_dict["tool"],
            component=request_dict["component"],
            parameters=request_dict.get("parameters", {}),
            justification=request_dict.get("justification", ""),
            mode=MCPMode(request_dict.get("mode", config.mcp_mode)),
            timestamp=time.time(),
            metadata=request_dict.get("metadata", {})
        )
    
    def _validate_request(self, request: MCPRequest) -> Dict[str, Any]:
        """Validate MCP request"""
        errors = []
        
        # Check if tool exists
        if request.tool not in self.registered_tools:
            errors.append(f"Unknown tool: {request.tool}")
        
        # Check if component is valid
        if not request.component or len(request.component) > 255:
            errors.append("Invalid component")
        
        # Check justification length
        if len(request.justification) < 10:
            errors.append("Justification too short (min 10 characters)")
        
        # Check parameters
        if not isinstance(request.parameters, dict):
            errors.append("Parameters must be a dictionary")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": []
        }
    
    def _check_permissions(self, request: MCPRequest) -> bool:
        """Check permissions for request"""
        # Placeholder implementation
        # In production, integrate with your authentication/authorization system
        
        # Check safety blacklist
        if request.tool.upper() in self.safety_guardrails["action_blacklist"]:
            logger.warning(f"Tool {request.tool} is in safety blacklist")
            return False
        
        # Check component permissions
        # This is a simple example - implement based on your needs
        restricted_components = ["database", "auth-service", "payment-service"]
        if request.component in restricted_components and self.mode == MCPMode.AUTONOMOUS:
            logger.warning(f"Component {request.component} requires approval in autonomous mode")
            return False
        
        return True
    
    def _check_cooldown(self, tool: str, component: str) -> Dict[str, Any]:
        """Check if tool is in cooldown period"""
        key = f"{component}:{tool}"
        current_time = time.time()
        
        if key in self.cooldowns:
            cooldown_until = self.cooldowns[key]
            remaining = cooldown_until - current_time
            
            if remaining > 0:
                return {
                    "allowed": False,
                    "remaining": remaining,
                    "cooldown_until": cooldown_until
                }
        
        return {"allowed": True, "remaining": 0}
    
    def _handle_advisory_mode(self, request: MCPRequest) -> MCPResponse:
        """Handle advisory mode (OSS default - no execution)"""
        return MCPResponse(
            request_id=request.request_id,
            status=MCPRequestStatus.COMPLETED,
            message=f"Advisory: Would execute {request.tool} on {request.component}",
            executed=False,
            result={
                "mode": "advisory",
                "would_execute": True,
                "justification": request.justification,
                "validation": "All checks passed"
            }
        )
    
    async def _handle_approval_mode(self, request: MCPRequest) -> MCPResponse:
        """Handle approval mode (human-in-loop)"""
        approval_id = str(uuid.uuid4())
        
        # Store approval request
        self.approval_requests[approval_id] = request
        
        # In production, this would trigger an approval workflow
        # (Slack, email, webhook, etc.)
        logger.info(
            f"Approval required for {request.tool} on {request.component}: "
            f"approval_id={approval_id}"
        )
        
        return MCPResponse(
            request_id=request.request_id,
            status=MCPRequestStatus.PENDING,
            message=f"Pending approval for {request.tool}",
            executed=False,
            approval_id=approval_id
        )
    
    async def _handle_autonomous_mode(self, request: MCPRequest) -> MCPResponse:
        """Handle autonomous mode (Enterprise - with guardrails)"""
        # Get tool instance
        tool = self.registered_tools.get(request.tool)
        if not tool:
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.REJECTED,
                message=f"Tool not found: {request.tool}",
                executed=False
            )
        
        # Create tool context
        context = ToolContext(
            component=request.component,
            parameters=request.parameters,
            environment=request.metadata.get("environment", "production"),
            metadata=request.metadata,
            safety_guardrails=self.safety_guardrails
        )
        
        # Validate tool execution
        validation_result = tool.validate(context)
        if not validation_result.valid:
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.REJECTED,
                message=f"Validation failed: {validation_result.errors}",
                executed=False,
                result={"validation_result": validation_result}
            )
        
        # Check safety guardrails
        safety_check = self._check_safety_guardrails(request, validation_result)
        if not safety_check["safe"]:
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.REJECTED,
                message=f"Safety check failed: {safety_check['reason']}",
                executed=False,
                result={"safety_check": safety_check}
            )
        
        # Execute tool with timeout
        try:
            result = await asyncio.wait_for(
                tool.execute(context),
                timeout=tool.timeout_seconds
            )
            
            # Update cooldown
            self._update_cooldown(request.tool, request.component)
            
            # Record execution
            self._record_execution(request, result, validation_result)
            
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.COMPLETED,
                message=result.message,
                executed=True,
                result={
                    "tool_result": result,
                    "validation_result": validation_result,
                    "safety_checks": safety_check
                }
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Tool {request.tool} timeout after {tool.timeout_seconds}s")
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.TIMEOUT,
                message=f"Tool execution timeout after {tool.timeout_seconds}s",
                executed=False
            )
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.FAILED,
                message=f"Tool execution failed: {str(e)}",
                executed=False
            )
    
    def _check_safety_guardrails(self, request: MCPRequest, validation_result: ValidationResult) -> Dict[str, Any]:
        """Check safety guardrails for autonomous execution"""
        checks = {}
        
        # Check blacklist
        checks["not_blacklisted"] = (
            request.tool.upper() not in self.safety_guardrails["action_blacklist"]
        )
        
        # Check blast radius
        affected_services = request.metadata.get("affected_services", [request.component])
        checks["blast_radius"] = (
            len(affected_services) <= self.safety_guardrails["max_blast_radius"]
        )
        
        # Check time of day (avoid production changes during business hours)
        import datetime
        now = datetime.datetime.now()
        if 9 <= now.hour <= 17:  # Business hours
            checks["safe_time"] = False
            checks["business_hours"] = True
        else:
            checks["safe_time"] = True
            checks["business_hours"] = False
        
        # Check validation safety checks
        if validation_result.safety_checks:
            checks.update(validation_result.safety_checks)
        
        # Overall safety
        safe = all(checks.values())
        
        return {
            "safe": safe,
            "reason": "All safety checks passed" if safe else "Safety checks failed",
            "checks": checks
        }
    
    def _update_cooldown(self, tool: str, component: str):
        """Update cooldown for tool"""
        key = f"{component}:{tool}"
        self.cooldowns[key] = time.time() + config.mpc_cooldown_seconds
        
        # Clean up old cooldowns
        current_time = time.time()
        expired_keys = [
            k for k, v in self.cooldowns.items()
            if current_time > v
        ]
        for k in expired_keys:
            del self.cooldowns[k]
    
    def _record_execution(self, request: MCPRequest, result: ToolResult, validation_result: ValidationResult):
        """Record execution in history"""
        execution_record = {
            "request_id": request.request_id,
            "timestamp": time.time(),
            "tool": request.tool,
            "component": request.component,
            "mode": request.mode.value,
            "success": result.success,
            "execution_time_seconds": result.execution_time_seconds,
            "validation_passed": validation_result.valid,
            "safety_checks": validation_result.safety_checks,
            "metadata": request.metadata
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    async def approve_request(self, approval_id: str, approved: bool = True, comment: str = "") -> MCPResponse:
        """
        Approve or reject a pending request
        
        Args:
            approval_id: Approval request ID
            approved: Whether to approve
            comment: Approval/rejection comment
            
        Returns:
            MCPResponse with result
        """
        if approval_id not in self.approval_requests:
            return MCPResponse(
                request_id=approval_id,
                status=MCPRequestStatus.REJECTED,
                message=f"Approval request not found: {approval_id}",
                executed=False
            )
        
        request = self.approval_requests.pop(approval_id)
        
        if not approved:
            return MCPResponse(
                request_id=request.request_id,
                status=MCPRequestStatus.REJECTED,
                message=f"Request rejected: {comment}",
                executed=False
            )
        
        # Execute the approved request
        request.mode = MCPMode.AUTONOMOUS  # Switch to autonomous for execution
        return await self._handle_autonomous_mode(request)
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get MCP server statistics"""
        return {
            "mode": self.mode.value,
            "registered_tools": len(self.registered_tools),
            "active_cooldowns": len(self.cooldowns),
            "pending_approvals": len(self.approval_requests),
            "execution_history_count": len(self.execution_history),
            "safety_guardrails": self.safety_guardrails,
            "uptime_seconds": time.time() - getattr(self, "_start_time", time.time())
        }
    
    def get_tool_info(self, tool_name: str = None) -> Dict[str, Any]:
        """Get information about tools"""
        if tool_name:
            tool = self.registered_tools.get(tool_name)
            if tool:
                return tool.get_tool_info()
            return {}
        
        return {
            tool_name: tool.get_tool_info()
            for tool_name, tool in self.registered_tools.items()
        }
