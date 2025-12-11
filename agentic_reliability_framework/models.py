"""
Data Models for Enterprise Agentic Reliability Framework
Fixed version with security patches and validation improvements
"""

from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict
from typing import Optional, List, Literal
from enum import Enum
from datetime import datetime, timezone
import hashlib
import re


class EventSeverity(Enum):
    """Event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealingAction(Enum):
    """Available healing actions for policy engine"""
    RESTART_CONTAINER = "restart_container"
    SCALE_OUT = "scale_out"
    TRAFFIC_SHIFT = "traffic_shift"
    CIRCUIT_BREAKER = "circuit_breaker"
    ROLLBACK = "rollback"
    ALERT_TEAM = "alert_team"
    NO_ACTION = "no_action"


class HealthStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class PolicyCondition(BaseModel):
    """
    Structured policy condition - replaces Dict[str, Any]
    Provides type safety and validation
    """
    metric: Literal["latency_p99", "error_rate", "cpu_util", "memory_util", "throughput"]
    operator: Literal["gt", "lt", "eq", "gte", "lte"]
    threshold: float = Field(ge=0)
    
    model_config = ConfigDict(frozen=True)


class ReliabilityEvent(BaseModel):
    """
    Core reliability event model with comprehensive validation
    
    SECURITY FIX: Changed timestamp from str to datetime
    SECURITY FIX: Changed fingerprint from MD5 to SHA-256
    IMPROVEMENT: Added frozen=True for immutability
    IMPROVEMENT: Added validators for all fields
    """
    
    # FIXED: timestamp is now datetime instead of string
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp in UTC"
    )
    
    component: str = Field(
        min_length=1,
        max_length=255,
        description="Component identifier (alphanumeric and hyphens only)"
    )
    
    service_mesh: str = Field(
        default="default",
        min_length=1,
        max_length=100
    )
    
    # Metrics with proper bounds
    latency_p99: float = Field(
        ge=0,
        lt=300000,  # 5 minutes max
        description="P99 latency in milliseconds"
    )
    
    error_rate: float = Field(
        ge=0,
        le=1,
        description="Error rate between 0 and 1"
    )
    
    throughput: float = Field(
        ge=0,
        description="Requests per second"
    )
    
    cpu_util: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="CPU utilization (0-1)"
    )
    
    memory_util: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Memory utilization (0-1)"
    )
    
    revenue_impact: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated revenue impact in dollars"
    )
    
    user_impact: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of affected users"
    )
    
    upstream_deps: List[str] = Field(
        default_factory=list,
        description="List of upstream dependencies"
    )
    
    downstream_deps: List[str] = Field(
        default_factory=list,
        description="List of downstream dependencies"
    )
    
    severity: EventSeverity = EventSeverity.LOW
    
    # FIXED: Frozen model means no mutable fingerprint field
    # Use computed_field instead
    
    model_config = ConfigDict(
        frozen=True,  # Immutability for data integrity
        validate_assignment=True
    )
    
    @field_validator("component")
    @classmethod
    def validate_component_id(cls, v: str) -> str:
        """Validate component ID format (alphanumeric and hyphens only)"""
        if not re.match(r"^[a-z0-9-]+$", v):
            raise ValueError(
                "Component ID must contain only lowercase letters, numbers, and hyphens"
            )
        return v
    
    @field_validator("upstream_deps", "downstream_deps")
    @classmethod
    def validate_dependency_format(cls, v: List[str]) -> List[str]:
        """Validate dependency names"""
        for dep in v:
            if not re.match(r"^[a-z0-9-]+$", dep):
                raise ValueError(
                    f"Dependency '{dep}' must contain only lowercase letters, numbers, and hyphens"
                )
        return v
    
    @computed_field  # FIXED: Use computed_field instead of __init__ override
    @property
    def fingerprint(self) -> str:
        """
        Generate deterministic fingerprint for event deduplication
        
        SECURITY FIX: Changed from MD5 to SHA-256
        IMPROVEMENT: Removed timestamp from fingerprint for determinism
        """
        components = [
            self.component,
            self.service_mesh,
            f"{self.latency_p99:.2f}",
            f"{self.error_rate:.4f}",
            f"{self.throughput:.2f}"
        ]
        
        fingerprint_str = ":".join(components)
        
        # SECURITY FIX: SHA-256 instead of MD5
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    def model_post_init(self, __context) -> None:
        """Validate cross-field constraints after initialization"""
        # Check for circular dependencies
        upstream_set = set(self.upstream_deps)
        downstream_set = set(self.downstream_deps)
        
        circular = upstream_set & downstream_set
        if circular:
            raise ValueError(
                f"Circular dependencies detected: {circular}. "
                "A component cannot be both upstream and downstream."
            )


class HealingPolicy(BaseModel):
    """
    Policy definition for automated healing actions
    
    IMPROVEMENT: Changed conditions from Dict[str, Any] to List[PolicyCondition]
    """
    
    name: str = Field(
        min_length=1,
        max_length=255,
        description="Policy name"
    )
    
    # FIXED: Structured conditions instead of Dict[str, Any]
    conditions: List[PolicyCondition] = Field(
        min_length=1,
        description="List of conditions (all must match)"
    )
    
    actions: List[HealingAction] = Field(
        min_length=1,
        description="Actions to execute when policy triggers"
    )
    
    priority: int = Field(
        ge=1,
        le=5,
        default=3,
        description="Policy priority (1=highest, 5=lowest)"
    )
    
    cool_down_seconds: int = Field(
        ge=0,
        default=300,
        description="Cooldown period between executions"
    )
    
    enabled: bool = Field(
        default=True,
        description="Whether policy is active"
    )
    
    max_executions_per_hour: int = Field(
        ge=1,
        default=10,
        description="Rate limit: max executions per hour"
    )
    
    model_config = ConfigDict(frozen=True)


class AnomalyResult(BaseModel):
    """Result from anomaly detection"""
    
    is_anomaly: bool
    confidence: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)
    affected_metrics: List[str] = Field(default_factory=list)
    detection_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    model_config = ConfigDict(frozen=True)


class ForecastResult(BaseModel):
    """Result from predictive forecasting"""
    
    metric: str
    predicted_value: float
    confidence: float = Field(ge=0, le=1)
    trend: Literal["increasing", "decreasing", "stable"]
    time_to_threshold: Optional[float] = Field(
        default=None,
        description="Minutes until threshold breach"
    )
    risk_level: Literal["low", "medium", "high", "critical"]
    forecast_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    model_config = ConfigDict(frozen=True)