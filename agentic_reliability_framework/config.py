"""
Configuration Management for Agentic Reliability Framework - OSS EDITION
Apache 2.0 Licensed - Enterprise features require commercial license
"""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """
    OSS Edition Configuration with hard limits
    
    V3 Features (OSS Limited):
    - RAG Graph Configuration (1k incident limit)
    - MCP Server Configuration (advisory only)  
    - No Learning Loop (Enterprise only)
    - No Beta Testing (Enterprise only)
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"  # Strict mode - no extra fields allowed
    )
    
    # === API Configuration ===
    hf_api_key: str = Field(default="", description="HuggingFace API key")
    hf_api_url: str = Field(
        default="https://router.huggingface.co/hf-inference/v1/completions",
        description="HuggingFace API endpoint"
    )
    
    # === System Configuration ===
    max_events_stored: int = Field(
        default=1000, 
        description="Maximum events to store in memory (OSS limit: 1000)",
        ge=1,
        le=1000  # OSS hard limit
    )
    faiss_batch_size: int = Field(
        default=10, 
        description="FAISS batch size for async writes"
    )
    vector_dim: int = Field(
        default=384, 
        description="Vector dimension for embeddings"
    )
    
    # === Business Metrics ===
    base_revenue_per_minute: float = Field(
        default=100.0, 
        description="Base revenue per minute for impact calculation"
    )
    base_users: int = Field(
        default=1000, 
        description="Base user count for impact calculation"
    )
    
    # === Anomaly Detection Thresholds ===
    latency_critical: float = Field(
        default=300.0, 
        description="Critical latency threshold (ms)"
    )
    latency_warning: float = Field(
        default=150.0, 
        description="Warning latency threshold (ms)"
    )
    latency_extreme: float = Field(
        default=500.0, 
        description="Extreme latency threshold (ms)"
    )
    
    cpu_critical: float = Field(
        default=0.9, 
        description="Critical CPU threshold",
        ge=0.0,
        le=1.0
    )
    memory_critical: float = Field(
        default=0.9, 
        description="Critical memory threshold",
        ge=0.0,
        le=1.0
    )
    
    error_rate_critical: float = Field(
        default=0.3, 
        description="Critical error rate threshold",
        ge=0.0,
        le=1.0
    )
    error_rate_high: float = Field(
        default=0.15, 
        description="High error rate threshold",
        ge=0.0,
        le=1.0
    )
    error_rate_warning: float = Field(
        default=0.05, 
        description="Warning error rate threshold",
        ge=0.0,
        le=1.0
    )
    
    # === Forecasting Constants ===
    forecast_lookahead_minutes: int = Field(
        default=15, 
        description="Forecast lookahead in minutes"
    )
    forecast_min_data_points: int = Field(
        default=5, 
        description="Minimum data points for forecast"
    )
    slope_threshold_increasing: float = Field(
        default=5.0, 
        description="Increasing trend threshold"
    )
    slope_threshold_decreasing: float = Field(
        default=-2.0, 
        description="Decreasing trend threshold"
    )
    cache_expiry_minutes: int = Field(
        default=15, 
        description="Cache expiry in minutes"
    )
    
    # === Rate Limiting ===
    max_requests_per_minute: int = Field(
        default=60, 
        description="Maximum requests per minute"
    )
    max_requests_per_hour: int = Field(
        default=500, 
        description="Maximum requests per hour"
    )
    
    # === Logging ===
    log_level: str = Field(
        default="INFO", 
        description="Logging level"
    )
    
    # === File Paths ===
    index_file: str = Field(
        default="data/faiss_index.bin", 
        description="FAISS index file path"
    )
    incident_texts_file: str = Field(
        default="data/incident_texts.json", 
        description="FAISS incident texts file path"
    )
    
    # === v3 FEATURE FLAGS & CONFIGURATION (OSS LIMITED) ===
    
    # Phase 1: RAG Graph (OSS - Limited)
    rag_enabled: bool = Field(
        default=False, 
        description="Enable RAG Graph features"
    )
    rag_similarity_threshold: float = Field(
        default=0.3, 
        description="Minimum similarity threshold for RAG retrieval",
        ge=0.0,
        le=1.0
    )
    rag_max_incident_nodes: int = Field(
        default=1000,  # OSS HARD LIMIT
        description="Maximum incident nodes in RAG graph (OSS limit: 1000)",
        ge=1,
        le=1000  # OSS hard limit - cannot be increased
    )
    rag_max_outcome_nodes: int = Field(
        default=5000,  # OSS HARD LIMIT
        description="Maximum outcome nodes in RAG graph (OSS limit: 5000)",
        ge=1,
        le=5000  # OSS hard limit - cannot be increased
    )
    rag_cache_size: int = Field(
        default=100, 
        description="RAG similarity cache size"
    )
    rag_embedding_dim: int = Field(
        default=384, 
        description="RAG embedding dimension"
    )
    
    # Phase 2: MCP Server (OSS - Advisory Only)
    mcp_mode: str = Field(
        default="advisory", 
        description="MCP execution mode: advisory ONLY (Enterprise required for approval/autonomous)",
        pattern="^advisory$"  # Only 'advisory' allowed in OSS
    )
    mcp_enabled: bool = Field(
        default=False, 
        description="Enable MCP Server for analysis boundaries (OSS: advisory only)"
    )
    mcp_host: str = Field(
        default="localhost", 
        description="MCP Server host"
    )
    mcp_port: int = Field(
        default=8000, 
        description="MCP Server port",
        ge=1,
        le=65535
    )
    mcp_timeout_seconds: int = Field(
        default=10, 
        description="MCP request timeout"
    )
    mpc_cooldown_seconds: int = Field(
        default=60, 
        description="MCP tool cooldown period"
    )
    
    # Phase 3: Learning Loop (REMOVED - Enterprise Only)
    # No learning_enabled, learning_min_data_points, etc. in OSS
    
    # === Performance & Safety ===
    agent_timeout_seconds: int = Field(
        default=5, 
        description="Agent timeout in seconds"
    )
    circuit_breaker_failures: int = Field(
        default=3, 
        description="Circuit breaker failure threshold"
    )
    circuit_breaker_timeout: int = Field(
        default=30, 
        description="Circuit breaker recovery timeout"
    )
    
    # === Demo Mode ===
    demo_mode: bool = Field(
        default=False, 
        description="Enable demo mode with pre-configured scenarios"
    )
    
    # === Rollout Configuration (REMOVED - Enterprise Only) ===
    # No rollout_percentage or beta_testing_enabled in OSS
    
    # === Safety Guardrails ===
    safety_action_blacklist: str = Field(
        default="DATABASE_DROP,FULL_ROLLOUT,SYSTEM_SHUTDOWN",
        description="Comma-separated list of actions to never execute autonomously"
    )
    safety_max_blast_radius: int = Field(
        default=3,
        description="Maximum number of services that can be affected by an action"
    )
    safety_rag_timeout_ms: int = Field(
        default=100,
        description="RAG search timeout in milliseconds before circuit breaker"
    )
    
    # === OSS EDITION PROPERTIES ===
    
    @property
    def is_oss_edition(self) -> bool:
        """Always True for OSS configuration"""
        return True
    
    @property
    def requires_enterprise_upgrade(self) -> bool:
        """Check if configuration requires Enterprise upgrade"""
        # In OSS, only advisory mode is allowed
        if self.mcp_mode != "advisory":
            return True
        # MCP execution requires Enterprise
        if self.mcp_enabled and self.mcp_mode != "advisory":
            return True
        # RAG limits at maximum
        if self.rag_max_incident_nodes >= 1000 or self.rag_max_outcome_nodes >= 5000:
            return True
        return False
    
    @property
    def v3_features(self) -> Dict[str, Any]:
        """Get v3 feature status - OSS edition with upgrade info"""
        return {
            "rag_enabled": self.rag_enabled,
            "mcp_enabled": self.mcp_enabled,
            "learning_enabled": False,  # OSS: Always False
            "beta_testing": False,      # OSS: Always False
            "rollout_active": False,    # OSS: Always False
            "edition": "oss",
            "oss_limits": {
                "max_incident_nodes": 1000,
                "max_outcome_nodes": 5000,
                "mcp_mode": "advisory",
                "execution_allowed": False,
            },
            "enterprise_upgrade_required": self.requires_enterprise_upgrade,
            "enterprise_features": [
                "autonomous_execution",
                "approval_workflows",
                "learning_engine",
                "persistent_storage",
                "unlimited_rag_nodes",
                "audit_trails",
                "compliance_reports",
                "sso_integration",
                "24_7_support"
            ],
            "upgrade_url": "https://arf.dev/enterprise"
        }
    
    @property
    def safety_guardrails(self) -> Dict[str, Any]:
        """Get safety guardrails configuration"""
        return {
            "action_blacklist": [action.strip() for action in self.safety_action_blacklist.split(",")],
            "max_blast_radius": self.safety_max_blast_radius,
            "rag_timeout_ms": self.safety_rag_timeout_ms,
            "circuit_breaker": {
                "failures": self.circuit_breaker_failures,
                "timeout": self.circuit_breaker_timeout,
            },
            "edition": "oss",
            "execution_blocked": True  # OSS never executes
        }
    
    def validate_oss_constraints(self) -> None:
        """
        Validate OSS edition constraints
        
        Raises:
            ValueError: If configuration violates OSS boundaries
            OSSBoundaryError: If Enterprise features are detected
        """
        violations = []
        
        # Validate MCP mode (must be advisory)
        if self.mcp_mode != "advisory":
            violations.append(
                f"MCP mode must be 'advisory' in OSS edition. "
                f"Got: '{self.mcp_mode}'. "
                f"Upgrade to Enterprise for approval/autonomous modes."
            )
        
        # Validate RAG limits
        if self.rag_max_incident_nodes > 1000:
            violations.append(
                f"rag_max_incident_nodes exceeds OSS limit (1000): {self.rag_max_incident_nodes}. "
                f"Enterprise edition supports unlimited nodes."
            )
        
        if self.rag_max_outcome_nodes > 5000:
            violations.append(
                f"rag_max_outcome_nodes exceeds OSS limit (5000): {self.rag_max_outcome_nodes}. "
                f"Enterprise edition supports unlimited nodes."
            )
        
        # Validate no execution capability
        if self.mcp_enabled and self.mcp_mode != "advisory":
            violations.append(
                "MCP execution requires Enterprise edition. "
                "OSS edition only supports advisory (analysis) mode."
            )
        
        # Check for Enterprise-only environment variables
        enterprise_env_vars = [
            "LEARNING_ENABLED",
            "LEARNING_MIN_DATA_POINTS", 
            "LEARNING_CONFIDENCE_THRESHOLD",
            "LEARNING_RETENTION_DAYS",
            "ROLLOUT_PERCENTAGE",
            "BETA_TESTING_ENABLED",
        ]
        
        for env_var in enterprise_env_vars:
            if os.getenv(env_var):
                violations.append(
                    f"Environment variable {env_var} is Enterprise-only. "
                    f"Remove it or upgrade to Enterprise edition."
                )
        
        if violations:
            error_msg = (
                "OSS CONFIGURATION VIOLATIONS DETECTED:\n\n" +
                "\n".join(f"• {v}" for v in violations) +
                "\n\nUpgrade to Enterprise Edition for these features:\n"
                "https://arf.dev/enterprise\n\n"
                "Or fix configuration to comply with OSS limits."
            )
            from arf_core.constants import OSSBoundaryError
            raise OSSBoundaryError(error_msg)
    
    def get_oss_limits(self) -> Dict[str, Any]:
        """Get OSS edition limits for documentation"""
        return {
            "edition": "oss",
            "license": "Apache 2.0",
            "limits": {
                "max_events_stored": 1000,
                "rag_max_incident_nodes": 1000,
                "rag_max_outcome_nodes": 5000,
                "mcp_mode": "advisory",
                "execution_allowed": False,
                "learning_enabled": False,
                "persistent_storage": False,
            },
            "capabilities": {
                "rag_analysis": True,
                "mcp_advisory": True,
                "anomaly_detection": True,
                "business_impact": True,
                "forecasting": True,
            },
            "enterprise_upgrade_available": True,
            "upgrade_url": "https://arf.dev/enterprise"
        }
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables with OSS validation"""
        env_vars: Dict[str, Any] = {}
        
        # Map environment variables to config fields (OSS ONLY)
        field_mapping = {
            "HF_API_KEY": "hf_api_key",
            "HF_API_URL": "hf_api_url",
            "MAX_EVENTS_STORED": "max_events_stored",
            "FAISS_BATCH_SIZE": "faiss_batch_size",
            "VECTOR_DIM": "vector_dim",
            "BASE_REVENUE_PER_MINUTE": "base_revenue_per_minute",
            "BASE_USERS": "base_users",
            "MAX_REQUESTS_PER_MINUTE": "max_requests_per_minute",
            "MAX_REQUESTS_PER_HOUR": "max_requests_per_hour",
            "LOG_LEVEL": "log_level",
            "INDEX_FILE": "index_file",
            "TEXTS_FILE": "incident_texts_file",
            
            # Anomaly thresholds
            "LATENCY_CRITICAL": "latency_critical",
            "LATENCY_WARNING": "latency_warning",
            "LATENCY_EXTREME": "latency_extreme",
            "CPU_CRITICAL": "cpu_critical",
            "MEMORY_CRITICAL": "memory_critical",
            "ERROR_RATE_CRITICAL": "error_rate_critical",
            "ERROR_RATE_HIGH": "error_rate_high",
            "ERROR_RATE_WARNING": "error_rate_warning",
            
            # Forecasting
            "FORECAST_LOOKAHEAD_MINUTES": "forecast_lookahead_minutes",
            "FORECAST_MIN_DATA_POINTS": "forecast_min_data_points",
            "SLOPE_THRESHOLD_INCREASING": "slope_threshold_increasing",
            "SLOPE_THRESHOLD_DECREASING": "slope_threshold_decreasing",
            "CACHE_EXPIRY_MINUTES": "cache_expiry_minutes",
            
            # v3 Features (OSS Limited)
            "RAG_ENABLED": "rag_enabled",
            "RAG_SIMILARITY_THRESHOLD": "rag_similarity_threshold",
            "RAG_MAX_INCIDENT_NODES": "rag_max_incident_nodes",
            "RAG_MAX_OUTCOME_NODES": "rag_max_outcome_nodes",
            "RAG_CACHE_SIZE": "rag_cache_size",
            "RAG_EMBEDDING_DIM": "rag_embedding_dim",
            
            "MCP_MODE": "mcp_mode",
            "MCP_ENABLED": "mcp_enabled",
            "MCP_HOST": "mcp_host",
            "MCP_PORT": "mcp_port",
            "MCP_TIMEOUT_SECONDS": "mcp_timeout_seconds",
            "MPC_COOLDOWN_SECONDS": "mpc_cooldown_seconds",
            
            # No learning environment variables in OSS
            # No rollout environment variables in OSS
            
            "AGENT_TIMEOUT_SECONDS": "agent_timeout_seconds",
            "CIRCUIT_BREAKER_FAILURES": "circuit_breaker_failures",
            "CIRCUIT_BREAKER_TIMEOUT": "circuit_breaker_timeout",
            
            "DEMO_MODE": "demo_mode",
            
            "SAFETY_ACTION_BLACKLIST": "safety_action_blacklist",
            "SAFETY_MAX_BLAST_RADIUS": "safety_max_blast_radius",
            "SAFETY_RAG_TIMEOUT_MS": "safety_rag_timeout_ms",
        }
        
        for env_name, field_name in field_mapping.items():
            env_value = os.getenv(env_name)
            if env_value is not None:
                # Get field type from model fields
                field_info = cls.model_fields.get(field_name)
                if field_info is None:
                    # Skip fields that don't exist in OSS edition
                    print(f"⚠️  Warning: {env_name} maps to non-existent field {field_name} in OSS edition")
                    continue
                    
                field_type = field_info.annotation
                
                try:
                    if field_type is bool:
                        env_vars[field_name] = env_value.lower() in ("true", "1", "yes", "y", "t", "on", "enabled")
                    elif field_type is int:
                        env_vars[field_name] = int(env_value)
                    elif field_type is float:
                        env_vars[field_name] = float(env_value)
                    else:
                        env_vars[field_name] = env_value
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Warning: Failed to parse {env_name}={env_value}: {e}")
                    # Use default if conversion fails
                    continue
        
        # Create config instance
        config = cls(**env_vars)
        
        # Validate OSS constraints
        try:
            config.validate_oss_constraints()
        except ValueError as e:
            # Log the violation but don't crash - use defaults
            print(f"🚨 OSS Configuration Violation: {str(e)[:200]}...")
            print("🔧 Using default OSS configuration")
            # Re-create with defaults to ensure OSS compliance
            config = cls()
        
        return config


# Custom exception for OSS boundary violations
class OSSBoundaryError(ValueError):
    """Raised when OSS boundaries are violated"""
    pass


# Global configuration instance with OSS validation
config = Config.from_env()

# Validate on module load
try:
    config.validate_oss_constraints()
except ValueError as e:
    print(f"🚨 CRITICAL: OSS boundary violation on startup: {e}")
    print("🔄 Falling back to safe OSS defaults")
    config = Config()  # Use safe defaults


# Update MemoryConstants with config values
def update_memory_constants():
    """Update memory constants from config (with OSS limits)"""
    try:
        from .memory.constants import MemoryConstants
        
        # Update FAISS constants
        if hasattr(MemoryConstants, 'FAISS_BATCH_SIZE'):
            MemoryConstants.FAISS_BATCH_SIZE = config.faiss_batch_size
        if hasattr(MemoryConstants, 'VECTOR_DIM'):
            MemoryConstants.VECTOR_DIM = config.vector_dim
        
        # Update RAG constants with OSS limits
        if hasattr(MemoryConstants, 'MAX_INCIDENT_NODES'):
            # Enforce OSS limit
            MemoryConstants.MAX_INCIDENT_NODES = min(
                config.rag_max_incident_nodes, 
                1000  # OSS hard limit
            )
        if hasattr(MemoryConstants, 'MAX_OUTCOME_NODES'):
            # Enforce OSS limit
            MemoryConstants.MAX_OUTCOME_NODES = min(
                config.rag_max_outcome_nodes,
                5000  # OSS hard limit
            )
        if hasattr(MemoryConstants, 'GRAPH_CACHE_SIZE'):
            MemoryConstants.GRAPH_CACHE_SIZE = config.rag_cache_size
        if hasattr(MemoryConstants, 'SIMILARITY_THRESHOLD'):
            MemoryConstants.SIMILARITY_THRESHOLD = config.rag_similarity_threshold
        
        # Add OSS edition flag
        if hasattr(MemoryConstants, 'OSS_EDITION'):
            MemoryConstants.OSS_EDITION = True
            
    except ImportError:
        pass  # MemoryConstants module might not exist yet
    except Exception as e:
        print(f"⚠️  Warning: Failed to update memory constants: {e}")


# Initialize constants on module load
update_memory_constants()

# Export OSS edition information
OSS_EDITION = True
OSS_LICENSE = "Apache 2.0"
ENTERPRISE_UPGRADE_URL = "https://arf.dev/enterprise"

# Print OSS edition info on import (development only)
if __name__ != "__main__":
    print(f"✅ Agentic Reliability Framework - OSS Edition (Apache 2.0)")
    if config.requires_enterprise_upgrade:
        print(f"⚠️  Configuration requires Enterprise upgrade: {ENTERPRISE_UPGRADE_URL}")
