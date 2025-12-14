"""
Configuration Management for Agentic Reliability Framework
Updated with v3 RAG Graph, MCP Server, and Learning Loop features
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """
    Application configuration with environment variable support
    
    V3 Features:
    - RAG Graph Configuration
    - MCP Server Configuration  
    - Learning Loop Configuration
    - Feature Flags for gradual rollout
    """
    
    model_config = ConfigDict(
        env_prefix="",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore"
    )
    
    # === API Configuration ===
    hf_api_key: str = Field(default="", description="HuggingFace API key")
    hf_api_url: str = Field(
        default="https://router.huggingface.co/hf-inference/v1/completions",
        description="HuggingFace API endpoint"
    )
    
    # === System Configuration ===
    max_events_stored: int = Field(default=1000, description="Maximum events to store in memory")
    faiss_batch_size: int = Field(default=10, description="FAISS batch size for async writes")
    vector_dim: int = Field(default=384, description="Vector dimension for embeddings")
    
    # === Business Metrics ===
    base_revenue_per_minute: float = Field(default=100.0, description="Base revenue per minute for impact calculation")
    base_users: int = Field(default=1000, description="Base user count for impact calculation")
    
    # === Anomaly Detection Thresholds ===
    latency_critical: float = Field(default=300.0, description="Critical latency threshold (ms)")
    latency_warning: float = Field(default=150.0, description="Warning latency threshold (ms)")
    latency_extreme: float = Field(default=500.0, description="Extreme latency threshold (ms)")
    
    cpu_critical: float = Field(default=0.9, description="Critical CPU threshold")
    memory_critical: float = Field(default=0.9, description="Critical memory threshold")
    
    error_rate_critical: float = Field(default=0.3, description="Critical error rate threshold")
    error_rate_high: float = Field(default=0.15, description="High error rate threshold")
    error_rate_warning: float = Field(default=0.05, description="Warning error rate threshold")
    
    # === Forecasting Constants ===
    forecast_lookahead_minutes: int = Field(default=15, description="Forecast lookahead in minutes")
    forecast_min_data_points: int = Field(default=5, description="Minimum data points for forecast")
    slope_threshold_increasing: float = Field(default=5.0, description="Increasing trend threshold")
    slope_threshold_decreasing: float = Field(default=-2.0, description="Decreasing trend threshold")
    cache_expiry_minutes: int = Field(default=15, description="Cache expiry in minutes")
    
    # === Rate Limiting ===
    max_requests_per_minute: int = Field(default=60, description="Maximum requests per minute")
    max_requests_per_hour: int = Field(default=500, description="Maximum requests per hour")
    
    # === Logging ===
    log_level: str = Field(default="INFO", description="Logging level")
    
    # === File Paths ===
    index_file: str = Field(default="data/faiss_index.bin", description="FAISS index file path")
    incident_texts_file: str = Field(default="data/incident_texts.json", description="FAISS incident texts file path")
    
    # === v3 FEATURE FLAGS & CONFIGURATION ===
    # Phase 1: RAG Graph
    rag_enabled: bool = Field(default=False, description="Enable RAG Graph features")
    rag_similarity_threshold: float = Field(default=0.3, description="Minimum similarity threshold for RAG retrieval")
    rag_max_incident_nodes: int = Field(default=1000, description="Maximum incident nodes in RAG graph")
    rag_max_outcome_nodes: int = Field(default=5000, description="Maximum outcome nodes in RAG graph")
    rag_cache_size: int = Field(default=100, description="RAG similarity cache size")
    rag_embedding_dim: int = Field(default=384, description="RAG embedding dimension")
    
    # Phase 2: MCP Server
    mcp_mode: str = Field(default="advisory", description="MCP execution mode: advisory, approval, or autonomous")
    mcp_enabled: bool = Field(default=False, description="Enable MCP Server for execution boundaries")
    mcp_host: str = Field(default="localhost", description="MCP Server host")
    mcp_port: int = Field(default=8000, description="MCP Server port")
    mcp_timeout_seconds: int = Field(default=10, description="MCP request timeout")
    mpc_cooldown_seconds: int = Field(default=60, description="MCP tool cooldown period")
    
    # Phase 3: Learning Loop
    learning_enabled: bool = Field(default=False, description="Enable learning loop from outcomes")
    learning_min_data_points: int = Field(default=10, description="Minimum data points before learning")
    learning_confidence_threshold: float = Field(default=0.7, description="Confidence threshold for learned patterns")
    learning_retention_days: int = Field(default=30, description="Days to retain learning data")
    
    # === Performance & Safety ===
    agent_timeout_seconds: int = Field(default=5, description="Agent timeout in seconds")
    circuit_breaker_failures: int = Field(default=3, description="Circuit breaker failure threshold")
    circuit_breaker_timeout: int = Field(default=30, description="Circuit breaker recovery timeout")
    
    # === Demo Mode ===
    demo_mode: bool = Field(default=False, description="Enable demo mode with pre-configured scenarios")
    
    # === Rollout Configuration ===
    rollout_percentage: int = Field(default=0, description="Percentage of traffic to enable v3 features for (0-100)")
    beta_testing_enabled: bool = Field(default=False, description="Enable beta testing features")
    
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
    
    @property
    def v3_features(self) -> Dict[str, bool]:
        """Get v3 feature status"""
        return {
            "rag_enabled": self.rag_enabled,
            "mcp_enabled": self.mcp_enabled,
            "learning_enabled": self.learning_enabled,
            "beta_testing": self.beta_testing_enabled,
            "rollout_active": self.rollout_percentage > 0,
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
            }
        }
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        env_vars = {}
        
        # Map environment variables to config fields
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
            
            # v3 Features
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
            
            "LEARNING_ENABLED": "learning_enabled",
            "LEARNING_MIN_DATA_POINTS": "learning_min_data_points",
            "LEARNING_CONFIDENCE_THRESHOLD": "learning_confidence_threshold",
            "LEARNING_RETENTION_DAYS": "learning_retention_days",
            
            "AGENT_TIMEOUT_SECONDS": "agent_timeout_seconds",
            "CIRCUIT_BREAKER_FAILURES": "circuit_breaker_failures",
            "CIRCUIT_BREAKER_TIMEOUT": "circuit_breaker_timeout",
            
            "DEMO_MODE": "demo_mode",
            
            "ROLLOUT_PERCENTAGE": "rollout_percentage",
            "BETA_TESTING_ENABLED": "beta_testing_enabled",
            
            "SAFETY_ACTION_BLACKLIST": "safety_action_blacklist",
            "SAFETY_MAX_BLAST_RADIUS": "safety_max_blast_radius",
            "SAFETY_RAG_TIMEOUT_MS": "safety_rag_timeout_ms",
        }
        
        for env_name, field_name in field_mapping.items():
            env_value = os.getenv(env_name)
            if env_value is not None:
                # Convert string to appropriate type
                field_type = cls.__annotations__.get(field_name, str)
                
                try:
                    if field_type is bool:
                        env_vars[field_name] = env_value.lower() in ("true", "1", "yes", "y", "t", "on")
                    elif field_type is int:
                        env_vars[field_name] = int(env_value)
                    elif field_type is float:
                        env_vars[field_name] = float(env_value)
                    else:
                        env_vars[field_name] = env_value
                except (ValueError, TypeError):
                    # Use default if conversion fails
                    continue
        
        return cls(**env_vars)


# Global configuration instance
config = Config.from_env()

# Update MemoryConstants with config values
def update_memory_constants():
    """Update memory constants from config"""
    try:
        from .memory.constants import MemoryConstants
        
        # Update FAISS constants
        if hasattr(MemoryConstants, 'FAISS_BATCH_SIZE'):
            MemoryConstants.FAISS_BATCH_SIZE = config.faiss_batch_size
        if hasattr(MemoryConstants, 'VECTOR_DIM'):
            MemoryConstants.VECTOR_DIM = config.vector_dim
        
        # Update RAG constants
        if hasattr(MemoryConstants, 'MAX_INCIDENT_NODES'):
            MemoryConstants.MAX_INCIDENT_NODES = config.rag_max_incident_nodes
        if hasattr(MemoryConstants, 'MAX_OUTCOME_NODES'):
            MemoryConstants.MAX_OUTCOME_NODES = config.rag_max_outcome_nodes
        if hasattr(MemoryConstants, 'GRAPH_CACHE_SIZE'):
            MemoryConstants.GRAPH_CACHE_SIZE = config.rag_cache_size
        if hasattr(MemoryConstants, 'SIMILARITY_THRESHOLD'):
            MemoryConstants.SIMILARITY_THRESHOLD = config.rag_similarity_threshold
        
    except ImportError:
        pass  # MemoryConstants module might not exist yet


# Initialize constants on module load
update_memory_constants()
