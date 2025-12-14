"""
Configuration Management for Agentic Reliability Framework
Updated with v3 RAG Graph features
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """
    Application configuration with environment variable support
    
    Updated for v3: Added RAG Graph and MCP Server configuration
    """
    
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
    
    # === Rate Limiting ===
    max_requests_per_minute: int = Field(default=60, description="Maximum requests per minute")
    max_requests_per_hour: int = Field(default=500, description="Maximum requests per hour")
    
    # === Logging ===
    log_level: str = Field(default="INFO", description="Logging level")
    
    # === File Paths ===
    index_file: str = Field(default="data/faiss_index.bin", description="FAISS index file path")
    incident_texts_file: str = Field(default="data/incident_texts.json", description="FAISS incident texts file path")
    
    # === v3 FEATURE FLAGS ===
    # RAG Graph Configuration
    rag_enabled: bool = Field(default=False, description="Enable RAG Graph features")
    rag_similarity_threshold: float = Field(default=0.3, description="Minimum similarity threshold for RAG retrieval")
    rag_max_incident_nodes: int = Field(default=1000, description="Maximum incident nodes in RAG graph")
    rag_max_outcome_nodes: int = Field(default=5000, description="Maximum outcome nodes in RAG graph")
    rag_cache_size: int = Field(default=100, description="RAG similarity cache size")
    
    # MCP Server Configuration
    mcp_mode: str = Field(default="advisory", description="MCP execution mode: advisory, approval, or autonomous")
    mcp_enabled: bool = Field(default=False, description="Enable MCP Server for execution boundaries")
    mcp_host: str = Field(default="localhost", description="MCP Server host")
    mcp_port: int = Field(default=8000, description="MCP Server port")
    
    # Learning Loop Configuration
    learning_enabled: bool = Field(default=False, description="Enable learning loop from outcomes")
    learning_min_data_points: int = Field(default=10, description="Minimum data points before learning")
    
    # === Performance ===
    agent_timeout_seconds: int = Field(default=5, description="Agent timeout in seconds")
    cache_expiry_minutes: int = Field(default=15, description="Cache expiry time in minutes")
    
    # === Demo Mode ===
    demo_mode: bool = Field(default=False, description="Enable demo mode with pre-configured scenarios")
    
    class Config:
        env_prefix = ""
        case_sensitive = False
    
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
            # v3 Features
            "RAG_ENABLED": "rag_enabled",
            "RAG_SIMILARITY_THRESHOLD": "rag_similarity_threshold",
            "RAG_MAX_INCIDENT_NODES": "rag_max_incident_nodes",
            "RAG_MAX_OUTCOME_NODES": "rag_max_outcome_nodes",
            "RAG_CACHE_SIZE": "rag_cache_size",
            "MCP_MODE": "mcp_mode",
            "MCP_ENABLED": "mcp_enabled",
            "MCP_HOST": "mcp_host",
            "MCP_PORT": "mcp_port",
            "LEARNING_ENABLED": "learning_enabled",
            "LEARNING_MIN_DATA_POINTS": "learning_min_data_points",
            "AGENT_TIMEOUT_SECONDS": "agent_timeout_seconds",
            "CACHE_EXPIRY_MINUTES": "cache_expiry_minutes",
            "DEMO_MODE": "demo_mode",
        }
        
        for env_name, field_name in field_mapping.items():
            env_value = os.getenv(env_name)
            if env_value is not None:
                # Convert string to appropriate type
                field_type = cls.__fields__[field_name].type_
                if field_type == bool:
                    env_vars[field_name] = env_value.lower() in ("true", "1", "yes", "y", "t")
                elif field_type == int:
                    env_vars[field_name] = int(env_value)
                elif field_type == float:
                    env_vars[field_name] = float(env_value)
                else:
                    env_vars[field_name] = env_value
        
        return cls(**env_vars)


# Global configuration instance
config = Config.from_env()

# Update MemoryConstants with config values if needed
def update_memory_constants():
    """Update memory constants from config"""
    from .memory.constants import MemoryConstants
    
    # Only update if constants exist
    if hasattr(MemoryConstants, '__annotations__'):
        # Update FAISS constants from config
        if hasattr(MemoryConstants, 'FAISS_BATCH_SIZE'):
            MemoryConstants.FAISS_BATCH_SIZE = config.faiss_batch_size
        if hasattr(MemoryConstants, 'VECTOR_DIM'):
            MemoryConstants.VECTOR_DIM = config.vector_dim
        
        # Update RAG constants from config
        if hasattr(MemoryConstants, 'MAX_INCIDENT_NODES'):
            MemoryConstants.MAX_INCIDENT_NODES = config.rag_max_incident_nodes
        if hasattr(MemoryConstants, 'MAX_OUTCOME_NODES'):
            MemoryConstants.MAX_OUTCOME_NODES = config.rag_max_outcome_nodes
        if hasattr(MemoryConstants, 'GRAPH_CACHE_SIZE'):
            MemoryConstants.GRAPH_CACHE_SIZE = config.rag_cache_size
        if hasattr(MemoryConstants, 'SIMILARITY_THRESHOLD'):
            MemoryConstants.SIMILARITY_THRESHOLD = config.rag_similarity_threshold


# Initialize constants on module load
update_memory_constants()
