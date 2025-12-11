"""
Configuration management for Agentic Reliability Framework

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


@dataclass
class AppConfig:
    """Application configuration from environment variables"""
    
    # API Configuration
    hf_api_key: str = os.getenv("HF_API_KEY", "")
    hf_api_url: str = os.getenv(
        "HF_API_URL", 
        "https://router.huggingface.co/hf-inference/v1/completions"
    )
    
    # System Configuration
    max_events_stored: int = int(os.getenv("MAX_EVENTS_STORED", "1000"))
    faiss_batch_size: int = int(os.getenv("FAISS_BATCH_SIZE", "10"))
    faiss_save_interval: int = int(os.getenv("FAISS_SAVE_INTERVAL_SECONDS", "30"))
    vector_dim: int = int(os.getenv("VECTOR_DIM", "384"))
    
    # Business Metrics
    base_revenue_per_minute: float = float(os.getenv("BASE_REVENUE_PER_MINUTE", "100.0"))
    base_users: int = int(os.getenv("BASE_USERS", "1000"))
    
    # Rate Limiting
    max_requests_per_minute: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
    max_requests_per_hour: int = int(os.getenv("MAX_REQUESTS_PER_HOUR", "500"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Thresholds
    latency_warning: float = float(os.getenv("LATENCY_WARNING", "150.0"))
    latency_critical: float = float(os.getenv("LATENCY_CRITICAL", "300.0"))
    latency_extreme: float = float(os.getenv("LATENCY_EXTREME", "500.0"))
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables"""
        return cls()
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.vector_dim <= 0:
            raise ValueError("VECTOR_DIM must be positive")
        if self.max_events_stored <= 0:
            raise ValueError("MAX_EVENTS_STORED must be positive")
        return True


# Global config instance
config = AppConfig.from_env()
config.validate()
