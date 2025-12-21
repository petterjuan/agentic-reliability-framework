# File: arf_core/config/oss_config.py
"""
OSS Configuration Wrapper - Enforces OSS boundaries on existing configuration
Apache 2.0 Licensed - Enterprise features require commercial license

This module provides a clean wrapper around the existing config.py
that enforces OSS edition boundaries at runtime.
"""

import os
import warnings
from typing import Dict, Any, Optional, List, Union, cast
from dataclasses import dataclass, field

from ..constants import (
    MAX_INCIDENT_HISTORY,
    MAX_RAG_LOOKBACK_DAYS,
    MCP_MODES_ALLOWED,
    EXECUTION_ALLOWED,
    GRAPH_STORAGE,
    MAX_INCIDENT_NODES,
    MAX_OUTCOME_NODES,
    FAISS_INDEX_TYPE,
    EMBEDDING_DIM,
    STORAGE_BACKEND,
    OSS_EDITION,
    OSS_LICENSE,
    OSS_VERSION,
    ENTERPRISE_UPGRADE_URL,
    OSSBoundaryError,
    validate_oss_config,
    get_oss_capabilities,
    check_oss_compliance,
)


@dataclass
class OSSConfig:
    """
    OSS Configuration Wrapper with boundary enforcement
    
    This class wraps the existing configuration and ensures:
    1. OSS hard limits are never exceeded
    2. Enterprise-only features are blocked or downgraded
    3. Runtime validation of OSS boundaries
    4. Clear upgrade prompts for Enterprise features
    
    Usage:
        from arf_core.config.oss_config import oss_config
        # oss_config is a singleton instance with OSS boundaries applied
    """
    
    # Original configuration (lazy-loaded)
    _original_config: Optional[Any] = field(default=None, init=False, repr=False)
    
    # OSS boundary violations detected
    _violations: List[str] = field(default_factory=list, init=False, repr=False)
    
    # Cache for performance - FIXED: Added explicit type annotation
    _config_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize OSS configuration with boundary checking"""
        # Check OSS compliance
        if not check_oss_compliance():
            warnings.warn(
                f"Environment appears to be Enterprise but using OSS configuration. "
                f"Some features may be limited. Use Enterprise configuration for full features.",
                RuntimeWarning
            )
        
        # Load and validate original configuration
        self._load_and_validate()
    
    def _load_and_validate(self) -> None:
        """Load original configuration and apply OSS boundaries"""
        try:
            # Import original config (lazy to avoid circular imports)
            from agentic_reliability_framework.config import config as original_config
            
            self._original_config = original_config
            
            # Convert to dictionary for validation
            config_dict = self._get_config_dict()
            
            # Validate against OSS boundaries
            try:
                validate_oss_config(config_dict)
            except OSSBoundaryError as e:
                self._violations.append(str(e))
                warnings.warn(
                    f"OSS configuration boundary violation: {str(e)[:200]}...",
                    RuntimeWarning
                )
            
            # Apply OSS limits to cache
            self._apply_oss_limits()
            
        except ImportError as e:
            warnings.warn(
                f"Could not load original configuration: {e}. "
                f"Using OSS defaults.",
                ImportWarning
            )
            self._original_config = None
            self._apply_default_oss_config()
    
    def _get_config_dict(self) -> Dict[str, Any]:
        """Convert original config to dictionary"""
        if self._original_config is None:
            return {}
        
        # Try to convert using model_dump (Pydantic v2)
        if hasattr(self._original_config, 'model_dump'):
            return cast(Dict[str, Any], self._original_config.model_dump())
        
        # Try to convert using dict method
        elif hasattr(self._original_config, 'dict'):
            return cast(Dict[str, Any], self._original_config.dict())
        
        # Try to access as dict
        elif isinstance(self._original_config, dict):
            return self._original_config.copy()
        
        # Fallback: access attributes directly
        else:
            config_dict: Dict[str, Any] = {}
            for attr_name in dir(self._original_config):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(self._original_config, attr_name)
                        if not callable(attr_value):
                            config_dict[attr_name] = attr_value
                    except:
                        pass
            return config_dict
    
    def _apply_oss_limits(self) -> None:
        """Apply OSS limits to configuration cache"""
        if self._original_config is None:
            return
        
        config_dict = self._get_config_dict()
        
        # === ENFORCE OSS BOUNDARIES ===
        
        # 1. Execution boundaries
        self._config_cache["mcp_mode"] = "advisory"  # Force advisory mode
        self._config_cache["mcp_enabled"] = False  # No execution in OSS
        self._config_cache["execution_allowed"] = EXECUTION_ALLOWED
        
        # 2. Storage boundaries
        self._config_cache["max_events_stored"] = min(
            config_dict.get("max_events_stored", 1000),
            MAX_INCIDENT_HISTORY
        )
        
        self._config_cache["rag_max_incident_nodes"] = min(
            config_dict.get("rag_max_incident_nodes", 1000),
            MAX_INCIDENT_NODES
        )
        
        self._config_cache["rag_max_outcome_nodes"] = min(
            config_dict.get("rag_max_outcome_nodes", 5000),
            MAX_OUTCOME_NODES
        )
        
        # 3. Memory/RAG boundaries
        self._config_cache["graph_storage"] = STORAGE_BACKEND
        self._config_cache["faiss_index_type"] = FAISS_INDEX_TYPE
        self._config_cache["vector_dim"] = EMBEDDING_DIM
        self._config_cache["rag_embedding_dim"] = EMBEDDING_DIM
        
        # 4. Disable Enterprise-only features
        self._config_cache["learning_enabled"] = False
        self._config_cache["beta_testing_enabled"] = False
        self._config_cache["rollout_percentage"] = 0
        self._config_cache["demo_mode"] = False  # OSS has no demo mode
        
        # 5. Copy safe configuration values
        safe_keys = [
            "hf_api_key", "hf_api_url", "faiss_batch_size",
            "base_revenue_per_minute", "base_users",
            "latency_critical", "latency_warning", "latency_extreme",
            "cpu_critical", "memory_critical",
            "error_rate_critical", "error_rate_high", "error_rate_warning",
            "forecast_lookahead_minutes", "forecast_min_data_points",
            "slope_threshold_increasing", "slope_threshold_decreasing",
            "cache_expiry_minutes", "max_requests_per_minute", "max_requests_per_hour",
            "log_level", "index_file", "incident_texts_file",
            "rag_enabled", "rag_similarity_threshold", "rag_cache_size",
            "mcp_host", "mcp_port", "mcp_timeout_seconds", "mpc_cooldown_seconds",
            "agent_timeout_seconds", "circuit_breaker_failures", "circuit_breaker_timeout",
            "safety_action_blacklist", "safety_max_blast_radius", "safety_rag_timeout_ms",
        ]
        
        for key in safe_keys:
            if key in config_dict:
                self._config_cache[key] = config_dict[key]
        
        # 6. Add OSS metadata
        self._config_cache["oss_edition"] = OSS_EDITION
        self._config_cache["oss_license"] = OSS_LICENSE
        self._config_cache["oss_version"] = OSS_VERSION
        self._config_cache["enterprise_upgrade_url"] = ENTERPRISE_UPGRADE_URL
        self._config_cache["requires_enterprise_upgrade"] = self.requires_enterprise_upgrade
    
    def _apply_default_oss_config(self) -> None:
        """Apply default OSS configuration when original config is unavailable"""
        self._config_cache = {
            # OSS boundaries (enforced)
            "mcp_mode": "advisory",
            "mcp_enabled": False,
            "execution_allowed": EXECUTION_ALLOWED,
            "max_events_stored": MAX_INCIDENT_HISTORY,
            "rag_max_incident_nodes": MAX_INCIDENT_NODES,
            "rag_max_outcome_nodes": MAX_OUTCOME_NODES,
            "graph_storage": STORAGE_BACKEND,
            "faiss_index_type": FAISS_INDEX_TYPE,
            "vector_dim": EMBEDDING_DIM,
            "rag_embedding_dim": EMBEDDING_DIM,
            
            # Disabled Enterprise features
            "learning_enabled": False,
            "beta_testing_enabled": False,
            "rollout_percentage": 0,
            "demo_mode": False,
            
            # Safe defaults
            "hf_api_key": "",
            "hf_api_url": "https://router.huggingface.co/hf-inference/v1/completions",
            "faiss_batch_size": 10,
            "base_revenue_per_minute": 100.0,
            "base_users": 1000,
            "latency_critical": 300.0,
            "latency_warning": 150.0,
            "latency_extreme": 500.0,
            "cpu_critical": 0.9,
            "memory_critical": 0.9,
            "error_rate_critical": 0.3,
            "error_rate_high": 0.15,
            "error_rate_warning": 0.05,
            "forecast_lookahead_minutes": 15,
            "forecast_min_data_points": 5,
            "slope_threshold_increasing": 5.0,
            "slope_threshold_decreasing": -2.0,
            "cache_expiry_minutes": 15,
            "max_requests_per_minute": 60,
            "max_requests_per_hour": 500,
            "log_level": "INFO",
            "index_file": "data/faiss_index.bin",
            "incident_texts_file": "data/incident_texts.json",
            "rag_enabled": False,
            "rag_similarity_threshold": 0.3,
            "rag_cache_size": 100,
            "mcp_host": "localhost",
            "mcp_port": 8000,
            "mcp_timeout_seconds": 10,
            "mpc_cooldown_seconds": 60,
            "agent_timeout_seconds": 5,
            "circuit_breaker_failures": 3,
            "circuit_breaker_timeout": 30,
            "safety_action_blacklist": "DATABASE_DROP,FULL_ROLLOUT,SYSTEM_SHUTDOWN",
            "safety_max_blast_radius": 3,
            "safety_rag_timeout_ms": 100,
            
            # OSS metadata
            "oss_edition": OSS_EDITION,
            "oss_license": OSS_LICENSE,
            "oss_version": OSS_VERSION,
            "enterprise_upgrade_url": ENTERPRISE_UPGRADE_URL,
            "requires_enterprise_upgrade": True,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with OSS boundaries applied"""
        # Check cache first
        if key in self._config_cache:
            return self._config_cache[key]
        
        # Try to get from original config with OSS limits
        if self._original_config is not None:
            try:
                value = getattr(self._original_config, key, default)
                
                # Apply OSS limits based on key
                value = self._apply_oss_limit_to_value(key, value)
                
                # Cache for future use
                self._config_cache[key] = value
                return value
                
            except (AttributeError, KeyError):
                pass
        
        # Return default if not found
        return default
    
    def _apply_oss_limit_to_value(self, key: str, value: Any) -> Any:
        """Apply OSS-specific limits to a configuration value"""
        if value is None:
            return value
        
        # Map keys to OSS limits
        oss_limits: Dict[str, Any] = {
            "mcp_mode": lambda v: "advisory",
            "mcp_enabled": lambda v: False,
            "execution_allowed": lambda v: False,
            "max_events_stored": lambda v: min(v, MAX_INCIDENT_HISTORY),
            "rag_max_incident_nodes": lambda v: min(v, MAX_INCIDENT_NODES),
            "rag_max_outcome_nodes": lambda v: min(v, MAX_OUTCOME_NODES),
            "graph_storage": lambda v: STORAGE_BACKEND,
            "faiss_index_type": lambda v: FAISS_INDEX_TYPE,
            "vector_dim": lambda v: EMBEDDING_DIM,
            "rag_embedding_dim": lambda v: EMBEDDING_DIM,
            "learning_enabled": lambda v: False,
            "beta_testing_enabled": lambda v: False,
            "rollout_percentage": lambda v: 0,
            "demo_mode": lambda v: False,
        }
        
        if key in oss_limits:
            return oss_limits[key](value)
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary with OSS boundaries applied"""
        return self._config_cache.copy()
    
    @property
    def original_config(self) -> Optional[Any]:
        """Get the original configuration (if available)"""
        return self._original_config
    
    @property
    def violations(self) -> List[str]:
        """Get list of OSS boundary violations detected"""
        return self._violations.copy()
    
    @property
    def is_oss_compliant(self) -> bool:
        """Check if configuration is OSS compliant"""
        return len(self._violations) == 0 and check_oss_compliance()
    
    @property
    def requires_enterprise_upgrade(self) -> bool:
        """Check if any configuration requires Enterprise upgrade"""
        # Check for any non-OSS values in cache
        enterprise_indicators = [
            self._config_cache.get("mcp_mode") != "advisory",
            self._config_cache.get("mcp_enabled", False) and self._config_cache.get("mcp_mode") != "advisory",
            self._config_cache.get("learning_enabled", False),
            self._config_cache.get("beta_testing_enabled", False),
            self._config_cache.get("rollout_percentage", 0) > 0,
            self._config_cache.get("graph_storage") != "in_memory",
            self._config_cache.get("faiss_index_type") != "IndexFlatL2",
        ]
        
        return any(enterprise_indicators)
    
    @property
    def v3_features(self) -> Dict[str, Any]:
        """Get v3 feature status - OSS edition with upgrade info"""
        return {
            "rag_enabled": self._config_cache.get("rag_enabled", False),
            "mcp_enabled": self._config_cache.get("mcp_enabled", False),
            "learning_enabled": False,  # OSS: Always False
            "beta_testing": False,      # OSS: Always False
            "rollout_active": False,    # OSS: Always False
            "edition": OSS_EDITION,
            "oss_limits": {
                "max_incident_nodes": MAX_INCIDENT_NODES,
                "max_outcome_nodes": MAX_OUTCOME_NODES,
                "mcp_mode": "advisory",
                "execution_allowed": False,
                "graph_storage": "in_memory",
                "faiss_index_type": "IndexFlatL2",
            },
            "enterprise_upgrade_required": self.requires_enterprise_upgrade,
            "enterprise_features": get_oss_capabilities().get("enterprise_features", []),
            "upgrade_url": ENTERPRISE_UPGRADE_URL,
        }
    
    @property
    def safety_guardrails(self) -> Dict[str, Any]:
        """Get safety guardrails configuration with OSS enforcement"""
        action_blacklist = self._config_cache.get("safety_action_blacklist", "")
        if isinstance(action_blacklist, str):
            actions = [action.strip() for action in action_blacklist.split(",")]
        else:
            actions = action_blacklist
        
        return {
            "action_blacklist": actions,
            "max_blast_radius": self._config_cache.get("safety_max_blast_radius", 3),
            "rag_timeout_ms": self._config_cache.get("safety_rag_timeout_ms", 100),
            "circuit_breaker": {
                "failures": self._config_cache.get("circuit_breaker_failures", 3),
                "timeout": self._config_cache.get("circuit_breaker_timeout", 30),
            },
            "edition": OSS_EDITION,
            "execution_blocked": True,  # OSS never executes
            "oss_restricted": True,
        }
    
    def validate(self) -> None:
        """Validate configuration against OSS boundaries"""
        config_dict = self.to_dict()
        validate_oss_config(config_dict)
        
        if self._violations:
            raise OSSBoundaryError(
                f"OSS configuration has {len(self._violations)} violations:\n" +
                "\n".join(f"  • {v[:100]}..." for v in self._violations[:3])
            )
    
    def get_oss_limits(self) -> Dict[str, Any]:
        """Get OSS edition limits for documentation"""
        return {
            "edition": OSS_EDITION,
            "license": OSS_LICENSE,
            "version": OSS_VERSION,
            "limits": {
                "max_events_stored": MAX_INCIDENT_HISTORY,
                "rag_max_incident_nodes": MAX_INCIDENT_NODES,
                "rag_max_outcome_nodes": MAX_OUTCOME_NODES,
                "mcp_mode": "advisory",
                "execution_allowed": False,
                "learning_enabled": False,
                "persistent_storage": False,
                "graph_storage": "in_memory",
                "faiss_index_type": "IndexFlatL2",
            },
            "capabilities": {
                "rag_analysis": self._config_cache.get("rag_enabled", False),
                "mcp_advisory": self._config_cache.get("mcp_enabled", False),
                "anomaly_detection": True,
                "business_impact": True,
                "forecasting": True,
            },
            "enterprise_upgrade_available": True,
            "upgrade_url": ENTERPRISE_UPGRADE_URL,
        }
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables with OSS validation"""
        try:
            # Try to reload original config from environment
            from agentic_reliability_framework.config import Config
            
            original = Config.from_env()
            self._original_config = original
            
            # Clear cache and re-validate
            self._config_cache.clear()
            self._violations.clear()
            self._load_and_validate()
            
        except ImportError as e:
            warnings.warn(f"Could not reload configuration from env: {e}", RuntimeWarning)
        except Exception as e:
            warnings.warn(f"Error reloading configuration: {e}", RuntimeWarning)
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to configuration values"""
        # Try to get from cache
        if name in self._config_cache:
            return self._config_cache[name]
        
        # Try to get from original config with OSS limits
        if self._original_config is not None:
            try:
                value = getattr(self._original_config, name)
                value = self._apply_oss_limit_to_value(name, value)
                self._config_cache[name] = value
                return value
            except AttributeError:
                pass
        
        # Special properties
        if name == "is_oss_edition":
            return True
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Singleton instance for easy import
oss_config = OSSConfig()

# Export
__all__ = [
    "OSSConfig",
    "oss_config",
]
