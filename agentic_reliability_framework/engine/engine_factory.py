"""
Engine Factory - OSS Edition Only
Creates OSS-compatible reliability engines with hard limits
Apache 2.0 Licensed

Copyright 2025 Juan Petter

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the complete language governing permissions and
limitations under the License.
"""

import logging
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Type aliases
EngineConfig = Dict[str, Any]

if TYPE_CHECKING:
    from .reliability import V3ReliabilityEngine as BaseV3ReliabilityEngine
    from .v3_reliability import V3ReliabilityEngine as EnhancedV3ReliabilityEngine
else:
    BaseV3ReliabilityEngine = Any
    EnhancedV3ReliabilityEngine = Any


class OSSV3ReliabilityEngine:
    """OSS wrapper for V3ReliabilityEngine with OSS metadata"""
    
    def __init__(self, base_engine: BaseV3ReliabilityEngine) -> None:
        self._engine = base_engine
        self._oss_edition = True
        self._requires_enterprise = False
        self._oss_capabilities = {
            "edition": "oss",
            "license": "Apache 2.0",
            "upgrade_url": "https://arf.dev/enterprise",
        }
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the base engine"""
        try:
            return getattr(self._engine, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                f"Base engine type: {type(self._engine).__name__}"
            )
    
    def __dir__(self) -> list[str]:
        """Include both engine attributes and OSS attributes"""
        engine_dir = dir(self._engine)
        return sorted(set(engine_dir + list(self.__dict__.keys())))
    
    @property
    def oss_edition(self) -> bool:
        """Return OSS edition flag"""
        return self._oss_edition
    
    @property
    def oss_capabilities(self) -> Dict[str, Any]:
        """Return OSS capabilities"""
        return self._oss_capabilities.copy()


class OSSEnhancedV3ReliabilityEngine:
    """OSS wrapper for Enhanced V3ReliabilityEngine with OSS metadata"""
    
    def __init__(self, base_engine: EnhancedV3ReliabilityEngine, enable_rag: bool = False, rag_nodes_limit: int = 1000) -> None:
        self._engine = base_engine
        self._oss_edition = True
        self._requires_enterprise = False
        self._oss_capabilities = {
            "rag_enabled": enable_rag,
            "rag_nodes_limit": rag_nodes_limit,
            "learning_enabled": False,
            "execution_enabled": False,
            "upgrade_available": rag_nodes_limit >= 1000,
            "upgrade_url": "https://arf.dev/enterprise",
        }
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the base engine"""
        try:
            return getattr(self._engine, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                f"Base engine type: {type(self._engine).__name__}"
            )
    
    def __dir__(self) -> list[str]:
        """Include both engine attributes and OSS attributes"""
        engine_dir = dir(self._engine)
        return sorted(set(engine_dir + list(self.__dict__.keys())))
    
    @property
    def oss_edition(self) -> bool:
        """Return OSS edition flag"""
        return self._oss_edition
    
    @property
    def oss_capabilities(self) -> Dict[str, Any]:
        """Return OSS capabilities"""
        return self._oss_capabilities.copy()


class EngineFactory:
    """Factory for creating reliability engines - OSS Edition"""
    
    def __init__(self) -> None:
        self._engines_created = 0
        self._config = self._get_config()
        logger.info("Initialized EngineFactory (OSS Edition)")
    
    def _get_config(self) -> Any:
        """Safely get config module"""
        try:
            from ..config import config
            return config
        except ImportError:
            logger.warning("Config module not available, using defaults")
            # Return a minimal config object
            class DefaultConfig:
                rag_enabled = False
                mcp_enabled = False
                index_file = None
                
                def __getattr__(self, name):
                    return None
            
            return DefaultConfig()
    
    def _get_faiss_index(self):
        """Get or create FAISS index for RAG"""
        try:
            # Try to import from memory module
            from ..memory.faiss_index import create_faiss_index
            
            # Check if we should use in-memory or persistent index
            if hasattr(self._config, 'index_file') and self._config.index_file:
                logger.info(f"Creating FAISS index from {self._config.index_file}")
                return create_faiss_index(persist_path=self._config.index_file)
            else:
                logger.info("Creating in-memory FAISS index")
                return create_faiss_index()
                
        except ImportError as e:
            logger.warning(f"FAISS not available: {e}")
            # Return a complete mock for compatibility
            class MockFAISS:
                def __init__(self):
                    self._count = 0
                    self._vectors = []
                
                def get_count(self):
                    return self._count
                
                def add_async(self, vectors):
                    self._vectors.extend(vectors)
                    self._count += len(vectors)
                    return self._count
                
                def query(self, vector, top_k=5):
                    return []
                
                def add_text(self, text, embedding):
                    self._count += 1
                    return self._count
                
                def search(self, query_vector, k=5):
                    return []
                
                def get_dimension(self):
                    return 768
                
                def is_trained(self):
                    return True
                
                def save(self, path):
                    pass
                
                def load(self, path):
                    pass
            
            return MockFAISS()
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return None
    
    def create_engine(self, engine_config: Optional[EngineConfig] = None) -> Union[OSSV3ReliabilityEngine, OSSEnhancedV3ReliabilityEngine]:
        """
        Create a reliability engine instance
        
        OSS EDITION: Only creates OSS-compatible engines
        
        Args:
            engine_config: Engine configuration dictionary
            
        Returns:
            Configured reliability engine instance (wrapped in OSS wrapper)
        """
        try:
            # Import here to avoid circular imports
            from .reliability import V3ReliabilityEngine as BaseV3Engine
            from .v3_reliability import V3ReliabilityEngine as EnhancedV3Engine
            
            # Determine which engine to create based on config
            use_enhanced = False
            
            if engine_config:
                # Check if enhanced features are requested
                rag_enabled = engine_config.get("rag_enabled", False)
                if rag_enabled and self._config.rag_enabled:
                    use_enhanced = True
            
            # Create engine
            if use_enhanced:
                logger.info("Creating EnhancedV3ReliabilityEngine (OSS limits apply)")
                
                # Get FAISS index for RAG
                faiss_index = None
                if self._config.rag_enabled:
                    faiss_index = self._get_faiss_index()
                    if faiss_index is None:
                        logger.warning("FAISS index not available, RAG disabled")
                        # Fall back to base engine
                        return self._create_base_engine()
                
                # Get MCP for enhanced engine
                mcp_server = None
                if self._config.mcp_enabled:
                    try:
                        from ..lazy import get_mcp_server
                        mcp_server = get_mcp_server()
                    except ImportError as e:
                        logger.warning(f"MCP server not available: {e}")
                        mcp_server = None
                
                # FIXED: Create enhanced engine with proper parameters
                # Check EnhancedV3Engine constructor signature
                enhanced_base_engine: EnhancedV3Engine
                try:
                    # Try with both parameters as positional
                    enhanced_base_engine = EnhancedV3Engine(faiss_index, mcp_server)
                except TypeError:
                    # Fall back to keyword arguments if positional fails
                    logger.debug("Using keyword arguments for EnhancedV3Engine")
                    enhanced_base_engine = EnhancedV3Engine(
                        faiss_index=faiss_index,
                        mcp_server=mcp_server
                    )
                
                # Wrap in OSS wrapper
                enhanced_engine: OSSEnhancedV3ReliabilityEngine = OSSEnhancedV3ReliabilityEngine(enhanced_base_engine)
                
                self._engines_created += 1
                
                # Log OSS capabilities
                logger.info(f"OSS Engine Created: {type(enhanced_engine).__name__}")
                logger.info(f"OSS Limits: 1000 incident nodes max, advisory mode only")
                
                if hasattr(enhanced_engine, '_requires_enterprise') and enhanced_engine._requires_enterprise:
                    logger.info(
                        "💡 Upgrade to Enterprise for more features: "
                        "https://arf.dev/enterprise"
                    )
                
                return enhanced_engine
                
            else:
                return self._create_base_engine()
            
        except ImportError as e:
            logger.error(f"Failed to import engine modules: {e}")
            # Try to create basic engine as fallback
            try:
                return self._create_base_engine()
            except Exception as inner_e:
                logger.error(f"Complete engine creation failure: {inner_e}")
                raise RuntimeError(
                    f"Cannot create any engine. "
                    f"Check dependencies and configuration. "
                    f"Original error: {e}, Fallback error: {inner_e}"
                )
        except Exception as e:
            logger.error(f"Failed to create engine: {e}")
            # Try to create basic engine as fallback
            try:
                return self._create_base_engine()
            except Exception:
                # Re-raise original error if fallback also fails
                raise RuntimeError(f"Failed to create engine: {e}")
    
    def _create_base_engine(self) -> OSSV3ReliabilityEngine:
        """Create base V3 engine with OSS wrapper"""
        try:
            from .reliability import V3ReliabilityEngine as BaseV3Engine
        except ImportError as e:
            logger.error(f"Failed to import BaseV3Engine: {e}")
            raise ImportError(f"Cannot create base engine: {e}")
        
        logger.info("Creating V3ReliabilityEngine (OSS Edition)")
        try:
            base_engine: BaseV3Engine = BaseV3Engine()
        except Exception as e:
            logger.error(f"Failed to instantiate BaseV3Engine: {e}")
            raise RuntimeError(f"Cannot instantiate base engine: {e}")
        
        # Wrap in OSS wrapper
        oss_engine: OSSV3ReliabilityEngine = OSSV3ReliabilityEngine(base_engine)
        
        self._engines_created += 1
        
        # Log OSS capabilities
        logger.info(f"OSS Engine Created: {type(oss_engine).__name__}")
        logger.info(f"OSS Limits: 1000 incident nodes max, advisory mode only")
        
        if hasattr(oss_engine, '_requires_enterprise') and oss_engine._requires_enterprise:
            logger.info(
                "💡 Upgrade to Enterprise for more features: "
                "https://arf.dev/enterprise"
            )
        
        return oss_engine
    
    def create_enhanced_engine(
        self, 
        enable_rag: bool = False,
        rag_nodes_limit: int = 1000
    ) -> OSSEnhancedV3ReliabilityEngine:
        """
        Create enhanced V3 engine with specific features
        
        OSS EDITION: Learning disabled, RAG limited to 1000 nodes
        
        Args:
            enable_rag: Enable RAG graph (OSS limited to 1000 nodes)
            rag_nodes_limit: Maximum RAG nodes (capped at 1000 in OSS)
            
        Returns:
            Enhanced V3 reliability engine (wrapped in OSS wrapper)
        """
        # OSS: Cap RAG nodes
        if rag_nodes_limit > 1000:
            logger.warning(
                f"RAG nodes limit capped at 1000 (OSS max). "
                f"Requested: {rag_nodes_limit}"
            )
            rag_nodes_limit = 1000
        
        # Get FAISS index for RAG if enabled
        faiss_index = None
        if enable_rag and self._config.rag_enabled:
            faiss_index = self._get_faiss_index()
            if faiss_index is None:
                logger.warning("FAISS index not available, RAG will be disabled")
                enable_rag = False
        
        # Get MCP server
        mcp_server = None
        if self._config.mcp_enabled:
            try:
                from ..lazy import get_mcp_server
                mcp_server = get_mcp_server()
            except ImportError as e:
                logger.warning(f"MCP server not available: {e}")
                mcp_server = None
        
        try:
            from .v3_reliability import V3ReliabilityEngine as EnhancedV3Engine
        except ImportError as e:
            logger.error(f"Failed to import EnhancedV3Engine: {e}")
            raise ImportError(f"Cannot create enhanced engine: {e}")
        
        # FIXED: Create enhanced engine with proper parameters
        enhanced_base_engine: EnhancedV3Engine
        try:
            # Try with both parameters as positional
            enhanced_base_engine = EnhancedV3Engine(faiss_index, mcp_server)
        except TypeError:
            # Fall back to keyword arguments if positional fails
            logger.debug("Using keyword arguments for EnhancedV3Engine")
            enhanced_base_engine = EnhancedV3Engine(
                faiss_index=faiss_index,
                mcp_server=mcp_server
            )
        
        # Wrap in OSS wrapper with capabilities
        oss_enhanced_engine: OSSEnhancedV3ReliabilityEngine = OSSEnhancedV3ReliabilityEngine(
            enhanced_base_engine,
            enable_rag=enable_rag and faiss_index is not None,  # Only enable if we have FAISS
            rag_nodes_limit=rag_nodes_limit
        )
        
        return oss_enhanced_engine
    
    def get_oss_engine_capabilities(self) -> Dict[str, Any]:
        """
        Get OSS engine capabilities and limits
        
        Returns:
            Dictionary of OSS capabilities
        """
        return {
            "edition": "oss",
            "license": "Apache 2.0",
            "engines_available": {
                "V3ReliabilityEngine": True,
                "EnhancedV3ReliabilityEngine": self._config.rag_enabled or self._config.mcp_enabled,
            },
            "limits": {
                "max_rag_incident_nodes": 1000,
                "max_rag_outcome_nodes": 5000,
                "mcp_modes": ["advisory"],
                "learning_enabled": False,
                "persistent_storage": False,
            },
            "capabilities": {
                "rag_analysis": self._config.rag_enabled,
                "anomaly_detection": True,
                "business_impact": True,
                "forecasting": True,
                "self_healing_advisory": True,
                "self_healing_execution": False,  # OSS: Advisory only
            },
            "requires_enterprise": (
                getattr(self._config, 'rag_max_incident_nodes', 0) >= 1000 or
                getattr(self._config, 'rag_max_outcome_nodes', 0) >= 5000 or
                getattr(self._config, 'mcp_mode', 'advisory') != "advisory"
            ),
            "enterprise_features": [
                "autonomous_execution",
                "approval_workflows",
                "learning_engine",
                "persistent_storage",
                "unlimited_rag_nodes",
                "audit_trails",
            ],
            "upgrade_url": "https://arf.dev/enterprise",
        }
    
    def validate_oss_compatibility(self, engine_config: EngineConfig) -> Dict[str, Any]:
        """
        Validate engine configuration for OSS compatibility
        
        Args:
            engine_config: Engine configuration to validate
            
        Returns:
            Validation results
        """
        violations: list[str] = []
        warnings: list[str] = []
        
        # Check RAG limits
        rag_nodes = engine_config.get("rag_max_incident_nodes", 0)
        if rag_nodes > 1000:
            violations.append(
                f"rag_max_incident_nodes exceeds OSS limit (1000): {rag_nodes}"
            )
        
        rag_outcomes = engine_config.get("rag_max_outcome_nodes", 0)
        if rag_outcomes > 5000:
            violations.append(
                f"rag_max_outcome_nodes exceeds OSS limit (5000): {rag_outcomes}"
            )
        
        # Check MCP mode
        mcp_mode = engine_config.get("mcp_mode", "advisory")
        if mcp_mode != "advisory":
            violations.append(
                f"MCP mode must be 'advisory' in OSS, got: {mcp_mode}"
            )
        
        # Check for Enterprise-only features
        if engine_config.get("learning_enabled", False):
            violations.append("learning_enabled requires Enterprise edition")
        
        if engine_config.get("beta_testing_enabled", False):
            violations.append("beta_testing_enabled requires Enterprise edition")
        
        if engine_config.get("rollout_percentage", 0) > 0:
            violations.append("rollout_percentage requires Enterprise edition")
        
        # Add warnings for features that will be ignored in OSS
        if engine_config.get("execution_enabled", False):
            warnings.append("execution_enabled will be ignored in OSS (advisory only)")
        
        if engine_config.get("autonomous_mode", False):
            warnings.append("autonomous_mode will be ignored in OSS (advisory only)")
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "requires_enterprise": len(violations) > 0,
            "oss_compatible": len(violations) == 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get factory statistics"""
        return {
            "engines_created": self._engines_created,
            "edition": "oss",
            "oss_compliant": True,
        }


# Factory function for backward compatibility
def create_engine(engine_config: Optional[EngineConfig] = None) -> Union[OSSV3ReliabilityEngine, OSSEnhancedV3ReliabilityEngine]:
    """
    Create engine - backward compatibility function
    
    OSS EDITION: Returns OSS-compatible engine only
    """
    factory = EngineFactory()
    return factory.create_engine(engine_config)


def get_engine(engine_config: Optional[EngineConfig] = None) -> Union[OSSV3ReliabilityEngine, OSSEnhancedV3ReliabilityEngine]:
    """Alias for create_engine - backward compatibility"""
    return create_engine(engine_config)


# Convenience functions that delegate to EngineFactory
def get_oss_engine_capabilities() -> Dict[str, Any]:
    """Get OSS engine capabilities and limits"""
    factory = EngineFactory()
    return factory.get_oss_engine_capabilities()


def validate_oss_compatibility(engine_config: EngineConfig) -> Dict[str, Any]:
    """Validate engine configuration for OSS compatibility"""
    factory = EngineFactory()
    return factory.validate_oss_compatibility(engine_config)


# Export
__all__ = [
    "EngineFactory",
    "create_engine",
    "get_engine",
    "get_oss_engine_capabilities",
    "validate_oss_compatibility",
    "OSSV3ReliabilityEngine",
    "OSSEnhancedV3ReliabilityEngine",
]
