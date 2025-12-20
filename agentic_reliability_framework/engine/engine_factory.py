# agentic_reliability_framework/engine/engine_factory.py
# REPLACE THE ENTIRE FILE WITH:

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
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Dict, Any, Optional, Union, Type, cast, overload

# Handle Literal for different Python versions
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..config import config

logger = logging.getLogger(__name__)

# Type aliases
EngineConfig = Dict[str, Any]


class EngineFactory:
    """Factory for creating reliability engines - OSS Edition"""
    
    def __init__(self):
        self._engines_created = 0
        logger.info("Initialized EngineFactory (OSS Edition)")
    
    def create_engine(self, engine_config: Optional[EngineConfig] = None) -> Any:
        """
        Create a reliability engine instance
        
        OSS EDITION: Only creates OSS-compatible engines
        
        Args:
            engine_config: Engine configuration dictionary
            
        Returns:
            Configured reliability engine instance
        """
        try:
            from .reliability import V3ReliabilityEngine
            from .v3_reliability import V3ReliabilityEngine as EnhancedV3ReliabilityEngine
            
            # Determine which engine to create based on config
            use_enhanced = False
            
            if engine_config:
                # Check if enhanced features are requested
                rag_enabled = engine_config.get("rag_enabled", False)
                if rag_enabled and config.rag_enabled:
                    use_enhanced = True
            
            # Create engine
            if use_enhanced:
                logger.info("Creating EnhancedV3ReliabilityEngine (OSS limits apply)")
                
                # Get RAG and MCP for enhanced engine
                rag_graph = None
                mcp_server = None
                
                try:
                    from ..memory.rag_graph import RAGGraphMemory
                    from ..lazy import get_rag_graph
                    rag_graph = get_rag_graph()
                except ImportError:
                    logger.warning("RAGGraphMemory not available")
                
                try:
                    from ..lazy import get_mcp_server
                    mcp_server = get_mcp_server()
                except ImportError:
                    logger.warning("MCP server not available")
                
                # Create enhanced engine
                engine = EnhancedV3ReliabilityEngine(
                    rag_graph=rag_graph,
                    mcp_server=mcp_server
                )
                
                # Add OSS metadata to engine instance
                engine._oss_edition = True
                engine._requires_enterprise = False  # OSS doesn't require enterprise
                
            else:
                logger.info("Creating V3ReliabilityEngine (OSS Edition)")
                engine = V3ReliabilityEngine()
                
                # Add OSS metadata to engine instance
                engine._oss_edition = True
                engine._requires_enterprise = False  # OSS doesn't require enterprise
            
            self._engines_created += 1
            
            # Log OSS capabilities
            logger.info(f"OSS Engine Created: {engine.__class__.__name__}")
            logger.info(f"OSS Limits: 1000 incident nodes max, advisory mode only")
            
            if hasattr(engine, '_requires_enterprise') and engine._requires_enterprise:
                logger.info(
                    "💡 Upgrade to Enterprise for more features: "
                    "https://arf.dev/enterprise"
                )
            
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create engine: {e}")
            # Fallback to basic engine
            from .reliability import V3ReliabilityEngine
            engine = V3ReliabilityEngine()
            engine._oss_edition = True
            return engine
    
    def create_enhanced_engine(
        self, 
        enable_rag: bool = False,
        rag_nodes_limit: int = 1000
    ) -> Any:
        """
        Create enhanced V3 engine with specific features
        
        OSS EDITION: Learning disabled, RAG limited to 1000 nodes
        
        Args:
            enable_rag: Enable RAG graph (OSS limited to 1000 nodes)
            rag_nodes_limit: Maximum RAG nodes (capped at 1000 in OSS)
            
        Returns:
            Enhanced V3 reliability engine
        """
        # OSS: Cap RAG nodes
        if rag_nodes_limit > 1000:
            logger.warning(
                f"RAG nodes limit capped at 1000 (OSS max). "
                f"Requested: {rag_nodes_limit}"
            )
            rag_nodes_limit = 1000
        
        # Get RAG and MCP for enhanced engine
        rag_graph = None
        mcp_server = None
        
        if enable_rag:
            try:
                from ..lazy import get_rag_graph
                rag_graph = get_rag_graph()
            except ImportError:
                logger.warning("RAG graph not available")
        
        try:
            from ..lazy import get_mcp_server
            mcp_server = get_mcp_server()
        except ImportError:
            logger.warning("MCP server not available")
        
        from .v3_reliability import V3ReliabilityEngine as EnhancedV3ReliabilityEngine
        
        engine = EnhancedV3ReliabilityEngine(
            rag_graph=rag_graph,
            mcp_server=mcp_server
        )
        
        # Add OSS capabilities info as instance attributes
        engine._oss_capabilities = {
            "rag_enabled": enable_rag,
            "rag_nodes_limit": rag_nodes_limit,
            "learning_enabled": False,
            "execution_enabled": False,
            "upgrade_available": rag_nodes_limit >= 1000,
            "upgrade_url": "https://arf.dev/enterprise",
        }
        
        engine._oss_edition = True
        engine._requires_enterprise = False
        
        return engine
    
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
                "EnhancedV3ReliabilityEngine": True,
            },
            "limits": {
                "max_rag_incident_nodes": 1000,
                "max_rag_outcome_nodes": 5000,
                "mcp_modes": ["advisory"],
                "learning_enabled": False,
                "persistent_storage": False,
            },
            "capabilities": {
                "rag_analysis": config.rag_enabled,
                "anomaly_detection": True,
                "business_impact": True,
                "forecasting": True,
                "self_healing_advisory": True,
                "self_healing_execution": False,  # OSS: Advisory only
            },
            "requires_enterprise": (
                config.rag_max_incident_nodes >= 1000 or
                config.rag_max_outcome_nodes >= 5000 or
                config.mcp_mode != "advisory"
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
        violations = []
        warnings = []
        
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
def create_engine(engine_config: Optional[EngineConfig] = None) -> Any:
    """
    Create engine - backward compatibility function
    
    OSS EDITION: Returns OSS-compatible engine only
    """
    factory = EngineFactory()
    return factory.create_engine(engine_config)


def get_engine(engine_config: Optional[EngineConfig] = None) -> Any:
    """Alias for create_engine - backward compatibility"""
    return create_engine(engine_config)


# Export
__all__ = [
    "EngineFactory",
    "create_engine",
    "get_engine",
    "get_oss_engine_capabilities",
    "validate_oss_compatibility",
]
