"""
Engine Factory for ARF v3

Creates appropriate engine instance based on configuration
Implements feature flags and gradual rollout
"""

import logging
from typing import Optional, Dict, Any

from .reliability import EnhancedReliabilityEngine as V2Engine
from .v3_reliability import V3ReliabilityEngine
from .mcp_server import MCPServer, MCPMode
from ..memory.rag_graph import RAGGraphMemory
from ..config import config

logger = logging.getLogger(__name__)


class EngineFactory:
    """
    Factory for creating reliability engines based on configuration
    
    Implements Phase A: Silent Launch (Feature Flags)
    """
    
    @staticmethod
    def create_engine() -> Any:
        """
        Create appropriate reliability engine based on configuration
        
        Returns:
            V2Engine, V3ReliabilityEngine, or hybrid based on feature flags
        """
        # Check if any v3 features are enabled
        v3_features_enabled = (
            config.rag_enabled or 
            config.mcp_enabled or 
            config.learning_enabled or
            config.rollout_percentage > 0
        )
        
        if not v3_features_enabled:
            logger.info("Creating v2 engine (all v3 features disabled)")
            return V2Engine()
        
        # Create v3 components based on feature flags
        rag_graph = None
        mcp_server = None
        
        # Create RAG graph if enabled
        if config.rag_enabled:
            try:
                from ..lazy import get_faiss_index
                faiss_index = get_faiss_index()
                rag_graph = RAGGraphMemory(faiss_index)
                logger.info(f"Created RAG graph with {faiss_index.get_count()} vectors")
            except Exception as e:
                logger.error(f"Failed to create RAG graph: {e}", exc_info=True)
                rag_graph = None
        
        # Create MCP server if enabled
        if config.mcp_enabled:
            try:
                mcp_mode = MCPMode(config.mcp_mode)
                mcp_server = MCPServer(mode=mcp_mode)
                logger.info(f"Created MCP server in {mcp_mode.value} mode")
            except Exception as e:
                logger.error(f"Failed to create MCP server: {e}", exc_info=True)
                mcp_server = None
        
        # Check if we should create v3 engine
        should_create_v3 = (
            (config.rag_enabled and rag_graph is not None) or
            (config.mcp_enabled and mcp_server is not None) or
            config.learning_enabled
        )
        
        if should_create_v3:
            logger.info(
                f"Creating v3 engine: RAG={rag_graph is not None}, "
                f"MCP={mcp_server is not None}, Learning={config.learning_enabled}"
            )
            return V3ReliabilityEngine(
                rag_graph=rag_graph,
                mcp_server=mcp_server
            )
        else:
            logger.warning(
                "v3 features enabled but components failed to initialize, "
                "falling back to v2 engine"
            )
            return V2Engine()
    
    @staticmethod
    def get_engine_info() -> Dict[str, Any]:
        """Get information about the current engine configuration"""
        v3_features = config.v3_features
        safety_guardrails = config.safety_guardrails
        
        # Check which components are actually available
        components_available = {
            "rag": False,
            "mcp": False,
            "learning": config.learning_enabled
        }
        
        # Try to import components to check availability
        try:
            from ..memory.rag_graph import RAGGraphMemory
            from ..lazy import get_faiss_index
            faiss_index = get_faiss_index()
            components_available["rag"] = faiss_index is not None
        except:
            pass
        
        try:
            from .mcp_server import MCPServer
            components_available["mcp"] = True
        except:
            pass
        
        return {
            "version": "v3" if any(v3_features.values()) else "v2",
            "v3_features": v3_features,
            "components_available": components_available,
            "safety_guardrails": safety_guardrails,
            "rollout_percentage": config.rollout_percentage,
            "beta_testing": config.beta_testing_enabled,
            "config_source": "environment" if any(v3_features.values()) else "defaults"
        }
    
    @staticmethod
    def update_configuration(new_config: Dict[str, Any]) -> bool:
        """
        Update configuration at runtime
        
        Args:
            new_config: Dictionary with configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update rollout percentage
            if "rollout_percentage" in new_config:
                config.rollout_percentage = new_config["rollout_percentage"]
                logger.info(f"Updated rollout percentage to {config.rollout_percentage}%")
            
            # Update feature flags
            feature_flags = ["rag_enabled", "mcp_enabled", "learning_enabled", "beta_testing_enabled"]
            for flag in feature_flags:
                if flag in new_config:
                    setattr(config, flag, new_config[flag])
                    logger.info(f"Updated {flag} to {new_config[flag]}")
            
            # Update MCP mode
            if "mcp_mode" in new_config:
                config.mcp_mode = new_config["mcp_mode"]
                logger.info(f"Updated MCP mode to {config.mcp_mode}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}", exc_info=True)
            return False
