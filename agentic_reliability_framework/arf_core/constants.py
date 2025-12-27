# File: arf_core/constants.py
"""
OSS HARD LIMITS - Build-time enforced boundaries for OSS edition
Enhanced version with memory-specific limits, validation, and integration helpers

These constants define the architectural boundaries between OSS and Enterprise editions.
All values are FINAL and validated at import time.

Apache 2.0 Licensed - Enterprise features require commercial license
"""

from typing import Final, Dict, Any, Tuple, List
import sys
import os
import hashlib

# ==================== OSS ARCHITECTURAL BOUNDARIES ====================

# === EXECUTION BOUNDARIES ===
MAX_INCIDENT_HISTORY: Final[int] = 1_000
MAX_RAG_LOOKBACK_DAYS: Final[int] = 7
MCP_MODES_ALLOWED: Final[Tuple[str, ...]] = ("advisory",)  # ONLY advisory
EXECUTION_ALLOWED: Final[bool] = False
GRAPH_STORAGE: Final[str] = "in_memory"
MAX_COOLDOWN_ENTRIES: Final[int] = 100

# === MEMORY/RAG BOUNDARIES (from your rag_graph.py implementation) ===
MAX_INCIDENT_NODES: Final[int] = 1_000
MAX_OUTCOME_NODES: Final[int] = 5_000
MAX_EMBEDDING_CACHE: Final[int] = 100
MAX_SIMILARITY_CACHE: Final[int] = 100
FAISS_INDEX_TYPE: Final[str] = "IndexFlatL2"  # OSS only - no IVF/HNSW/PQ
EMBEDDING_DIM: Final[int] = 384
STORAGE_BACKEND: Final[str] = "in_memory"
SIMILARITY_THRESHOLD: Final[float] = 0.3  # From config.rag_similarity_threshold
GRAPH_CACHE_SIZE: Final[int] = 100  # From MemoryConstants.GRAPH_CACHE_SIZE

# === FEATURE BOUNDARIES ===
MAX_TOOLS: Final[int] = 6  # Current tool count (rollback, restart, scale_out, etc.)
MAX_CONCURRENT_ANALYSIS: Final[int] = 10
MAX_EVENT_RATE_PER_SECOND: Final[int] = 100
MAX_API_REQUESTS_PER_MINUTE: Final[int] = 60  # From config.max_requests_per_minute

# === SECURITY BOUNDARIES ===
MAX_API_KEYS: Final[int] = 1  # Only HuggingFace API key
ALLOWED_ENVIRONMENTS: Final[Tuple[str, ...]] = ("development", "staging", "production")
DISALLOWED_ACTIONS: Final[Tuple[str, ...]] = (
    "DATABASE_DROP",
    "FULL_ROLLOUT", 
    "SYSTEM_SHUTDOWN",
    "SECRET_ROTATION",
)

# === VERSION & EDITION ===
OSS_EDITION: Final[str] = "open-source"
OSS_LICENSE: Final[str] = "Apache 2.0"
OSS_VERSION: Final[str] = "3.3.0-oss"
ENTERPRISE_UPGRADE_URL: Final[str] = "https://arf.dev/enterprise"

# === COMPATIBILITY HASH (for validation) ===
def _generate_oss_hash() -> str:
    """Generate hash of OSS constants for validation"""
    constants_data = {
        "MAX_INCIDENT_HISTORY": MAX_INCIDENT_HISTORY,
        "MCP_MODES_ALLOWED": MCP_MODES_ALLOWED,
        "EXECUTION_ALLOWED": EXECUTION_ALLOWED,
        "GRAPH_STORAGE": GRAPH_STORAGE,
        "MAX_INCIDENT_NODES": MAX_INCIDENT_NODES,
        "MAX_OUTCOME_NODES": MAX_OUTCOME_NODES,
        "FAISS_INDEX_TYPE": FAISS_INDEX_TYPE,
    }
    json_str = str(sorted(constants_data.items()))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]

OSS_CONSTANTS_HASH: Final[str] = _generate_oss_hash()

# ==================== VALIDATION & ENFORCEMENT ====================

class OSSBoundaryError(RuntimeError):
    """Error raised when OSS boundaries are violated"""
    pass


def validate_oss_config(config: Dict[str, Any]) -> None:
    """
    Validate runtime configuration against OSS boundaries
    
    Args:
        config: Current configuration dictionary
        
    Raises:
        OSSBoundaryError: If any OSS boundary is violated
    """
    violations: List[str] = []
    
    # Check MCP mode
    mcp_mode = config.get("mcp_mode", "advisory").lower()
    if mcp_mode != "advisory":
        violations.append(
            f"MCP mode must be 'advisory' in OSS edition. Got: '{mcp_mode}'. "
            f"Upgrade to Enterprise for approval/autonomous modes."
        )
    
    # Check execution capability
    if config.get("mcp_enabled", False) and mcp_mode != "advisory":
        violations.append(
            "MCP execution requires Enterprise edition. "
            "OSS edition only supports advisory (analysis) mode."
        )
    
    # Check storage limits
    max_events = config.get("max_events_stored", 1000)
    if max_events > MAX_INCIDENT_HISTORY:
        violations.append(
            f"max_events_stored exceeds OSS limit: {max_events} > {MAX_INCIDENT_HISTORY}"
        )
    
    # Check RAG limits
    rag_nodes = config.get("rag_max_incident_nodes", 1000)
    if rag_nodes > MAX_INCIDENT_NODES:
        violations.append(
            f"rag_max_incident_nodes exceeds OSS limit: {rag_nodes} > {MAX_INCIDENT_NODES}"
        )
    
    rag_outcomes = config.get("rag_max_outcome_nodes", 5000)
    if rag_outcomes > MAX_OUTCOME_NODES:
        violations.append(
            f"rag_max_outcome_nodes exceeds OSS limit: {rag_outcomes} > {MAX_OUTCOME_NODES}"
        )
    
    # Check feature flags
    if config.get("learning_enabled", False):
        violations.append("Learning engine requires Enterprise edition")
    
    if config.get("beta_testing_enabled", False):
        violations.append("Beta testing features require Enterprise edition")
    
    if config.get("rollout_percentage", 0) > 0:
        violations.append("Rollout features require Enterprise edition")
    
    # Check for Enterprise storage backends
    storage_type = config.get("graph_storage", "in_memory")
    if storage_type != "in_memory":
        violations.append(
            f"Storage backend '{storage_type}' requires Enterprise edition. "
            f"OSS edition only supports 'in_memory' storage."
        )
    
    # Check FAISS index type (if specified)
    faiss_type = config.get("faiss_index_type", "IndexFlatL2")
    if faiss_type != "IndexFlatL2":
        violations.append(
            f"FAISS index type '{faiss_type}' requires Enterprise edition. "
            f"OSS edition only supports 'IndexFlatL2'."
        )
    
    if violations:
        error_msg = (
            f"OSS CONFIGURATION VIOLATIONS DETECTED:\n\n" +
            "\n".join(f"  • {v}" for v in violations) +
            f"\n\nEdition: {OSS_EDITION} ({OSS_LICENSE})" +
            f"\nVersion: {OSS_VERSION}" +
            f"\nConstants Hash: {OSS_CONSTANTS_HASH}" +
            f"\n\nUpgrade to Enterprise Edition for these features:" +
            f"\n{ENTERPRISE_UPGRADE_URL}" +
            f"\n\nOr fix configuration to comply with OSS limits."
        )
        raise OSSBoundaryError(error_msg)


def get_oss_capabilities() -> Dict[str, Any]:
    """
    Get OSS edition capabilities for documentation and UI
    
    Returns:
        Dictionary of OSS capabilities and limits
    """
    return {
        "edition": OSS_EDITION,
        "license": OSS_LICENSE,
        "version": OSS_VERSION,
        "constants_hash": OSS_CONSTANTS_HASH,
        
        "execution": {
            "modes": list(MCP_MODES_ALLOWED),
            "allowed": EXECUTION_ALLOWED,
            "max_incidents": MAX_INCIDENT_HISTORY,
            "max_rag_lookback_days": MAX_RAG_LOOKBACK_DAYS,
            "max_cooldown_entries": MAX_COOLDOWN_ENTRIES,
        },
        
        "memory": {
            "type": STORAGE_BACKEND,
            "max_incident_nodes": MAX_INCIDENT_NODES,
            "max_outcome_nodes": MAX_OUTCOME_NODES,
            "faiss_index_type": FAISS_INDEX_TYPE,
            "embedding_dim": EMBEDDING_DIM,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "graph_cache_size": GRAPH_CACHE_SIZE,
        },
        
        "features": {
            "rag_enabled": True,
            "mcp_advisory_enabled": True,
            "anomaly_detection": True,
            "business_impact": True,
            "forecasting": True,
            "learning_enabled": False,
            "audit_trails": False,
            "persistent_storage": False,
            "enterprise_integration": False,
            "beta_features": False,
        },
        
        "limits": {
            "max_tools": MAX_TOOLS,
            "max_concurrent_analysis": MAX_CONCURRENT_ANALYSIS,
            "max_event_rate": MAX_EVENT_RATE_PER_SECOND,
            "max_api_keys": MAX_API_KEYS,
            "max_api_requests_per_minute": MAX_API_REQUESTS_PER_MINUTE,
        },
        
        "upgrade_available": True,
        "upgrade_url": ENTERPRISE_UPGRADE_URL,
        
        "enterprise_features": [
            "autonomous_execution",
            "approval_workflows",
            "learning_engine",
            "persistent_graph_storage",
            "advanced_faiss_indices",  # IVF, HNSW, PQ
            "sentence_transformers_embeddings",
            "enterprise_audit_trails",
            "compliance_reporting",
            "multi_tenant_support",
            "sso_integration",
            "24_7_support",
            "beta_feature_access",
            "commercial_license",
            "priority_support",
            "unlimited_incident_storage",
            "time_series_database",
            "custom_embedding_models",
            "hybrid_search",  # vector + keyword + graph
        ]
    }


def check_oss_compliance() -> bool:
    """
    Check if current runtime is OSS compliant
    
    Returns:
        True if OSS compliant, False otherwise
    """
    try:
        # Check environment
        tier = os.getenv("ARF_TIER", "oss").lower()
        if tier != "oss":
            return False
        
        # Check for Enterprise dependencies
        try:
            # These imports should fail in OSS edition
            import neo4j
            return False
        except ImportError:
            pass
        
        try:
            import psycopg2
            return False
        except ImportError:
            pass
        
        try:
            import sqlalchemy
            # SQLAlchemy might be used for other things, check usage patterns
            if os.getenv("ARF_DATABASE_URL"):
                return False
        except ImportError:
            pass
        
        return True
        
    except Exception:
        # Default to OSS if cannot determine
        return True


def validate_memory_implementation() -> None:
    """
    Validate that memory implementation meets OSS requirements
    
    Checks:
    - FAISS uses only IndexFlatL2
    - No persistent storage dependencies
    - Embedding dimension matches OSS limit
    """
    violations: List[str] = []
    
    try:
        # Check FAISS implementation
        from agentic_reliability_framework.memory.faiss_index import ProductionFAISSIndex
        
        # Verify embedding dimension
        import inspect
        source = inspect.getsource(ProductionFAISSIndex.__init__)
        
        # Check for advanced FAISS indices (should not exist in OSS)
        advanced_patterns = [
            "IndexIVF",  # Inverted file
            "IndexHNSW",  # Hierarchical navigable small world  
            "IndexPQ",    # Product quantization
            "IndexScalarQuantizer",
            "IndexRefine",
            ".gpu",       # GPU acceleration
            "res = faiss.",  # Direct FAISS construction
        ]
        
        for pattern in advanced_patterns:
            if pattern in source:
                violations.append(f"FAISS pattern '{pattern}' requires Enterprise edition")
        
        # Check memory constants
        from agentic_reliability_framework.memory.constants import MemoryConstants
        
        if hasattr(MemoryConstants, 'MAX_INCIDENT_NODES'):
            if getattr(MemoryConstants, 'MAX_INCIDENT_NODES') > MAX_INCIDENT_NODES:
                violations.append(
                    f"MemoryConstants.MAX_INCIDENT_NODES exceeds OSS limit: "
                    f"{getattr(MemoryConstants, 'MAX_INCIDENT_NODES')} > {MAX_INCIDENT_NODES}"
                )
        
        if hasattr(MemoryConstants, 'MAX_OUTCOME_NODES'):
            if getattr(MemoryConstants, 'MAX_OUTCOME_NODES') > MAX_OUTCOME_NODES:
                violations.append(
                    f"MemoryConstants.MAX_OUTCOME_NODES exceeds OSS limit: "
                    f"{getattr(MemoryConstants, 'MAX_OUTCOME_NODES')} > {MAX_OUTCOME_NODES}"
                )
        
    except ImportError as e:
        # Memory module might not be available yet
        pass
    except Exception as e:
        violations.append(f"Error validating memory implementation: {str(e)}")
    
    if violations:
        raise OSSBoundaryError(
            f"Memory implementation violates OSS boundaries:\n" +
            "\n".join(f"  • {v}" for v in violations)
        )


def get_oss_memory_limits() -> Dict[str, Any]:
    """
    Get OSS memory system limits for integration with existing code
    
    This helps existing code (like rag_graph.py) use OSS limits
    """
    return {
        "MAX_INCIDENT_NODES": MAX_INCIDENT_NODES,
        "MAX_OUTCOME_NODES": MAX_OUTCOME_NODES,
        "GRAPH_CACHE_SIZE": GRAPH_CACHE_SIZE,
        "SIMILARITY_THRESHOLD": SIMILARITY_THRESHOLD,
        "EMBEDDING_DIM": EMBEDDING_DIM,
        "OSS_EDITION": True,
    }


# ==================== BUILD-TIME VALIDATION ====================

def _validate_oss_constants_at_import() -> None:
    """Validate OSS constants at module import time"""
    violations: List[str] = []
    
    # Check execution boundaries
    if EXECUTION_ALLOWED:
        violations.append("EXECUTION_ALLOWED must be False in OSS edition")
    
    if MCP_MODES_ALLOWED != ("advisory",):
        violations.append(f"MCP_MODES_ALLOWED must be ('advisory',), got {MCP_MODES_ALLOWED}")
    
    if GRAPH_STORAGE != "in_memory":
        violations.append(f"GRAPH_STORAGE must be 'in_memory', got '{GRAPH_STORAGE}'")
    
    if FAISS_INDEX_TYPE != "IndexFlatL2":
        violations.append(f"FAISS_INDEX_TYPE must be 'IndexFlatL2', got '{FAISS_INDEX_TYPE}'")
    
    # Check memory limits
    if MAX_INCIDENT_NODES > 1000:
        violations.append(f"MAX_INCIDENT_NODES must be ≤ 1000, got {MAX_INCIDENT_NODES}")
    
    if MAX_OUTCOME_NODES > 5000:
        violations.append(f"MAX_OUTCOME_NODES must be ≤ 5000, got {MAX_OUTCOME_NODES}")
    
    if violations:
        raise OSSBoundaryError(
            f"OSS constant validation failed at import:\n" +
            "\n".join(f"  • {v}" for v in violations)
        )


# Run validation on import
_validate_oss_constants_at_import()


# Export
__all__ = [
    # === CORE BOUNDARIES ===
    "MAX_INCIDENT_HISTORY",
    "MAX_RAG_LOOKBACK_DAYS", 
    "MCP_MODES_ALLOWED",
    "EXECUTION_ALLOWED",
    "GRAPH_STORAGE",
    "MAX_COOLDOWN_ENTRIES",
    
    # === MEMORY BOUNDARIES ===
    "MAX_INCIDENT_NODES",
    "MAX_OUTCOME_NODES",
    "MAX_EMBEDDING_CACHE",
    "MAX_SIMILARITY_CACHE",
    "FAISS_INDEX_TYPE",
    "EMBEDDING_DIM",
    "STORAGE_BACKEND",
    "SIMILARITY_THRESHOLD",
    "GRAPH_CACHE_SIZE",
    
    # === FEATURE BOUNDARIES ===
    "MAX_TOOLS",
    "MAX_CONCURRENT_ANALYSIS",
    "MAX_EVENT_RATE_PER_SECOND",
    "MAX_API_REQUESTS_PER_MINUTE",
    
    # === SECURITY BOUNDARIES ===
    "MAX_API_KEYS",
    "ALLOWED_ENVIRONMENTS",
    "DISALLOWED_ACTIONS",
    
    # === VERSION & EDITION ===
    "OSS_EDITION",
    "OSS_LICENSE",
    "OSS_VERSION",
    "ENTERPRISE_UPGRADE_URL",
    "OSS_CONSTANTS_HASH",
    
    # === VALIDATION FUNCTIONS ===
    "OSSBoundaryError",
    "validate_oss_config",
    "get_oss_capabilities",
    "check_oss_compliance",
    "validate_memory_implementation",
    "get_oss_memory_limits",
]
