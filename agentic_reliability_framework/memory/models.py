"""
Graph Data Models for RAG Memory
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the RAG graph"""
    INCIDENT = "incident"
    OUTCOME = "outcome"
    ACTION = "action"
    COMPONENT = "component"


class EdgeType(Enum):
    """Types of edges in the RAG graph"""
    SIMILAR_TO = "similar_to"
    LEADS_TO = "leads_to"
    RESOLVED_BY = "resolved_by"
    CAUSED_BY = "caused_by"
    PRECEDES = "precedes"
    RELATED_TO = "related_to"


@dataclass
class IncidentNode:
    """Node representing an incident in the graph"""
    
    incident_id: str
    component: str
    severity: str
    timestamp: str
    metrics: Dict[str, float]
    agent_analysis: Dict[str, Any]
    embedding_id: Optional[int] = None
    faiss_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed fields
    def __post_init__(self) -> None:
        """Initialize IncidentNode with default values"""
        self.node_type = NodeType.INCIDENT
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "node_type": self.node_type.value,
            "incident_id": self.incident_id,
            "component": self.component,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "agent_analysis": self.agent_analysis,
            "embedding_id": self.embedding_id,
            "faiss_index": self.faiss_index,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncidentNode":
        """Create from dictionary"""
        return cls(
            incident_id=data["incident_id"],
            component=data["component"],
            severity=data["severity"],
            timestamp=data["timestamp"],
            metrics=data["metrics"],
            agent_analysis=data["agent_analysis"],
            embedding_id=data.get("embedding_id"),
            faiss_index=data.get("faiss_index"),
            metadata=data.get("metadata", {})
        )


@dataclass
class OutcomeNode:
    """Node representing the outcome/resolution of an incident"""
    
    outcome_id: str
    incident_id: str
    actions_taken: List[str]
    resolution_time_minutes: float
    success: bool
    lessons_learned: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize OutcomeNode with default values"""
        self.node_type = NodeType.OUTCOME
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "node_type": self.node_type.value,
            "outcome_id": self.outcome_id,
            "incident_id": self.incident_id,
            "actions_taken": self.actions_taken,
            "resolution_time_minutes": self.resolution_time_minutes,
            "success": self.success,
            "lessons_learned": self.lessons_learned,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutcomeNode":
        """Create from dictionary"""
        return cls(
            outcome_id=data["outcome_id"],
            incident_id=data["incident_id"],
            actions_taken=data["actions_taken"],
            resolution_time_minutes=data["resolution_time_minutes"],
            success=data["success"],
            lessons_learned=data["lessons_learned"],
            metadata=data.get("metadata", {})
        )


@dataclass
class GraphEdge:
    """Edge connecting nodes in the RAG graph"""
    
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize GraphEdge with default values"""
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        """Create from dictionary"""
        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class SimilarityResult:
    """Result of similarity search"""
    
    incident_node: IncidentNode
    similarity_score: float
    raw_score: float
    faiss_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "incident": self.incident_node.to_dict(),
            "similarity_score": self.similarity_score,
            "raw_score": self.raw_score,
            "faiss_index": self.faiss_index
        }
