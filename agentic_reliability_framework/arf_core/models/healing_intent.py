# File: arf_core/models/healing_intent.py
"""
Healing Intent - OSS creates, Enterprise executes
Enhanced with OSS limits and RAG integration

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

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, ClassVar
from datetime import datetime
import hashlib
import json
import time
import uuid
from enum import Enum

from ..constants import (
    OSS_EDITION,
    OSS_LICENSE,
    ENTERPRISE_UPGRADE_URL,
    EXECUTION_ALLOWED
)


class HealingIntentError(Exception):
    """Base exception for HealingIntent errors"""
    pass


class SerializationError(HealingIntentError):
    """Error during serialization/deserialization"""
    pass


class ValidationError(HealingIntentError):
    """Error during intent validation"""
    pass


class IntentSource(str, Enum):
    """Source of the healing intent"""
    OSS_ANALYSIS = "oss_analysis"
    HUMAN_OVERRIDE = "human_override"
    AUTOMATED_LEARNING = "automated_learning"  # Enterprise only
    SCHEDULED_ACTION = "scheduled_action"  # Enterprise only
    RAG_SIMILARITY = "rag_similarity"  # New: From RAG graph similarity


class IntentStatus(str, Enum):
    """Status of the healing intent"""
    CREATED = "created"  # OSS created, not yet sent to Enterprise
    PENDING_EXECUTION = "pending_execution"  # Sent to Enterprise, waiting
    EXECUTING = "executing"  # Enterprise is executing
    COMPLETED = "completed"  # Enterprise executed successfully
    FAILED = "failed"  # Enterprise execution failed
    REJECTED = "rejected"  # Enterprise rejected the intent
    CANCELLED = "cancelled"  # Intent was cancelled
    OSS_ADVISORY_ONLY = "oss_advisory_only"  # New: OSS can only advise


@dataclass(frozen=True, slots=True)
class HealingIntent:
    """
    OSS-generated healing recommendation for Enterprise execution
    
    Enhanced with:
    - OSS edition metadata
    - RAG integration capabilities
    - OSS boundary validation
    
    This is the clean boundary between OSS intelligence and Enterprise execution:
    - OSS creates HealingIntent through analysis (advisory only)
    - Enterprise executes HealingIntent through MCP server
    - Immutable (frozen) to ensure consistency across OSS→Enterprise handoff
    """
    
    # === CORE ACTION FIELDS (Sent to Enterprise) ===
    action: str                          # Tool name, e.g., "restart_container"
    component: str                       # Target component
    parameters: Dict[str, Any] = field(default_factory=dict)  # Action parameters
    justification: str = ""              # OSS reasoning chain
    
    # === CONFIDENCE & METADATA ===
    confidence: float = 0.85             # OSS confidence score (0.0 to 1.0)
    incident_id: str = ""                # Source incident identifier
    detected_at: float = field(default_factory=time.time)  # When OSS detected
    
    # === OSS ANALYSIS CONTEXT (Stays in OSS) ===
    reasoning_chain: Optional[List[Dict[str, Any]]] = None
    similar_incidents: Optional[List[Dict[str, Any]]] = None
    rag_similarity_score: Optional[float] = None
    source: IntentSource = IntentSource.OSS_ANALYSIS
    
    # === IMMUTABLE IDENTIFIERS ===
    intent_id: str = field(default_factory=lambda: f"intent_{uuid.uuid4().hex[:16]}")
    created_at: float = field(default_factory=time.time)
    
    # === EXECUTION METADATA (Set by Enterprise) ===
    status: IntentStatus = IntentStatus.CREATED
    execution_id: Optional[str] = None
    executed_at: Optional[float] = None
    execution_result: Optional[Dict[str, Any]] = None
    enterprise_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # === OSS EDITION METADATA ===
    oss_edition: str = OSS_EDITION
    oss_license: str = OSS_LICENSE
    requires_enterprise: bool = True  # Always True for OSS-generated intents
    execution_allowed: bool = EXECUTION_ALLOWED  # From OSS constants
    
    # Class constants for validation
    MIN_CONFIDENCE: ClassVar[float] = 0.0
    MAX_CONFIDENCE: ClassVar[float] = 1.0
    MAX_JUSTIFICATION_LENGTH: ClassVar[int] = 1000
    MAX_PARAMETERS_SIZE: ClassVar[int] = 100  # Maximum number of parameters
    MAX_SIMILAR_INCIDENTS: ClassVar[int] = 10  # Limit RAG context size
    VERSION: ClassVar[str] = "1.1.0"  # Bumped for OSS integration
    
    def __post_init__(self):
        """Validate HealingIntent after initialization with OSS boundaries"""
        self._validate_oss_boundaries()
    
    def _validate_oss_boundaries(self) -> None:
        """Validate all fields of the HealingIntent against OSS limits"""
        errors: List[str] = []
        
        # Validate confidence range
        if not (self.MIN_CONFIDENCE <= self.confidence <= self.MAX_CONFIDENCE):
            errors.append(
                f"Confidence must be between {self.MIN_CONFIDENCE} and "
                f"{self.MAX_CONFIDENCE}, got {self.confidence}"
            )
        
        # Validate justification length
        if len(self.justification) > self.MAX_JUSTIFICATION_LENGTH:
            errors.append(
                f"Justification exceeds max length {self.MAX_JUSTIFICATION_LENGTH}"
            )
        
        # Validate action and component
        if not self.action.strip():
            errors.append("Action cannot be empty")
        
        if not self.component.strip():
            errors.append("Component cannot be empty")
        
        # Validate parameters size
        if len(self.parameters) > self.MAX_PARAMETERS_SIZE:
            errors.append(
                f"Too many parameters: {len(self.parameters)} > {self.MAX_PARAMETERS_SIZE}"
            )
        
        # Validate parameters are JSON serializable
        try:
            json.dumps(self.parameters)
        except (TypeError, ValueError) as e:
            errors.append(f"Parameters must be JSON serializable: {e}")
        
        # SIMPLEST FIX: Validate similar incidents structure and size (OSS limit)
        if self.similar_incidents:
            if len(self.similar_incidents) > self.MAX_SIMILAR_INCIDENTS:
                errors.append(
                    f"Too many similar incidents: {len(self.similar_incidents)} > "
                    f"{self.MAX_SIMILAR_INCIDENTS}"
                )
            
            for i, incident_item in enumerate(self.similar_incidents):
                # SIMPLEST FIX: Remove the 'continue' statement that confuses MyPy
                if not isinstance(incident_item, dict):
                    errors.append(f"Similar incident {i} must be a dictionary")
                # Check similarity only if item is a dict AND has similarity key
                elif isinstance(incident_item, dict) and "similarity" in incident_item:
                    similarity = incident_item["similarity"]
                    # Check if similarity is numeric
                    if isinstance(similarity, (int, float)):
                        similarity_float = float(similarity)
                        if not (0.0 <= similarity_float <= 1.0):
                            errors.append(
                                f"Similar incident {i} similarity must be between 0.0 and 1.0, "
                                f"got {similarity}"
                            )
                    else:
                        errors.append(
                            f"Similar incident {i} similarity must be a number, "
                            f"got {type(similarity).__name__}"
                        )
        
        # Validate OSS edition restrictions
        if self.execution_allowed and self.oss_edition == OSS_EDITION:
            errors.append("Execution not allowed in OSS edition")
        
        if self.status == IntentStatus.EXECUTING and self.oss_edition == OSS_EDITION:
            errors.append("EXECUTING status not allowed in OSS edition")
        
        if errors:
            raise ValidationError(
                f"HealingIntent validation failed:\n" +
                "\n".join(f"  • {error}" for error in errors)
            )
    
    @property
    def deterministic_id(self) -> str:
        """
        Deterministic ID for idempotency based on action + component + parameters
        
        This ensures the same action on the same component with the same parameters
        generates the same intent ID, preventing duplicate executions.
        """
        data = {
            "action": self.action,
            "component": self.component,
            "parameters": self._normalize_parameters(self.parameters),
            "incident_id": self.incident_id,
            "detected_at": int(self.detected_at),
            "oss_edition": self.oss_edition,
        }
        
        # Sort keys for deterministic JSON
        json_str = json.dumps(data, sort_keys=True, default=str)
        
        # Create hash-based ID
        hash_digest = hashlib.sha256(json_str.encode()).hexdigest()
        return f"intent_{hash_digest[:16]}"
    
    @property
    def age_seconds(self) -> float:
        """Get age of intent in seconds"""
        return time.time() - self.created_at
    
    @property
    def is_executable(self) -> bool:
        """Check if intent is ready for execution"""
        # In OSS edition, nothing is executable
        if self.oss_edition == OSS_EDITION:
            return False
        
        return self.status in [IntentStatus.CREATED, IntentStatus.PENDING_EXECUTION]
    
    @property
    def is_oss_advisory(self) -> bool:
        """Check if this is an OSS advisory-only intent"""
        return self.oss_edition == OSS_EDITION and not self.execution_allowed
    
    @property
    def requires_enterprise_upgrade(self) -> bool:
        """Check if intent requires Enterprise upgrade"""
        return self.requires_enterprise and self.oss_edition == OSS_EDITION
    
    def to_enterprise_request(self) -> Dict[str, Any]:
        """
        Convert to Enterprise API request format
        
        Returns only the data needed for Enterprise execution.
        OSS analysis context stays in OSS.
        """
        return {
            # Core execution fields
            "intent_id": self.deterministic_id,
            "action": self.action,
            "component": self.component,
            "parameters": self.parameters,
            "justification": self.justification,
            
            # OSS metadata for Enterprise context
            "confidence": self.confidence,
            "incident_id": self.incident_id,
            "detected_at": self.detected_at,
            "created_at": self.created_at,
            "source": self.source.value,
            
            # OSS edition information
            "oss_edition": self.oss_edition,
            "oss_license": self.oss_license,
            "requires_enterprise": self.requires_enterprise,
            "execution_allowed": self.execution_allowed,
            "version": self.VERSION,
            
            # Minimal OSS context (for debugging only)
            "oss_metadata": {
                "similar_incidents_count": len(self.similar_incidents) if self.similar_incidents else 0,
                "rag_similarity_score": self.rag_similarity_score,
                "has_reasoning_chain": self.reasoning_chain is not None,
                "source": self.source.value,
                "is_oss_advisory": self.is_oss_advisory,
            },
            
            # Upgrade information
            "upgrade_url": ENTERPRISE_UPGRADE_URL,
            "enterprise_features": [
                "autonomous_execution",
                "approval_workflows",
                "persistent_storage",
                "learning_engine",
                "audit_trails",
                "compliance_reports",
                "multi_tenant_support",
                "sso_integration",
                "24_7_support"
            ]
        }
    
    def to_mcp_request(self) -> Dict[str, Any]:
        """
        Convert to existing MCP request format for backward compatibility
        
        Returns:
            Dictionary compatible with existing MCPServer.execute_tool()
        """
        return {
            "request_id": self.deterministic_id,
            "tool": self.action,
            "component": self.component,
            "parameters": self.parameters,
            "justification": self.justification,
            "mode": "advisory",  # Will be overridden by Enterprise
            "timestamp": self.detected_at,
            "metadata": {
                "intent_id": self.deterministic_id,
                "oss_confidence": self.confidence,
                "requires_enterprise": True,
                "oss_generated": True,
                "oss_edition": self.oss_edition,
                "version": self.VERSION,
                "source": self.source.value,
            }
        }
    
    def to_dict(self, include_oss_context: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        
        Args:
            include_oss_context: Whether to include OSS analysis context
                (should be False when sending to Enterprise)
        
        Returns:
            Dictionary representation of the intent
        """
        data = asdict(self)
        
        # Convert enums to strings
        data["source"] = self.source.value
        data["status"] = self.status.value
        
        # Remove OSS context if not needed
        if not include_oss_context:
            data.pop("reasoning_chain", None)
            data.pop("similar_incidents", None)
            data.pop("rag_similarity_score", None)
        
        # Add computed properties
        data["deterministic_id"] = self.deterministic_id
        data["age_seconds"] = self.age_seconds
        data["is_executable"] = self.is_executable
        data["is_oss_advisory"] = self.is_oss_advisory
        data["requires_enterprise_upgrade"] = self.requires_enterprise_upgrade
        data["version"] = self.VERSION
        
        return data
    
    def with_execution_result(
        self,
        execution_id: str,
        executed_at: float,
        result: Dict[str, Any],
        status: IntentStatus = IntentStatus.COMPLETED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "HealingIntent":
        """
        Create a new HealingIntent with execution results (used by Enterprise)
        
        This is how Enterprise updates the intent after execution.
        Returns a new immutable intent with execution results.
        """
        # Create a new dataclass with updated fields
        return HealingIntent(
            # Core fields (copied)
            action=self.action,
            component=self.component,
            parameters=self.parameters,
            justification=self.justification,
            confidence=self.confidence,
            incident_id=self.incident_id,
            detected_at=self.detected_at,
            
            # OSS context (copied)
            reasoning_chain=self.reasoning_chain,
            similar_incidents=self.similar_incidents,
            rag_similarity_score=self.rag_similarity_score,
            source=self.source,
            
            # Identifiers (copied)
            intent_id=self.intent_id,
            created_at=self.created_at,
            
            # OSS metadata (copied)
            oss_edition=self.oss_edition,
            oss_license=self.oss_license,
            requires_enterprise=self.requires_enterprise,
            execution_allowed=self.execution_allowed,
            
            # Updated execution fields
            status=status,
            execution_id=execution_id,
            executed_at=executed_at,
            execution_result=result,
            enterprise_metadata={**(self.enterprise_metadata or {}), **(metadata or {})}
        )
    
    def mark_as_sent_to_enterprise(self) -> "HealingIntent":
        """
        Mark intent as sent to Enterprise (used by OSS)
        
        Returns a new intent with status updated to PENDING_EXECUTION
        """
        return HealingIntent(
            # Core fields (copied)
            action=self.action,
            component=self.component,
            parameters=self.parameters,
            justification=self.justification,
            confidence=self.confidence,
            incident_id=self.incident_id,
            detected_at=self.detected_at,
            
            # OSS context (copied)
            reasoning_chain=self.reasoning_chain,
            similar_incidents=self.similar_incidents,
            rag_similarity_score=self.rag_similarity_score,
            source=self.source,
            
            # Identifiers (copied)
            intent_id=self.intent_id,
            created_at=self.created_at,
            
            # OSS metadata (copied)
            oss_edition=self.oss_edition,
            oss_license=self.oss_license,
            requires_enterprise=self.requires_enterprise,
            execution_allowed=self.execution_allowed,
            
            # Updated status
            status=IntentStatus.PENDING_EXECUTION,
            execution_id=self.execution_id,
            executed_at=self.executed_at,
            execution_result=self.execution_result,
            enterprise_metadata=self.enterprise_metadata
        )
    
    def mark_as_oss_advisory(self) -> "HealingIntent":
        """
        Mark intent as OSS advisory only
        
        Used when OSS creates an intent that can only be advisory
        """
        return HealingIntent(
            # Core fields (copied)
            action=self.action,
            component=self.component,
            parameters=self.parameters,
            justification=self.justification,
            confidence=self.confidence,
            incident_id=self.incident_id,
            detected_at=self.detected_at,
            
            # OSS context (copied)
            reasoning_chain=self.reasoning_chain,
            similar_incidents=self.similar_incidents,
            rag_similarity_score=self.rag_similarity_score,
            source=IntentSource.OSS_ANALYSIS,
            
            # Identifiers (copied)
            intent_id=self.intent_id,
            created_at=self.created_at,
            
            # OSS metadata (copied)
            oss_edition=self.oss_edition,
            oss_license=self.oss_license,
            requires_enterprise=self.requires_enterprise,
            execution_allowed=False,  # Force no execution in OSS
            
            # Updated status
            status=IntentStatus.OSS_ADVISORY_ONLY,
            execution_id=self.execution_id,
            executed_at=self.executed_at,
            execution_result=self.execution_result,
            enterprise_metadata=self.enterprise_metadata
        )
    
    @classmethod
    def from_mcp_request(cls, request: Dict[str, Any]) -> "HealingIntent":
        """
        Create HealingIntent from existing MCP request
        
        Provides backward compatibility with existing code
        """
        metadata = request.get("metadata", {})
        
        return cls(
            action=request.get("tool", ""),
            component=request.get("component", ""),
            parameters=request.get("parameters", {}),
            justification=request.get("justification", ""),
            incident_id=metadata.get("incident_id", ""),
            detected_at=request.get("timestamp", time.time()),
            intent_id=metadata.get("intent_id", f"intent_{uuid.uuid4().hex[:16]}"),
            source=IntentSource(metadata.get("source", IntentSource.OSS_ANALYSIS.value)),
            oss_edition=metadata.get("oss_edition", OSS_EDITION),
            requires_enterprise=metadata.get("requires_enterprise", True),
            execution_allowed=metadata.get("execution_allowed", False),
        )
    
    @classmethod
    def from_analysis(
        cls,
        action: str,
        component: str,
        parameters: Dict[str, Any],
        justification: str,
        confidence: float,
        similar_incidents: Optional[List[Dict[str, Any]]] = None,
        reasoning_chain: Optional[List[Dict[str, Any]]] = None,
        incident_id: str = "",
        source: IntentSource = IntentSource.OSS_ANALYSIS,
        rag_similarity_score: Optional[float] = None,
    ) -> "HealingIntent":
        """
        Factory method for creating HealingIntent from OSS analysis
        
        This is the primary way OSS creates intents.
        Enhanced with RAG similarity integration.
        """
        # Apply OSS limits to similar incidents
        if similar_incidents and len(similar_incidents) > cls.MAX_SIMILAR_INCIDENTS:
            similar_incidents = similar_incidents[:cls.MAX_SIMILAR_INCIDENTS]
        
        # Calculate enhanced confidence based on similar incidents
        enhanced_confidence = confidence
        if similar_incidents:
            # Boost confidence if we have historical context
            similarity_scores = [
                inc.get("similarity", 0.0)
                for inc in similar_incidents
                if "similarity" in inc
            ]
            if similarity_scores:
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                # Cap the boost to prevent overconfidence
                confidence_boost = min(0.2, avg_similarity * 0.3)
                enhanced_confidence = min(confidence * (1.0 + confidence_boost), cls.MAX_CONFIDENCE)
        
        # Use provided RAG score or calculate from similar incidents
        final_rag_score = rag_similarity_score
        if final_rag_score is None and similar_incidents and len(similar_incidents) > 0:
            final_rag_score = sum(
                inc.get("similarity", 0.0)
                for inc in similar_incidents[:3]
            ) / min(3, len(similar_incidents))
        
        return cls(
            action=action,
            component=component,
            parameters=parameters,
            justification=justification,
            confidence=enhanced_confidence,
            incident_id=incident_id,
            similar_incidents=similar_incidents,
            reasoning_chain=reasoning_chain,
            rag_similarity_score=final_rag_score,
            source=source,
            oss_edition=OSS_EDITION,
            requires_enterprise=True,
            execution_allowed=False,  # OSS never executes
        )
    
    @classmethod
    def from_rag_recommendation(
        cls,
        action: str,
        component: str,
        parameters: Dict[str, Any],
        rag_similarity_score: float,
        similar_incidents: List[Dict[str, Any]],
        justification_template: str = "Based on {count} similar historical incidents with {success_rate:.0%} success rate",
        success_rate: Optional[float] = None,
    ) -> "HealingIntent":
        """
        Create HealingIntent from RAG graph recommendation
        
        Specialized factory for RAG-based recommendations
        """
        if not similar_incidents:
            raise ValidationError("RAG recommendation requires similar incidents")
        
        # Calculate success rate if not provided
        if success_rate is None:
            successful = sum(1 for inc in similar_incidents if inc.get("success", False))
            # similar_incidents is guaranteed to be truthy here (otherwise exception raised above)
            success_rate = successful / len(similar_incidents)
        
        # Generate justification
        justification = justification_template.format(
            count=len(similar_incidents),
            success_rate=success_rate or 0.0,
            action=action,
            component=component,
        )
        
        # Calculate confidence based on RAG similarity
        base_confidence = rag_similarity_score * 0.8  # Scale similarity to confidence
        if success_rate:
            base_confidence = base_confidence * (0.7 + success_rate * 0.3)
        
        return cls.from_analysis(
            action=action,
            component=component,
            parameters=parameters,
            justification=justification,
            confidence=min(base_confidence, 0.95),  # Cap at 95%
            similar_incidents=similar_incidents,
            incident_id=similar_incidents[0].get("incident_id", "") if similar_incidents else "",
            source=IntentSource.RAG_SIMILARITY,
            rag_similarity_score=rag_similarity_score,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealingIntent":
        """
        Create from dictionary (deserialize)
        
        Handles versioning and field conversion
        """
        # Handle versioning
        version = data.get("version", "1.0.0")
        
        # Convert string enums back to Enum instances
        if "source" in data and isinstance(data["source"], str):
            data["source"] = IntentSource(data["source"])
        
        if "status" in data and isinstance(data["status"], str):
            data["status"] = IntentStatus(data["status"])
        
        # Remove computed fields that shouldn't be in constructor
        data.pop("deterministic_id", None)
        data.pop("age_seconds", None)
        data.pop("is_executable", None)
        data.pop("is_oss_advisory", None)
        data.pop("requires_enterprise_upgrade", None)
        data.pop("version", None)
        
        return cls(**data)
    
    def _normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameters for deterministic hashing
        
        Ensures that parameter order and minor format differences
        don't affect the deterministic ID.
        """
        normalized: Dict[str, Any] = {}
        
        for key, value in sorted(params.items()):
            if isinstance(value, (int, float, str, bool, type(None))):
                normalized[key] = value
            elif isinstance(value, (list, tuple)):
                # Recursively normalize list items
                normalized_items: List[Any] = []
                for v in value:
                    if isinstance(v, dict):
                        normalized_items.append(self._normalize_parameters(v))
                    else:
                        normalized_items.append(v)
                normalized[key] = tuple(normalized_items)
            elif isinstance(value, dict):
                normalized[key] = self._normalize_parameters(value)
            else:
                # Convert to string representation for other types
                normalized[key] = str(value)
        
        return normalized
    
    def get_oss_context(self) -> Dict[str, Any]:
        """
        Get OSS analysis context (stays in OSS)
        
        This data never leaves the OSS environment for privacy and IP protection.
        """
        return {
            "reasoning_chain": self.reasoning_chain,
            "similar_incidents": self.similar_incidents,
            "rag_similarity_score": self.rag_similarity_score,
            "analysis_timestamp": datetime.fromtimestamp(self.detected_at).isoformat(),
            "source": self.source.value,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "oss_edition": self.oss_edition,
            "is_oss_advisory": self.is_oss_advisory,
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get execution summary (public information)
        
        Safe to share externally
        """
        summary = {
            "intent_id": self.deterministic_id,
            "action": self.action,
            "component": self.component,
            "confidence": self.confidence,
            "status": self.status.value,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "age_seconds": self.age_seconds,
            "oss_edition": self.oss_edition,
            "requires_enterprise": self.requires_enterprise,
            "is_oss_advisory": self.is_oss_advisory,
        }
        
        if self.executed_at:
            summary["executed_at"] = datetime.fromtimestamp(self.executed_at).isoformat()
            summary["execution_duration_seconds"] = self.executed_at - self.created_at
        
        if self.execution_result:
            summary["execution_success"] = self.execution_result.get("success", False)
            summary["execution_message"] = self.execution_result.get("message", "")
        
        if self.rag_similarity_score:
            summary["rag_similarity_score"] = self.rag_similarity_score
        
        if self.similar_incidents:
            summary["similar_incidents_count"] = len(self.similar_incidents)
        
        return summary


class HealingIntentSerializer:
    """
    Versioned serialization for HealingIntent
    
    Enhanced with OSS edition support and RAG integration.
    
    Supports:
    - JSON serialization/deserialization
    - Version compatibility
    - Schema validation
    - OSS/Enterprise edition detection
    """
    
    SCHEMA_VERSION: ClassVar[str] = "1.1.0"
    
    @classmethod
    def serialize(cls, intent: HealingIntent, version: str = "1.1.0") -> Dict[str, Any]:
        """
        Serialize HealingIntent with versioning
        
        Args:
            intent: HealingIntent to serialize
            version: Schema version to use
        
        Returns:
            Versioned serialization dictionary
        
        Raises:
            SerializationError: If serialization fails
        """
        try:
            if version == "1.1.0":
                return {
                    "version": version,
                    "schema_version": cls.SCHEMA_VERSION,
                    "data": intent.to_dict(include_oss_context=True),
                    "metadata": {
                        "serialized_at": time.time(),
                        "deterministic_id": intent.deterministic_id,
                        "is_executable": intent.is_executable,
                        "is_oss_advisory": intent.is_oss_advisory,
                        "requires_enterprise_upgrade": intent.requires_enterprise_upgrade,
                        "oss_edition": intent.oss_edition,
                    }
                }
            elif version == "1.0.0":
                # Backward compatibility with v1.0.0
                data = intent.to_dict(include_oss_context=True)
                # Remove OSS-specific fields for v1.0.0 compatibility
                data.pop("oss_edition", None)
                data.pop("oss_license", None)
                data.pop("requires_enterprise", None)
                data.pop("execution_allowed", None)
                
                return {
                    "version": version,
                    "schema_version": "1.0.0",
                    "data": data,
                    "metadata": {
                        "serialized_at": time.time(),
                        "deterministic_id": intent.deterministic_id,
                        "is_executable": intent.is_executable,
                    }
                }
            else:
                raise SerializationError(f"Unsupported version: {version}")
        
        except Exception as e:
            raise SerializationError(f"Failed to serialize HealingIntent: {e}") from e
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> HealingIntent:
        """
        Deserialize HealingIntent with version detection
        
        Args:
            data: Serialized data
        
        Returns:
            Deserialized HealingIntent
        
        Raises:
            SerializationError: If deserialization fails
        """
        try:
            version = data.get("version", "1.0.0")
            
            if version in ["1.1.0", "1.0.0"]:
                intent_data = data["data"]
                
                # Handle version differences
                if version == "1.0.0":
                    # Add default OSS fields for v1.0.0 data
                    intent_data.setdefault("oss_edition", OSS_EDITION)
                    intent_data.setdefault("oss_license", OSS_LICENSE)
                    intent_data.setdefault("requires_enterprise", True)
                    intent_data.setdefault("execution_allowed", False)
                
                return HealingIntent.from_dict(intent_data)
            else:
                raise SerializationError(f"Unsupported version: {version}")
        
        except KeyError as e:
            raise SerializationError(f"Missing required field in serialized data: {e}") from e
        except Exception as e:
            raise SerializationError(f"Failed to deserialize HealingIntent: {e}") from e
    
    @classmethod
    def to_json(cls, intent: HealingIntent, pretty: bool = False) -> str:
        """Convert HealingIntent to JSON string"""
        try:
            serialized = cls.serialize(intent)
            if pretty:
                return json.dumps(serialized, indent=2, default=str)
            else:
                return json.dumps(serialized, default=str)
        except Exception as e:
            raise SerializationError(f"Failed to convert to JSON: {e}") from e
    
    @classmethod
    def from_json(cls, json_str: str) -> HealingIntent:
        """Create HealingIntent from JSON string"""
        try:
            data = json.loads(json_str)
            return cls.deserialize(data)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON: {e}") from e
        except Exception as e:
            raise SerializationError(f"Failed to parse JSON: {e}") from e
    
    @classmethod
    def to_enterprise_json(cls, intent: HealingIntent) -> str:
        """
        Convert to Enterprise-ready JSON (excludes OSS context)
        
        This is what should be sent to the Enterprise API
        """
        try:
            enterprise_request = intent.to_enterprise_request()
            return json.dumps(enterprise_request, default=str)
        except Exception as e:
            raise SerializationError(f"Failed to create Enterprise JSON: {e}") from e
    
    @classmethod
    def validate_for_oss(cls, intent: HealingIntent) -> bool:
        """
        Validate that HealingIntent complies with OSS boundaries
        
        Returns:
            True if intent is valid for OSS edition
        """
        try:
            # Check OSS edition
            if intent.oss_edition != OSS_EDITION:
                return False
            
            # Check execution restrictions
            if intent.execution_allowed:
                return False
            
            # Check similar incidents limit
            if intent.similar_incidents and len(intent.similar_incidents) > intent.MAX_SIMILAR_INCIDENTS:
                return False
            
            return True
            
        except Exception:
            return False


# Factory functions for common use cases
def create_rollback_intent(
    component: str,
    revision: str = "previous",
    justification: str = "",
    incident_id: str = "",
    similar_incidents: Optional[List[Dict[str, Any]]] = None,
    rag_similarity_score: Optional[float] = None,
) -> HealingIntent:
    """Create a rollback healing intent with OSS limits"""
    if not justification:
        justification = f"Rollback {component} to {revision} revision"
    
    return HealingIntent.from_analysis(
        action="rollback",
        component=component,
        parameters={"revision": revision},
        justification=justification,
        confidence=0.9,
        similar_incidents=similar_incidents,
        incident_id=incident_id,
        rag_similarity_score=rag_similarity_score,
    ).mark_as_oss_advisory()


def create_restart_intent(
    component: str,
    container_id: Optional[str] = None,
    justification: str = "",
    incident_id: str = "",
    similar_incidents: Optional[List[Dict[str, Any]]] = None,
    rag_similarity_score: Optional[float] = None,
) -> HealingIntent:
    """Create a container restart healing intent with OSS limits"""
    parameters = {}
    if container_id:
        parameters["container_id"] = container_id
    
    if not justification:
        justification = f"Restart container for {component}"
    
    return HealingIntent.from_analysis(
        action="restart_container",
        component=component,
        parameters=parameters,
        justification=justification,
        confidence=0.85,
        similar_incidents=similar_incidents,
        incident_id=incident_id,
        rag_similarity_score=rag_similarity_score,
    ).mark_as_oss_advisory()


def create_scale_out_intent(
    component: str,
    scale_factor: int = 2,
    justification: str = "",
    incident_id: str = "",
    similar_incidents: Optional[List[Dict[str, Any]]] = None,
    rag_similarity_score: Optional[float] = None,
) -> HealingIntent:
    """Create a scale-out healing intent with OSS limits"""
    if not justification:
        justification = f"Scale out {component} by factor {scale_factor}"
    
    return HealingIntent.from_analysis(
        action="scale_out",
        component=component,
        parameters={"scale_factor": scale_factor},
        justification=justification,
        confidence=0.8,
        similar_incidents=similar_incidents,
        incident_id=incident_id,
        rag_similarity_score=rag_similarity_score,
    ).mark_as_oss_advisory()


def create_oss_advisory_intent(
    action: str,
    component: str,
    parameters: Dict[str, Any],
    justification: str,
    confidence: float = 0.85,
    incident_id: str = "",
) -> HealingIntent:
    """
    Create a generic OSS advisory-only intent
    
    Used when OSS wants to recommend an action without execution capability
    """
    return HealingIntent(
        action=action,
        component=component,
        parameters=parameters,
        justification=justification,
        confidence=confidence,
        incident_id=incident_id,
        oss_edition=OSS_EDITION,
        requires_enterprise=True,
        execution_allowed=False,
        status=IntentStatus.OSS_ADVISORY_ONLY,
    )


# Export
__all__ = [
    # Main class
    "HealingIntent",
    
    # Enums
    "IntentSource",
    "IntentStatus",
    
    # Serializer
    "HealingIntentSerializer",
    
    # Exceptions
    "HealingIntentError",
    "SerializationError",
    "ValidationError",
    
    # Factory functions
    "create_rollback_intent",
    "create_restart_intent",
    "create_scale_out_intent",
    "create_oss_advisory_intent",
]
