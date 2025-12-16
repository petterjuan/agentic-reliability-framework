"""
Protocol interfaces for v3 integration - breaks circular dependencies
"""

from typing import Protocol, Optional, Any, Dict, List
from ..models import ReliabilityEvent


class ReliabilityEngineProtocol(Protocol):
    """Protocol for reliability engines to avoid circular imports"""

    def process_event(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """Process a reliability event"""
        ...

    async def process_event_enhanced(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Process event with enhanced v2/v3 features"""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        ...


class MCPProtocol(Protocol):
    """Protocol for MCP servers"""

    async def execute_tool(self, request_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool"""
        ...

    def get_server_stats(self) -> Dict[str, Any]:
        """Get MCP server statistics"""
        ...


class RAGProtocol(Protocol):
    """Protocol for RAG graph memory"""

    def find_similar(self, event: ReliabilityEvent, k: int = 5) -> List[ReliabilityEvent]:
        """Find similar incidents"""
        ...

    def store_incident(self, event: ReliabilityEvent, analysis: Dict[str, Any]) -> str:
        """Store incident in graph"""
        ...

    def store_outcome(
        self,
        incident_id: str,
        actions_taken: List[str],
        success: bool,
        resolution_time_minutes: float,
        lessons_learned: Optional[List[str]] = None,
    ) -> str:
        """Store outcome for incident"""
        ...
