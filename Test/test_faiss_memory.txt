"""
Test suite for FAISS vector memory and incident similarity search

Save this as: tests/test_faiss_memory.py
"""

import pytest
import os
import tempfile
from datetime import datetime
from app import FAISSIncidentMemory


class TestFAISSIncidentMemory:
    """Test FAISS vector memory functionality"""
    
    @pytest.fixture
    def temp_index_file(self):
        """Create a temporary FAISS index file"""
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            temp_file = f.name
        yield temp_file
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    @pytest.fixture
    def memory(self, temp_index_file):
        """Create FAISSIncidentMemory instance with temp storage"""
        return FAISSIncidentMemory(index_file=temp_index_file)
    
    @pytest.mark.asyncio
    async def test_faiss_add_and_search_incident(self, memory):
        """Test adding an incident and searching for similar ones"""
        # Add incident
        incident_text = "database connection pool exhausted causing high latency"
        metadata = {
            "service": "api",
            "severity": "critical",
            "timestamp": datetime.now().isoformat()
        }
        
        await memory.add_incident(incident_text, metadata)
        
        # Search for similar
        query = "connection pool issues with database"
        results = await memory.search_similar(query, k=1)
        
        assert len(results) > 0, "Should find similar incident"
        assert results[0]['text'] == incident_text
        assert results[0]['score'] > 0.5, "Similarity score should be reasonably high"
    
    @pytest.mark.asyncio
    async def test_faiss_semantic_similarity(self, memory):
        """Test that FAISS finds semantically similar incidents"""
        # Add multiple incidents
        incidents = [
            ("database connection pool exhausted", {"type": "db"}),
            ("redis cache timeout", {"type": "cache"}),
            ("memory leak in worker process", {"type": "memory"}),
            ("connection refused to postgres", {"type": "db"}),
        ]
        
        for text, meta in incidents:
            await memory.add_incident(text, meta)
        
        # Search for DB-related issue
        query = "database connectivity problems"
        results = await memory.search_similar(query, k=2)
        
        assert len(results) >= 2
        # Top results should be DB-related
        assert any("database" in r['text'] or "postgres" in r['text'] for r in results[:2])
    
    @pytest.mark.asyncio
    async def test_faiss_persistence(self, memory, temp_index_file):
        """Test that FAISS index persists to disk"""
        # Add incident
        incident_text = "critical memory exhaustion"
        await memory.add_incident(incident_text, {"test": "persistence"})
        
        # Save explicitly
        await memory.save()
        
        # Create new instance (should load from disk)
        new_memory = FAISSIncidentMemory(index_file=temp_index_file)
        
        # Search in new instance
        results = await new_memory.search_similar("memory issues", k=1)
        
        assert len(results) > 0, "Should load persisted data"
        assert "memory" in results[0]['text'].lower()
    
    @pytest.mark.asyncio
    async def test_faiss_handles_empty_query(self, memory):
        """Test that empty queries are handled gracefully"""
        results = await memory.search_similar("", k=5)
        
        assert isinstance(results, list)
        # Should return empty or handle gracefully
    
    @pytest.mark.asyncio
    async def test_faiss_returns_top_k_results(self, memory):
        """Test that k parameter limits results correctly"""
        # Add 10 incidents
        for i in range(10):
            await memory.add_incident(f"incident number {i}", {"id": i})
        
        # Search for k=3
        results = await memory.search_similar("incident", k=3)
        
        assert len(results) <= 3, "Should return at most k results"
    
    @pytest.mark.asyncio
    async def test_faiss_scores_decrease_with_relevance(self, memory):
        """Test that similarity scores are ordered (highest first)"""
        # Add incidents with varying similarity to query
        await memory.add_incident("database connection timeout", {})
        await memory.add_incident("redis cache miss", {})
        await memory.add_incident("database pool exhausted", {})
        
        query = "database connection issues"
        results = await memory.search_similar(query, k=3)
        
        if len(results) > 1:
            # Scores should be in descending order
            for i in range(len(results) - 1):
                assert results[i]['score'] >= results[i+1]['score'], \
                    "Scores should be ordered from highest to lowest"
    
    @pytest.mark.asyncio
    async def test_faiss_thread_safety_single_writer(self, memory):
        """Test single-writer pattern for thread safety"""
        import asyncio
        
        # Simulate concurrent writes
        async def add_incident(idx):
            await memory.add_incident(f"incident {idx}", {"id": idx})
        
        # Execute 10 concurrent adds
        await asyncio.gather(*[add_incident(i) for i in range(10)])
        
        # Verify all were added (no race conditions)
        results = await memory.search_similar("incident", k=10)
        assert len(results) == 10, "All incidents should be added safely"


class TestFAISSMemoryIntegration:
    """Test FAISS integration with agent workflow"""
    
    @pytest.fixture
    def memory(self):
        """Create in-memory FAISS instance"""
        return FAISSIncidentMemory()
    
    @pytest.mark.asyncio
    async def test_faiss_helps_diagnostician(self, memory):
        """Test that FAISS memory helps with root cause analysis"""
        # Add historical incident
        await memory.add_incident(
            "database connection pool exhausted caused 500ms latency spike",
            {
                "root_cause": "db pool exhaustion",
                "resolution": "increased pool size from 10 to 50"
            }
        )
        
        # Search for similar current incident
        query = "seeing high latency and connection errors to database"
        results = await memory.search_similar(query, k=1)
        
        assert len(results) > 0
        assert "database" in results[0]['text'].lower()
        assert "connection" in results[0]['text'].lower()
        
        # Metadata should contain resolution hint
        assert 'root_cause' in results[0]
    
    @pytest.mark.asyncio
    async def test_faiss_recalls_past_incidents(self, memory):
        """Test that FAISS can recall incidents from different time periods"""
        from datetime import timedelta
        
        # Add incidents from different times
        now = datetime.now()
        
        await memory.add_incident(
            "memory leak in worker",
            {"timestamp": (now - timedelta(days=30)).isoformat()}
        )
        await memory.add_incident(
            "redis timeout spike",
            {"timestamp": (now - timedelta(days=7)).isoformat()}
        )
        await memory.add_incident(
            "database slow query",
            {"timestamp": now.isoformat()}
        )
        
        # Should find all incidents regardless of age
        results = await memory.search_similar("system issues", k=10)
        
        assert len(results) == 3, "Should recall all incidents"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
