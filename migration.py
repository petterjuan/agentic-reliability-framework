"""
Migration script from v2 to v3

Phase 1: Convert existing FAISS storage to RAG Graph
"""

import json
import logging
import os
from typing import List, Dict, Any
from datetime import datetime

from .memory.rag_graph import RAGGraphMemory
from .models import ReliabilityEvent, EventSeverity
from .config import config

logger = logging.getLogger(__name__)


def migrate_v2_to_v3() -> RAGGraphMemory:
    """
    Convert existing FAISS storage to RAG Graph
    
    Returns:
        RAGGraphMemory instance with migrated data
    """
    logger.info("Starting v2 to v3 migration...")
    
    try:
        # 1. Load existing FAISS vectors
        from .lazy import get_faiss_index
        faiss_index = get_faiss_index()
        
        if not faiss_index:
            logger.error("FAISS index not available")
            raise ValueError("FAISS index not available")
        
        # 2. Load existing text metadata (if any)
        incident_texts = []
        if hasattr(faiss_index, 'texts'):
            incident_texts = faiss_index.texts
        elif os.path.exists(config.incident_texts_file):
            with open(config.incident_texts_file, 'r') as f:
                incident_texts = json.load(f)
        
        logger.info(f"Found {len(incident_texts)} incident texts to migrate")
        
        # 3. Create RAG Graph with existing data
        rag = RAGGraphMemory(faiss_index)
        
        migrated_count = 0
        for i, text in enumerate(incident_texts):
            try:
                # Parse v2 text format: "component latency error_rate analysis_text"
                parts = text.split(" ", 3)
                
                if len(parts) >= 3:
                    component = parts[0]
                    
                    try:
                        latency = float(parts[1])
                    except ValueError:
                        latency = 0.0
                    
                    try:
                        error_rate = float(parts[2])
                    except ValueError:
                        error_rate = 0.0
                    
                    analysis_text = parts[3] if len(parts) > 3 else ""
                    
                    # Create synthetic event
                    event = ReliabilityEvent(
                        component=component,
                        latency_p99=latency,
                        error_rate=error_rate,
                        throughput=1000.0,
                        severity=EventSeverity.MEDIUM if latency > 300 or error_rate > 0.1 else EventSeverity.LOW,
                        timestamp=datetime.now()
                    )
                    
                    # Create synthetic analysis
                    analysis = {
                        "incident_summary": {
                            "severity": event.severity.value.upper(),
                            "anomaly_confidence": 0.7 if latency > 300 or error_rate > 0.1 else 0.3,
                            "primary_metrics_affected": ["latency"] if latency > 300 else ["error_rate"] if error_rate > 0.1 else []
                        },
                        "migration_note": f"Migrated from v2 text: {analysis_text[:100]}..."
                    }
                    
                    # Store in RAG
                    incident_id = rag.store_incident(event, analysis)
                    
                    if incident_id:
                        migrated_count += 1
                        
                        # Log progress
                        if migrated_count % 100 == 0:
                            logger.info(f"Migrated {migrated_count}/{len(incident_texts)} incidents")
                    
                else:
                    logger.warning(f"Skipping malformed text: {text[:50]}...")
                    
            except Exception as e:
                logger.error(f"Error migrating incident {i}: {e}")
                continue
        
        logger.info(f"Migration complete: {migrated_count} incidents migrated to RAG")
        
        # 4. Save migration summary
        migration_summary = {
            "migration_timestamp": datetime.now().isoformat(),
            "total_incidents_migrated": migrated_count,
            "total_incidents_found": len(incident_texts),
            "success_rate": migrated_count / len(incident_texts) if incident_texts else 0,
            "rag_graph_stats": rag.get_graph_stats()
        }
        
        # Save summary to file
        summary_file = "migration_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(migration_summary, f, indent=2)
        
        logger.info(f"Migration summary saved to {summary_file}")
        
        return rag
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise


def run_migration():
    """Run migration from command line"""
    try:
        rag = migrate_v2_to_v3()
        print(f"Migration successful! RAG graph created with {rag.get_graph_stats()['incident_nodes']} incidents.")
        return rag
    except Exception as e:
        print(f"Migration failed: {e}")
        return None


if __name__ == "__main__":
    run_migration()
