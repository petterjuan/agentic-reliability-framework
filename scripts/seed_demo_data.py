#!/usr/bin/env python3
"""
Seed demo data for ARF
Creates sample incidents for FAISS index and JSON storage
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from datetime import datetime, timedelta

def create_demo_incidents():
    """Create realistic demo incidents for testing/demo"""
    
    incidents = {
        "database_connection_pool_exhaustion": {
            "description": "Database connection pool exhausted due to sudden traffic spike",
            "root_cause": "Connection pool size too small for peak load",
            "solution": "Increase connection pool size from 50 to 200",
            "metrics": {
                "latency_p99": 850,
                "error_rate": 0.45,
                "cpu_util": 0.65,
                "memory_util": 0.88
            },
            "business_impact": {
                "revenue_loss_per_minute": 1250.0,
                "affected_users": 15000
            },
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat()
        },
        "api_rate_limit_exceeded": {
            "description": "Third-party API rate limit exceeded causing payment failures",
            "root_cause": "Missing exponential backoff in retry logic",
            "solution": "Implement circuit breaker with 429 response handling",
            "metrics": {
                "latency_p99": 320,
                "error_rate": 0.28,
                "cpu_util": 0.45,
                "memory_util": 0.52
            },
            "business_impact": {
                "revenue_loss_per_minute": 850.0,
                "affected_users": 8200
            },
            "timestamp": (datetime.now() - timedelta(days=5)).isoformat()
        },
        "memory_leak_in_cache_layer": {
            "description": "Memory leak in Redis cache client causing OOM kills",
            "root_cause": "Unbounded cache growth without TTL",
            "solution": "Add TTL to cache entries and memory limits",
            "metrics": {
                "latency_p99": 420,
                "error_rate": 0.35,
                "cpu_util": 0.72,
                "memory_util": 0.96
            },
            "business_impact": {
                "revenue_loss_per_minute": 950.0,
                "affected_users": 10500
            },
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat()
        }
    }
    
    return incidents

def seed_faiss_index():
    """Create and seed FAISS index with demo incidents"""
    
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    incidents = create_demo_incidents()
    texts = []
    embeddings = []
    
    for incident_id, data in incidents.items():
        # Create embedding text
        text = f"{data['description']}. Root cause: {data['root_cause']}. Solution: {data['solution']}"
        texts.append(text)
        
        # Generate embedding
        embedding = model.encode(text, normalize_embeddings=True)
        embeddings.append(embedding)
    
    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    
    # Convert to numpy array and add to index
    embedding_array = np.array(embeddings).astype('float32')
    index.add(embedding_array)
    
    # Save index
    os.makedirs('data', exist_ok=True)
    faiss.write_index(index, 'data/incident_vectors.index')
    
    # Save incident texts
    with open('data/incident_texts.json', 'w') as f:
        json.dump({str(i): texts[i] for i in range(len(texts))}, f, indent=2)
    
    print(f"✅ Seeded FAISS index with {len(embeddings)} demo incidents")
    print(f"   Dimension: {dimension}")
    print("   Saved to: data/incident_vectors.index")  # FIXED: Removed f prefix
    
    return incidents

if __name__ == "__main__":
    seed_faiss_index()
