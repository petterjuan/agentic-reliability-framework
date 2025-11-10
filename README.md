🧠 Agentic Reliability Framework

Adaptive anomaly detection + AI-driven self-healing + persistent FAISS memory for reliability-aware systems.

The Agentic Reliability Framework (ARF) provides a prototype for self-healing operational intelligence — integrating vector-based observability, adaptive anomaly detection, and cloud inference for autonomous incident handling.

**Autonomous detect → diagnose → heal reliability framework** with adaptive anomaly detection, AI-driven root cause analysis, and persistent vector memory for cloud infrastructure.

---

## 🚀 Overview

A real-time reliability engineering system that continuously monitors telemetry data, detects anomalies using adaptive thresholds, performs AI-powered root cause analysis, and simulates self-healing actions with persistent memory of past incidents.

---

## 🛠️ Core Features

| Feature | Description |
|---------|-------------|
| **🔍 Adaptive Anomaly Detection** | Dynamic threshold-based detection with latency (>150ms) and error rate (>5%) monitoring |
| **🧠 AI Root Cause Analysis** | Integration with Mistral-8x7B via Hugging Face Inference API for intelligent incident analysis |
| **💾 Vector Memory (FAISS)** | Persistent storage of incident embeddings using sentence-transformers for similarity search |
| **⚡ Self-Healing Simulation** | Automated corrective actions (restart, scale, clear backlog) with historical context |
| **📊 Real-time Dashboard** | Gradio UI for telemetry submission and incident visualization |
| **🔐 Secure API** | FastAPI backend with environment-based configuration |

---

## 🏗️ Architecture
                   ┌──────────────────────────────┐
                   │        Gradio UI (Web)        │
                   │  ──── Real-time telemetry ─── │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │       FastAPI Endpoint        │
                   │    /add-event + API Key Auth   │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Hugging Face Router API       │
                   │  Mixtral-8x7B → Root Cause NLP │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  FAISS + Sentence Transformers│
                   │  Persistent Memory & Similarity│
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Adaptive Healing Simulation   │
                   │ (Scale, Restart, Queue Clear)  │
                   └──────────────────────────────┘
                   
⚙️ Tech Stack

| Layer                 | Component                                                     | Description                                                             |
| --------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **UI / Control**      | `Gradio 5.49.1`                                               | Interactive telemetry dashboard for real-time anomaly visualization     |
| **Inference Gateway** | `Hugging Face Router API (Mixtral 8x7B Instruct)`             | Contextual incident analysis via dynamic inference routing              |
| **Memory Layer**      | `FAISS + Sentence Transformers (all-MiniLM-L6-v2)`            | Persistent semantic memory of past incidents for retrieval & clustering |
| **API Layer**         | `FastAPI + Starlette`                                         | RESTful telemetry ingestion (`/add-event`) with `X-API-Key` security    |
| **Runtime**           | `Python 3.10` + `NumPy`, `Pandas`, `Scikit-learn`, `Tenacity` | Adaptive computation and fault tolerance                                |
| **Deployment**        | GitHub Codespaces / Hugging Face Spaces                       | Containerized, GPU-optional environment for live experimentation        |

🚀 Features

| Capability                     | Description                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Adaptive Anomaly Detection** | Dynamically learns thresholds from latency & error-rate signals                   |
| **AI Root Cause Analysis**     | Integrates with Mixtral-8x7B via Hugging Face Router for context-rich diagnostics |
| **Self-Healing Simulation**    | Executes adaptive healing routines (restart, scale, clear queue)                  |
| **Persistent FAISS Memory**    | Stores embeddings of incidents for cross-similarity search                        |
| **Secure REST API**            | `/add-event` endpoint with API key validation for automation integration          |
| **Interactive Dashboard**      | Visualizes system events, anomalies, and remediation logs in real time            |

🎯 Use Cases
Cloud Infrastructure Monitoring - Real-time anomaly detection for microservices

Incident Response - AI-assisted root cause analysis and decision support

Reliability Engineering - Continuous learning from past incidents

DevOps Automation - Self-healing system simulation and validation

💾 Data Persistence
FAISS Index: incident_vectors.index - Vector embeddings of incidents

JSON Metadata: incident_texts.json - Textual descriptions and metadata

FileLock: Concurrent write safety for multi-user environments

🧠 Example Output

✅ Event Processed (Anomaly)

Component: api-service
Latency: 224 ms
Error Rate: 0.062
Status: Anomaly
Analysis: Error 404 - Missing upstream dependency
Healing Action: Restarted container (Found 3 similar incidents)

🧾 API Usage

Endpoint

POST /add-event

Headers

X-API-Key: <your_api_key>

Body

{
  "component": "api-service",
  "latency": 200,
  "error_rate": 0.04
}

Response

{
  "status": "ok",
  "event": {
    "timestamp": "2025-11-09 21:14:03",
    "component": "api-service",
    "status": "Anomaly",
    "analysis": "Error 404: Not Found",
    "healing_action": "Restarted container (Found 3 similar incidents)"
  }
}

🧩 Quickstart

git clone https://github.com/petterjuan/agentic-reliability-framework.git
cd agentic-reliability-framework
pip install -r requirements.txt
python app.py

Open your browser at: http://localhost:7860

🌐 Live Demo
Hugging Face Space: Launch Demo

GitHub Repository: Source Code

⚙️ Code Improvement Plan
Here’s how to evolve the MVP into a production-ready reliability agent:
| Focus                           | Next Steps                                                                    | Description                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **1. Real-time Feedback Loops** | Integrate WebSocket channels for live anomaly updates in Gradio               | Enables anomaly preemption & operator notifications                               |
| **2. Distributed Inference**    | Move Hugging Face inference calls behind an async task queue (Celery + Redis) | Prevents blocking and improves fault tolerance                                    |
| **3. Memory Management**        | Shard FAISS indexes & periodically prune vector memory                        | Keeps inference cost predictable over time                                        |
| **4. Observability Layer**      | Add Prometheus metrics + Grafana dashboard hooks                              | Enables monitoring of anomaly rates, inference latency, and healing effectiveness |
| **5. Multi-Agent Coordination** | Add a `Coordinator Agent` to manage self-healing priorities                   | Converts reactive healing → proactive orchestration                               |
| **6. Model Adaptation**         | Swap MiniLM for `all-distilroberta-v1` and test semantic retention            | Improves similarity clustering accuracy                                           |
| **7. Config & Secrets**         | Use `.env` loader and structured settings via `pydantic.BaseSettings`         | Cleaner configuration management across environments                              |

🤝 Contributing
This is an active research project exploring agentic reliability patterns. Issues and pull requests are welcome!

🧭 Author

Juan D. Petter
AI Engineer & Cloud Architect
Building Agentic Systems for Scalable Automation | ex-NetApp
🔗 LinkedIn
 • GitHub

🪪 License

MIT License © 2025 Juan D. Petter
