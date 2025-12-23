# Agentic Reliability Framework (ARF) — Quick Start

Welcome to ARF v3.3.0 — a hybrid intelligence framework for autonomous system reliability, self-healing, and context-aware incident management.

This guide shows you how to get started quickly with **event processing, safety, and self-healing**.

---

## 1. Installation

### OSS Edition (Advisory Only)

```bash
pip install agentic-reliability-framework
```
*   Runs advisory-only MCP mode
    
*   In-memory RAG graph + FAISS for incident similarity
    
*   No actions executed (safe for testing)
    

### Enterprise Edition (Full Execution)
```
pip install arf-enterprise
docker-compose up -d neo4j redis
arf-enterprise --license-key YOUR_KEY
```

*   Full MCP modes: Advisory → Approval → Autonomous
    
*   Persistent graph memory
    
*   Outcome tracking and audit trails
    
*   2\. Basic Event Processing
    

Use the engine.process\_event\_enhanced(...) interface:
```
from arf import Engine

engine = Engine()

result = await engine.process_event_enhanced(
    component="api-service",
    latency_p99=320.0,      # ms
    error_rate=0.18,        # 18% errors
    throughput=1250.0,      # req/sec
    cpu_util=0.87,          # 0–1
    memory_util=0.92        # 0–1
)
print(result)
```

*   Key Notes:
    
*   result contains advisory recommendations or executed outcomes depending on MCP mode.
    
*   Exceptions are raised for invalid inputs or policy violations.
    
*   Logs are emitted for each step, ensuring **traceability**.

*   3\. Safety & Self-Healing
    
*   ARF enforces a **five-layer safety system**:

  ```
safety_system = {
    "layer_1": "Action Blacklisting",
    "layer_2": "Blast Radius Limiting",
    "layer_3": "Human Approval Workflows",
    "layer_4": "Business Hour Restrictions",
    "layer_5": "Circuit Breakers & Cooldowns"
}
```
*   Prevents dangerous or unsafe automated actions
    
*   Applies human-in-the-loop approvals where required
    
*   Limits the scope and timing of self-healing actions
    
*   Ensures audit-ready execution for compliance
    

### Self-Healing Flow

1.  **Event Ingestion** – Telemetry validated via Pydantic models
    
2.  **Multi-Agent Analysis** – Detection, recall, and decision agents run in parallel
    
3.  **RAG Context Retrieval** – Historical incidents and outcomes retrieved
    
4.  **Policy Evaluation** – Deterministic and AI-informed guardrails applied
    
5.  **Action Execution** – Healing intents executed (Enterprise only)
    
6.  **Outcome Recording** – Updates RAG memory for continuous learning
    

For details, see [self-healing patterns](self-healing.md) and [architecture](architecture.md).

4\. Error Handling
------------------

*   **Advisory Mode:** All events processed safely; errors are logged but no actions executed
    
*   **Enterprise Mode:** Failed actions trigger rollback policies; blast radius and circuit breakers prevent cascading failures
    
*   **API Exceptions:** Each function documents possible exceptions in [API reference](api.md)
    

5\. Recommended Usage
---------------------

*   Start **OSS** in advisory mode to test your telemetry safely
    
*   Gradually enable Enterprise approval/autonomous modes once confident
    
*   Monitor logs and audit trails continuously
    
*   Use the **CLI** or programmatic API to simulate and test events
    

6\. Additional Resources
------------------------

*   [API Documentation](api.md)
    
*   [Architecture Overview](architecture.md)
    
*   [Self-Healing Patterns](self-healing.md)
    
*   [Business Impact Metrics](business-metrics.md)
    

7\. Contact & Support
---------------------

*   **Email:** petter2025us@outlook.com
    
*   **LinkedIn:** [Juan Petter](https://linkedin.com/in/petterjuan)
    
*   **GitHub Issues:** [Submit bugs or questions](https://github.com/petterjuan/agentic-reliability-framework/issues)
    

**Tip:** Always start in advisory mode to validate your system before enabling any execution capabilities.
