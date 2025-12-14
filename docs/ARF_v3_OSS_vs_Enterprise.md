# ARF v3 — OSS vs Enterprise Separation

**Goal:** Maximize adoption and monetization without contaminating the core.

---

## 1. First-Principles Boundary Definition

**Core Rule**

- OSS owns **reasoning & recommendation**
- Enterprise owns **execution, governance, and learning at scale**

This separation is architectural, not licensing-based.

---

## 2. Layered Architecture View

```
┌──────────────────────────────┐
│   Enterprise Control Plane   │
│  - Autonomous MCP execution  │
│  - Policy governance         │
│  - Graph persistence         │
│  - Multi-tenant isolation    │
│  - Compliance & audit        │
└──────────────▲───────────────┘
               │ HARD BOUNDARY (API / Network)
┌──────────────┴───────────────┐
│        OSS Intelligence      │
│  - Detection & analysis      │
│  - RAG-based reasoning       │
│  - Advisory decisions        │
│  - Explainability            │
└──────────────────────────────┘
```

---

## 3. OSS Surface Area (Free Forever)

### Included in Open Source

- Event ingestion
- Anomaly detection agents
- Policy evaluation engine
- In-memory RAG Graph
- FAISS similarity search
- MCP client
- MCP advisory execution mode
- Explainability, traces, and reasoning artifacts

### Explicit OSS Constraints

```
MAX_INCIDENT_HISTORY = 1_000
MAX_RAG_LOOKBACK_DAYS = 7
MCP_MODES_ALLOWED = ["advisory"]
GRAPH_STORAGE = "in_memory"
EXECUTION_ALLOWED = False
```

These limits are architectural ceilings, not license checks.

---

## 4. Enterprise Surface Area (Monetized)

### Paid Capabilities

- Autonomous MCP execution
- Human-in-the-loop approval workflows
- Persistent graph memory (Neo4j / managed graph)
- Cross-incident learning and optimization
- Blast-radius enforcement
- Compliance and audit logging
- Multi-tenant isolation
- Outcome-based policy optimization

---

## 5. Hard Technical Enforcement

### Execution Boundary

OSS emits **intent only**.  
Enterprise owns **execution**.

```
class HealingIntent:
    action: str
    justification: str
    confidence: float
```

Only the Enterprise MCP Server can execute tools.

---

### Storage Boundary (Bridge Pattern)

OSS:
```
class InMemoryGraphStorage(GraphStorage):
    pass
```

Enterprise:
```
class PersistentGraphStorage(GraphStorage):
    pass
```

No shared state. Interface reuse only.

---

### Learning Boundary

OSS:
```
find_similar_incidents(event)
```

Enterprise:
```
optimize_policy(similar_incidents, outcomes)
```

OSS observes history.  
Enterprise changes future behavior.

---

## 6. MCP Mode Gating

```
MCPMode:
- advisory   (OSS)
- approval   (Enterprise)
- autonomous (Enterprise)
```

Enforced at build time, runtime, and deployment.

---

## 7. Data Ownership & Compliance

| Data Type | OSS | Enterprise |
|----------|-----|------------|
| Incident metadata | Yes | Yes |
| Vector embeddings | Yes | Yes |
| Outcomes & resolution metrics | No | Yes |
| Audit logs | No | Yes |
| Operator identities | No | Yes |
| Change approvals | No | Yes |

---

## 8. Repository Strategy

```
arf-core/           (Apache 2.0)
arf-rag/            (Apache 2.0)
arf-mcp-client/     (Apache 2.0)

arf-mcp-server/     (Commercial)
arf-graph-store/    (Commercial)
arf-governance/     (Commercial)
```

No dual-licensed files.  
No gray-area feature flags.

---

## 9. Strategic Outcome

- OSS adoption increases enterprise value
- Execution and learning remain defensible
- No future rewrite required to monetize
- Clear enterprise risk-reduction story

---

## 10. Revenue Mapping

| Tier | Monetized Value |
|-----|-----------------|
| Team | Approval-mode MCP + persistence |
| Business | Autonomous MCP + blast radius |
| Enterprise | Compliance, SSO, audit |
| Platform | Outcome-optimized learning loops |

---

**Conclusion**

ARF v3 cleanly separates intelligence from authority.  
The more the OSS is used, the more valuable the enterprise layer becomes — without giving it away.
