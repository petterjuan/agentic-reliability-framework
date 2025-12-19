# ARF v3: Complete OSS vs Enterprise Separation & Implementation Plan

**Based on analysis of the existing codebase**  
**Version:** 3.3.0  
**Author:** Juan Petter - AI Engineer  
**Date:** December 19, 2024  

---

## Executive Summary

Your current codebase **already contains Enterprise-grade functionality mixed into OSS code**.  
The MCP server supports advisory, approval, and autonomous modes, and tools can actually execute (or simulate execution).

The goal is **separation, not rewrite**, while preserving functionality and enabling a clean commercial path.

### Critical Findings

1. **Current State:** Single repository with mixed OSS and Enterprise logic  
2. **MCP Server:** Fully implemented with execution capability  
3. **Configuration:** Runtime-based (`config.mcp_mode`), not build-time enforced  
4. **Tools:** Capable of execution or simulation  
5. **License:** Currently MIT → moving to **Apache 2.0 (OSS)** + **Commercial (Enterprise)**  

---

## 1. Current Architecture Analysis

### 1.1 What Is Already Working

```text
✅ Full MCP server with 3 modes
✅ Tool implementations (rollback, restart, scale_out, etc.)
✅ RAG graph memory (FAISS-based)
✅ Reliability engine with learning
✅ Metrics export API (Prometheus, JSON, CSV)
✅ Post-mortem benchmark suite
✅ Comprehensive models and validation
```

### 1.2 What Needs Separation

```text
🚫 MCP execution modes mixed in same code
🚫 Tool execution not restricted
🚫 No build-time enforcement
🚫 No license validation
🚫 OSS and Enterprise in same repo
```

---

## 2. Immediate Action Plan (5 Days)

### Day 1: Repository Restructuring & Build-Time Boundaries

**Morning (4h)**

1. Split repository into OSS and Enterprise  
2. Create OSS constants with hard limits  
3. Implement build-time enforcement scripts  

**Afternoon (4h)**

4. Create bridge interfaces  
5. Implement `HealingIntent` boundary pattern  
6. Set up CI/CD boundary checks  

#### Key Files (Day 1)

```text
agentic-reliability-framework/        # Apache 2.0 (OSS)
├── arf-core/
│   ├── src/
│   │   ├── constants.py
│   │   ├── boundaries.py
│   │   ├── models/healing_intent.py
│   │   └── security/oss_auth.py
│   └── tests/test_oss_boundaries.py
├── arf-mcp-client/                   # Advisory only
└── scripts/enforce_oss_constants.py

arf-enterprise/                       # Commercial
├── arf-mcp-server/
├── arf-graph-store/
└── arf-learning/
```

---

## 3. Detailed Implementation Guide

### 3.1 OSS Constants & Boundaries

```python
from typing import Final

MAX_INCIDENT_HISTORY: Final[int] = 1000
MAX_RAG_LOOKBACK_DAYS: Final[int] = 7
MCP_MODES_ALLOWED: Final[tuple] = ("advisory",)
EXECUTION_ALLOWED: Final[bool] = False
GRAPH_STORAGE: Final[str] = "in_memory"

def _validate_oss_constants():
    violations = []
    if MAX_INCIDENT_HISTORY > 1000:
        violations.append("MAX_INCIDENT_HISTORY > 1000")
    if MCP_MODES_ALLOWED != ("advisory",):
        violations.append("Invalid MCP mode")
    if EXECUTION_ALLOWED:
        violations.append("Execution not allowed in OSS")
    if GRAPH_STORAGE != "in_memory":
        violations.append("Invalid storage backend")

    if violations:
        raise RuntimeError(f"OSS violations: {violations}")

_validate_oss_constants()
```

---

## 4. Success Criteria

### OSS Success

- Advisory-only enforced  
- No execution paths  
- CI boundary checks passing  

### Enterprise Success

- License validation active  
- All MCP modes functional  
- Audit + learning enabled  
