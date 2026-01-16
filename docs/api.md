# ARF API Reference (v1)

**Agentic Reliability Framework (ARF)** provides a production-grade API for incident management, execution policy evaluation, and safe rollback operations.

This document describes **API v1**, aligned with the current FastAPI codebase under `src/api/v1`.

---

## Base Information

- **Base URL:** `/api/v1`
- **API Style:** REST + graph operations
- **Authentication:** JWT / role-based dependencies
- **Roles:**
  - **Viewer** – read-only access
  - **Operator** – create/update operational data
  - **Admin** – destructive and system-level operations

---

## Authentication & Authorization

Authentication is enforced per endpoint using dependency injection:

- `require_viewer`
- `require_operator`
- `require_admin`
- `get_current_user_optional`

Some read-only endpoints allow **anonymous access** with restricted visibility.

---

## Common Conventions

### Pagination
Most list endpoints support:
- `page` (default: 1)
- `page_size` or `limit`
- `has_more` boolean

### Sorting
Where applicable:
- `sort_by`
- `sort_order` (`asc | desc`)

### Time
All timestamps are UTC and ISO-8601 formatted.

---

# Incidents API

**Tag:** `incidents`  
**Prefix:** `/incidents`

Incident endpoints support filtering, pagination, caching, and role-aware access control.

---

## List Incidents

`GET /incidents`

**Authentication:** Optional  
**Authorization:**
- Anonymous users see LOW / MEDIUM severity only
- Authenticated users see all

### Query Parameters
| Name | Type | Description |
|----|----|----|
| page | int | Page number (≥1) |
| page_size | int | Results per page (1–100) |
| severity | enum | LOW, MEDIUM, HIGH, CRITICAL |
| status | enum | Incident status |
| incident_type | enum | Incident category |
| agent_id | string | Filter by agent |
| start_date | datetime | Created after |
| end_date | datetime | Created before |
| sort_by | string | created_at \| severity \| status |
| sort_order | string | asc \| desc |

### Response
```json
{
  "incidents": [],
  "total": 0,
  "page": 1,
  "page_size": 50,
  "has_more": false
}
```

Get Incident
------------

GET /incidents/{incident\_id}

**Authentication:** Optional**Authorization:**Anonymous users cannot access HIGH / CRITICAL incidents.

Create Incident
---------------

POST /incidents

**Authentication:** Required**Authorization:** Operator+

Creates a new incident.created\_by metadata is automatically injected.

Update Incident
---------------

PUT /incidents/{incident\_id}

**Authentication:** Required**Authorization:** Operator+

Automatically updates:

*   updated\_by
    
*   updated\_at
    
*   resolved\_at (when applicable)
    

Delete Incident (Soft Delete)
-----------------------------

DELETE /incidents/{incident\_id}

**Authentication:** Required**Authorization:** Admin+

Marks the incident as CLOSED and records deletion metadata.

Export Incidents (Admin)
------------------------

GET /incidents/admin/export

**Authentication:** Required**Authorization:** Admin+

### Query Parameters

*   format: json | csv
    

Execution Ladder API
====================

**Tag:** execution-ladder**Prefix:** /execution-ladder

Graph-based execution and policy evaluation backed by Neo4j.

Create Execution Graph
----------------------

POST /execution-ladder/graphs

**Authentication:** Required**Authorization:** Admin+

Get Execution Graph
-------------------

GET /execution-ladder/graphs/{graph\_id}

**Authentication:** Required**Authorization:** Operator+

Update Execution Graph
----------------------

PUT /execution-ladder/graphs/{graph\_id}

**Authentication:** Required**Authorization:** Admin+

Add Node to Graph
-----------------

POST /execution-ladder/graphs/{graph\_id}/nodes

**Authentication:** Required**Authorization:** Admin+

Create Edge
-----------

POST /execution-ladder/graphs/{graph\_id}/edges

**Authentication:** Required**Authorization:** Admin+

Evaluate Policies
-----------------

POST /execution-ladder/evaluate

**Authentication:** Optional**Authorization:** None

Evaluates execution policies against a provided context and returns:

*   evaluations
    
*   execution trace
    
*   final decision
    
*   confidence score
    

Get Execution Trace
-------------------

GET /execution-ladder/traces/{trace\_id}

**Status:** Not yet implemented (501)

Graph Utilities
---------------

*   GET /graphs/{graph\_id}/path
    
*   GET /graphs/{graph\_id}/statistics
    
*   POST /graphs/{graph\_id}/clone
    
*   GET /nodes/{node\_id}/connections
    

Execution Ladder Health
-----------------------

GET /execution-ladder/health

Performs a Neo4j connectivity check.

Rollback API
============

**Tag:** rollback**Prefix:** /rollback

Provides auditable, risk-aware rollback operations.

Log Action
----------

POST /rollback/actions

**Authentication:** Required**Authorization:** Operator+

Logs an action for future rollback.

Get Action
----------

GET /rollback/actions/{action\_id}

**Authentication:** Required**Authorization:** Operator+

Analyze Rollback
----------------

POST /rollback/actions/{action\_id}/analyze

**Authentication:** Required**Authorization:** Operator+

Returns feasibility, risk, and rollback strategy.

Execute Rollback
----------------

POST /rollback/actions/{action\_id}/execute

**Authentication:** Required**Authorization:** Admin+

Bulk Rollback
-------------

POST /rollback/bulk

**Authentication:** Required**Authorization:** Admin+

Search Actions
--------------

GET /rollback/actions

Supports filtering by:

*   action\_type
    
*   status
    
*   risk\_level
    
*   time range
    

Rollback Statistics
-------------------

GET /rollback/statistics

**Authentication:** Required**Authorization:** Operator+

### Time Ranges

1d | 7d | 30d | 90d | all

Cleanup Expired Actions
-----------------------

POST /rollback/cleanup

**Authentication:** Required**Authorization:** Admin+

Rollback Dashboard
------------------

GET /rollback/dashboard

**Authentication:** Required**Authorization:** Operator+

Returns:

*   recent actions
    
*   high-risk actions
    
*   success rates
    
*   time-series stats
    

Export Rollback Data (Admin)
----------------------------

GET /rollback/admin/export

**Authentication:** Required**Authorization:** Admin+

Rollback Health
---------------

GET /rollback/health

Performs a full service self-test.

Versioning
----------

*   This document applies to **API v1**
    
*   Breaking changes will be introduced under /api/v2
    

Source of Truth
---------------

This documentation is derived directly from:

*   src/api/v1/incidents.py
    
*   src/api/v1/execution\_ladder.py
    
*   src/api/v1/rollback.py
    

Any mismatch should be treated as a documentation bug.
