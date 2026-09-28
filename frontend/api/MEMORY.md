# Global State Schema & State Persistence

## 1. LangGraph State Object Definition
The state machine passes a unified typed state dictionary `GraphRAGState` across all execution nodes:

```python
from typing import TypedDict, List, Dict, Any, Optional

class GraphRAGState(TypedDict):
    query_raw: str
    query_rewritten: str
    complexity_score: float
    routing_strategy: str  # "vector" | "graph" | "hybrid"
    entities: List[str]
    estimated_depth: int
    local_density: float
    retrieved_vector_chunks: List[Dict[str, Any]]
    retrieved_graph_triples: List[Dict[str, Any]]
    fused_context: str
    generated_response: str
    verification_score: float
    metric_breakdown: Dict[str, float]
    retry_count: int
    memory_logs: List[str]
    execution_trace: List[Dict[str, Any]]
    fallback_invoked: bool
```

---

## 2. Memory Log Format & Reflection Context
When the Verifier agent flags a response because $S_{\text{total}} < T_{\text{verify}}$, the system generates a structured memory log entry that is appended to `memory_logs`:

```json
{
  "retry_iteration": 1,
  "timestamp": "2026-09-19T10:30:00Z",
  "rejected_response": "TSMC was founded in 1987 in Tokyo...",
  "failure_reason": "Low faithfulness score S_faith=0.42. Hallucinated founding location.",
  "score_total": 0.58,
  "metric_breakdown": {
    "faithfulness": 0.42,
    "answer_relevance": 0.81,
    "temporal_validity": 0.65
  },
  "suggested_query_modification": "Focus retrieval on TSMC founding location and historical incorporation records."
}
```

The Planner agent consumes these memory logs on reflection iterations to adjust entity queries, search depth, or routing strategy.

---

## 3. Execution Trace Telemetry
Every node appends an event to `execution_trace` recording:
- `node_name`: Name of executed node (`planner`, `retriever`, `generator`, `verifier`, `fallback`).
- `timestamp`: UTC execution time.
- `duration_ms`: Wall-clock elapsed execution time.
- `metadata`: Inputs/outputs relevant for auditing.

This ensures full provenance and reproducibility for compliance and debugging.
