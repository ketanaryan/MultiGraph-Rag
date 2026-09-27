# Testing Strategy & Test Suites

## 1. Testing Philosophy
The Adaptive Agentic GraphRAG repository employs a multi-tiered testing strategy ensuring mathematical correctness, async storage resiliency, multi-agent reflection convergence, and deterministic fallback compliance.

## 2. Test Suites Overview

### Suite 1: Query Complexity & Policy Routing (`tests/test_routing.py`)
- **Single-hop Boundary Tests:** Ensures queries with low entity count and hop depth produce $C(Q) < 0.35$ and map to `"vector"`.
- **Multi-hop Boundary Tests:** Ensures multi-entity, deep dependency queries produce $C(Q) \ge 0.70$ and map to `"graph"`.
- **Hybrid Domain Tests:** Verifies queries with intermediate complexity fall into $[0.35, 0.70)$ and invoke hybrid RRF fusion.
- **RRF Math Verification:** Validates that Reciprocal Rank Fusion correctly scores and orders intersecting items from vector and graph result sets with smoothing constant $k = 60$.

### Suite 2: Temporal Graph Edge Decay & Storage (`tests/test_graph.py`)
- **Decay Formula Verification:** Validates $w(e, t) = w_0 \cdot \exp(-\lambda \Delta t)$ across various elapsed time values.
- **Edge Filtering Rule:** Verifies edges where $w(e, t) < 0.30$ are filtered out from retrieved subgraphs.
- **Cypher Parameterization Check:** Validates that queries avoid string concatenation and use parameter dictionaries.
- **Async Driver Resilience:** Verifies graceful fallback to in-memory graph structures when live Neo4j daemon is unreachable.

### Suite 3: Multi-Agent LangGraph Workflow (`tests/test_workflow.py`)
- **Direct Pass Path:** Verifies execution when verifier scores $S_{\text{total}} \ge 0.75$ on attempt 0, terminating at `END`.
- **Reflection Loop Path:** Tests that a low score triggers reflection, increments `retry_count`, logs critique into `memory_logs`, and re-enters `planner_node`.
- **Hard Fallback Termination:** Simulates persistent low verification score and verifies that at `retry_count == 3` the system transitions to `fallback_node` with deterministic fallback output.

---

## 3. Running Test Suites

Execute all tests with verbose output:
```bash
pytest tests/ -v
```

Execute tests with test coverage reporting:
```bash
pytest --cov=src --cov-report=term-missing tests/
```

Execute individual test suite:
```bash
pytest tests/test_routing.py -v
pytest tests/test_graph.py -v
pytest tests/test_workflow.py -v
```
