# Agent System Specifications

## 1. Multi-Agent Architecture
Adaptive Agentic GraphRAG leverages a deterministic multi-agent pipeline governed by a LangGraph State Machine. The pipeline decouples planning, contextual synthesis, and algorithmic verification into specialized autonomous nodes.

```
┌─────────────────┐
│  Planner Agent  │  <-- Analyzes query, estimates complexity, extracts entities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retriever Node  │  <-- Vector (ChromaDB) / Graph (Neo4j) / Hybrid (RRF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generator Agent │  <-- Strict context-grounded synthesis with triple citations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Verifier Agent  │  <-- Programmatic scoring: S_faith, S_ans_rel, S_temp
└────────┬────────┘
         │
   Pass S_total >= 0.75?
   ├── YES ──> [END]
   └── NO ───> Retry Count < 3?
               ├── YES ──> [Planner Agent (Reflection)]
               └── NO ───> [Deterministic Fallback]
```

---

## 2. Agent Roles & Specifications

### A. Planner Agent
- **Purpose:** Analyzes the raw query $Q$, extracts key named entities, estimates structural hop depth $\text{depth}_Q$, calculates local graph density $D_Q$, computes routing complexity $C(Q)$, and rewrites the query for maximum retrieval efficacy. On retry loops, incorporates verifier feedback to adjust query phrasing.
- **Input:**
  - `query_raw`: Raw user query string.
  - `memory_logs`: Historical trace of previous verification failures (if reflection loop).
  - `retry_count`: Current execution iteration counter.
- **Output (`PlannerOutput`):**
  - `query_rewritten`: Disambiguated and expanded search query.
  - `entities`: List of identified focal entities.
  - `estimated_depth`: Integer estimating multi-hop depth ($\ge 1$).
  - `local_density`: Float representing local neighborhood degree $D_Q$.
  - `complexity_score`: Calculated $C(Q) \in [0.0, 1.0]$.
  - `routing_strategy`: `"vector" | "graph" | "hybrid"`.
  - `reflection_notes`: Optional diagnostic critique explaining plan adjustments.

### B. Generator Agent
- **Purpose:** Synthesizes clear, factual responses based strictly and exclusively on the fused context retrieved from ChromaDB and Neo4j. It is prevented from using parametric pre-training assumptions.
- **Input:**
  - `query_rewritten`: Refined query.
  - `fused_context`: Context block combining ranked vector passages and pruned graph triples.
- **Guardrails & Constraints:**
  - "Synthesize an answer using ONLY the retrieved context. Do NOT use parametric memory. Cite source triple IDs and document chunk references."
  - Output must explicitly decline to answer if the context is insufficient.
- **Output:**
  - `generated_response`: Grounded response with inline citations (`[T1]`, `[Doc3]`).

### C. Verifier Agent
- **Purpose:** Programmatically evaluates the candidate answer without subjective LLM self-evaluation. Calculates factual grounding against retrieved sources, semantic alignment with the user's intent, and temporal freshness of evidence.
- **Input:**
  - `generated_response`: Output $Y$ from Generator.
  - `fused_context`: Retrieved context $K$.
  - `query_raw`: Original query $Q$.
- **Calculations:**
  - $S_{\text{faith}}(Y, K)$: Sentence-level atomic claim verification against retrieved evidence.
  - $S_{\text{ans\_rel}}(Y, Q)$: Cosine similarity between embedding vectors of $Y$ and $Q$.
  - $S_{\text{temp}}(K)$: Exponential decay recency score of retrieved context triples.
  - $S_{\text{total}} = 0.50 \cdot S_{\text{faith}} + 0.30 \cdot S_{\text{ans\_rel}} + 0.20 \cdot S_{\text{temp}}$.
- **Output (`VerificationOutput`):**
  - `score_total`: Composite verification score $S_{\text{total}} \in [0.0, 1.0]$.
  - `is_verified`: Boolean flag (`true` if $S_{\text{total}} \ge 0.75$).
  - `metric_breakdown`: Detailed dictionary of sub-scores.
  - `critique`: Actionable diagnostic critique passed to reflection node upon failure.

---

## 3. Inter-Agent Communication Protocol
Agents communicate exclusively via the immutable typed state dictionary `GraphRAGState`. Direct agent-to-agent peer invocation is prohibited; all control flow transitions are handled deterministically by the LangGraph orchestrator.
