# Architectural Decision Records (ADRs)

## ADR-001: Orchestration Framework Selection — LangGraph vs. CrewAI / AutoGen
- **Status:** Accepted
- **Context:** Adaptive Agentic GraphRAG requires cyclical control flows (reflection loops), conditional branching, immutable typed state passing, and guaranteed bounded retries.
- **Decision:** Select **LangGraph** as the orchestration engine.
- **Rationale:**
  - Unlike chat-oriented conversational frameworks (CrewAI, AutoGen), LangGraph models agent interactions as a deterministic cyclic directed graph with first-class state checkpointing.
  - Bounded loops ($N_{\text{retry}} < 3$) can be strictly enforced using conditional edge predicates.
  - Granular node-level debugging and trace telemetry map directly to state machine states.

---

## ADR-002: Dynamic Query Routing Function — Logistic Sigmoid over Heuristic Rule Engine
- **Status:** Accepted
- **Context:** Fixed if/else rule sets for routing queries to vector or graph storage are brittle and do not scale when query length, entity count, and domain complexity vary.
- **Decision:** Implement a continuous logistic complexity formulation:
  $$C(Q) = \sigma\left( w_c^T \cdot E(Q) + b_c \cdot \frac{E_Q \cdot \text{depth}_Q}{1 + \log(D_Q)} \right)$$
- **Rationale:**
  - Continuous scoring enables smooth tri-modal partitioning: dense vector for simple facts ($< 0.35$), multi-hop graph traversal for structural questions ($\ge 0.70$), and hybrid reciprocal rank fusion for intermediate queries ($[0.35, 0.70)$).
  - Graph density normalization $1 + \log(D_Q)$ prevents densely connected hubs from skewing traversal depth estimates.

---

## ADR-003: Hybrid Context Fusion — Reciprocal Rank Fusion (RRF) vs. Linear Score Normalization
- **Status:** Accepted
- **Context:** Combining dense vector cosine similarity scores with graph path scores is challenging due to incompatible score scales and uncalibrated distance metrics.
- **Decision:** Use **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
- **Rationale:**
  - RRF is non-parametric and rank-based, eliminating the requirement to calibrate arbitrary distance metrics between ChromaDB and Neo4j.
  - Proven robust in information retrieval benchmarks when fusing disparate modalities.

---

## ADR-004: Temporal Knowledge Graph Edge Decay Pruning
- **Status:** Accepted
- **Context:** Enterprise knowledge graphs accumulate contradictory facts over time. Stale facts lead to hallucinated or superseded answers.
- **Decision:** Apply an exponential decay filter $w(e, t) = w_0 \cdot \exp(-\lambda (t_0 - t_e))$ with threshold $\tau_{\text{decay}} = 0.30$.
- **Rationale:**
  - Allows recent relationships to carry primary weight while gracefully diminishing stale historical relationships.
  - Computationally efficient to compute either within Cypher expressions or as an async post-traversal filter.

---

## ADR-005: Extrinsic Non-LLM Algorithmic Verification
- **Status:** Accepted
- **Context:** LLM self-reflection suffers from blind-spot confirmation bias (the model often agrees with its own previous answer).
- **Decision:** Enforce composite non-LLM algorithmic verification combining RAGAS claim faithfulness, semantic embedding cosine relevance, and graph temporal freshness.
- **Rationale:**
  - Provides objective, reproducible gating thresholds ($T_{\text{verify}} \ge 0.75$).
  - Prevents hallucinated claims from bypassing the verifier.
