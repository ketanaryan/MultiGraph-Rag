# System Architecture Specification

## 1. End-to-End System Diagram

```
                              User Query (Q)
                                    │
                                    ▼
                         ┌────────────────────┐
                         │   Planner Agent    │
                         │  (src/agents/      │
                         │   planner.py)      │
                         └──────────┬─────────┘
                                    │ Computes C(Q), Rewrites Q,
                                    │ Extracts Entities & Depth
                                    ▼
                         ┌────────────────────┐
                         │ Routing Evaluation │
                         │ (src/routing.py)   │
                         └──────────┬─────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │ C(Q) < 0.35              │ 0.35 <= C(Q) < 0.70      │ C(Q) >= 0.70
         ▼                          ▼                          ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Vector Search  │        │  Hybrid Search  │        │  Graph Search   │
│   (ChromaDB)    │        │ (Chroma + Neo4j)│        │ (Neo4j Cypher)  │
│ Dense Cosine    │        │  RRF Algorithm  │        │ Temporal Pruning│
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                         ┌────────────────────┐
                         │   Retriever Node   │
                         │ (Fuses Context K)  │
                         └──────────┬─────────┘
                                    ▼
                         ┌────────────────────┐
                         │  Generator Agent   │
                         │  (src/agents/      │
                         │   generator.py)    │
                         └──────────┬─────────┘
                                    │ Produces Answer Y with Citations
                                    ▼
                         ┌────────────────────┐
                         │   Verifier Agent   │
                         │  (src/agents/      │
                         │   verifier.py)     │
                         └──────────┬─────────┘
                                    │ Computes S_total = α·S_faith + β·S_ans_rel + γ·S_temp
                                    ▼
                     ┌──────────────────────────────┐
                     │ S_total >= T_verify (0.75)?  │
                     └──────────────┬───────────────┘
                                    │
                   ├── YES ─────────┴───────── NO ───┐
                   ▼                                 ▼
           ┌──────────────┐                 ┌──────────────────┐
           │ Final Output │                 │ Retry Count < 3? │
           │   [END]      │                 └────────┬─────────┘
           └──────────────┘                          │
                                    ├── YES ─────────┴───────── NO ───┐
                                    ▼                                 ▼
                         ┌────────────────────┐            ┌──────────────────────┐
                         │ Reflection Loop    │            │ Deterministic        │
                         │ -> Planner Node    │            │ Fallback Node        │
                         │ (N_retry = N + 1)  │            │ -> [END]             │
                         └────────────────────┘            └──────────────────────┘
```

---

## 2. Core Mathematical Formulations

### 2.1 Query Complexity Scoring Engine
The query complexity $C(Q) \in [0, 1]$ determines whether semantic proximity (vector) or relational topological traversal (graph) is optimal:

$$C(Q) = \sigma \left( w_c^T \cdot E(Q) + b_c \cdot \frac{E_Q \cdot \text{depth}_Q}{1 + \log(D_Q)} \right)$$

Where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the standard logistic sigmoid function.
- $E(Q)$ is the dense embedding feature representation of the query string.
- $w_c$ is the learned weight vector for query semantics.
- $b_c$ is the scaling bias coefficient for structural features (default: $0.85$).
- $E_Q$ is the count of named entities recognized in query $Q$.
- $\text{depth}_Q$ is the estimated reasoning traversal hop depth ($\ge 1$).
- $D_Q$ is the local neighborhood graph density ($D_Q \ge 1.0$ to ensure $\log(D_Q) \ge 0$).

#### Routing Decision Boundaries:
- **Single-hop:** $C(Q) < 0.35 \implies$ Route to Vector Store (`ChromaDB`).
- **Hybrid Fusion:** $0.35 \le C(Q) < 0.70 \implies$ Route to Reciprocal Rank Fusion over Vector + Graph.
- **Multi-hop:** $C(Q) \ge 0.70 \implies$ Route to Knowledge Graph Store (`Neo4j`).

---

### 2.2 Reciprocal Rank Fusion (RRF)
For queries in the hybrid window, results from vector similarity and graph subpaths are ranked and fused using:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $M = \{\text{vector}, \text{graph}\}$, $k = 60$ (standard smoothing constant), and $r_m(d)$ is the 1-based rank of item $d$ in modality $m$.

---

### 2.3 Temporal Knowledge Graph Edge Decay
Knowledge graphs degrade over time as facts evolve. Each edge $e = (u, r, v)$ created or updated at timestamp $t_e$ has a time-dependent weight $w(e, t)$ evaluated at reference time $t_0$:

$$w(e, t) = w_0 \cdot \exp(-\lambda (t_0 - t_e))$$

Where:
- $w_0$ is the intrinsic confidence weight of the triple ($w_0 \in [0.0, 1.0]$, default: $1.0$).
- $\lambda$ is the exponential decay constant ($\lambda > 0$, default: $0.015$ per day).
- $t_0 - t_e$ is the elapsed time in elapsed time units (days).
- **Pruning Rule:** An edge is dropped from Cypher traversal if:
  $$w(e, t) < \tau_{\text{decay}} \quad (\text{default } \tau_{\text{decay}} = 0.30)$$

---

### 2.4 Extrinsic Composite Verification Gating
Candidate answers $Y$ generated from fused context $K$ are scored programmatically:

$$S_{\text{total}} = \alpha \cdot S_{\text{faith}}(Y, K) + \beta \cdot S_{\text{ans\_rel}}(Y, Q) + \gamma \cdot S_{\text{temp}}(K)$$

Subject to constraint $\alpha + \beta + \gamma = 1.0$. Defaults:
- $\alpha = 0.50$ (Faithfulness / Grounding weight)
- $\beta = 0.30$ (Answer Relevance weight)
- $\gamma = 0.20$ (Temporal Validity weight)

Sub-metrics:
1. **Faithfulness ($S_{\text{faith}}$):** Sentence-level claim decomposition of $Y$ verified against statements in $K$.
2. **Answer Relevance ($S_{\text{ans\_rel}}$):** Semantic cosine similarity:
   $$S_{\text{ans\_rel}} = \frac{E(Y) \cdot E(Q)}{\|E(Y)\| \|E(Q)\|}$$
3. **Temporal Freshness ($S_{\text{temp}}$):** Mean normalized edge weight of retrieved triples:
   $$S_{\text{temp}}(K) = \frac{1}{|E_K|} \sum_{e \in E_K} w(e, t)$$

Threshold: If $S_{\text{total}} < T_{\text{verify}}$ ($0.75$), reject answer and trigger reflection.
