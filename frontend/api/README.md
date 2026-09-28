# MultiAgent_GrapeRAG: Adaptive Agentic GraphRAG

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Multi--Agent-LangGraph-FF6F00?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%203.5%20%2F%203.6-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Neo4j](https://img.shields.io/badge/Graph-Neo4j%20Cypher-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![ChromaDB](https://img.shields.io/badge/Vector-ChromaDB-purple)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-19%2F19%20Passing-brightgreen)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**An enterprise-grade, hallucination-resistant Retrieval-Augmented Generation (RAG) framework combining multi-agent reflection loops, dynamic query complexity routing, temporal knowledge graph decay, and multi-hop reasoning.**

[Features](#-key-features) • [System Architecture](#-system-architecture) • [Mathematical Formulations](#-mathematical-formulations) • [Quickstart Guide](#-quickstart-guide) • [Multi-Hop Reasoning](#-multi-hop-pdf-reasoning) • [Project Structure](#-project-structure)

</div>

---

## 🌟 Key Features

- 🧠 **Dynamic Policy Complexity Routing ($C(Q)$)**: Continuously evaluates query complexity via a logistic sigmoid model, dynamically steering retrieval between:
  - **Single-hop Vector Search** (Dense cosine similarity via ChromaDB)
  - **Multi-hop Graph Traversal** (Cypher path expansion via Neo4j)
  - **Hybrid Retrieval** (Reciprocal Rank Fusion — RRF)
- ⏳ **Temporal Edge Decay & Graph Pruning**: Penalizes outdated relationships using continuous exponential decay: $w(e, t) = w_0 \cdot \exp(-\lambda (t_0 - t_e))$. Prunes any edges falling below threshold ($w < 0.30$).
- 🛡️ **Extrinsic Verification Gating ($S_{\text{total}}$)**: Programmatically validates synthesized answers across faithfulness, relevance, temporal freshness, and mathematical reconciliation ($S_{\text{total}} \ge 0.75$).
- 🔄 **LangGraph Self-Correction Reflection Loop**: Enforces reflection cycles bounded by a strict retry limit ($N_{\text{retry}} = 3$) with deterministic fallback if validation fails.
- ⚡ **Multi-Model Gemini Fallback Cascade**: Production-ready LLM generator featuring automatic failover across active Gemini models (`gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3-flash-preview`) with exponential backoff for HTTP 503/429 resilience.
- 📄 **Multi-PDF Concept Triplet Extraction**: Automatically parses uploaded PDFs (invoices, challans, resumes, technical specs) into atomic, domain-specific $(s, p, o)$ knowledge triplets and dense vector embeddings.
- 📊 **Interactive Streamlit Telemetry Console**: Dark-mode glassmorphic UI featuring live GraphViz subgraph visualization, execution latency metrics, routing modality badges, verification scorecards, and node-by-node execution traces.
- 🚀 **Zero-Infrastructure In-Memory Fallbacks**: Runs instantly out of the box without requiring Docker, Neo4j, or external vector servers.

---

## 🏗️ System Architecture

```
                             User Query (Q)
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │    Planner Agent    │
                        │ (src/agents/planner)│
                        └──────────┬──────────┘
                                   │ Rewrites Q, Computes C(Q),
                                   │ Identifies Entities & Traversal Depth
                                   ▼
                        ┌─────────────────────┐
                        │   Policy Routing    │
                        │  (src/routing.py)   │
                        └──────────┬──────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │ C(Q) < 0.35              │ 0.35 <= C(Q) < 0.70      │ C(Q) >= 0.70
        ▼                          ▼                          ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│ Vector Search │          │ Hybrid Search │          │ Graph Search  │
│  (ChromaDB)   │          │ (Chroma+Neo4j)│          │ (Neo4j Cypher)│
│ Dense Cosine  │          │ RRF Algorithm │          │Temporal Prune │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
                        ┌─────────────────────┐
                        │ Context Fusion Node │
                        │  Fuses K Passages + │
                        │  Knowledge Triples  │
                        └──────────┬──────────┘
                                   ▼
                        ┌─────────────────────┐
                        │   Generator Agent   │
                        │   Gemini Cascade +  │
                        │ Backoff & Citations │
                        └──────────┬──────────┘
                                   │ Candidate Answer Y
                                   ▼
                        ┌─────────────────────┐
                        │   Verifier Agent    │
                        │ RAGAS Faithfulness, │
                        │ Relevance & Freshness
                        └──────────┬──────────┘
                                   │ Computes S_total
                                   ▼
                    ┌──────────────────────────────┐
                    │ S_total >= T_verify (0.75)?  │
                    └──────────────┬───────────────┘
                                   │
                 ├── YES ──────────┴────────── NO ───┐
                 ▼                                   ▼
         ┌──────────────┐                   ┌──────────────────┐
         │ Final Output │                   │ Retry Count < 3? │
         │    [END]     │                   └────────┬─────────┘
         └──────────────┘                            │
                                   ├── YES ──────────┴────────── NO ───┐
                                   ▼                                   ▼
                        ┌─────────────────────┐             ┌──────────────────────┐
                        │  Reflection Loop    │             │ Deterministic        │
                        │  -> Re-plan Query   │             │ Grounded Fallback    │
                        │  (N_retry = N + 1)  │             │ -> [END]             │
                        └─────────────────────┘             └──────────────────────┘
```

---

## 📐 Mathematical Formulations

### 1. Dynamic Complexity Metric $C(Q)$
$$\mathcal{L}(Q) = w_1 \cdot \text{len}(Q) + w_2 \cdot |\mathcal{E}_Q| + w_3 \cdot D_{\text{hop}}(Q) + w_4 \cdot \mathbb{I}_{\text{rel}}(Q)$$
$$C(Q) = \sigma\left(\frac{\mathcal{L}(Q) - \mu_c}{\tau_c}\right) = \frac{1}{1 + \exp\left(-\frac{\mathcal{L}(Q) - \mu_c}{\tau_c}\right)}$$
- $C(Q) < 0.35 \implies$ **Vector Retrieval** (ChromaDB)
- $0.35 \le C(Q) < 0.70 \implies$ **Hybrid Retrieval** (RRF Fusion)
- $C(Q) \ge 0.70 \implies$ **Graph Multi-Hop Traversal** (Neo4j Cypher)

### 2. Temporal Edge Weight Decay
Outdated relationships lose confidence exponentially over time:
$$w(e, t) = w_0 \cdot \exp(-\lambda (t_0 - t_e))$$
Where:
- $w_0$: Initial edge confidence ($1.0$).
- $\lambda$: Decay rate ($\lambda = 0.015$).
- $t_0 - t_e$: Elapsed days since edge formation.
- Edges with $w(e, t) < 0.30$ are pruned from the active subgraph.

### 3. Reciprocal Rank Fusion (RRF)
$$\text{RRF}(d) = \sum_{m \in \{\text{vector}, \text{graph}\}} \frac{1}{k + r_m(d)}$$
Where $k = 60$ and $r_m(d)$ is the ordinal rank of document $d$ in retrieval modality $m$.

### 4. Extrinsic Verification Score $S_{\text{total}}$
$$S_{\text{total}} = \alpha \cdot S_{\text{faith}} + \beta \cdot S_{\text{ans\_rel}} + \gamma \cdot S_{\text{temp}}$$
- Weights: $\alpha = 0.40$, $\beta = 0.35$, $\gamma = 0.25$.
- Threshold: $T_{\text{verify}} = 0.75$.
- Automatic mathematical reconciliation penalty if numerical claims disagree with parsed source documents.

---

## 🚀 Quickstart Guide

### Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, and 3.14)
- **Git**
- *(Optional)* **Docker & Docker Compose** (for hosting live Neo4j and ChromaDB servers)
- **Google Gemini API Key** ([Get a free key from Google AI Studio](https://aistudio.google.com/))

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/Geeky-ash/MultiAgent_GrapeRAG.git
cd MultiAgent_GrapeRAG
```

---

### Step 2: Set Up Virtual Environment

**On Windows (PowerShell):**
```powershell
cd adaptive_graphrag
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
cd adaptive_graphrag
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```

Open `.env` in your editor and provide your Gemini API key:
```ini
# --- Gemini API Configuration ---
GEMINI_API_KEY="your_actual_gemini_api_key_here"
GEMINI_MODEL="gemini-3.5-flash"

# --- Storage Layer Settings (Default In-Memory Fallbacks Active) ---
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag_password
NEO4J_DATABASE=neo4j

CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHROMA_COLLECTION_NAME=adaptive_graphrag_chunks
```

> [!NOTE]
> **No Docker required to start!** If Neo4j or ChromaDB instances are not running, the system automatically engages high-performance in-memory graph and vector backends seamlessly.

---

### Step 5: Generate Sample Multi-Hop PDFs
Generate interlinked test PDFs demonstrating cross-document multi-hop reasoning:
```bash
python scripts/generate_multihop_pdfs.py
```
This generates 3 linked documents in `sample_pdfs/`:
1. `Doc_1_Architecture.pdf`: Project Aether $\rightarrow$ Subsystem-X9
2. `Doc_2_Hardware.pdf`: Subsystem-X9 $\rightarrow$ Helios Processing Core
3. `Doc_3_Supplier.pdf`: Helios Processing Core $\rightarrow$ QuantumTech Foundry

---

### Step 6: Run the Test Suite
Verify that all unit tests, routing boundaries, temporal formulas, and state machines pass:
```bash
python -m pytest tests/ -v
```
Expected output:
```
======================= 19 passed, 1 warning in ~50s =======================
```

---

### Step 7: Launch the Streamlit Dashboard
```bash
python -m streamlit run ui/app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

### Step 8: (Optional) Launch Dedicated Storage Backends
If you wish to use production Dockerized Neo4j and ChromaDB containers:
```bash
docker compose up -d
```
- **Neo4j Browser**: `http://localhost:7474` (User: `neo4j`, Password: `graphrag_password`)
- **ChromaDB Endpoint**: `http://localhost:8000`

To shut down containers:
```bash
docker compose down
```

---

## 🔍 Multi-Hop PDF Reasoning

Adaptive Agentic GraphRAG excels at resolving cross-document transitive relationships that traditional vector RAG misses:

```
[Doc 1: Architecture]           [Doc 2: Hardware]              [Doc 3: Supplier]
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│  Project Aether  │──(runs)──>│   Subsystem-X9   │──(powered)─>│   Helios Core    │──(made by)──> QuantumTech Foundry
└──────────────────┘           └──────────────────┘           └──────────────────┘
```

### Try This Query in the UI:
> *"Who is the primary manufacturer responsible for fabricating the hardware unit that powers Project Aether?"*

**Expected Grounded Answer:**
> **QuantumTech Foundry** is the primary manufacturer.
>
> **Logical Chain:**
> 1. *Project Aether* operates on *Subsystem-X9* (`Doc_1_Architecture.pdf`).
> 2. *Subsystem-X9* is powered by the *Helios Processing Core* (`Doc_2_Hardware.pdf`).
> 3. The *Helios Processing Core* is manufactured exclusively by *QuantumTech Foundry* (`Doc_3_Supplier.pdf`).

The interactive GraphViz viewer will render the 3-hop connected path with clean relationship badges and verification scores!

---

## 📁 Project Structure

```
MultiAgent_GrapeRAG/
├── .gitignore                   # Root Git ignore rules
├── README.md                    # Main repository documentation
└── adaptive_graphrag/
    ├── .env.example             # Environment template
    ├── docker-compose.yml       # Neo4j and ChromaDB services
    ├── requirements.txt         # Core Python dependencies
    ├── ARCHITECTURE.md          # Full architectural specification
    ├── PRD.md                   # Product Requirements Document
    ├── DESIGN.md                # System design guidelines
    ├── RULES.md                 # Development & LOC constraints (<250 LOC)
    ├── TESTING.md               # Test coverage & verification specs
    ├── sample_pdfs/             # Generated multi-hop test documents
    ├── scripts/
    │   └── generate_multihop_pdfs.py  # Multi-hop PDF generator
    ├── src/
    │   ├── config.py            # Centralized settings & Pydantic models
    │   ├── schema.py            # StateGraph TypedDict schema definitions
    │   ├── routing.py           # Complexity evaluation & RRF algorithms
    │   ├── storage/
    │   │   ├── graph_store.py   # Neo4j / In-memory graph with temporal decay
    │   │   ├── vector_store.py  # ChromaDB / In-memory dense cosine search
    │   │   └── pdf_parser.py    # Multi-PDF triplet extractor & parser
    │   ├── agents/
    │   │   ├── planner.py       # Query rewriting & plan formulation
    │   │   ├── generator.py     # Gemini cascade with backoff & citations
    │   │   └── verifier.py      # Faithfulness & relevance verification
    │   ├── workflow/
    │   │   └── state_machine.py # Compiled LangGraph StateGraph pipeline
    │   └── utils/
    │       └── metrics.py       # RAGAS metrics & math reconciliation
    ├── tests/
    │   ├── conftest.py          # Pytest fixtures & setup
    │   ├── test_routing.py      # Routing policy & RRF tests
    │   ├── test_graph.py        # Temporal decay & subgraph query tests
    │   ├── test_pdf_parser.py   # Triplet extraction & entity tests
    │   └── test_workflow.py     # End-to-end LangGraph execution tests
    └── ui/
        ├── app.py               # Streamlit application entry point
        └── styles.py            # Editorial dark-mode UI stylesheet
```

---

## ⚙️ Configuration Reference

Key settings configurable via `adaptive_graphrag/.env`:

| Parameter | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(Required)* | Google Gemini API key for generation & extraction |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Default Gemini model to attempt first |
| `ROUTING_TAU_1` | `0.35` | Lower threshold: Below this routes to Vector search |
| `ROUTING_TAU_2` | `0.70` | Upper threshold: Above this routes to Graph search |
| `TEMPORAL_DECAY_LAMBDA` | `0.015` | Exponential decay rate $\lambda$ for edge confidence |
| `VERIFICATION_THRESHOLD` | `0.75` | Minimum $S_{\text{total}}$ score required to pass verification |
| `MAX_REFLECTIONS` | `3` | Maximum retry attempts for self-correcting reflection loop |

---

## 🧪 Benchmarking

Benchmark routing efficiency, retrieval latency, and answer accuracy across synthetic queries:
```bash
python benchmarks/run_benchmark.py
```

---

## 🛡️ Quality Standards

This codebase strictly adheres to enterprise engineering standards:
- **Strict LOC Budget**: Every single Python file is strictly under **250 lines of code** for maximal maintainability.
- **Zero Mock Fallbacks**: Production answers are synthesized directly from verified context and knowledge graph evidence.
- **100% Type Annotated**: Full `typing` and Pydantic schema validation across all inputs and outputs.
- **Self-Healing Resilience**: Automatically cascades across alternative Gemini models and uses exponential backoff during high-demand periods.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">
Developed with ❤️ by <a href="https://github.com/Geeky-ash">Ashirwad Sharma</a>
</div>
