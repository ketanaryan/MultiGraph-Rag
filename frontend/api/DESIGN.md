# UI & Design System Specifications

## 1. Design Philosophy
The Streamlit interface for Adaptive Agentic GraphRAG is built as a state-of-the-art developer and executive intelligence console. It merges functional clarity with a modern dark-mode aesthetic, vibrant modality cues, real-time agent execution telemetry, and interactive knowledge graph visualization.

## 2. Design Tokens & Color Palette

### Theme Palette (HSL / Hex)
- **Background Primary:** `#0E1117` (Deep Obsidian Dark)
- **Surface Card / Container:** `#1A1F2C` (Navy Slate with subtle glassmorphism border `#2E3856`)
- **Text Primary:** `#F8FAFC` (Slate 50)
- **Text Secondary:** `#94A3B8` (Slate 400)
- **Border / Divider:** `#334155` (Slate 700)

### Modality Accent Badges
- **Vector Search Modality:**
  - Accent Color: `#38BDF8` (Electric Cyan)
  - Badge: `[⚡ Dense Vector Search | ChromaDB]`
- **Graph Traversal Modality:**
  - Accent Color: `#A855F7` (Deep Amethyst Purple)
  - Badge: `[🕸️ Multi-Hop Graph Traversal | Neo4j]`
- **Hybrid Fusion Modality:**
  - Accent Color: `#F59E0B` (Amber Flame)
  - Badge: `[🔀 Hybrid RRF Fusion | Vector + Graph]`

### Verification Status Tokens
- **Verified Pass ($S_{\text{total}} \ge 0.75$):** `#10B981` (Emerald Green)
- **Verification Failed / Reflection Loop:** `#EF4444` (Coral Crimson)
- **Deterministic Fallback:** `#EAB308` (Warning Gold)

---

## 3. UI Component Hierarchy & Layout

### Header & System Telemetry Bar
- System Title & Status Pills:
  - Neo4j Connection State: `● Online (Bolt localhost:7687)` / `○ In-Memory Mock`
  - ChromaDB Indexing State: `● Ready (4 collections)`
  - Active LLM Engine: `gpt-4o-mini / text-embedding-3-small`

### Main Two-Column Viewport (Ratio 45% : 55%)

#### Left Column: Query Console & Parameter Configuration
1. **Interactive Query Box:** Multi-line text input with sample preset queries:
   - *Single-Hop:* "What is the primary role of an attention head in Transformers?"
   - *Hybrid:* "Compare Apple and Microsoft patent strategies in edge AI."
   - *Multi-Hop:* "Find all supply chain dependencies connecting TSMC to ASML and Qualcomm."
2. **Dynamic Routing Badge:** Displays real-time predicted complexity $C(Q)$ and modality badge.
3. **Advanced Threshold Controls (Sidebar / Expander):**
   - Verification Threshold ($T_{\text{verify}}$) slider: `0.50 - 0.95` (Default: `0.75`).
   - Max Reflection Retries ($N_{\text{retry}}$) slider: `1 - 5` (Default: `3`).
   - Time-decay Half-Life ($\lambda$) slider: `0.005 - 0.050` (Default: `0.015`).
4. **Execution Action Button:** `Execute Adaptive Pipeline` with animated loading indicators.

#### Right Column: Visual Telemetry & Metric Inspection
1. **State Machine Execution Progress:** Step-by-step progress cards showing active agent node (`Planner` -> `Retriever` -> `Generator` -> `Verifier`).
2. **Verification Metric Scorecard:**
   - 4-metric score gauge: $S_{\text{total}}$, $S_{\text{faith}}$, $S_{\text{ans\_rel}}$, $S_{\text{temp}}$.
   - Metric Radar/Bar visualization with pass/fail threshold line ($0.75$).
3. **Interactive Subgraph Visualizer (GraphViz):**
   - Nodes color-coded by entity type (Organization, Product, Person, Concept).
   - Edges labeled with relation types and time-decay weights $w(e, t)$.
4. **Storage Debug Panel (Tabbed):**
   - Tab 1: Cypher Query & Retrieved Triples with temporal weights.
   - Tab 2: ChromaDB Chunks with Cosine Distance scores.
   - Tab 3: Fused Reciprocal Rank Fusion (RRF) list.

### Bottom Panel: Full Auditable State Inspector
- Collapsible expandable JSON inspector containing the complete immutable `GraphRAGState` snapshot for auditable compliance and debugging.
