# 🧠 Adaptive GraphRAG: Enterprise Multi-Agent Architecture

## 🚀 Overview
**Adaptive GraphRAG** is an enterprise-grade, multi-agent AI architecture designed to eliminate hallucinations in large language models. Unlike traditional AI chatbots (like ChatGPT or standard Gemini) that rely purely on semantic probability to guess the next word, this system dynamically constructs a **deterministic mathematical Knowledge Graph** from your uploaded documents and algorithmically verifies answers before they are shown to the user.

---

## 🆚 Why is this different from ChatGPT / Normal Chatbots?

### 1. The Hallucination Problem (Traditional Chatbots)
Standard chatbots and traditional RAG (Retrieval-Augmented Generation) systems use basic **Vector Search**. They slice documents into chunks and find paragraphs that have similar keywords to the user's question. 
* **The Flaw:** If a user asks a complex multi-hop question (e.g., *"Does the manufacturer of the iPhone chip also license technology from Arm?"*), traditional vector search fails because the answer is scattered across multiple different documents. The AI is forced to "guess" the connection, leading to hallucinations.

### 2. The Deterministic Solution (Adaptive GraphRAG)
Our architecture solves this by extracting **Entities and Relationships** directly into a **Neo4j Graph Database**. 
Instead of just matching keywords, our AI traces literal mathematical paths across nodes (e.g., `[Apple] <-(SUPPLIES)- [TSMC] -(LICENSES)-> [Arm]`). 

Furthermore, a dedicated **Verifier Agent** audits the generated answer against the Graph Database. If the answer is not mathematically proven by the graph, the Verifier rejects the answer, rewrites it, and tries again. **This guarantees 100% factual auditability.**

---

## 🛠️ The Technology Stack

### 1. Cloud Infrastructure & Hosting
* **Vercel (Serverless Edge):** The entire application operates on a decoupled serverless architecture. Vercel hosts both the React frontend and handles the Python API routing, ensuring infinite scalability with zero idle server costs.
* **Neo4j AuraDB (Managed Cloud Graph):** A persistent, highly-available graph database that permanently stores the extracted Knowledge Graph (`Nodes` and `Edges`).

### 2. Frontend Layer
* **Next.js & React:** Provides a lightning-fast, reactive user interface.
* **Tailwind CSS:** For premium, responsive enterprise styling.
* **Server-Sent Events (SSE):** Utilizes `app.astream()` to stream the internal "thoughts" of the autonomous agents directly to the UI in real-time, proving the system's reasoning process visually.

### 3. Backend & AI Orchestration
* **Python FastAPI:** A high-performance async API that serves as the bridge between the frontend and the AI logic.
* **LangGraph (State Machine):** The core intelligence engine. Instead of a single linear prompt, LangGraph orchestrates a team of specialized AI agents that pass a shared state back and forth.
* **Google Gemini 1.5 Flash:** The primary LLM used for high-speed Information Extraction (NER) and final answer synthesis.

---

## 🤖 The Multi-Agent Workflow

When a user asks a question, it doesn't just go to a single AI. It goes through a corporate "assembly line" of autonomous agents:

1. **The Planner Agent:** Analyzes the user's query and decides the routing strategy. If the query is simple, it routes to standard vector search. If it requires deep reasoning, it triggers a Multi-Hop Graph Traversal.
2. **The Retriever Agent:** Executes complex Cypher queries against the Neo4j Cloud Database, tracing relationships across multiple documents to gather raw factual evidence.
3. **The Generator Agent:** Synthesizes the raw mathematical facts into a fluent, human-readable answer.
4. **The Verifier Agent:** The strict auditor. It uses a custom Temporal Decay formula (`w(e,t) = w0 * exp(-λ * Δt)`) to ensure the facts used are up-to-date and accurate. If the verification score falls below `0.75`, the answer is rejected and regenerated.

---

## 📊 Temporal Edge Decay (Algorithmic Verification)
A unique feature of this architecture is time-aware knowledge. Information degrades over time. If a document from 2021 says "TSMC manufactures 5nm chips", and a document from 2024 says "TSMC manufactures 3nm chips", our Graph Database applies an exponential decay function to the 2021 edge, naturally prioritizing the most recent structural reality without deleting historical context.
