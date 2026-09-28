# Coding Standards and Architecture Rules

## 1. Python Code Standards
- **Strict Type Hints:** Every function parameter and return type must be explicitly annotated using `typing` primitives (`Optional`, `Union`, `Dict`, `List`, `Tuple`, `Any`) and Pydantic v2 models.
- **Modular Limits (< 250 Lines of Code):** Under no circumstance shall any individual Python source file exceed 250 lines of code. Files approaching this limit must be refactored and decoupled across submodules in `src/`.
- **Async-First Execution:** Database drivers and long-running network operations (Neo4j driver sessions, ChromaDB queries, external API calls) must utilize `asyncio` and `async / await` syntax to avoid blocking the state machine thread.

## 2. LLM & Prompt Engineering Protocol
- **Zero Hardcoded Prompts:** Prompts, system instructions, and few-shot templates must NOT be embedded as inline strings within agent business logic. All prompt templates must reside centrally in `src/config.py`.
- **Structured JSON Output:** Agents producing structured data (Planner, Verifier) must enforce output schema compliance using Pydantic models with validation. Unstructured string outputs from LLMs are prohibited for control decisions.

## 3. Storage Layer Standards
- **Cypher Query Parameterization:** All Neo4j Cypher queries MUST use parameter binding (e.g., `MATCH (n:Entity {id: $entity_id})`) to prevent Cypher injection vulnerabilities. Direct string concatenation of user-provided tokens into Cypher queries is strictly forbidden.
- **Temporal Pruning Compliance:** Every graph edge query must incorporate the temporal decay filter $w(e, t) \ge \tau_{\text{decay}}$ directly in Cypher or via the storage post-filter pipeline before context fusion.
- **Vector Embedding Consistency:** Vector collections in ChromaDB must be queried with normalized cosine distance using designated embedding models (`text-embedding-3-small` or `all-MiniLM-L6-v2`).

## 4. State Orchestration & State Machine Rules
- **State Immutability:** Nodes in the LangGraph graph must treat incoming state dictionaries as immutable and return delta updates. Direct in-place mutation of input dictionaries is prohibited.
- **Bounded Reflection Depth:** The retry counter `retry_count` must be strictly incremented upon every reflection edge transition. Reaching `retry_count >= 3` MUST trigger the deterministic fallback node to ensure finite termination.
