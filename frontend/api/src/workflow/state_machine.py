"""LangGraph State Machine orchestrating Adaptive Agentic GraphRAG nodes."""

import time
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from src.config import settings, PromptTemplates
from src.schema import GraphRAGState
from src.routing import reciprocal_rank_fusion
from src.storage.vector_store import get_vector_store
from src.storage.graph_store import get_graph_store
from src.agents.planner import PlannerAgent
from src.agents.generator import GeneratorAgent
from src.agents.verifier import VerifierAgent

planner = PlannerAgent()
generator = GeneratorAgent()
verifier = VerifierAgent()
vector_store = get_vector_store()
graph_store = get_graph_store()


async def planner_node(state: GraphRAGState) -> Dict[str, Any]:
    """Analyzes query, estimates depth & density, calculates C(Q), sets routing."""
    start_t = time.time()
    plan_res = await planner.plan(
        query_raw=state["query_raw"],
        retry_count=state.get("retry_count", 0),
        memory_logs=state.get("memory_logs", []),
    )
    duration = round((time.time() - start_t) * 1000, 2)
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "planner", "duration_ms": duration, "strategy": plan_res.routing_strategy})

    return {
        "query_rewritten": plan_res.query_rewritten,
        "entities": plan_res.entities,
        "estimated_depth": plan_res.estimated_depth,
        "local_density": plan_res.local_density,
        "complexity_score": plan_res.complexity_score,
        "routing_strategy": plan_res.routing_strategy,
        "execution_trace": trace,
    }


async def retriever_node(state: GraphRAGState) -> Dict[str, Any]:
    """Retrieves context via Vector, Graph, or Hybrid fusion based on strategy."""
    start_t = time.time()
    strategy = state.get("routing_strategy", "vector")
    query = state.get("query_rewritten", state["query_raw"])
    entities = state.get("entities", [])
    depth = max(3, state.get("estimated_depth", 3))
    uploaded_docs = state.get("uploaded_documents", [])

    # Dynamic Modality Override: if user files are uploaded, always run hybrid retrieval
    if uploaded_docs:
        strategy = "hybrid"

    v_chunks: List[Dict[str, Any]] = []
    g_triples: List[Dict[str, Any]] = []

    if strategy in ("vector", "hybrid"):
        v_chunks = await vector_store.search(
            query=query,
            top_k=6,
            source_filter=uploaded_docs if uploaded_docs else None,
        )

    if strategy in ("graph", "hybrid"):
        g_triples = await graph_store.query_subgraph(
            entities=entities,
            depth=depth,
            source_filter=uploaded_docs if uploaded_docs else None,
        )

    # Ensure extracted triples from uploaded documents are always retrieved
    if uploaded_docs and not g_triples:
        g_triples = await graph_store.query_subgraph(
            entities=[],
            depth=depth,
            source_filter=uploaded_docs,
        )

    # Fuse context representation with clean document source labels
    context_blocks: List[str] = []
    if strategy == "hybrid":
        fused = reciprocal_rank_fusion(v_chunks, g_triples)
        for item in fused[:12]:
            doc_label = item.get("doc_name") or item.get("id")
            context_blocks.append(f"[{doc_label}]: {item['text']}")
    else:
        for c in v_chunks:
            doc_label = c.get("metadata", {}).get("source") or c.get("chunk_id")
            context_blocks.append(f"[{doc_label}]: {c['text']}")
        for t in g_triples:
            doc_label = t.get("source") or t.get("triple_id", "T")
            context_blocks.append(
                f"[{doc_label}]: ({t.get('subject')}) -[{t.get('predicate')}]-> ({t.get('object')}) (weight: {t.get('weight', 1.0):.2f})"
            )

    fused_context = "\n".join(context_blocks)
    duration = round((time.time() - start_t) * 1000, 2)
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "retriever", "duration_ms": duration, "chunks": len(v_chunks), "triples": len(g_triples)})

    return {
        "routing_strategy": strategy,
        "retrieved_vector_chunks": v_chunks,
        "retrieved_graph_triples": g_triples,
        "fused_context": fused_context,
        "execution_trace": trace,
    }


async def generator_node(state: GraphRAGState) -> Dict[str, Any]:
    """Synthesizes answer grounded exclusively in retrieved fused context."""
    start_t = time.time()
    answer = await generator.generate(
        query_raw=state["query_raw"],
        query_rewritten=state.get("query_rewritten", state["query_raw"]),
        fused_context=state.get("fused_context", ""),
    )
    duration = round((time.time() - start_t) * 1000, 2)
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "generator", "duration_ms": duration})

    return {"generated_response": answer, "execution_trace": trace}


async def verifier_node(state: GraphRAGState) -> Dict[str, Any]:
    """Calculates S_total = alpha*S_faith + beta*S_ans_rel + gamma*S_temp."""
    start_t = time.time()
    v_output = await verifier.verify(
        query_raw=state["query_raw"],
        generated_response=state.get("generated_response", ""),
        fused_context=state.get("fused_context", ""),
        retrieved_triples=state.get("retrieved_graph_triples", []),
        numerical_extractions=state.get("numerical_extractions", {}),
    )
    duration = round((time.time() - start_t) * 1000, 2)
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "verifier", "duration_ms": duration, "score": v_output.score_total, "passed": v_output.is_verified})

    updates: Dict[str, Any] = {
        "verification_score": v_output.score_total,
        "metric_breakdown": v_output.metric_breakdown.model_dump(),
        "execution_trace": trace,
    }

    if not v_output.is_verified:
        curr_retry = state.get("retry_count", 0)
        logs = list(state.get("memory_logs", []))
        logs.append(f"Iteration {curr_retry}: {v_output.critique}")
        updates["retry_count"] = curr_retry + 1
        updates["memory_logs"] = logs

    return updates


async def fallback_node(state: GraphRAGState) -> Dict[str, Any]:
    """Deterministic fallback node executed when retries are exhausted."""
    start_t = time.time()
    fallback_text = PromptTemplates.FALLBACK_RESPONSE_TEMPLATE.format(
        max_retries=settings.max_retry_count,
        score=state.get("verification_score", 0.0),
        threshold=settings.verification_threshold,
    )
    duration = round((time.time() - start_t) * 1000, 2)
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "fallback", "duration_ms": duration})

    return {
        "generated_response": fallback_text,
        "fallback_invoked": True,
        "execution_trace": trace,
    }


def verifier_conditional_router(state: GraphRAGState) -> str:
    """Routes to END if passed, planner_node if retry < 3, else fallback_node."""
    score = state.get("verification_score", 0.0)
    if score >= settings.verification_threshold:
        return "end"
    if state.get("retry_count", 0) < settings.max_retry_count:
        return "reflect"
    return "fallback"


def build_graphrag_graph() -> StateGraph:
    """Constructs compiled LangGraph state graph."""
    workflow = StateGraph(GraphRAGState)

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("retriever_node", retriever_node)
    workflow.add_node("generator_node", generator_node)
    workflow.add_node("verifier_node", verifier_node)
    workflow.add_node("fallback_node", fallback_node)

    workflow.set_entry_point("planner_node")
    workflow.add_edge("planner_node", "retriever_node")
    workflow.add_edge("retriever_node", "generator_node")
    workflow.add_edge("generator_node", "verifier_node")

    workflow.add_conditional_edges(
        "verifier_node",
        verifier_conditional_router,
        {
            "end": END,
            "reflect": "planner_node",
            "fallback": "fallback_node",
        },
    )
    workflow.add_edge("fallback_node", END)
    return workflow.compile()


async def execute_graphrag_pipeline(
    query: str,
    numerical_extractions: Optional[Dict[str, Any]] = None,
    uploaded_documents: Optional[List[str]] = None,
) -> GraphRAGState:
    """Executes state machine pipeline and returns final state."""
    app = build_graphrag_graph()
    initial_state: GraphRAGState = {
        "query_raw": query,
        "query_rewritten": query,
        "complexity_score": 0.0,
        "routing_strategy": "vector",
        "entities": [],
        "estimated_depth": 1,
        "local_density": 1.0,
        "retrieved_vector_chunks": [],
        "retrieved_graph_triples": [],
        "fused_context": "",
        "generated_response": "",
        "verification_score": 0.0,
        "metric_breakdown": {},
        "retry_count": 0,
        "memory_logs": [],
        "execution_trace": [],
        "fallback_invoked": False,
        "numerical_extractions": numerical_extractions or {},
        "math_verification": {},
        "uploaded_documents": uploaded_documents or [],
    }
    final_state = await app.ainvoke(initial_state)
    return final_state
