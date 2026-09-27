"""Routing policy and complexity scoring engine for Adaptive Agentic GraphRAG."""

import re
import math
from typing import List, Dict, Any, Optional
from src.config import settings


def sigmoid(z: float) -> float:
    """Standard logistic sigmoid function with numerical stability protection."""
    if z < -50.0:
        return 0.0
    if z > 50.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def extract_query_semantic_bias(query: str) -> float:
    """Calculates w_c^T * E(Q) from semantic indicators and token complexity."""
    query_lower = query.lower()
    
    # Multi-hop relational indicator triggers
    multi_hop_triggers = [
        "relate", "connect", "influence", "dependency", "depend on",
        "impact of", "pathway", "chain", "compare", "difference between",
        "ancestor", "caused by", "lead to", "relationship", "subsequent"
    ]
    
    # Single-hop direct fact indicators
    single_hop_triggers = [
        "what is", "who is", "define", "where is", "when did",
        "which year", "meaning of", "capital of", "name of"
    ]
    
    score = 0.0
    for trigger in multi_hop_triggers:
        if trigger in query_lower:
            score += 0.35
            
    for trigger in single_hop_triggers:
        if trigger in query_lower:
            score -= 0.30

    # Token length factor: longer questions typically convey relational complexity
    token_count = len(re.findall(r"\b\w+\b", query))
    if token_count > 15:
        score += 0.20
    elif token_count < 6:
        score -= 0.25

    return max(-2.0, min(2.0, score))


def estimate_traversal_depth(query: str) -> int:
    """Estimates the structural relation traversal depth (depth_Q >= 1)."""
    query_lower = query.lower()
    two_hop_patterns = [
        r"\b(and|or)\b.*\b(connect|relat|affect|impact)\b",
        r"\bbetween\b.+\band\b",
        r"\bhow does\b.+\baffect\b",
        r"\bchain of\b",
        r"\bpath from\b.+\bto\b"
    ]
    three_hop_patterns = [
        r"\ball supply chain dependencies\b",
        r"\bindirect effects?\b",
        r"\btransitive\b",
        r"\bmulti-step\b"
    ]

    for pattern in three_hop_patterns:
        if re.search(pattern, query_lower):
            return 3

    for pattern in two_hop_patterns:
        if re.search(pattern, query_lower):
            return 2

    return 1


def calculate_query_complexity(
    query: str,
    entities: Optional[List[str]] = None,
    depth: Optional[int] = None,
    density: float = 1.0,
) -> float:
    """Calculates query complexity C(Q) via logistic sigmoid formulation.
    
    C(Q) = sigmoid( w_c^T * E(Q) + b_c * (E_Q * depth_Q) / (1 + log(D_Q)) )
    """
    # 1. Semantic bias w_c^T * E(Q)
    w_c_eq = extract_query_semantic_bias(query)

    # 2. Entity count E_Q
    if entities is not None:
        eq = len(entities)
    else:
        # Fallback entity heuristic: capitalized word sequences or quotes
        matches = re.findall(r"\b[A-Z][a-z0-9]+\b|\"[^\"]+\"", query)
        eq = len(matches) if matches else 1
    eq = max(1, eq)

    # 3. Traversal depth depth_Q
    depth_q = depth if depth is not None else estimate_traversal_depth(query)
    depth_q = max(1, depth_q)

    # 4. Local graph density D_Q >= 1.0
    d_q = max(1.0, float(density))

    # 5. Composite logit argument
    b_c = settings.routing_bias_c
    structural_component = b_c * ((eq * depth_q) / (1.0 + math.log(d_q)))
    
    # Standard normalization offset to center default single-hop questions around 0.20-0.30
    logit = (w_c_eq * 0.8) + (structural_component * 0.5) - 1.2
    return round(float(sigmoid(logit)), 4)


def determine_routing_strategy(complexity_score: float) -> str:
    """Maps complexity score to retrieval strategy according to thresholds:
    - C(Q) < 0.35 -> 'vector'
    - 0.35 <= C(Q) < 0.70 -> 'hybrid'
    - C(Q) >= 0.70 -> 'graph'
    """
    if complexity_score < settings.routing_tau_1:
        return "vector"
    if complexity_score >= settings.routing_tau_2:
        return "graph"
    return "hybrid"


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    graph_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merges vector chunks and graph triples using Reciprocal Rank Fusion (RRF).
    
    RRF(d) = sum_{m in M} (1 / (k + rank_m(d)))
    """
    rrf_scores: Dict[str, float] = {}
    content_map: Dict[str, Dict[str, Any]] = {}

    # Rank vector results (1-indexed)
    for rank, item in enumerate(vector_results, start=1):
        item_id = str(item.get("chunk_id", f"vec_{rank}"))
        doc_src = item.get("metadata", {}).get("source") or item.get("source") or item_id
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (k + rank))
        content_map[item_id] = {
            "source": "vector",
            "id": item_id,
            "doc_name": doc_src,
            "text": item.get("text", ""),
            "original_score": item.get("similarity_score", 0.0),
        }

    # Rank graph results (1-indexed)
    for rank, item in enumerate(graph_results, start=1):
        item_id = str(item.get("triple_id", f"trip_{rank}"))
        doc_src = item.get("source") or item_id
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (k + rank))
        content_map[item_id] = {
            "source": "graph",
            "id": item_id,
            "doc_name": doc_src,
            "text": f"({item.get('subject')}) -[{item.get('predicate')}]-> ({item.get('object_') or item.get('object')})",
            "original_score": item.get("weight", 1.0),
        }

    # Sort items descending by combined RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused_results: List[Dict[str, Any]] = []
    for item_id in sorted_ids:
        entry = content_map[item_id]
        entry["rrf_score"] = round(rrf_scores[item_id], 6)
        fused_results.append(entry)

    return fused_results
