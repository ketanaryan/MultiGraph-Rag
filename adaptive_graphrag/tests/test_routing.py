"""Unit tests for query complexity function and routing strategy selection."""

import pytest
from src.routing import (
    sigmoid,
    calculate_query_complexity,
    determine_routing_strategy,
    reciprocal_rank_fusion,
    estimate_traversal_depth,
)


def test_sigmoid_monotonicity_and_bounds():
    """Validates that sigmoid(z) output remains strictly in (0, 1) and is monotonic."""
    assert sigmoid(-100.0) == 0.0
    assert sigmoid(100.0) == 1.0
    assert abs(sigmoid(0.0) - 0.5) < 1e-5
    assert sigmoid(-2.0) < sigmoid(0.0) < sigmoid(2.0)


def test_estimate_traversal_depth():
    """Tests hop depth estimation heuristic on single-hop and multi-hop queries."""
    single_hop = "What is the capital of France?"
    assert estimate_traversal_depth(single_hop) == 1

    two_hop = "What is the relationship between Apple and TSMC?"
    assert estimate_traversal_depth(two_hop) >= 2

    multi_hop = "Find all supply chain dependencies connecting ASML to TSMC and Apple."
    assert estimate_traversal_depth(multi_hop) == 3


def test_single_hop_vector_routing():
    """Queries with simple factual structure should have C(Q) < 0.35 and route to 'vector'."""
    query = "What is an attention head?"
    complexity = calculate_query_complexity(query, entities=["Attention_Head"], depth=1, density=1.0)
    assert complexity < 0.35
    assert determine_routing_strategy(complexity) == "vector"


def test_multi_hop_graph_routing():
    """Queries with high entity count and deep traversal depth should have C(Q) >= 0.70 and route to 'graph'."""
    query = "Trace all supply chain dependencies connecting ASML to TSMC, Nvidia, and Apple."
    complexity = calculate_query_complexity(
        query,
        entities=["ASML", "TSMC", "Nvidia", "Apple"],
        depth=3,
        density=2.6,
    )
    assert complexity >= 0.70
    assert determine_routing_strategy(complexity) == "graph"


def test_hybrid_routing_boundary():
    """Intermediate relational queries should map to 'hybrid' (0.35 <= C(Q) < 0.70)."""
    complexity = 0.52
    assert determine_routing_strategy(complexity) == "hybrid"


def test_reciprocal_rank_fusion(sample_vector_chunks, sample_graph_triples):
    """Validates RRF scoring formula and correct ordering."""
    fused = reciprocal_rank_fusion(sample_vector_chunks, sample_graph_triples, k=60)
    
    assert len(fused) == len(sample_vector_chunks) + len(sample_graph_triples)
    # Check that highest RRF items come first
    scores = [item["rrf_score"] for item in fused]
    assert scores == sorted(scores, reverse=True)

    # Rank 1 item in single list should have 1 / (60 + 1) = ~0.016393
    top_score = fused[0]["rrf_score"]
    assert top_score >= round(1.0 / 61.0, 5)
