"""Unit tests for temporal graph decay and GraphStore filtering."""

import math
import pytest
from src.storage.graph_store import (
    calculate_temporal_edge_weight,
    GraphStore,
    get_graph_store,
)


def test_temporal_edge_weight_formula():
    """Validates w(e,t) = w_0 * exp(-lambda * (t_0 - t_e))."""
    w_0 = 1.0
    lam = 0.015
    t_0 = 100.0

    # At t_e = t_0, delta = 0 -> weight = 1.0
    assert calculate_temporal_edge_weight(w_0=w_0, t_e=100.0, t_0=t_0, decay_lambda=lam) == 1.0

    # At delta = 50 days -> weight = exp(-0.015 * 50) = exp(-0.75) ~ 0.4724
    expected_50 = round(math.exp(-0.015 * 50), 4)
    calculated_50 = calculate_temporal_edge_weight(w_0=w_0, t_e=50.0, t_0=t_0, decay_lambda=lam)
    assert abs(calculated_50 - expected_50) < 1e-3

    # At delta = 200 days -> weight = exp(-3.0) ~ 0.0498
    expected_200 = round(math.exp(-0.015 * 200), 4)
    calculated_200 = calculate_temporal_edge_weight(w_0=w_0, t_e=-100.0, t_0=t_0, decay_lambda=lam)
    assert abs(calculated_200 - expected_200) < 1e-3


def test_edge_pruning_threshold():
    """Validates that relationships with weight < 0.30 are filtered out."""
    store = GraphStore()
    t_0 = 100.0
    triples = [
        {"triple_id": "T_fresh", "subject": "TSMC", "object": "Apple", "t_e": 98.0, "w0": 1.0},
        {"triple_id": "T_medium", "subject": "ASML", "object": "TSMC", "t_e": 50.0, "w0": 1.0},  # ~0.47
        {"triple_id": "T_stale", "subject": "OldCo", "object": "Apple", "t_e": 10.0, "w0": 1.0},  # ~0.25 (< 0.30)
    ]

    filtered = store.filter_triples_by_temporal_decay(triples, t_0=t_0, tau_decay=0.30)
    surviving_ids = [t["triple_id"] for t in filtered]

    assert "T_fresh" in surviving_ids
    assert "T_medium" in surviving_ids
    assert "T_stale" not in surviving_ids
    assert all(t["weight"] >= 0.30 for t in filtered)


@pytest.mark.asyncio
async def test_graph_store_subgraph_query():
    """Validates subgraph querying and entity filtering."""
    store = get_graph_store()
    results = await store.query_subgraph(entities=["TSMC", "Apple"], depth=2)
    
    assert len(results) > 0
    # Every surviving edge must have weight >= tau_decay
    for item in results:
        assert item["weight"] >= 0.30
        assert "subject" in item
        assert "object" in item
        assert "predicate" in item
