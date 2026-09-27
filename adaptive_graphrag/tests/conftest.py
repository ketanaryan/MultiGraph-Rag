"""Pytest fixtures and configuration setup for Adaptive Agentic GraphRAG."""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_vector_chunks():
    """Provides sample vector chunks for retrieval and fusion testing."""
    return [
        {
            "chunk_id": "Doc1",
            "text": "The attention head computes scaled dot-product attention in Transformer architectures.",
            "similarity_score": 0.88,
        },
        {
            "chunk_id": "Doc2",
            "text": "TSMC manufactures advanced microchips for Apple and Nvidia.",
            "similarity_score": 0.82,
        },
        {
            "chunk_id": "Doc3",
            "text": "ASML provides lithography equipment to TSMC.",
            "similarity_score": 0.75,
        },
    ]


@pytest.fixture
def sample_graph_triples():
    """Provides sample graph triples with temporal weights."""
    return [
        {
            "triple_id": "T1",
            "subject": "TSMC",
            "predicate": "SUPPLIES_CHIPS_TO",
            "object": "Apple",
            "t_e": 100.0,
            "weight": 0.95,
        },
        {
            "triple_id": "T2",
            "subject": "ASML",
            "predicate": "PROVIDES_EUV_LITHOGRAPHY_TO",
            "object": "TSMC",
            "t_e": 95.0,
            "weight": 0.88,
        },
        {
            "triple_id": "T3",
            "subject": "Intel",
            "predicate": "LEGACY_CONTRACT",
            "object": "Apple",
            "t_e": 20.0,
            "weight": 0.22,  # Stale edge below 0.30
        },
    ]
