"""Tests for HybridRetriever and Reciprocal Rank Fusion."""

from nexus_graphrag.core.models import Community, DocumentChunk, RetrievalMode
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retriever(populated_index: GraphIndex):
    chunks = [
        DocumentChunk(
            text="Google DeepMind developed AlphaFold with PyTorch in London.",
            chunk_index=0,
            id="c_0",
        ),
        DocumentChunk(
            text="PyTorch is widely adopted across industry AI labs.",
            chunk_index=1,
            id="c_1",
        ),
    ]
    comms = [
        Community(
            id="comm_0",
            level=0,
            title="Deep Learning Frameworks",
            summary="Ecosystem of neural network libraries and AI research.",
            member_entities=["PyTorch", "AlphaFold"],
            findings=["PyTorch powers modern foundation models."],
        )
    ]

    hybrid = HybridRetriever(populated_index, comms, chunks=chunks)
    res = hybrid.retrieve("How does AlphaFold utilize PyTorch?", max_entities=4)

    assert res.mode == RetrievalMode.HYBRID
    assert len(res.entities) > 0
    assert "Macro Thematic Clusters" in res.context
    assert "Micro-Level Relational Triplet Paths" in res.context
    assert "Verbatim Grounding Evidence" in res.context
    assert res.confidence_score > 0.9
    assert "latency_ms" in res.metrics
