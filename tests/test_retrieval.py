"""Tests for BM25, Vector Search, LocalRetriever, and GlobalRetriever."""

from nexus_graphrag.core.models import Community, DocumentChunk, RetrievalMode
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.retrieval.global_retriever import GlobalRetriever
from nexus_graphrag.retrieval.local_retriever import LocalRetriever
from nexus_graphrag.retrieval.search import BM25Index, VectorSearchIndex


def test_bm25_search():
    index = BM25Index()
    docs = [
        "Machine learning models predict patterns in datasets.",
        "Deep learning architectures use neural networks and backpropagation.",
        "Graph neural networks operate on nodes and edges.",
    ]
    ids = ["d1", "d2", "d3"]
    index.index_documents(ids, docs)

    hits = index.search("neural networks", top_k=2)
    assert len(hits) >= 1
    # Check top hit is d2 or d3 which contain neural networks
    top_id = hits[0][0]
    assert top_id in ["d2", "d3"]


def test_vector_search():
    index = VectorSearchIndex(n_features=256)
    docs = [
        "Differential privacy injects calibrated noise into queries.",
        "Synthetic data generators emulate multi-type tabular distributions.",
    ]
    ids = ["dp", "synth"]
    index.index_documents(ids, docs)

    hits = index.search("privacy noise", top_k=1)
    assert len(hits) == 1
    assert hits[0][0] == "dp"


def test_local_retriever(populated_index: GraphIndex):
    chunks = [
        DocumentChunk(
            text="AlphaFold was created by DeepMind to predict 3D protein structures.",
            chunk_index=0,
            id="chk_af",
        )
    ]
    retriever = LocalRetriever(populated_index, chunks=chunks)

    result = retriever.retrieve("What is AlphaFold?", max_entities=3)
    assert result.mode == RetrievalMode.LOCAL
    assert len(result.entities) > 0
    assert any("AlphaFold" in e.name for e in result.entities)
    assert "AlphaFold" in result.context
    assert result.metrics["latency_ms"] >= 0


def test_global_retriever(populated_index: GraphIndex):
    comms = [
        Community(
            id="c1",
            level=0,
            title="Bio-AI & Protein Modeling",
            summary="Focuses on molecular modeling, AlphaFold, and computational biology.",
            member_entities=["AlphaFold", "PyTorch"],
            findings=["AlphaFold maps protein conformations."],
        )
    ]
    retriever = GlobalRetriever(comms, populated_index)

    result = retriever.retrieve("Summarize molecular protein modeling", top_k=1)
    assert result.mode == RetrievalMode.GLOBAL
    assert len(result.communities) == 1
    assert "Bio-AI & Protein Modeling" in result.context
    assert "findings" in result.context.lower() or "finding" in result.context.lower()
