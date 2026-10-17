"""Tests for NexusGraphRAG unified engine."""

import os
import tempfile

from nexus_graphrag.core.models import RetrievalMode
from nexus_graphrag.engine import NexusGraphRAG


def test_nexus_graphrag_index_text(sample_text: str):
    rag = NexusGraphRAG(chunk_size=300, chunk_overlap=50)
    summary = rag.index_text(sample_text, doc_id="bio_ai")

    assert summary["status"] == "success"
    assert summary["chunks_added"] >= 1
    assert summary["total_entities"] > 0
    assert summary["total_relationships"] > 0
    assert summary["communities_count"] >= 1


def test_nexus_graphrag_multi_document_indexing():
    rag = NexusGraphRAG()
    docs = [
        "Nitesh Beniwal created EvalGuard AI in Python for RAG evaluation.",
        "EvalGuard AI uses Google Gemini 2.0 Flash and local metrics to measure faithfulness.",
        "SynthoTab provides Differential Privacy and Gaussian Copula modeling for tabular data.",
    ]
    summary = rag.index_documents(docs)

    assert summary["documents_indexed"] == 3
    assert summary["total_entities"] >= 4
    assert summary["total_relationships"] >= 2


def test_nexus_graphrag_query_modes(sample_text: str):
    rag = NexusGraphRAG()
    rag.index_text(sample_text)

    # Local mode
    local_res = rag.query("Who developed AlphaFold?", mode=RetrievalMode.LOCAL)
    assert local_res.mode == RetrievalMode.LOCAL
    assert len(local_res.context) > 0

    # Global mode
    global_res = rag.query("Summarize main themes", mode="global")
    assert global_res.mode == RetrievalMode.GLOBAL

    # Hybrid mode
    hybrid_res = rag.query("How does AlphaFold utilize PyTorch?", mode="hybrid")
    assert hybrid_res.mode == RetrievalMode.HYBRID
    assert "Macro Thematic Clusters" in hybrid_res.context


def test_nexus_graphrag_save_load_lifecycle(sample_text: str):
    rag = NexusGraphRAG()
    rag.index_text(sample_text)

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "saved_rag.json")
        rag.save(json_path)

        loaded_rag = NexusGraphRAG.load(json_path)
        assert len(loaded_rag.index.entities) == len(rag.index.entities)
        assert len(loaded_rag.index.relationships) == len(rag.index.relationships)
        assert len(loaded_rag.communities) == len(rag.communities)

        # Test querying reloaded instance
        res = loaded_rag.query("AlphaFold", mode="hybrid")
        assert len(res.entities) > 0

        # Export html
        html_path = os.path.join(tmp_dir, "graph.html")
        loaded_rag.export_html(html_path)
        assert os.path.exists(html_path)
