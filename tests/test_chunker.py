"""Tests for TextChunker."""

import pytest

from nexus_graphrag.core.chunker import TextChunker


def test_chunker_empty_and_whitespace():
    chunker = TextChunker()
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\t  ") == []


def test_chunker_small_text():
    chunker = TextChunker(chunk_size=500)
    text = "Short sentence that easily fits in one chunk."
    chunks = chunker.split_text(text, doc_id="test_doc")
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].id == "test_doc_chunk_0"


def test_chunker_sentence_splitting():
    chunker = TextChunker(chunk_size=120, chunk_overlap=30, respect_sentences=True)
    text = (
        "First sentence discussing knowledge graphs. "
        "Second sentence covering multi-hop traversal. "
        "Third sentence exploring Louvain community modularity. "
        "Fourth sentence demonstrating hybrid RAG retrieval."
    )
    chunks = chunker.split_text(text, doc_id="doc_sentence")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) > 0
        assert c.metadata.get("doc_id") == "doc_sentence"


def test_chunker_fixed_splitting():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10, respect_sentences=False)
    text = "A" * 120
    chunks = chunker.split_text(text)
    assert len(chunks) >= 3


def test_chunker_invalid_arguments():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=-5)
