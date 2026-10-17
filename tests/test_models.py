"""Tests for core domain data models."""

from nexus_graphrag.core.models import (
    Community,
    DocumentChunk,
    Entity,
    EntityType,
    Relationship,
    RetrievalMode,
    RetrievalResult,
    slugify,
)


def test_slugify():
    assert slugify("Google DeepMind") == "google_deepmind"
    assert slugify("Python 3.13 / AI!") == "python_313_ai"
    assert slugify("  Alpha-Fold  ") == "alpha_fold"


def test_entity_creation_and_serialization():
    ent = Entity(
        name="Nexus-GraphRAG",
        entity_type=EntityType.TECHNOLOGY,
        description="Knowledge graph engine",
        weight=2.5,
        source_chunks=["chk_1", "chk_2"],
    )
    assert ent.id == "nexus_graphrag"
    assert ent.entity_type == EntityType.TECHNOLOGY
    assert ent.weight == 2.5

    data = ent.to_dict()
    assert data["name"] == "Nexus-GraphRAG"
    assert data["entity_type"] == "TECHNOLOGY"

    loaded = Entity.from_dict(data)
    assert loaded.name == ent.name
    assert loaded.id == ent.id
    assert loaded.weight == ent.weight


def test_entity_type_normalization():
    assert EntityType.from_string("person") == EntityType.PERSON
    assert EntityType.from_string("Technology") == EntityType.TECHNOLOGY
    assert EntityType.from_string("UNKNOWN_TYPE") == EntityType.OTHER


def test_relationship_creation_and_serialization():
    rel = Relationship(
        source="Google DeepMind",
        target="AlphaFold",
        relation_type="DEVELOPS",
        description="DeepMind develops AlphaFold",
        weight=1.5,
    )
    assert "google_deepmind" in rel.id
    assert "alphafold" in rel.id

    data = rel.to_dict()
    assert data["source"] == "Google DeepMind"
    assert data["target"] == "AlphaFold"
    assert data["relation_type"] == "DEVELOPS"

    loaded = Relationship.from_dict(data)
    assert loaded.source == rel.source
    assert loaded.target == rel.target
    assert loaded.weight == rel.weight


def test_document_chunk_creation():
    chunk = DocumentChunk(text="This is a test document slice.", chunk_index=0)
    assert chunk.token_count == 6
    assert chunk.id.startswith("chunk_0_")

    data = chunk.to_dict()
    loaded = DocumentChunk.from_dict(data)
    assert loaded.text == chunk.text
    assert loaded.token_count == chunk.token_count


def test_community_creation_and_serialization():
    comm = Community(
        id="comm_0",
        level=0,
        title="AI Research Community",
        summary="A cluster of AI tools and research labs.",
        member_entities=["Google DeepMind", "AlphaFold"],
        findings=["AlphaFold predicts protein structures."],
        weight=2.0,
    )
    data = comm.to_dict()
    assert data["id"] == "comm_0"
    assert len(data["member_entities"]) == 2

    loaded = Community.from_dict(data)
    assert loaded.title == comm.title
    assert loaded.findings == comm.findings


def test_retrieval_mode_normalization():
    assert RetrievalMode.from_string("local") == RetrievalMode.LOCAL
    assert RetrievalMode.from_string("GLOBAL") == RetrievalMode.GLOBAL
    assert RetrievalMode.from_string("hybrid") == RetrievalMode.HYBRID
    assert RetrievalMode.from_string("any_other") == RetrievalMode.HYBRID


def test_retrieval_result_serialization():
    res = RetrievalResult(
        query="What is Nexus-GraphRAG?",
        mode=RetrievalMode.HYBRID,
        context="Sample context",
        confidence_score=0.95,
        metrics={"latency_ms": 12.4},
    )
    data = res.to_dict()
    assert data["query"] == "What is Nexus-GraphRAG?"
    assert data["mode"] == "hybrid"
    assert data["metrics"]["latency_ms"] == 12.4
