"""Tests for DeterministicExtractor and LLMExtractor."""

from nexus_graphrag.core.models import DocumentChunk, EntityType
from nexus_graphrag.extractors.deterministic import DeterministicExtractor
from nexus_graphrag.extractors.llm_extractor import LLMExtractor


def test_deterministic_extractor_basic(sample_text: str):
    extractor = DeterministicExtractor()
    chunk = DocumentChunk(text=sample_text, chunk_index=0, id="chk_0")
    entities, relationships = extractor.extract([chunk])

    assert len(entities) > 0
    ent_names = [e.name for e in entities]
    # Check that key entities like PyTorch or DeepMind were recognized
    assert any("PyTorch" in n for n in ent_names)
    assert any("DeepMind" in n or "AlphaFold" in n for n in ent_names)

    # Check relationships exist
    assert len(relationships) > 0
    rel_types = [r.relation_type for r in relationships]
    assert any(t in ["DEVELOPS", "USES", "ASSOCIATED_WITH", "DEPENDS_ON"] for t in rel_types)


def test_deterministic_extractor_predicate_matching():
    extractor = DeterministicExtractor()
    text = "Google DeepMind develops AlphaFold. AlphaFold integrates with PyTorch."
    chunk = DocumentChunk(text=text, chunk_index=0, id="chk_pred")
    entities, relationships = extractor.extract([chunk])

    develops_rel = [r for r in relationships if r.relation_type == "DEVELOPS"]
    assert len(develops_rel) > 0

    integrates_rel = [r for r in relationships if r.relation_type == "INTEGRATES_WITH"]
    assert len(integrates_rel) > 0


def test_deterministic_extractor_deduplication():
    extractor = DeterministicExtractor()
    chunk1 = DocumentChunk(text="CrewAI is an agent framework.", chunk_index=0, id="chk_1")
    chunk2 = DocumentChunk(text="CrewAI integrates with Python.", chunk_index=1, id="chk_2")

    entities, relationships = extractor.extract([chunk1, chunk2])
    crewai_ents = [e for e in entities if "CrewAI" in e.name or "crewai" in e.id]
    assert len(crewai_ents) == 1
    assert "chk_1" in crewai_ents[0].source_chunks
    assert "chk_2" in crewai_ents[0].source_chunks
    assert crewai_ents[0].weight > 1.0


def test_llm_extractor_mock_success():
    def mock_llm_fn(prompt: str) -> str:
        return """```json
        {
          "entities": [
            {"name": "NexusRAG", "type": "TECHNOLOGY", "description": "Graph engine"},
            {"name": "Python", "type": "TECHNOLOGY", "description": "Programming language"}
          ],
          "relationships": [
            {"source": "NexusRAG", "target": "Python", "relation_type": "USES", "description": "NexusRAG uses Python"}
          ]
        }
        ```"""

    extractor = LLMExtractor(llm_fn=mock_llm_fn)
    chunk = DocumentChunk(text="NexusRAG uses Python.", chunk_index=0, id="chk_llm")
    entities, relationships = extractor.extract([chunk])

    assert len(entities) == 2
    assert entities[0].name == "NexusRAG"
    assert entities[0].entity_type == EntityType.TECHNOLOGY
    assert len(relationships) == 1
    assert relationships[0].relation_type == "USES"


def test_llm_extractor_fallback_on_error():
    def broken_llm_fn(prompt: str) -> str:
        raise RuntimeError("LLM rate limit reached")

    extractor = LLMExtractor(llm_fn=broken_llm_fn)
    chunk = DocumentChunk(text="Google DeepMind developed AlphaFold.", chunk_index=0, id="chk_err")
    entities, relationships = extractor.extract([chunk])

    # Should gracefully fall back to deterministic extraction
    assert len(entities) > 0
    assert any("AlphaFold" in e.name or "DeepMind" in e.name for e in entities)


def test_llm_extractor_none_llm():
    extractor = LLMExtractor(llm_fn=None)
    chunk = DocumentChunk(text="PyTorch integrates with NumPy.", chunk_index=0, id="chk_none")
    entities, relationships = extractor.extract([chunk])
    assert len(entities) > 0
