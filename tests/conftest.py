"""Pytest shared fixtures and mock datasets."""

import pytest

from nexus_graphrag.core.models import DocumentChunk, Entity, EntityType, Relationship
from nexus_graphrag.graph.index import GraphIndex

SAMPLE_DOC = """
Google DeepMind developed AlphaFold in London to predict 3D protein structures.
AlphaFold relies on PyTorch and Deep Learning algorithms to achieve atomic accuracy.
In 2024, Demis Hassabis was awarded the Nobel Prize in Chemistry for protein structure prediction.
PyTorch is an open-source machine learning framework maintained by the PyTorch Foundation.
Researchers use AlphaFold to design novel therapeutics and analyze molecular interactions.
"""


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_DOC


@pytest.fixture
def sample_entities() -> list[Entity]:
    return [
        Entity(name="Google DeepMind", entity_type=EntityType.ORGANIZATION),
        Entity(name="AlphaFold", entity_type=EntityType.TECHNOLOGY),
        Entity(name="PyTorch", entity_type=EntityType.TECHNOLOGY),
        Entity(name="Demis Hassabis", entity_type=EntityType.PERSON),
    ]


@pytest.fixture
def populated_index(sample_entities: list[Entity]) -> GraphIndex:
    idx = GraphIndex()
    for ent in sample_entities:
        idx.add_entity(ent)

    idx.add_relationship(
        Relationship(
            source="Google DeepMind",
            target="AlphaFold",
            relation_type="DEVELOPS",
            description="Google DeepMind developed AlphaFold.",
        )
    )
    idx.add_relationship(
        Relationship(
            source="AlphaFold",
            target="PyTorch",
            relation_type="USES",
            description="AlphaFold relies on PyTorch.",
        )
    )
    idx.add_relationship(
        Relationship(
            source="Demis Hassabis",
            target="Google DeepMind",
            relation_type="DEVELOPS",
            description="Demis Hassabis leads Google DeepMind.",
        )
    )
    return idx
