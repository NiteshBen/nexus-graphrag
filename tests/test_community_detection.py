"""Tests for CommunityDetector."""

from nexus_graphrag.core.models import Entity, EntityType, Relationship
from nexus_graphrag.graph.communities import CommunityDetector
from nexus_graphrag.graph.index import GraphIndex


def test_community_detection_empty():
    idx = GraphIndex()
    detector = CommunityDetector()
    comms = detector.detect_communities(idx)
    assert comms == []


def test_community_detection_populated(populated_index: GraphIndex):
    detector = CommunityDetector()
    comms = detector.detect_communities(populated_index)

    assert len(comms) >= 1
    for c in comms:
        assert c.id.startswith("comm_")
        assert len(c.member_entities) > 0
        assert len(c.title) > 0
        assert len(c.summary) > 0


def test_community_synthesis_profiles():
    idx = GraphIndex()
    # Create two disjoint clusters
    idx.add_relationship(Relationship(source="Python", target="FastAPI", relation_type="USES"))
    idx.add_relationship(Relationship(source="FastAPI", target="Uvicorn", relation_type="DEPENDS_ON"))

    idx.add_relationship(Relationship(source="Cardiology", target="Heart", relation_type="STUDIES"))
    idx.add_relationship(Relationship(source="Heart", target="ECG", relation_type="MEASURES"))

    detector = CommunityDetector()
    comms = detector.detect_communities(idx)

    assert len(comms) >= 2
    # Verify titles contain primary concepts
    all_titles = " ".join(c.title for c in comms)
    assert any(term in all_titles for term in ["Python", "FastAPI", "Heart", "Cardiology"])
