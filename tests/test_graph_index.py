"""Tests for GraphIndex and topology algorithms."""

import os
import tempfile

from nexus_graphrag.core.models import Entity, EntityType, Relationship
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.graph.metrics import compute_graph_metrics


def test_graph_index_add_and_lookup(populated_index: GraphIndex):
    ent = populated_index.get_entity("Google DeepMind")
    assert ent is not None
    assert ent.name == "Google DeepMind"

    # Case-insensitive lookup
    ent_lower = populated_index.get_entity("google deepmind")
    assert ent_lower is not None
    assert ent_lower.id == ent.id


def test_graph_index_neighbors(populated_index: GraphIndex):
    neighbors = populated_index.get_neighbors("Google DeepMind", max_depth=1)
    assert len(neighbors) >= 2
    names = [e.name for e, r in neighbors]
    assert "AlphaFold" in names
    assert "Demis Hassabis" in names


def test_graph_index_subgraph(populated_index: GraphIndex):
    sub_ents, sub_rels = populated_index.get_subgraph(["AlphaFold"], hops=1)
    ent_names = [e.name for e in sub_ents]
    assert "AlphaFold" in ent_names
    assert "PyTorch" in ent_names or "Google DeepMind" in ent_names
    assert len(sub_rels) >= 1


def test_graph_centrality_computations(populated_index: GraphIndex):
    pr = populated_index.compute_pagerank()
    assert len(pr) == len(populated_index.entities)
    assert sum(pr.values()) > 0.9  # Normalized sum close to 1.0

    deg = populated_index.compute_degree_centrality()
    assert len(deg) == len(populated_index.entities)

    bet = populated_index.compute_betweenness_centrality()
    assert len(bet) == len(populated_index.entities)


def test_graph_metrics(populated_index: GraphIndex):
    metrics = compute_graph_metrics(populated_index)
    assert metrics["total_nodes"] == len(populated_index.entities)
    assert metrics["total_edges"] >= 3
    assert metrics["density"] > 0
    assert metrics["connected_components_count"] >= 1
    assert len(metrics["top_pagerank"]) > 0


def test_graph_serialization_roundtrip(populated_index: GraphIndex):
    data = populated_index.to_dict()
    reloaded = GraphIndex.from_dict(data)

    assert len(reloaded.entities) == len(populated_index.entities)
    assert len(reloaded.relationships) == len(populated_index.relationships)
    assert reloaded.get_entity("AlphaFold") is not None

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_file = os.path.join(tmp_dir, "graph.json")
        populated_index.save_json(json_file)
        from_disk = GraphIndex.load_json(json_file)
        assert len(from_disk.entities) == len(populated_index.entities)
