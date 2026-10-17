"""Graph topology analysis and structural metrics suite."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from nexus_graphrag.graph.index import GraphIndex


def compute_graph_metrics(index: GraphIndex) -> Dict[str, Any]:
    """Computes comprehensive structural and topological metrics of the knowledge graph.

    Args:
        index: GraphIndex instance to evaluate.

    Returns:
        Dictionary containing node counts, density, centrality distributions,
        and component analyses.
    """
    g = index.graph
    num_nodes = len(g)
    num_edges = g.number_of_edges()

    if num_nodes == 0:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "connected_components_count": 0,
            "largest_component_size": 0,
            "diameter": 0,
            "top_pagerank": [],
            "top_degree": [],
            "entity_type_distribution": {},
            "relation_type_distribution": {},
        }

    # Density & Degrees
    density = nx.density(g)
    degrees = [deg for _, deg in g.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

    # Connected Components
    components = list(nx.connected_components(g))
    comp_count = len(components)
    largest_comp = max(components, key=len) if components else set()
    largest_size = len(largest_comp)

    # Diameter of largest connected component
    diameter = 0
    if largest_size > 1:
        sub_g = g.subgraph(largest_comp)
        try:
            diameter = nx.diameter(sub_g)
        except Exception:
            diameter = 0

    # Centralities
    pr_scores = index.compute_pagerank()
    deg_scores = index.compute_degree_centrality()

    # Map slug back to friendly display name
    top_pr: List[Tuple[str, float]] = []
    sorted_pr = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for slug, score in sorted_pr:
        ent = index.entities.get(slug)
        name = ent.name if ent else slug
        top_pr.append((name, round(score, 4)))

    top_deg: List[Tuple[str, float]] = []
    sorted_deg = sorted(deg_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for slug, score in sorted_deg:
        ent = index.entities.get(slug)
        name = ent.name if ent else slug
        top_deg.append((name, round(score, 4)))

    # Distributions
    type_counts = Counter(e.entity_type.value for e in index.entities.values())
    rel_counts = Counter(r.relation_type for r in index.relationships.values())

    return {
        "total_nodes": num_nodes,
        "total_edges": num_edges,
        "density": round(density, 4),
        "avg_degree": round(avg_degree, 2),
        "connected_components_count": comp_count,
        "largest_component_size": largest_size,
        "diameter": diameter,
        "top_pagerank": top_pr,
        "top_degree": top_deg,
        "entity_type_distribution": dict(type_counts),
        "relation_type_distribution": dict(rel_counts),
    }
