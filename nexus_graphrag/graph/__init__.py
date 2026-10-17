"""Knowledge graph indexing, topological metrics, and community detection."""

from nexus_graphrag.graph.communities import CommunityDetector
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.graph.metrics import compute_graph_metrics

__all__ = [
    "GraphIndex",
    "compute_graph_metrics",
    "CommunityDetector",
]
