"""Community detection and hierarchical knowledge summarization engine."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import networkx as nx

from nexus_graphrag.core.models import Community, Entity, Relationship
from nexus_graphrag.graph.index import GraphIndex

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Partitions the knowledge graph into dense modular semantic communities."""

    def __init__(self, resolution: float = 1.0, seed: int = 42) -> None:
        self.resolution = resolution
        self.seed = seed

    def detect_communities(
        self, index: GraphIndex, min_community_size: int = 1
    ) -> List[Community]:
        """Detects modular communities using Louvain modularity or fallback algorithms.

        Args:
            index: GraphIndex to partition.
            min_community_size: Minimum number of entities required in a community.

        Returns:
            List of Community instances with computed titles and member lists.
        """
        g = index.graph
        if len(g) == 0:
            return []

        # Calculate PageRank for ranking member prominence
        pr_scores = index.compute_pagerank()

        # Step 1: Detect clusters
        clusters: List[Set[str]] = []
        try:
            # Try Louvain modularity optimization
            clusters = list(
                nx.community.louvain_communities(
                    g,
                    weight="weight",
                    resolution=self.resolution,
                    seed=self.seed,
                )
            )
        except Exception as e:
            logger.debug(f"Louvain partitioning failed: {e}. Falling back to connected components.")
            clusters = [set(c) for c in nx.connected_components(g)]

        communities: List[Community] = []
        for c_idx, cluster_slugs in enumerate(clusters):
            if len(cluster_slugs) < min_community_size:
                continue

            # Sort member entities by internal prominence
            sorted_slugs = sorted(
                cluster_slugs, key=lambda s: pr_scores.get(s, 0.0), reverse=True
            )
            member_names = [
                index.entities[s].name for s in sorted_slugs if s in index.entities
            ]

            if not member_names:
                continue

            # Synthesize title and summary from member entities and intra-community edges
            title, summary, findings = self._synthesize_community_profile(
                c_idx, member_names, cluster_slugs, index
            )

            comm = Community(
                id=f"comm_{c_idx}",
                level=0,
                title=title,
                summary=summary,
                member_entities=member_names,
                findings=findings,
                weight=float(len(member_names)),
            )
            communities.append(comm)

        return communities

    def _synthesize_community_profile(
        self,
        comm_idx: int,
        member_names: List[str],
        cluster_slugs: Set[str],
        index: GraphIndex,
    ) -> tuple[str, str, List[str]]:
        """Synthesizes human-readable title, summary, and findings for the community."""
        primary_entities = member_names[:3]

        # Intra-community relationships
        intra_rels: List[Relationship] = []
        for u in cluster_slugs:
            for v in cluster_slugs:
                if u >= v:
                    continue
                edge_data = index.graph.get_edge_data(u, v)
                if edge_data:
                    rel_id = edge_data.get("rel_id")
                    if rel_id and rel_id in index.relationships:
                        intra_rels.append(index.relationships[rel_id])

        # Derive title
        if len(primary_entities) == 1:
            title = f"{primary_entities[0]} Ecosystem"
        elif len(primary_entities) == 2:
            title = f"{primary_entities[0]} & {primary_entities[1]} Domain"
        else:
            title = f"{primary_entities[0]}, {primary_entities[1]} & Related Systems"

        # Derive summary
        summary_lines = [
            f"Knowledge cluster containing {len(member_names)} entities centered on {', '.join(primary_entities)}."
        ]
        if intra_rels:
            rel_descriptions = [r.description for r in intra_rels[:3] if r.description]
            if rel_descriptions:
                summary_lines.append(" ".join(rel_descriptions))
        summary = " ".join(summary_lines)

        # Derive findings
        findings: List[str] = []
        for r in intra_rels[:4]:
            if r.description:
                findings.append(r.description)
            else:
                findings.append(f"{r.source} {r.relation_type.lower().replace('_', ' ')} {r.target}.")

        if not findings and member_names:
            findings.append(f"Primary focal concepts include {', '.join(primary_entities)}.")

        return title, summary, findings
