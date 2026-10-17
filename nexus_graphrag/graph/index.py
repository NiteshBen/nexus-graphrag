"""In-memory Knowledge Graph Index built on NetworkX with fast relational indexing."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from nexus_graphrag.core.models import (
    Entity,
    EntityType,
    Relationship,
    slugify,
)


class GraphIndex:
    """High-performance knowledge graph index with topology and neighborhood traversal."""

    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.entities: Dict[str, Entity] = {}  # slug -> Entity
        self.relationships: Dict[str, Relationship] = {}  # id -> Relationship
        self._name_to_slug: Dict[str, str] = {}
        self._pagerank_cache: Optional[Dict[str, float]] = None

    def add_entity(self, entity: Entity) -> None:
        """Inserts or merges an entity into the graph index."""
        slug = entity.id or slugify(entity.name)
        if slug in self.entities:
            existing = self.entities[slug]
            existing.weight = max(existing.weight, entity.weight) + 0.2
            for c in entity.source_chunks:
                if c not in existing.source_chunks:
                    existing.source_chunks.append(c)
            if entity.description and not existing.description:
                existing.description = entity.description
            entity = existing
        else:
            self.entities[slug] = entity

        self._name_to_slug[entity.name.lower()] = slug
        self._name_to_slug[slug] = slug

        # Update NetworkX node
        self.graph.add_node(
            slug,
            name=entity.name,
            entity_type=entity.entity_type.value,
            description=entity.description,
            weight=entity.weight,
        )
        self._pagerank_cache = None

    def add_relationship(self, relationship: Relationship) -> None:
        """Inserts or merges an edge into the graph index."""
        src_slug = slugify(relationship.source)
        tgt_slug = slugify(relationship.target)

        # Ensure source and target nodes exist in graph
        if src_slug not in self.entities:
            self.add_entity(Entity(name=relationship.source, id=src_slug))
        if tgt_slug not in self.entities:
            self.add_entity(Entity(name=relationship.target, id=tgt_slug))

        rel_id = relationship.id or f"{src_slug}--{slugify(relationship.relation_type)}-->{tgt_slug}"
        if rel_id in self.relationships:
            existing = self.relationships[rel_id]
            existing.weight += 0.5
            for c in relationship.evidence_chunks:
                if c not in existing.evidence_chunks:
                    existing.evidence_chunks.append(c)
        else:
            self.relationships[rel_id] = relationship

        # Update NetworkX edge (undirected graph representation for bidirectional discovery)
        current_weight = 1.0
        if self.graph.has_edge(src_slug, tgt_slug):
            current_weight = self.graph[src_slug][tgt_slug].get("weight", 1.0) + relationship.weight
        else:
            current_weight = relationship.weight

        self.graph.add_edge(
            src_slug,
            tgt_slug,
            weight=current_weight,
            relation_type=relationship.relation_type,
            description=relationship.description,
            rel_id=rel_id,
        )
        self._pagerank_cache = None

    def get_entity(self, name_or_slug: str) -> Optional[Entity]:
        """Resolves an entity by name, case-insensitive match, or slug."""
        lookup = name_or_slug.strip().lower()
        slug = self._name_to_slug.get(lookup, slugify(lookup))
        return self.entities.get(slug)

    def get_neighbors(
        self, entity_name: str, max_depth: int = 1
    ) -> List[Tuple[Entity, Relationship]]:
        """Finds neighboring entities and incident relationships up to max_depth."""
        ent = self.get_entity(entity_name)
        if not ent or ent.id not in self.graph:
            return []

        results: List[Tuple[Entity, Relationship]] = []
        visited: Set[str] = {ent.id}
        queue: List[Tuple[str, int]] = [(ent.id, 0)]

        while queue:
            curr_slug, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for neighbor_slug in self.graph.neighbors(curr_slug):
                edge_data = self.graph.get_edge_data(curr_slug, neighbor_slug)
                rel_id = edge_data.get("rel_id")
                rel = self.relationships.get(rel_id)
                if not rel:
                    rel = Relationship(
                        source=self.entities[curr_slug].name,
                        target=self.entities[neighbor_slug].name,
                        relation_type=edge_data.get("relation_type", "RELATED_TO"),
                        description=edge_data.get("description", ""),
                        weight=edge_data.get("weight", 1.0),
                    )

                neighbor_ent = self.entities.get(neighbor_slug)
                if neighbor_ent:
                    results.append((neighbor_ent, rel))

                if neighbor_slug not in visited:
                    visited.add(neighbor_slug)
                    queue.append((neighbor_slug, depth + 1))

        return results

    def get_subgraph(
        self, entity_names: List[str], hops: int = 1
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extracts an induced subgraph around a collection of seed entities."""
        seed_slugs: Set[str] = set()
        for name in entity_names:
            ent = self.get_entity(name)
            if ent and ent.id in self.graph:
                seed_slugs.add(ent.id)

        if not seed_slugs:
            return [], []

        expanded_nodes: Set[str] = set(seed_slugs)
        for _ in range(hops):
            current_nodes = list(expanded_nodes)
            for node in current_nodes:
                for neighbor in self.graph.neighbors(node):
                    expanded_nodes.add(neighbor)

        sub_entities: List[Entity] = [
            self.entities[s] for s in expanded_nodes if s in self.entities
        ]

        sub_rels: List[Relationship] = []
        seen_rels: Set[str] = set()
        for u, v, data in self.graph.subgraph(expanded_nodes).edges(data=True):
            rel_id = data.get("rel_id")
            if rel_id and rel_id in self.relationships and rel_id not in seen_rels:
                seen_rels.add(rel_id)
                sub_rels.append(self.relationships[rel_id])
            elif not rel_id:
                u_ent = self.entities.get(u)
                v_ent = self.entities.get(v)
                if u_ent and v_ent:
                    synthetic_rel = Relationship(
                        source=u_ent.name,
                        target=v_ent.name,
                        relation_type=data.get("relation_type", "RELATED_TO"),
                        description=data.get("description", ""),
                        weight=data.get("weight", 1.0),
                    )
                    sub_rels.append(synthetic_rel)

        return sub_entities, sub_rels

    def compute_pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """Calculates PageRank score for each entity in the graph."""
        if self._pagerank_cache is not None:
            return self._pagerank_cache

        if len(self.graph) == 0:
            return {}

        try:
            scores = nx.pagerank(self.graph, alpha=alpha, weight="weight")
        except Exception:
            # Fallback uniform scores for edge conditions
            n = len(self.graph)
            scores = {node: 1.0 / n for node in self.graph.nodes()}

        self._pagerank_cache = scores
        return scores

    def compute_degree_centrality(self) -> Dict[str, float]:
        """Calculates degree centrality for each node."""
        if len(self.graph) == 0:
            return {}
        return nx.degree_centrality(self.graph)

    def compute_betweenness_centrality(self) -> Dict[str, float]:
        """Calculates betweenness centrality for each node."""
        if len(self.graph) == 0:
            return {}
        return nx.betweenness_centrality(self.graph, weight="weight")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes complete graph index to dictionary."""
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relationships": [r.to_dict() for r in self.relationships.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphIndex":
        """Reconstructs GraphIndex from serialized dictionary."""
        idx = cls()
        for e_dict in data.get("entities", []):
            idx.add_entity(Entity.from_dict(e_dict))
        for r_dict in data.get("relationships", []):
            idx.add_relationship(Relationship.from_dict(r_dict))
        return idx

    def save_json(self, filepath: str) -> None:
        """Saves graph index to disk as formatted JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "GraphIndex":
        """Loads graph index from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
