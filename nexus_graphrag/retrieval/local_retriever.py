"""Local entity-centric and multi-hop ego-network knowledge retriever."""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Set, Tuple

from nexus_graphrag.core.models import (
    DocumentChunk,
    Entity,
    Relationship,
    RetrievalMode,
    RetrievalResult,
    slugify,
)
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.retrieval.search import BM25Index, tokenize


class LocalRetriever:
    """Executes high-precision, entity-centric local subgraph traversal."""

    def __init__(
        self,
        index: GraphIndex,
        chunks: Optional[List[DocumentChunk]] = None,
    ) -> None:
        self.index = index
        self.chunks = {c.id: c for c in (chunks or [])}
        self.entity_search = BM25Index()
        self._init_search_index()

    def _init_search_index(self) -> None:
        """Indexes all entity names and descriptions into BM25 for fuzzy seed discovery."""
        slugs = list(self.index.entities.keys())
        texts = [
            f"{e.name} {e.description} {e.entity_type.value}"
            for e in self.index.entities.values()
        ]
        self.entity_search.index_documents(slugs, texts)

    def find_seed_entities(self, query: str, top_k: int = 4) -> List[Entity]:
        """Identifies seed entities mentioned in or semantically closest to the query."""
        seeds: List[Entity] = []
        seen_slugs: Set[str] = set()
        q_lower = query.lower()

        # Step 1: Direct substring or alias matching
        for ent in self.index.entities.values():
            if re.search(rf"\b{re.escape(ent.name.lower())}\b", q_lower):
                if ent.id not in seen_slugs:
                    seen_slugs.add(ent.id)
                    seeds.append(ent)

        # Step 2: BM25 fuzzy match if direct matches are few
        if len(seeds) < top_k:
            fuzzy_results = self.entity_search.search(query, top_k=top_k * 2)
            for slug, score, _ in fuzzy_results:
                if slug not in seen_slugs and slug in self.index.entities:
                    seen_slugs.add(slug)
                    seeds.append(self.index.entities[slug])
                    if len(seeds) >= top_k:
                        break

        return seeds

    def retrieve(
        self,
        query: str,
        max_entities: int = 8,
        max_hops: int = 2,
    ) -> RetrievalResult:
        """Performs entity-anchored ego-network multi-hop retrieval for the query.

        Args:
            query: User's factual or relational inquiry.
            max_entities: Maximum number of focal entities to include in context.
            max_hops: Traversal depth from seed entities.

        Returns:
            RetrievalResult containing formatted markdown context and graph entities.
        """
        start_time = time.perf_counter()
        seeds = self.find_seed_entities(query, top_k=4)

        if not seeds:
            # Fallback: take top entities by PageRank
            pr = self.index.compute_pagerank()
            top_slugs = sorted(pr.keys(), key=lambda s: pr[s], reverse=True)[:max_entities]
            seeds = [self.index.entities[s] for s in top_slugs if s in self.index.entities]

        # Expand subgraph around seeds
        seed_names = [s.name for s in seeds]
        sub_entities, sub_rels = self.index.get_subgraph(seed_names, hops=max_hops)

        # Rank entities combining PageRank + seed proximity
        pr_scores = self.index.compute_pagerank()
        ranked_entities = sorted(
            sub_entities,
            key=lambda e: (e.id in {s.id for s in seeds}, pr_scores.get(e.id, 0.0), e.weight),
            reverse=True,
        )[:max_entities]

        # Filter relationships relevant to ranked entities
        ranked_entity_names = {e.name for e in ranked_entities}
        filtered_rels = [
            r for r in sub_rels
            if r.source in ranked_entity_names or r.target in ranked_entity_names
        ]
        # Sort relationships by weight and description richness
        filtered_rels = sorted(
            filtered_rels, key=lambda r: (r.weight, len(r.description)), reverse=True
        )[:12]

        # Gather evidence chunks
        evidence_ids: Set[str] = set()
        for e in ranked_entities:
            evidence_ids.update(e.source_chunks)
        for r in filtered_rels:
            evidence_ids.update(r.evidence_chunks)

        evidence_texts = [
            self.chunks[cid].text
            for cid in list(evidence_ids)[:3]
            if cid in self.chunks
        ]

        # Build structured markdown context
        context_lines: List[str] = [
            "### [Local Graph Context - Entity & Multi-Hop Traversal]",
            f"**Query:** {query}",
            "",
            "#### Focal Knowledge Entities:",
        ]
        for ent in ranked_entities:
            context_lines.append(
                f"- **{ent.name}** ({ent.entity_type.value}): {ent.description or 'No explicit description.'}"
            )

        if filtered_rels:
            context_lines.append("\n#### Relational Knowledge Paths:")
            for rel in filtered_rels:
                desc = rel.description or f"{rel.source} {rel.relation_type} {rel.target}"
                context_lines.append(f"- `{rel.source}` ──({rel.relation_type})──> `{rel.target}`: {desc}")

        if evidence_texts:
            context_lines.append("\n#### Supporting Source Evidence:")
            for idx, text in enumerate(evidence_texts, 1):
                clean_snippet = text.replace("\n", " ").strip()
                if len(clean_snippet) > 200:
                    clean_snippet = clean_snippet[:200] + "..."
                context_lines.append(f"{idx}. \"{clean_snippet}\"")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return RetrievalResult(
            query=query,
            mode=RetrievalMode.LOCAL,
            context="\n".join(context_lines),
            entities=ranked_entities,
            relationships=filtered_rels,
            confidence_score=0.92,
            metrics={
                "latency_ms": round(elapsed_ms, 2),
                "seeds_count": len(seeds),
                "entities_retrieved": len(ranked_entities),
                "relations_retrieved": len(filtered_rels),
            },
        )
