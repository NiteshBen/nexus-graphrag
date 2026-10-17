"""Global community-level holistic knowledge retriever for macro queries."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from nexus_graphrag.core.models import (
    Community,
    Entity,
    RetrievalMode,
    RetrievalResult,
)
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.retrieval.search import BM25Index, VectorSearchIndex


class GlobalRetriever:
    """Retrieves macro-thematic summaries across graph communities for holistic inquiries."""

    def __init__(
        self,
        communities: List[Community],
        index: GraphIndex,
    ) -> None:
        self.communities = communities
        self.comm_map: Dict[str, Community] = {c.id: c for c in communities}
        self.index = index
        self.bm25_index = BM25Index()
        self.vector_index = VectorSearchIndex()
        self._init_search_indexes()

    def _init_search_indexes(self) -> None:
        """Indexes community profiles into BM25 and vector spaces."""
        ids: List[str] = []
        texts: List[str] = []

        for c in self.communities:
            doc_str = (
                f"{c.title} {c.summary} "
                f"{' '.join(c.findings)} "
                f"{' '.join(c.member_entities)}"
            )
            ids.append(c.id)
            texts.append(doc_str)

        self.bm25_index.index_documents(ids, texts)
        self.vector_index.index_documents(ids, texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> RetrievalResult:
        """Finds most relevant community clusters and synthesizes an executive thematic overview.

        Args:
            query: High-level thematic query (e.g., "Summarize major components").
            top_k: Number of community clusters to include.

        Returns:
            RetrievalResult containing community abstracts and key findings.
        """
        start_time = time.perf_counter()

        if not self.communities:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return RetrievalResult(
                query=query,
                mode=RetrievalMode.GLOBAL,
                context="### [Global Graph Context]\nNo knowledge communities detected in graph.",
                confidence_score=0.0,
                metrics={"latency_ms": round(elapsed_ms, 2), "communities_count": 0},
            )

        # Dual search over communities
        bm25_hits = self.bm25_index.search(query, top_k=top_k * 2)
        vec_hits = self.vector_index.search(query, top_k=top_k * 2)

        # Merge scores via Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        for rank, (cid, score, _) in enumerate(bm25_hits):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))
        for rank, (cid, score, _) in enumerate(vec_hits):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))

        # Sort selected communities
        if rrf_scores:
            ranked_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        else:
            # Fallback to largest communities by entity count
            ranked_cids = [
                c.id
                for c in sorted(self.communities, key=lambda c: len(c.member_entities), reverse=True)[:top_k]
            ]

        selected_communities = [self.comm_map[cid] for cid in ranked_cids if cid in self.comm_map]

        # Collect entities across these communities
        all_member_names: List[str] = []
        for c in selected_communities:
            all_member_names.extend(c.member_entities)

        entities: List[Entity] = []
        for name in list(dict.fromkeys(all_member_names))[:10]:
            ent = self.index.get_entity(name)
            if ent:
                entities.append(ent)

        # Build structured markdown context
        context_lines: List[str] = [
            "### [Global Graph Context - Community Thematic Summaries]",
            f"**Holistic Query:** {query}",
            "",
            f"Synthesizing across {len(selected_communities)} top modular domain clusters:\n",
        ]

        for idx, comm in enumerate(selected_communities, 1):
            context_lines.append(f"#### Cluster {idx}: {comm.title}")
            context_lines.append(f"**Overview:** {comm.summary}")
            context_lines.append(f"**Key Entities:** {', '.join(comm.member_entities[:6])}")
            if comm.findings:
                context_lines.append("**Key Domain Findings:**")
                for f in comm.findings[:3]:
                    context_lines.append(f"  • {f}")
            context_lines.append("")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return RetrievalResult(
            query=query,
            mode=RetrievalMode.GLOBAL,
            context="\n".join(context_lines),
            entities=entities,
            communities=selected_communities,
            confidence_score=0.88,
            metrics={
                "latency_ms": round(elapsed_ms, 2),
                "communities_retrieved": len(selected_communities),
            },
        )
