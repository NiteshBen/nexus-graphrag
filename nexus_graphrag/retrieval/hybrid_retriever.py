"""Hybrid dual-branch RAG retriever fusing Local, Global, and Lexical channels."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set

from nexus_graphrag.core.models import (
    Community,
    DocumentChunk,
    Entity,
    Relationship,
    RetrievalMode,
    RetrievalResult,
)
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.retrieval.global_retriever import GlobalRetriever
from nexus_graphrag.retrieval.local_retriever import LocalRetriever
from nexus_graphrag.retrieval.search import BM25Index, VectorSearchIndex


class HybridRetriever:
    """Fuses local subgraph paths, global community summaries, and direct chunk search via RRF."""

    def __init__(
        self,
        index: GraphIndex,
        communities: List[Community],
        chunks: Optional[List[DocumentChunk]] = None,
    ) -> None:
        self.index = index
        self.communities = communities
        self.chunks_list = chunks or []
        self.chunks_map = {c.id: c for c in self.chunks_list}

        self.local_retriever = LocalRetriever(index, chunks=self.chunks_list)
        self.global_retriever = GlobalRetriever(communities, index)

        # Raw chunk search indexes
        self.chunk_bm25 = BM25Index()
        self.chunk_vector = VectorSearchIndex()
        self._init_chunk_indexes()

    def _init_chunk_indexes(self) -> None:
        """Indexes raw text chunks for direct citation retrieval."""
        if not self.chunks_list:
            return
        cids = [c.id for c in self.chunks_list]
        texts = [c.text for c in self.chunks_list]
        self.chunk_bm25.index_documents(cids, texts)
        self.chunk_vector.index_documents(cids, texts)

    def retrieve(
        self,
        query: str,
        max_entities: int = 6,
        max_communities: int = 2,
        max_chunks: int = 3,
    ) -> RetrievalResult:
        """Executes full hybrid search blending local paths, community abstracts, and verbatim chunks.

        Args:
            query: User's inquiry.
            max_entities: Max focal entities.
            max_communities: Max global clusters.
            max_chunks: Max source citations.

        Returns:
            Unified RetrievalResult ready for LLM consumption.
        """
        start_time = time.perf_counter()

        # Branch 1: Local Subgraph Retrieval
        local_res = self.local_retriever.retrieve(query, max_entities=max_entities, max_hops=2)

        # Branch 2: Global Community Retrieval
        global_res = self.global_retriever.retrieve(query, top_k=max_communities)

        # Branch 3: Direct Chunk Search
        chunk_hits_bm25 = self.chunk_bm25.search(query, top_k=max_chunks * 2)
        chunk_hits_vec = self.chunk_vector.search(query, top_k=max_chunks * 2)

        # Reciprocal Rank Fusion on chunks
        chunk_rrf: Dict[str, float] = {}
        for rank, (cid, score, _) in enumerate(chunk_hits_bm25):
            chunk_rrf[cid] = chunk_rrf.get(cid, 0.0) + (1.0 / (60.0 + rank))
        for rank, (cid, score, _) in enumerate(chunk_hits_vec):
            chunk_rrf[cid] = chunk_rrf.get(cid, 0.0) + (1.0 / (60.0 + rank))

        top_cids = sorted(chunk_rrf.keys(), key=lambda cid: chunk_rrf[cid], reverse=True)[:max_chunks]
        selected_chunks = [self.chunks_map[cid] for cid in top_cids if cid in self.chunks_map]

        # Merge entities and deduplicate
        merged_entities_map: Dict[str, Entity] = {}
        for e in local_res.entities + global_res.entities:
            merged_entities_map[e.id] = e
        merged_entities = list(merged_entities_map.values())[:max_entities + 2]

        merged_rels = local_res.relationships[:10]
        selected_comms = global_res.communities[:max_communities]

        # Synthesize Unified Hybrid Context
        context_parts: List[str] = [
            "### [Nexus-GraphRAG Hybrid Unified Context]",
            f"**Query:** {query}",
            "",
            "#### I. Macro Thematic Clusters (Global Synthesis):",
        ]

        if selected_comms:
            for idx, comm in enumerate(selected_comms, 1):
                context_parts.append(f"**Cluster {idx} ({comm.title}):** {comm.summary}")
                if comm.findings:
                    context_parts.append(f"  * Key finding: {comm.findings[0]}")
        else:
            context_parts.append("*(No global community clusters detected)*")

        context_parts.append("\n#### II. Micro-Level Relational Triplet Paths (Local Traversal):")
        if merged_rels:
            for rel in merged_rels[:6]:
                desc = rel.description or f"{rel.source} {rel.relation_type} {rel.target}"
                context_parts.append(f"- `{rel.source}` ──[{rel.relation_type}]──> `{rel.target}`: {desc}")
        elif merged_entities:
            for ent in merged_entities[:4]:
                context_parts.append(f"- **{ent.name}** ({ent.entity_type.value}): {ent.description}")
        else:
            context_parts.append("*(No multi-hop relational paths detected)*")

        context_parts.append("\n#### III. Verbatim Grounding Evidence (Chunk Retrieval):")
        if selected_chunks:
            for idx, chk in enumerate(selected_chunks, 1):
                clean_txt = chk.text.replace("\n", " ").strip()
                if len(clean_txt) > 220:
                    clean_txt = clean_txt[:220] + "..."
                context_parts.append(f"[{idx}] (Chunk {chk.chunk_index}): \"{clean_txt}\"")
        else:
            context_parts.append("*(No direct chunk excerpts matched)*")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return RetrievalResult(
            query=query,
            mode=RetrievalMode.HYBRID,
            context="\n".join(context_parts),
            entities=merged_entities,
            relationships=merged_rels,
            communities=selected_comms,
            confidence_score=0.96,
            metrics={
                "latency_ms": round(elapsed_ms, 2),
                "entities_count": len(merged_entities),
                "relations_count": len(merged_rels),
                "communities_count": len(selected_comms),
                "chunks_count": len(selected_chunks),
            },
        )
