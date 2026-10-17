"""Primary engine orchestrator for Nexus-GraphRAG."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

from nexus_graphrag.core.chunker import TextChunker
from nexus_graphrag.core.models import (
    Community,
    DocumentChunk,
    Entity,
    Relationship,
    RetrievalMode,
    RetrievalResult,
)
from nexus_graphrag.extractors.base import BaseExtractor
from nexus_graphrag.extractors.deterministic import DeterministicExtractor
from nexus_graphrag.graph.communities import CommunityDetector
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.graph.metrics import compute_graph_metrics
from nexus_graphrag.retrieval.global_retriever import GlobalRetriever
from nexus_graphrag.retrieval.hybrid_retriever import HybridRetriever
from nexus_graphrag.retrieval.local_retriever import LocalRetriever
from nexus_graphrag.visualizer.html_report import generate_html_report

logger = logging.getLogger(__name__)


class NexusGraphRAG:
    """Unified Graph Retrieval-Augmented Generation engine.

    Orchestrates boundary-aware text chunking, entity-relationship extraction,
    in-memory graph indexing, Louvain community detection, and tri-modal
    (Local, Global, Hybrid) retrieval.
    """

    def __init__(
        self,
        extractor: Optional[BaseExtractor] = None,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> None:
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.extractor = extractor or DeterministicExtractor()
        self.index = GraphIndex()
        self.community_detector = CommunityDetector()

        self.chunks: List[DocumentChunk] = []
        self.communities: List[Community] = []

        # Retrievers
        self._local_retriever: Optional[LocalRetriever] = None
        self._global_retriever: Optional[GlobalRetriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None

    def index_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingests raw text corpus, extracts knowledge graph elements, and partitions communities.

        Args:
            text: Unstructured input document string.
            doc_id: Optional identifier for source document.
            metadata: Optional dictionary of document metadata.

        Returns:
            Dictionary containing indexing summary statistics.
        """
        new_chunks = self.chunker.split_text(text, doc_id=doc_id, base_metadata=metadata)
        self.chunks.extend(new_chunks)

        # Extract entities and relationships
        entities, relationships = self.extractor.extract(new_chunks)

        for ent in entities:
            self.index.add_entity(ent)

        for rel in relationships:
            self.index.add_relationship(rel)

        # Update communities
        self.communities = self.community_detector.detect_communities(self.index)

        # Re-initialize retrievers with updated index
        self._init_retrievers()

        metrics = compute_graph_metrics(self.index)
        return {
            "status": "success",
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks),
            "entities_extracted": len(entities),
            "total_entities": metrics["total_nodes"],
            "relationships_extracted": len(relationships),
            "total_relationships": metrics["total_edges"],
            "communities_count": len(self.communities),
        }

    def index_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Ingests a collection of documents."""
        total_chunks = 0
        total_ents = 0
        total_rels = 0
        for idx, doc in enumerate(documents):
            res = self.index_text(doc, doc_id=f"doc_{idx}")
            total_chunks += res["chunks_added"]
            total_ents += res["entities_extracted"]
            total_rels += res["relationships_extracted"]

        return {
            "status": "success",
            "documents_indexed": len(documents),
            "chunks_added": total_chunks,
            "total_entities": len(self.index.entities),
            "total_relationships": len(self.index.relationships),
            "communities_count": len(self.communities),
        }

    def _init_retrievers(self) -> None:
        """Refreshes retriever instances with current graph state."""
        self._local_retriever = LocalRetriever(self.index, chunks=self.chunks)
        self._global_retriever = GlobalRetriever(self.communities, self.index)
        self._hybrid_retriever = HybridRetriever(
            self.index, self.communities, chunks=self.chunks
        )

    def query(
        self,
        query_text: str,
        mode: Union[str, RetrievalMode] = RetrievalMode.HYBRID,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Executes knowledge retrieval against the indexed knowledge graph.

        Args:
            query_text: User question or research prompt.
            mode: Retrieval mode ('local', 'global', or 'hybrid').
            **kwargs: Additional parameters forwarded to the underlying retriever.

        Returns:
            RetrievalResult containing synthesized markdown context and graph citations.
        """
        if not self._hybrid_retriever:
            self._init_retrievers()

        if isinstance(mode, str):
            mode = RetrievalMode.from_string(mode)

        if mode == RetrievalMode.LOCAL:
            assert self._local_retriever is not None
            return self._local_retriever.retrieve(query_text, **kwargs)
        elif mode == RetrievalMode.GLOBAL:
            assert self._global_retriever is not None
            return self._global_retriever.retrieve(query_text, **kwargs)
        else:
            assert self._hybrid_retriever is not None
            return self._hybrid_retriever.retrieve(query_text, **kwargs)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns topological and graph structural metrics."""
        return compute_graph_metrics(self.index)

    def export_html(
        self,
        filepath: str,
        title: str = "Nexus-GraphRAG Knowledge Explorer",
    ) -> str:
        """Renders and saves a standalone interactive HTML graph report to disk."""
        html_content = generate_html_report(self.index, self.communities, title=title)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return filepath

    def save(self, filepath: str) -> None:
        """Serializes entire Nexus-GraphRAG state (graph, chunks, communities) to JSON."""
        state = {
            "graph": self.index.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "communities": [c.to_dict() for c in self.communities],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(
        cls,
        filepath: str,
        extractor: Optional[BaseExtractor] = None,
    ) -> "NexusGraphRAG":
        """Reconstructs NexusGraphRAG instance from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        rag = cls(extractor=extractor)
        rag.index = GraphIndex.from_dict(data.get("graph", {}))
        rag.chunks = [DocumentChunk.from_dict(c) for c in data.get("chunks", [])]
        rag.communities = [Community.from_dict(c) for c in data.get("communities", [])]
        rag._init_retrievers()
        return rag
