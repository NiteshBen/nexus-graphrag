"""Nexus-GraphRAG: Production Local Knowledge Graph & Hybrid RAG Engine in Python."""

from nexus_graphrag.core.chunker import TextChunker
from nexus_graphrag.core.models import (
    Community,
    DocumentChunk,
    Entity,
    EntityType,
    Relationship,
    RetrievalMode,
    RetrievalResult,
)
from nexus_graphrag.engine import NexusGraphRAG
from nexus_graphrag.extractors.base import BaseExtractor
from nexus_graphrag.extractors.deterministic import DeterministicExtractor
from nexus_graphrag.extractors.llm_extractor import LLMExtractor
from nexus_graphrag.graph.communities import CommunityDetector
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.graph.metrics import compute_graph_metrics
from nexus_graphrag.retrieval.global_retriever import GlobalRetriever
from nexus_graphrag.retrieval.hybrid_retriever import HybridRetriever
from nexus_graphrag.retrieval.local_retriever import LocalRetriever
from nexus_graphrag.visualizer.html_report import generate_html_report

__version__ = "0.1.0"
__author__ = "Nitesh Beniwal"

__all__ = [
    "NexusGraphRAG",
    "Entity",
    "EntityType",
    "Relationship",
    "DocumentChunk",
    "Community",
    "RetrievalMode",
    "RetrievalResult",
    "TextChunker",
    "BaseExtractor",
    "DeterministicExtractor",
    "LLMExtractor",
    "GraphIndex",
    "CommunityDetector",
    "compute_graph_metrics",
    "LocalRetriever",
    "GlobalRetriever",
    "HybridRetriever",
    "generate_html_report",
]
