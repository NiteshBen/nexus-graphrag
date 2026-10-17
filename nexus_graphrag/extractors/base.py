"""Abstract base interface for knowledge extraction modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from nexus_graphrag.core.chunker import TextChunker
from nexus_graphrag.core.models import DocumentChunk, Entity, Relationship


class BaseExtractor(ABC):
    """Abstract interface defining entity and relationship extraction protocols."""

    @abstractmethod
    def extract(
        self, chunks: List[DocumentChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extracts entities and relationships from a collection of document chunks.

        Args:
            chunks: List of segmented document chunks.

        Returns:
            Tuple of (entities, relationships).
        """
        pass

    def extract_text(
        self, text: str, chunk_size: int = 600, chunk_overlap: int = 100
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Convenience method to chunk raw text and extract knowledge graph elements.

        Args:
            text: Raw input text.
            chunk_size: Window size in characters.
            chunk_overlap: Overlap in characters.

        Returns:
            Tuple of (entities, relationships).
        """
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.split_text(text)
        return self.extract(chunks)
