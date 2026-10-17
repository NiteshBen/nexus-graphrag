"""Robust text chunking and document segmentation engine."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from nexus_graphrag.core.models import DocumentChunk


class TextChunker:
    """Segments unstructured textual corpora into overlapping, boundary-aware chunks."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
        respect_sentences: bool = True,
    ) -> None:
        """Initializes chunker with size and overlap constraints.

        Args:
            chunk_size: Target character count per chunk.
            chunk_overlap: Number of overlapping characters between adjacent chunks.
            min_chunk_size: Minimum threshold for a chunk to be kept.
            respect_sentences: Whether to avoid breaking sentences mid-phrase.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be non-negative and < chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentences = respect_sentences

    def split_text(
        self,
        text: str,
        doc_id: Optional[str] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Splits raw text into a sequence of DocumentChunk instances.

        Args:
            text: Unstructured raw document text.
            doc_id: Optional identifier for source document.
            base_metadata: Optional dict of metadata attributes to attach.

        Returns:
            List of segmented DocumentChunk objects.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        base_meta = dict(base_metadata or {})
        if doc_id:
            base_meta["doc_id"] = doc_id

        if len(cleaned_text) <= self.chunk_size:
            return [
                DocumentChunk(
                    text=cleaned_text,
                    chunk_index=0,
                    metadata=base_meta,
                    id=f"{doc_id or 'doc'}_chunk_0",
                )
            ]

        if self.respect_sentences:
            return self._split_by_sentences(cleaned_text, doc_id, base_meta)
        return self._split_fixed(cleaned_text, doc_id, base_meta)

    def _split_fixed(
        self,
        text: str,
        doc_id: Optional[str],
        base_meta: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Fixed character window splitting."""
        chunks: List[DocumentChunk] = []
        start = 0
        idx = 0
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_str = text[start:end].strip()
            # If at the very end of text, keep remaining characters
            is_final = (end >= text_len)
            if chunk_str and (len(chunk_str) >= self.min_chunk_size or is_final or not chunks):
                meta = dict(base_meta)
                meta["start_char"] = start
                meta["end_char"] = end
                chunks.append(
                    DocumentChunk(
                        text=chunk_str,
                        chunk_index=idx,
                        metadata=meta,
                        id=f"{doc_id or 'doc'}_chunk_{idx}",
                    )
                )
                idx += 1
            if is_final:
                break
            start += step

        return chunks

    def _split_by_sentences(
        self,
        text: str,
        doc_id: Optional[str],
        base_meta: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Splits text preserving sentence boundaries where feasible."""
        # Split into sentences or paragraph blocks
        raw_sentences = re.split(r"(?<=[.!?\n])\s+", text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return self._split_fixed(text, doc_id, base_meta)

        chunks: List[DocumentChunk] = []
        current_sentences: List[str] = []
        current_len = 0
        idx = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.chunk_size and current_sentences:
                chunk_str = " ".join(current_sentences).strip()
                if len(chunk_str) >= self.min_chunk_size:
                    chunks.append(
                        DocumentChunk(
                            text=chunk_str,
                            chunk_index=idx,
                            metadata=dict(base_meta),
                            id=f"{doc_id or 'doc'}_chunk_{idx}",
                        )
                    )
                    idx += 1

                # Calculate overlap sentences
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                current_sentences = overlap_sentences + [sentence]
                current_len = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
            else:
                current_sentences.append(sentence)
                current_len += sentence_len + 1

        if current_sentences:
            final_str = " ".join(current_sentences).strip()
            if len(final_str) >= self.min_chunk_size or not chunks:
                chunks.append(
                    DocumentChunk(
                        text=final_str,
                        chunk_index=idx,
                        metadata=dict(base_meta),
                        id=f"{doc_id or 'doc'}_chunk_{idx}",
                    )
                )

        return chunks
