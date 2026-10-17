"""Fast, offline lexical (BM25) and vectorized search indexes using NumPy."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def tokenize(text: str) -> List[str]:
    """Extracts lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


class BM25Index:
    """Okapi BM25 search index implemented with vectorized NumPy computations."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_freqs: Dict[str, int] = Counter()
        self.tf_list: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def index_documents(self, doc_ids: List[str], doc_texts: List[str]) -> None:
        """Indexes an array of document texts."""
        self.doc_ids = list(doc_ids)
        self.doc_texts = list(doc_texts)
        self.corpus_size = len(doc_texts)

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        total_tokens = 0
        self.doc_lens = []
        self.tf_list = []
        self.doc_freqs = Counter()

        for text in doc_texts:
            tokens = tokenize(text)
            t_len = len(tokens)
            self.doc_lens.append(t_len)
            total_tokens += t_len
            tf = Counter(tokens)
            self.tf_list.append(tf)
            for token in tf:
                self.doc_freqs[token] += 1

        self.avg_doc_len = total_tokens / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate standard IDF
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            # BM25 IDF formulation with floor smoothing
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[token] = max(idf_val, 0.01)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Scores documents against query using BM25.

        Returns:
            List of (doc_id, score, doc_text) sorted in descending order of score.
        """
        if self.corpus_size == 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for token in q_tokens:
            if token not in self.idf:
                continue
            token_idf = self.idf[token]

            for i in range(self.corpus_size):
                tf = self.tf_list[i].get(token, 0)
                if tf == 0:
                    continue
                d_len = self.doc_lens[i]
                norm = 1.0 - self.b + self.b * (d_len / (self.avg_doc_len + 1e-6))
                term_score = token_idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * norm))
                scores[i] += term_score

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Tuple[str, float, str]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.doc_ids[idx], score, self.doc_texts[idx]))

        return results


class VectorSearchIndex:
    """Vectorized TF-IDF character & word n-gram semantic search index."""

    def __init__(self, n_features: int = 512) -> None:
        self.n_features = n_features
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.matrix: Optional[np.ndarray] = None

    def _embed(self, text: str) -> np.ndarray:
        """Generates a normalized hash-projected frequency vector for text."""
        vec = np.zeros(self.n_features, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec

        for t in tokens:
            h = hash(t) % self.n_features
            vec[h] += 1.0

        # Also add bigram features for order-awareness
        for i in range(len(tokens) - 1):
            h2 = hash(f"{tokens[i]}_{tokens[i+1]}") % self.n_features
            vec[h2] += 1.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def index_documents(self, doc_ids: List[str], doc_texts: List[str]) -> None:
        """Builds vector matrix across document corpus."""
        self.doc_ids = list(doc_ids)
        self.doc_texts = list(doc_texts)
        if not doc_texts:
            self.matrix = None
            return

        rows = [self._embed(t) for t in doc_texts]
        self.matrix = np.vstack(rows)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Computes cosine similarity against indexed vector representations."""
        if self.matrix is None or len(self.doc_ids) == 0:
            return []

        q_vec = self._embed(query)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        # Cosine similarity matrix multiplication: matrix @ q_vec
        sims = self.matrix @ q_vec
        top_indices = np.argsort(sims)[::-1][:top_k]

        results: List[Tuple[str, float, str]] = []
        for idx in top_indices:
            score = float(sims[idx])
            if score > 0.05:
                results.append((self.doc_ids[idx], score, self.doc_texts[idx]))

        return results
