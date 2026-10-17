"""High-performance, 100% offline deterministic entity and relationship extractor."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from nexus_graphrag.core.models import (
    DocumentChunk,
    Entity,
    EntityType,
    Relationship,
    slugify,
)
from nexus_graphrag.extractors.base import BaseExtractor

# Stopwords and common words to filter out from entity recognition
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "don", "should", "now", "it", "its", "their", "they", "we", "our", "you",
    "he", "she", "his", "her", "him", "my", "me", "i", "also", "into", "over",
    "after", "before", "between", "under", "above", "below", "up", "down",
    "traditional", "first", "second", "third", "fourth", "finally", "however",
    "therefore", "furthermore", "moreover", "overall", "meanwhile", "additionally",
    "instead", "similarly", "indeed", "specifically", "namely",
}

# Known heuristics for entity classification
TECH_KEYWORDS: Set[str] = {
    "python", "java", "c++", "pytorch", "tensorflow", "scikit-learn", "numpy",
    "pandas", "power bi", "gemini", "crewai", "networkx", "sqlite", "docker",
    "fastapi", "rag", "llm", "onnx", "pytest", "git", "github", "api", "rest",
    "graphrag", "vector", "embedding", "copula", "json", "sql", "html", "svg",
}

ORG_KEYWORDS: Set[str] = {
    "google", "deepmind", "microsoft", "meta", "openai", "amazon", "pwc",
    "university", "labs", "technologies", "inc", "corp", "corporation",
    "foundation", "consortium", "institute", "enterprise",
}

LOCATION_KEYWORDS: Set[str] = {
    "jaipur", "rajasthan", "india", "switzerland", "usa", "us", "uk",
    "california", "london", "europe", "delhi", "bangalore",
}

PREDICATE_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(is developed by|was developed by|developed by|created by|authored by)\b", "DEVELOPED_BY"),
    (r"\b(develops|developed|creates|created|authors|authored|builds|built)\b", "DEVELOPS"),
    (r"\b(integrates with|integrates into|connects to|combines with|interfaces with)\b", "INTEGRATES_WITH"),
    (r"\b(improves|enhances|optimizes|boosts|accelerates|streamlines)\b", "IMPROVES"),
    (r"\b(mitigates|prevents|eliminates|reduces|defends against|halts)\b", "MITIGATES"),
    (r"\b(uses|utilizes|employs|leverages|applies|executes)\b", "USES"),
    (r"\b(depends on|relies on|requires|mandates)\b", "DEPENDS_ON"),
    (r"\b(causes|leads to|results in|triggers|induces)\b", "CAUSES"),
    (r"\b(consists of|contains|includes|encompasses|comprises)\b", "CONTAINS"),
    (r"\b(is an?|is defined as|represents|acts as|serves as)\b", "IS_A"),
    (r"\b(evaluates|benchmarks|measures|audits|inspects|scores)\b", "EVALUATES"),
    (r"\b(solves|resolves|addresses|tackles|fixes)\b", "SOLVES"),
]


class DeterministicExtractor(BaseExtractor):
    """Zero-cloud, fast regex-and-grammar-based knowledge extractor."""

    def __init__(self, min_entity_length: int = 2, max_entity_words: int = 4) -> None:
        self.min_entity_length = min_entity_length
        self.max_entity_words = max_entity_words

    def extract(
        self, chunks: List[DocumentChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Processes chunks to extract deduplicated entities and typed relationships."""
        entity_map: Dict[str, Entity] = {}
        entity_display_names: Dict[str, str] = {}
        rel_map: Dict[str, Relationship] = {}
        chunk_entities: Dict[str, List[Entity]] = defaultdict(list)

        for chunk in chunks:
            raw_text = chunk.text
            sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", raw_text) if s.strip()]

            for sentence in sentences:
                found_entities = self._extract_entities_from_sentence(sentence, chunk.id)

                # Deduplicate and register in entity_map
                for ent in found_entities:
                    slug = ent.id
                    if slug not in entity_map:
                        entity_map[slug] = ent
                        entity_display_names[slug] = ent.name
                    else:
                        existing = entity_map[slug]
                        existing.weight += 0.5
                        if chunk.id not in existing.source_chunks:
                            existing.source_chunks.append(chunk.id)
                        if ent.description and not existing.description:
                            existing.description = ent.description
                        # Favor better casing if available
                        if ent.name.istitle() and not existing.name.istitle():
                            existing.name = ent.name

                    chunk_entities[chunk.id].append(entity_map[slug])

                # Extract relations within this sentence
                if len(found_entities) >= 2:
                    found_rels = self._extract_relationships_from_sentence(
                        sentence, found_entities, chunk.id
                    )
                    for rel in found_rels:
                        rel_id = rel.id
                        if rel_id not in rel_map:
                            rel_map[rel_id] = rel
                        else:
                            existing_rel = rel_map[rel_id]
                            existing_rel.weight += 0.5
                            if chunk.id not in existing_rel.evidence_chunks:
                                existing_rel.evidence_chunks.append(chunk.id)

        # Fallback: cross-entity co-occurrence for sparse relationships
        for chunk_id, ents in chunk_entities.items():
            unique_ents = list({e.id: e for e in ents}.values())
            for i in range(len(unique_ents)):
                for j in range(i + 1, min(i + 4, len(unique_ents))):
                    src = unique_ents[i]
                    tgt = unique_ents[j]
                    rel_id_fwd = f"{src.id}--associated_with-->{tgt.id}"
                    rel_id_bwd = f"{tgt.id}--associated_with-->{src.id}"

                    if rel_id_fwd not in rel_map and rel_id_bwd not in rel_map:
                        # Check if any relationship already exists between them
                        has_rel = any(
                            (r.source == src.name and r.target == tgt.name)
                            or (r.source == tgt.name and r.target == src.name)
                            for r in rel_map.values()
                        )
                        if not has_rel:
                            rel_map[rel_id_fwd] = Relationship(
                                source=src.name,
                                target=tgt.name,
                                relation_type="ASSOCIATED_WITH",
                                description=f"{src.name} co-occurs with {tgt.name} in document context.",
                                weight=0.8,
                                evidence_chunks=[chunk_id],
                                id=rel_id_fwd,
                            )

        return list(entity_map.values()), list(rel_map.values())

    def _extract_entities_from_sentence(
        self, sentence: str, chunk_id: str
    ) -> List[Entity]:
        """Extracts candidate entities using regex heuristics."""
        entities: List[Entity] = []
        seen_slugs: Set[str] = set()

        # Pattern 1: Title Case Single or Multi-word phrases: "AlphaFold", "Google DeepMind", "CrewAI"
        title_case_pattern = r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3}\b"
        for match in re.finditer(title_case_pattern, sentence):
            name = match.group(0).strip()
            if self._is_valid_entity(name):
                slug = slugify(name)
                if slug not in seen_slugs:
                    seen_slugs.add(slug)
                    ent_type = self._classify_entity(name)
                    entities.append(
                        Entity(
                            name=name,
                            entity_type=ent_type,
                            description=f"Entity identified in context: {name}",
                            source_chunks=[chunk_id],
                            id=slug,
                        )
                    )

        # Pattern 2: Acronyms and Technical Terms: "LLM", "RAG", "API", "Pytest", "Power BI", "GraphRAG"
        acronym_pattern = r"\b[A-Z]{2,6}(?:-[A-Za-z0-9]+)?\b"
        for match in re.finditer(acronym_pattern, sentence):
            name = match.group(0).strip()
            if self._is_valid_entity(name):
                slug = slugify(name)
                if slug not in seen_slugs:
                    seen_slugs.add(slug)
                    ent_type = self._classify_entity(name)
                    entities.append(
                        Entity(
                            name=name,
                            entity_type=ent_type,
                            description=f"Technical acronym or standard: {name}",
                            source_chunks=[chunk_id],
                            id=slug,
                        )
                    )

        # Pattern 3: Known Tech Terms matching regardless of case: "Python", "Docker", "PyTorch"
        for tech in TECH_KEYWORDS:
            pattern = rf"\b{re.escape(tech)}\b"
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                # Retrieve actual text with preserved case or capitalized
                matched_text = match.group(0)
                name = matched_text.title() if matched_text.islower() else matched_text
                slug = slugify(name)
                if slug not in seen_slugs:
                    seen_slugs.add(slug)
                    entities.append(
                        Entity(
                            name=name,
                            entity_type=EntityType.TECHNOLOGY,
                            description=f"Technology or tool: {name}",
                            source_chunks=[chunk_id],
                            id=slug,
                        )
                    )

        return entities

    def _extract_relationships_from_sentence(
        self, sentence: str, entities: List[Entity], chunk_id: str
    ) -> List[Relationship]:
        """Extracts directed or undirected relationships linking entities in the sentence."""
        relationships: List[Relationship] = []

        # Find positions of entities in the sentence
        ent_positions: List[Tuple[int, int, Entity]] = []
        for ent in entities:
            # Match entity name in sentence (case-insensitive)
            pattern = rf"\b{re.escape(ent.name)}\b"
            for m in re.finditer(pattern, sentence, re.IGNORECASE):
                ent_positions.append((m.start(), m.end(), ent))

        # Sort entities by start position
        ent_positions.sort(key=lambda x: x[0])

        # Test predicate matches between consecutive or nearby entity pairs
        for i in range(len(ent_positions)):
            for j in range(i + 1, min(i + 3, len(ent_positions))):
                start_i, end_i, src = ent_positions[i]
                start_j, end_j, tgt = ent_positions[j]

                if src.id == tgt.id:
                    continue

                # Segment between the two entities
                between_text = sentence[end_i:start_j].strip()
                if not between_text or len(between_text.split()) > 10:
                    continue

                # Match predicate pattern in between_text
                matched_predicate = None
                matched_rel_type = "RELATED_TO"

                for pattern_str, rel_type in PREDICATE_PATTERNS:
                    m = re.search(pattern_str, between_text, re.IGNORECASE)
                    if m:
                        matched_predicate = m.group(0)
                        matched_rel_type = rel_type
                        break

                if matched_predicate:
                    # Passive voice inversion: "developed by" means target develops source
                    if matched_rel_type == "DEVELOPED_BY":
                        s_name, t_name = tgt.name, src.name
                        rel_type = "DEVELOPS"
                    else:
                        s_name, t_name = src.name, tgt.name
                        rel_type = matched_rel_type

                    rel_id = f"{slugify(s_name)}--{slugify(rel_type)}-->{slugify(t_name)}"
                    desc = f"{s_name} {matched_predicate} {t_name}."

                    relationships.append(
                        Relationship(
                            source=s_name,
                            target=t_name,
                            relation_type=rel_type,
                            description=desc,
                            weight=1.5,
                            evidence_chunks=[chunk_id],
                            id=rel_id,
                        )
                    )

        return relationships

    def _is_valid_entity(self, name: str) -> bool:
        """Validates candidate entity string."""
        if len(name) < self.min_entity_length:
            return False
        words = name.split()
        if len(words) > self.max_entity_words:
            return False
        if len(words) == 1 and words[0].lower() in STOPWORDS:
            return False
        # Disallow purely numeric strings
        if name.replace(".", "").replace(",", "").isdigit():
            return False
        return True

    def _classify_entity(self, name: str) -> EntityType:
        """Assigns an EntityType based on name contents."""
        lower = name.lower()
        if lower in TECH_KEYWORDS or any(tech in lower for tech in ["api", "rag", "engine", "model", "python", "framework", "library"]):
            return EntityType.TECHNOLOGY
        if lower in ORG_KEYWORDS or any(org in lower for org in ["inc", "corp", "labs", "university", "google", "meta", "openai"]):
            return EntityType.ORGANIZATION
        if lower in LOCATION_KEYWORDS:
            return EntityType.LOCATION
        # Person heuristic: 2 capitalized words, not an org or tech
        words = name.split()
        if len(words) == 2 and all(w.istitle() for w in words) and not any(kw in lower for kw in ["system", "data", "pipeline", "agent"]):
            return EntityType.PERSON
        return EntityType.CONCEPT
