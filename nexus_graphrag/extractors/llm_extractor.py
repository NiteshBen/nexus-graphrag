"""Pluggable LLM-powered knowledge extractor with automatic deterministic fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable, List, Optional, Tuple

from nexus_graphrag.core.models import (
    DocumentChunk,
    Entity,
    EntityType,
    Relationship,
)
from nexus_graphrag.extractors.base import BaseExtractor
from nexus_graphrag.extractors.deterministic import DeterministicExtractor

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph extraction engine.
Analyze the following text and extract all meaningful entities and relationships.
Return ONLY valid JSON matching this exact structure:
{
  "entities": [
    {
      "name": "Entity Name",
      "type": "PERSON | ORGANIZATION | TECHNOLOGY | CONCEPT | LOCATION | EVENT | OTHER",
      "description": "Short factual description based strictly on the text"
    }
  ],
  "relationships": [
    {
      "source": "Exact Source Entity Name",
      "target": "Exact Target Entity Name",
      "relation_type": "UPPERCASE_VERB_PHRASE",
      "description": "Short explanation of how source and target interact"
    }
  ]
}
Do not include conversational preamble or markdown other than valid JSON.
"""


class LLMExtractor(BaseExtractor):
    """Hybrid extractor: uses an LLM if available, with deterministic fallback."""

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        fallback_extractor: Optional[BaseExtractor] = None,
    ) -> None:
        """Initializes LLMExtractor.

        Args:
            llm_fn: Optional callable taking a prompt string and returning string output.
            fallback_extractor: Extractor to utilize if LLM is unavailable or fails.
        """
        self.llm_fn = llm_fn
        self.fallback = fallback_extractor or DeterministicExtractor()

    def extract(
        self, chunks: List[DocumentChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extracts entities and relationships via LLM or graceful fallback."""
        if not self.llm_fn:
            logger.info("No LLM callable provided; utilizing deterministic extraction.")
            return self.fallback.extract(chunks)

        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []

        for chunk in chunks:
            prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nTEXT CHUNK:\n{chunk.text}"
            try:
                response = self.llm_fn(prompt)
                ents, rels = self._parse_llm_response(response, chunk.id)
                all_entities.extend(ents)
                all_relationships.extend(rels)
            except Exception as e:
                logger.warning(
                    f"LLM extraction failed for chunk {chunk.id}: {e}. Falling back to deterministic."
                )
                f_ents, f_rels = self.fallback.extract([chunk])
                all_entities.extend(f_ents)
                all_relationships.extend(f_rels)

        # Merge and deduplicate
        return self._deduplicate(all_entities, all_relationships)

    def _parse_llm_response(
        self, response: str, chunk_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Cleans and parses raw JSON from LLM output."""
        cleaned = response.strip()
        # Remove markdown code block fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)
        entities: List[Entity] = []
        relationships: List[Relationship] = []

        for item in data.get("entities", []):
            name = item.get("name", "").strip()
            if name:
                ent_type = EntityType.from_string(item.get("type", "CONCEPT"))
                desc = item.get("description", "")
                entities.append(
                    Entity(
                        name=name,
                        entity_type=ent_type,
                        description=desc,
                        source_chunks=[chunk_id],
                    )
                )

        for item in data.get("relationships", []):
            source = item.get("source", "").strip()
            target = item.get("target", "").strip()
            if source and target:
                rel_type = item.get("relation_type", "RELATED_TO").strip().upper()
                desc = item.get("description", "")
                relationships.append(
                    Relationship(
                        source=source,
                        target=target,
                        relation_type=rel_type,
                        description=desc,
                        evidence_chunks=[chunk_id],
                    )
                )

        return entities, relationships

    def _deduplicate(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Merges duplicate entities and edges."""
        ent_map: dict[str, Entity] = {}
        for ent in entities:
            if ent.id not in ent_map:
                ent_map[ent.id] = ent
            else:
                existing = ent_map[ent.id]
                existing.weight += 0.5
                for c in ent.source_chunks:
                    if c not in existing.source_chunks:
                        existing.source_chunks.append(c)

        rel_map: dict[str, Relationship] = {}
        for rel in relationships:
            if rel.id not in rel_map:
                rel_map[rel.id] = rel
            else:
                existing_rel = rel_map[rel.id]
                existing_rel.weight += 0.5
                for c in rel.evidence_chunks:
                    if c not in existing_rel.evidence_chunks:
                        existing_rel.evidence_chunks.append(c)

        return list(ent_map.values()), list(rel_map.values())
