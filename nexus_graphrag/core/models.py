"""Core data models for Nexus-GraphRAG knowledge representation."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EntityType(str, Enum):
    """Semantic category of an extracted knowledge entity."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    TECHNOLOGY = "TECHNOLOGY"
    CONCEPT = "CONCEPT"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    OTHER = "OTHER"

    @classmethod
    def from_string(cls, val: str) -> "EntityType":
        """Normalizes arbitrary string to known EntityType."""
        normalized = val.strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.OTHER


def slugify(text: str) -> str:
    """Generates clean ASCII slug identifier."""
    cleaned = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "_", cleaned)


@dataclass
class Entity:
    """Represents a discrete concept, actor, or object in the knowledge graph."""

    name: str
    entity_type: EntityType = EntityType.CONCEPT
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    source_chunks: List[str] = field(default_factory=list)
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            self.id = slugify(self.name)
        if isinstance(self.entity_type, str):
            self.entity_type = EntityType.from_string(self.entity_type)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes entity to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "properties": self.properties,
            "weight": self.weight,
            "source_chunks": list(self.source_chunks),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Instantiates entity from dictionary representation."""
        return cls(
            name=data["name"],
            entity_type=EntityType.from_string(data.get("entity_type", "CONCEPT")),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            weight=float(data.get("weight", 1.0)),
            source_chunks=list(data.get("source_chunks", [])),
            id=data.get("id", slugify(data["name"])),
        )


@dataclass
class Relationship:
    """Directed or undirected relational edge linking two entities."""

    source: str
    target: str
    relation_type: str = "RELATED_TO"
    description: str = ""
    weight: float = 1.0
    evidence_chunks: List[str] = field(default_factory=list)
    id: str = field(default="")

    def __post_init__(self) -> None:
        norm_source = slugify(self.source)
        norm_target = slugify(self.target)
        norm_rel = slugify(self.relation_type)
        if not self.id:
            self.id = f"{norm_source}--{norm_rel}-->{norm_target}"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes relationship to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "description": self.description,
            "weight": self.weight,
            "evidence_chunks": list(self.evidence_chunks),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Instantiates relationship from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=data.get("relation_type", "RELATED_TO"),
            description=data.get("description", ""),
            weight=float(data.get("weight", 1.0)),
            evidence_chunks=list(data.get("evidence_chunks", [])),
            id=data.get("id", ""),
        )


@dataclass
class DocumentChunk:
    """A granular segmented slice of source textual corpus."""

    text: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default="")
    token_count: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"chunk_{self.chunk_index}_{uuid.uuid4().hex[:8]}"
        if self.token_count == 0 and self.text:
            self.token_count = len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes chunk to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Instantiates chunk from dictionary."""
        return cls(
            text=data["text"],
            chunk_index=int(data.get("chunk_index", 0)),
            metadata=data.get("metadata", {}),
            id=data.get("id", ""),
            token_count=int(data.get("token_count", 0)),
        )


@dataclass
class Community:
    """Clustered group of interconnected entities reflecting a modular domain."""

    id: str
    level: int
    title: str
    summary: str
    member_entities: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes community cluster to dictionary."""
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "summary": self.summary,
            "member_entities": list(self.member_entities),
            "findings": list(self.findings),
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Community":
        """Instantiates community cluster from dictionary."""
        return cls(
            id=data["id"],
            level=int(data.get("level", 0)),
            title=data.get("title", f"Community {data['id']}"),
            summary=data.get("summary", ""),
            member_entities=list(data.get("member_entities", [])),
            findings=list(data.get("findings", [])),
            weight=float(data.get("weight", 1.0)),
        )


class RetrievalMode(str, Enum):
    """Retrieval modality for query execution."""

    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"

    @classmethod
    def from_string(cls, val: str) -> "RetrievalMode":
        """Normalizes retrieval mode."""
        clean = val.strip().lower()
        if clean == "local":
            return cls.LOCAL
        if clean == "global":
            return cls.GLOBAL
        return cls.HYBRID


@dataclass
class RetrievalResult:
    """Synthesized result from knowledge graph retrieval."""

    query: str
    mode: RetrievalMode
    context: str
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    communities: List[Community] = field(default_factory=list)
    confidence_score: float = 1.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes retrieval result."""
        return {
            "query": self.query,
            "mode": self.mode.value,
            "context": self.context,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "communities": [c.to_dict() for c in self.communities],
            "confidence_score": self.confidence_score,
            "metrics": self.metrics,
        }
