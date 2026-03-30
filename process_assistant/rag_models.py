from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    title: str
    file_type: str
    content: str
    last_modified: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source_path: str
    title: str
    char_start: int
    char_end: int
    heading_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    chunk_index: int
    score: float
    text: str
    heading_path: List[str] = field(default_factory=list)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    ref_id: str
    chunk_id: str
    source_path: str
    title: str
    chunk_index: int
    score: float
    quote: str


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    engine: str
    model: str
    query_rewrite: Dict[str, Any] = field(default_factory=dict)
    retrieval_policy: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
