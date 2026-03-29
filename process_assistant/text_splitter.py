from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .rag_models import Chunk, Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class TextSplitter:
    chunk_size: int = 700
    chunk_overlap: int = 100
    min_chunk_size: int = 120

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be >= 100")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_size < 1:
            raise ValueError("min_chunk_size must be >= 1")

    def split_documents(self, docs: Iterable[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in docs:
            chunks.extend(self._split_document(doc))
        return chunks

    def _split_document(self, doc: Document) -> List[Chunk]:
        sections = _split_sections(doc.content)
        if not sections:
            sections = [(doc.content, 0, [])]

        chunks: List[Chunk] = []
        idx = 0
        for text, section_start, heading_path in sections:
            cleaned = text.strip()
            if not cleaned:
                continue
            for start, end, piece in _sliding_windows(cleaned, self.chunk_size, self.chunk_overlap):
                piece = piece.strip()
                if not piece:
                    continue
                if len(piece) < self.min_chunk_size and end != len(cleaned):
                    continue
                idx += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}-{idx:04d}",
                        doc_id=doc.doc_id,
                        chunk_index=idx,
                        text=piece,
                        source_path=doc.source_path,
                        title=doc.title,
                        char_start=section_start + start,
                        char_end=section_start + end,
                        heading_path=list(heading_path),
                        metadata=dict(doc.metadata),
                    )
                )
        if not chunks:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-0001",
                    doc_id=doc.doc_id,
                    chunk_index=1,
                    text=doc.content.strip(),
                    source_path=doc.source_path,
                    title=doc.title,
                    char_start=0,
                    char_end=len(doc.content),
                    heading_path=[],
                    metadata=dict(doc.metadata),
                )
            )
        return chunks


def _split_sections(text: str) -> List[Tuple[str, int, List[str]]]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    sections: List[Tuple[str, int, List[str]]] = []
    heading_stack: List[str] = []
    heading_levels: List[int] = []
    current_lines: List[str] = []
    current_start = 0
    cursor = 0

    for line in lines:
        match = HEADING_RE.match(line.strip())
        if match:
            if current_lines:
                sections.append(("".join(current_lines), current_start, list(heading_stack)))
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_levels and heading_levels[-1] >= level:
                heading_levels.pop()
                heading_stack.pop()
            heading_levels.append(level)
            heading_stack.append(title)
            current_lines = []
            current_start = cursor + len(line)
        else:
            current_lines.append(line)
        cursor += len(line)

    if current_lines:
        sections.append(("".join(current_lines), current_start, list(heading_stack)))
    return sections


def _sliding_windows(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int, str]]:
    if len(text) <= chunk_size:
        return [(0, len(text), text)]

    windows: List[Tuple[int, int, str]] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        windows.append((start, end, text[start:end]))
        if end >= len(text):
            break
        start += step
    return windows

