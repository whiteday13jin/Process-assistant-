from __future__ import annotations

"""RAG 文本切分层。

这一层把整篇文档拆成适合检索的知识块。在可检索性和语义完整性之间做平衡：
先按标题段落分区，再滑窗补足长度控制，最后生成可回溯到原文位置的 Chunk。
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .rag_models import Chunk, Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class TextSplitter:
    """面向中文工艺文档的轻量切分器。"""

    chunk_size: int = 700
    chunk_overlap: int = 100
    min_chunk_size: int = 120

    def __post_init__(self) -> None:
        """在真正切分前先约束参数，避免出现无效窗口。"""

        if self.chunk_size < 100:
            raise ValueError("chunk_size must be >= 100")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_size < 1:
            raise ValueError("min_chunk_size must be >= 1")

    def split_documents(self, docs: Iterable[Document]) -> List[Chunk]:
        """批量切分文档，是建索引阶段最常用的入口。"""

        chunks: List[Chunk] = []
        for doc in docs:
            chunks.extend(self._split_document(doc))
        return chunks

    def _split_document(self, doc: Document) -> List[Chunk]:
        """单篇文档切分逻辑。

        这里先按标题层级划 section，再在每个 section 内按滑窗切块，
        这样做比“整篇文章直接硬切”更容易保留局部语义。
        """

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
                # 过短片段对检索帮助不大，还容易制造噪声，所以中途短块直接丢掉。
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
            # 兜底逻辑：即使标题解析失败，也至少保留一个整篇 chunk，避免文档彻底消失。
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
    """按 Markdown 标题切分 section，并保留标题路径。"""

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
            # 标题栈用来记录层级路径，后面引用来源时可以告诉用户“这段来自哪个章节”。
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
    """对超长段落做滑窗切分，兼顾长度限制和上下文连续性。"""

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
        # overlap 的作用是让相邻块共享一段上下文，减少跨块信息被截断。
        start += step
    return windows
