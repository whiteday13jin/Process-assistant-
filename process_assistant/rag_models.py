from __future__ import annotations

"""RAG 领域对象模型。

这些对象主要服务于文档问答链路，把“文档、切片、召回结果、引用、最终答案”
明确拆成不同层级，方便检索、展示和评测阶段各自使用。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class Document:
    """标准化后的原始文档对象，是切分器的直接输入。"""

    doc_id: str
    source_path: str
    title: str
    file_type: str
    content: str
    last_modified: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """文档切片对象，是向量化和检索的最小知识单元。"""

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
    """召回后的知识块，额外带上分数和分数组成，便于解释检索结果。"""

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
    """回答中的引用对象，用来把答案和原始证据片段重新绑定起来。"""

    ref_id: str
    chunk_id: str
    source_path: str
    title: str
    chunk_index: int
    score: float
    quote: str


@dataclass(frozen=True)
class RagAnswer:
    """RAG 最终输出。

    除了自然语言答案外，还保留召回片段、引用、查询改写和证据判断，
    这样既方便前端展示，也方便后续评测和调试。
    """

    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    engine: str
    model: str
    query_rewrite: Dict[str, Any] = field(default_factory=dict)
    retrieval_policy: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
