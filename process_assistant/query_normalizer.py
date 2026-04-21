from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


SYNONYM_GROUPS: Dict[str, List[str]] = {
    "虚焊": ["虚焊", "假焊", "焊接不实"],
    "焊盘氧化": ["焊盘氧化", "焊盘发暗", "焊盘发黑"],
    "助焊剂活性": ["助焊剂活性", "助焊剂老化", "助焊剂失效"],
    "起泡": ["起泡", "鼓包"],
    "分层": ["分层", "层间剥离"],
    "对位偏移": ["对位偏移", "偏位", "图形偏移"],
    "电阻偏高": ["电阻偏高", "阻值偏高", "阻值高"],
    "耐压不良": ["耐压不良", "耐压失效", "漏电"],
}


@dataclass(frozen=True)
class QueryRewrite:
    original_question: str
    normalized_question: str
    retrieval_query: str
    expanded_terms: List[str]
    matched_concepts: List[str]


def rewrite_query(question: str) -> QueryRewrite:
    original = question.strip()
    normalized = _normalize_text(original)

    expanded: List[str] = []
    matched_concepts: List[str] = []
    for canonical, variants in SYNONYM_GROUPS.items():
        if any(v in normalized for v in variants):
            matched_concepts.append(canonical)
            for term in variants:
                if term not in expanded:
                    expanded.append(term)

    retrieval_query = normalized
    append_terms = [t for t in expanded if t not in normalized]
    if append_terms:
        retrieval_query = f"{normalized} {' '.join(append_terms)}".strip()

    return QueryRewrite(
        original_question=original,
        normalized_question=normalized,
        retrieval_query=retrieval_query,
        expanded_terms=expanded,
        matched_concepts=matched_concepts,
    )


def _normalize_text(text: str) -> str:
    text = text.replace("，", ",").replace("。", ".").replace("？", "?").replace("！", "!")
    text = re.sub(r"\s+", " ", text).strip()
    return text

