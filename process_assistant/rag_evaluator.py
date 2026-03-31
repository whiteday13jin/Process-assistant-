from __future__ import annotations

"""RAG 轻量评测层。

这一层不是为了做学术级 benchmark，而是帮助我们快速回答三个工程问题：
检索有没有找到对的文档、回答有没有覆盖关键点、该拒答时有没有乱答。
对于演示项目来说，这类“能解释结果”的评测比复杂指标更实用。
"""

from statistics import mean
from typing import Any, Dict, List

from .rag_pipeline import RagPipeline

REFUSAL_HINTS = [
    "依据不足",
    "资料不足",
    "未找到依据",
    "无法从现有资料判断",
    "现有语料未覆盖",
]


class RagEvaluator:
    """围绕当前 `RagPipeline` 做离线批量评测。"""

    def __init__(self, pipeline: RagPipeline) -> None:
        self.pipeline = pipeline

    def evaluate(
        self,
        dataset: List[Dict[str, Any]],
        *,
        top_k: int = 4,
        max_context_chunks: int = 4,
        vector_weight: float = 0.70,
        lexical_weight: float = 0.20,
        title_weight: float = 0.10,
        max_per_doc: int = 2,
        min_final_score: float = 0.42,
        min_lexical_score: float = 0.02,
        min_title_score: float = 0.02,
    ) -> Dict[str, Any]:
        """对评测集逐条提问，并汇总成一份可读报告。"""

        if not dataset:
            raise ValueError("evaluation dataset is empty")

        details: List[Dict[str, Any]] = []
        hit_scores: List[float] = []
        recall_scores: List[float] = []
        keyword_scores: List[float] = []
        refusal_scores: List[float] = []

        for item in dataset:
            qid = str(item.get("id", "unknown"))
            question = str(item.get("question", "")).strip()
            expected_sources = [str(x) for x in item.get("expected_sources", [])]
            expected_keywords = [str(x) for x in item.get("expected_keywords", [])]
            answerable = bool(item.get("answerable", True))

            answer = self.pipeline.ask(
                question,
                top_k=top_k,
                max_context_chunks=max_context_chunks,
                vector_weight=vector_weight,
                lexical_weight=lexical_weight,
                title_weight=title_weight,
                max_per_doc=max_per_doc,
                min_final_score=min_final_score,
                min_lexical_score=min_lexical_score,
                min_title_score=min_title_score,
                temperature=0.0,
            )
            answer_payload = self.pipeline.to_payload(answer)
            # 这里把“检索命中”和“回答质量”拆开看，便于判断问题出在召回还是生成。
            hit, recall = _source_metrics(
                [c["source_path"] for c in answer_payload["retrieved_chunks"]],
                expected_sources,
            )
            keyword = _keyword_coverage(answer_payload["answer"], expected_keywords)
            refusal = _refusal_score(answer_payload["answer"], answerable)

            if answerable:
                hit_scores.append(hit)
                recall_scores.append(recall)
            keyword_scores.append(keyword)
            refusal_scores.append(refusal)

            details.append(
                {
                    "id": qid,
                    "question": question,
                    "answerable": answerable,
                    "hit_at_k": hit if answerable else None,
                    "source_recall_at_k": recall if answerable else None,
                    "keyword_coverage": keyword,
                    "refusal_score": refusal,
                    "retrieved_sources": [c["source_path"] for c in answer_payload["retrieved_chunks"]],
                    "answer": answer_payload["answer"],
                }
            )

        return {
            "engine": "rag_simple_evaluator_v1",
            "samples": len(dataset),
            "answerable_samples": sum(1 for x in dataset if bool(x.get("answerable", True))),
            "hit_at_k": round(_safe_mean(hit_scores), 4),
            "source_recall_at_k": round(_safe_mean(recall_scores), 4),
            "answer_keyword_coverage": round(_safe_mean(keyword_scores), 4),
            "refusal_accuracy": round(_safe_mean(refusal_scores), 4),
            "details": details,
        }


def _source_metrics(retrieved_sources: List[str], expected_sources: List[str]) -> tuple[float, float]:
    """评估检索结果是否至少命中过目标文档，以及召回比例如何。"""

    if not expected_sources:
        return 0.0, 0.0
    retrieved_set = {x.split("/")[-1] for x in retrieved_sources}
    expected_set = {x.split("/")[-1] for x in expected_sources}
    inter = retrieved_set.intersection(expected_set)
    hit = 1.0 if inter else 0.0
    recall = len(inter) / len(expected_set) if expected_set else 0.0
    return hit, recall


def _keyword_coverage(answer: str, keywords: List[str]) -> float:
    """用关键词覆盖率做一个足够轻量的回答完整度近似指标。"""

    if not keywords:
        return 1.0
    text = answer.lower()
    hit = 0
    for kw in keywords:
        if kw.lower() in text:
            hit += 1
    return hit / len(keywords)


def _refusal_score(answer: str, answerable: bool) -> float:
    """评估系统在“该答”与“该拒答”两种场景下是否表现合理。"""

    has_refusal = any(hint in answer for hint in REFUSAL_HINTS)
    if answerable:
        return 0.0 if has_refusal else 1.0
    return 1.0 if has_refusal else 0.0


def _safe_mean(values: List[float]) -> float:
    """空列表时返回 0，避免评测阶段出现无意义异常。"""

    if not values:
        return 0.0
    return mean(values)
