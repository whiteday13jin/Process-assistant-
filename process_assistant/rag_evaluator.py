from __future__ import annotations

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
    if not expected_sources:
        return 0.0, 0.0
    retrieved_set = {x.split("/")[-1] for x in retrieved_sources}
    expected_set = {x.split("/")[-1] for x in expected_sources}
    inter = retrieved_set.intersection(expected_set)
    hit = 1.0 if inter else 0.0
    recall = len(inter) / len(expected_set) if expected_set else 0.0
    return hit, recall


def _keyword_coverage(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    text = answer.lower()
    hit = 0
    for kw in keywords:
        if kw.lower() in text:
            hit += 1
    return hit / len(keywords)


def _refusal_score(answer: str, answerable: bool) -> float:
    has_refusal = any(hint in answer for hint in REFUSAL_HINTS)
    if answerable:
        return 0.0 if has_refusal else 1.0
    return 1.0 if has_refusal else 0.0


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return mean(values)
