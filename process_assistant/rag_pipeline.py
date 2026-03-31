from __future__ import annotations

"""自研版 RAG 主流程。

这是项目里“文档问答正式版”的核心文件。
和 LangChain 对照版相比，这里更强调我们对检索和回答链路的控制能力。
"""

import re
from dataclasses import asdict
from typing import Any, Dict, List

from .embedder import OpenAICompatibleClient
from .query_normalizer import rewrite_query
from .rag_models import Citation, RagAnswer, RetrievedChunk
from .vector_index import LocalVectorIndex


class RagPipeline:
    """自研版 RAG 主类。"""

    def __init__(self, index: LocalVectorIndex, client: OpenAICompatibleClient) -> None:
        self.index = index
        self.client = client
        self.engine = "rag_retrieval_generation_v1"

    def ask(
        self,
        question: str,
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
        temperature: float = 0.1,
    ) -> RagAnswer:
        """执行一次自研版 RAG 问答。

        主流程：清洗问题 -> 查询归一化 -> 检索与重排 -> 证据门控 -> 生成答案或拒答。
        """
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        if max_context_chunks < 1:
            raise ValueError("max_context_chunks must be >= 1")

        # 先做查询归一化，让“假焊 / 虚焊 / 焊接不实”这类现场说法能命中同一类知识。
        rewrite = rewrite_query(question)
        qvec = self.client.embed_texts([rewrite.retrieval_query], batch_size=1)[0]
        retrieved = self.index.search(
            qvec,
            top_k=top_k,
            query_text=rewrite.retrieval_query,
            vector_weight=vector_weight,
            lexical_weight=lexical_weight,
            title_weight=title_weight,
            max_per_doc=max_per_doc,
        )
        context_chunks = retrieved[: max_context_chunks]
        citations = _build_citations(context_chunks)

        # 检索结果不直接喂给模型，而是先做证据门控，减少“证据很弱却硬答”的情况。
        evidence = _evaluate_evidence(
            context_chunks,
            min_final_score=min_final_score,
            min_lexical_score=min_lexical_score,
            min_title_score=min_title_score,
        )
        if evidence["is_sufficient"]:
            answer = self._generate_answer(question, context_chunks, citations, temperature=temperature)
        else:
            answer = self._insufficient_evidence_answer(citations, evidence)

        return RagAnswer(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved,
            engine=self.engine,
            model=self.client.config.chat_model,
            query_rewrite={
                "original": rewrite.original_question,
                "normalized": rewrite.normalized_question,
                "retrieval_query": rewrite.retrieval_query,
                "expanded_terms": rewrite.expanded_terms,
                "matched_concepts": rewrite.matched_concepts,
            },
            retrieval_policy={
                "formula": "final = vector_weight*vector + lexical_weight*lexical + title_weight*title",
                "vector_weight": vector_weight,
                "lexical_weight": lexical_weight,
                "title_weight": title_weight,
                "max_per_doc": max_per_doc,
            },
            evidence=evidence,
        )

    @staticmethod
    def to_payload(answer: RagAnswer) -> Dict[str, Any]:
        return asdict(answer)

    def _generate_answer(
        self,
        question: str,
        context_chunks: List[RetrievedChunk],
        citations: List[Citation],
        *,
        temperature: float,
    ) -> str:
        """用检索到的片段生成最终答案。"""
        if not context_chunks:
            return "未检索到可用知识片段，当前无法基于知识库给出可靠答案。"

        source_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            heading = " > ".join(chunk.heading_path) if chunk.heading_path else "(无标题层级)"
            source_blocks.append(
                "\n".join(
                    [
                        f"[Source {idx}]",
                        f"File: {chunk.source_path}",
                        f"Title: {chunk.title}",
                        f"Chunk: {chunk.chunk_id}",
                        f"Heading: {heading}",
                        "Content:",
                        chunk.text,
                    ]
                )
            )
        sources_text = "\n\n".join(source_blocks)

        # system prompt 主要负责“约束边界”。
        system_prompt = (
            "你是工艺知识问答助手。"
            "只允许依据提供的来源内容回答，不得编造参数、规范或结论。"
            "仅当问题主干无法回答时，才写‘依据不足’，否则不要出现该词。"
            "回答中请使用 [1] [2] 这样的引用编号指向来源。"
        )
        # user prompt 主要负责“规定输出格式”。
        user_prompt = (
            f"问题：{question}\n\n"
            f"可用来源如下：\n{sources_text}\n\n"
            "请严格按以下模板输出：\n"
            "【结论】\n"
            "（2-4句，先回答主问题）\n\n"
            "【关键依据】\n"
            "- 依据点1 [1]\n"
            "- 依据点2 [2]\n\n"
            "【建议动作】\n"
            "1. 动作1\n"
            "2. 动作2\n\n"
            "要求：\n"
            "- 不要输出模板外标题\n"
            "- 不要输出 emoji\n"
            "- 对可回答问题，禁止写‘依据不足’\n"
            "- 对不可回答问题，写明‘依据不足’，并说明缺失信息"
        )
        generated = self.client.chat(system_prompt, user_prompt, temperature=temperature).strip()
        generated = _clean_answer_text(generated)
        if not _contains_reference_marker(generated):
            generated += "\n\n参考来源: " + " ".join(c.ref_id for c in citations)
        return generated

    @staticmethod
    def _insufficient_evidence_answer(citations: List[Citation], evidence: Dict[str, float | bool]) -> str:
        """证据不足时的统一回答模板。"""
        if citations:
            refs = " ".join(c.ref_id for c in citations)
            return (
                "【结论】\n"
                "依据不足，当前知识库证据强度不够，无法给出可靠结论。\n\n"
                "【关键依据】\n"
                f"- 当前最高证据分不足阈值或词法/标题命中较弱（top_final={evidence.get('top_final_score')}）\n"
                f"- 已参考候选来源：{refs}\n\n"
                "【建议动作】\n"
                "1. 补充更明确的问题关键词（工序、异常现象、条件）\n"
                "2. 补充对应工艺文档后再次检索"
            )
        return (
            "【结论】\n"
            "依据不足，未检索到可用证据。\n\n"
            "【关键依据】\n"
            "- 当前问题在知识库中缺少可用支撑片段\n\n"
            "【建议动作】\n"
            "1. 补充对应工艺文档\n"
            "2. 调整问题描述后重试"
        )


def _build_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
    """把检索片段转换成前端和报告可直接消费的引用对象。"""
    citations: List[Citation] = []
    for idx, chunk in enumerate(chunks, start=1):
        quote = _clip(chunk.text, 180)
        citations.append(
            Citation(
                ref_id=f"[{idx}]",
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                title=chunk.title,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                quote=quote,
            )
        )
    return citations


def _clip(text: str, size: int) -> str:
    text = " ".join(text.split())
    if len(text) <= size:
        return text
    return text[: size - 3] + "..."


def _contains_reference_marker(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text))


def _clean_answer_text(text: str) -> str:
    """清理模型输出里的多余符号和换行噪声。"""
    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"[✅❌⚠️⭐✨•●■□▶►▪️]", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _evaluate_evidence(
    chunks: List[RetrievedChunk],
    *,
    min_final_score: float,
    min_lexical_score: float,
    min_title_score: float,
) -> Dict[str, float | bool]:
    """评估当前检索证据是否足以支撑回答。"""
    if not chunks:
        return {
            "is_sufficient": False,
            "top_final_score": 0.0,
            "top_lexical_score": 0.0,
            "top_title_score": 0.0,
        }

    top = chunks[0]
    breakdown = top.score_breakdown or {}
    top_final = float(breakdown.get("final", top.score))
    top_lexical = float(breakdown.get("lexical", 0.0))
    top_title = float(breakdown.get("title", 0.0))
    score_ok = top_final >= min_final_score
    support_ok = top_lexical >= min_lexical_score or top_title >= min_title_score

    return {
        "is_sufficient": bool(score_ok and support_ok),
        "top_final_score": round(top_final, 6),
        "top_lexical_score": round(top_lexical, 6),
        "top_title_score": round(top_title, 6),
        "min_final_score": min_final_score,
        "min_lexical_score": min_lexical_score,
        "min_title_score": min_title_score,
    }
