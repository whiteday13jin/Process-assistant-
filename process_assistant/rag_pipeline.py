from __future__ import annotations

import re
from dataclasses import asdict
from typing import Dict, List

from .embedder import OpenAICompatibleClient
from .rag_models import Citation, RagAnswer, RetrievedChunk
from .vector_index import LocalVectorIndex


class RagPipeline:
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
        hybrid_alpha: float = 0.82,
        max_per_doc: int = 2,
        temperature: float = 0.1,
    ) -> RagAnswer:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        if max_context_chunks < 1:
            raise ValueError("max_context_chunks must be >= 1")

        qvec = self.client.embed_texts([question], batch_size=1)[0]
        retrieved = self.index.search(
            qvec,
            top_k=top_k,
            query_text=question,
            hybrid_alpha=hybrid_alpha,
            max_per_doc=max_per_doc,
        )
        context_chunks = retrieved[: max_context_chunks]
        citations = _build_citations(context_chunks)
        answer = self._generate_answer(question, context_chunks, citations, temperature=temperature)
        return RagAnswer(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved,
            engine=self.engine,
            model=self.client.config.chat_model,
        )

    @staticmethod
    def to_payload(answer: RagAnswer) -> Dict:
        return asdict(answer)

    def _generate_answer(
        self,
        question: str,
        context_chunks: List[RetrievedChunk],
        citations: List[Citation],
        *,
        temperature: float,
    ) -> str:
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

        system_prompt = (
            "你是工艺知识问答助手。"
            "只允许依据提供的来源内容回答，不得编造参数、规范或结论。"
            "仅当问题主干无法回答时，才写“依据不足”。"
            "回答中请使用 [1] [2] 这样的引用编号指向来源。"
        )
        user_prompt = (
            f"问题：{question}\n\n"
            f"可用来源如下：\n{sources_text}\n\n"
            "请输出：\n"
            "1) 简洁结论\n"
            "2) 关键判断依据（附引用编号）\n"
            "3) 建议动作（若依据不足请明确说明）"
        )
        generated = self.client.chat(system_prompt, user_prompt, temperature=temperature).strip()
        if not _contains_reference_marker(generated):
            generated += "\n\n参考来源: " + " ".join(c.ref_id for c in citations)
        return generated


def _build_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
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
