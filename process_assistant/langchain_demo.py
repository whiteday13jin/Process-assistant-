from __future__ import annotations

"""LangChain 对照版 RAG。

在不破坏主链路的前提下，做了一个独立 LangChain 分支。
主要是把 Loader、Splitter、Embedding、VectorStore、Retriever、Prompt、LLM 
这些通用组件标准化了。
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_loader import load_documents_from_dir
from .env_loader import load_project_env


class LangChainRagDemo:
    """LangChain 对照版主类。"""

    INDEX_NAME = "lc_faiss"
    MANIFEST_NAME = "manifest.json"

    def __init__(self, index_dir: str | Path) -> None:
        load_project_env()
        self.index_dir = Path(index_dir)

    @classmethod
    def build(
        cls,
        docs_dir: str | Path,
        index_dir: str | Path,
        *,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """构建 LangChain 版索引。

        主流程：读文档 -> 转 LangChain Document -> 切分 -> Embeddings + FAISS 建库 -> 落盘。
        """
        load_project_env()
        docs_dir = Path(docs_dir)
        index_dir = Path(index_dir)

        # Loader 这里复用项目现有文档接入层，重点把 LangChain 价值放在后续组件上。
        raw_docs = load_documents_from_dir(docs_dir, recursive=recursive)
        langchain_docs = [_to_langchain_document(item) for item in raw_docs]

        # Splitter：选用 LangChain 标准切分器。
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " "],
            keep_separator=True,
        )
        chunks = splitter.split_documents(langchain_docs)
        for idx, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = f"LC-{idx:04d}"
            chunk.metadata["chunk_index"] = idx
            chunk.page_content = _normalize_content(chunk.page_content)

        index_dir.mkdir(parents=True, exist_ok=True)
        # VectorStore：FAISS
        vectorstore = FAISS.from_documents(chunks, _build_embeddings())
        vectorstore.save_local(str(index_dir), index_name=cls.INDEX_NAME)

        manifest = {
            "engine": "langchain_demo",
            "docs_dir": str(docs_dir.resolve()),
            "index_dir": str(index_dir.resolve()),
            "doc_count": len(raw_docs),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retriever": {
                "type": "mmr",
                "why": "Use framework-native diversified retrieval to keep the demo simple and explainable.",
            },
            "chat_model": _chat_model_name(),
            "embed_model": _embed_model_name(),
            "built_at": datetime.now().isoformat(timespec="seconds"),
        }
        (index_dir / cls.MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def ask(
        self,
        question: str,
        *,
        top_k: int = 4,
        max_context_chunks: int = 4,
        temperature: float = 0.1,
        lambda_mult: float = 0.55,
    ) -> dict[str, Any]:
        """执行一次 LangChain 版问答。"""
        prompt_question = (question or "").strip()
        if not prompt_question:
            raise ValueError("question is required")

        vectorstore = FAISS.load_local(
            str(self.index_dir),
            _build_embeddings(),
            index_name=self.INDEX_NAME,
            allow_dangerous_deserialization=True,
        )
        # Retriever：使用 MMR，标准。
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": max(top_k, 1),
                "fetch_k": max(top_k * 4, 12),
                "lambda_mult": lambda_mult,
            },
        )
        retrieved_docs = retriever.invoke(prompt_question)
        context_docs = retrieved_docs[: max(max_context_chunks, 1)]
        citations = _build_citations(context_docs)
        context = _format_context(citations)

        if not citations:
            answer_text = "【结论】\n当前未检索到可用片段，LangChain 演示链路暂时无法基于知识库回答。\n\n【关键依据】\n未召回到有效文档。\n\n【建议动作】\n请先补充语料或扩大检索范围后再提问。"
        else:
            # Prompt：显式规定结构化输出，避免展示时回答风格波动太大。
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是工艺知识问答助手。只能基于提供的参考片段回答。请严格输出三个小节：\n【结论】\n【关键依据】\n【建议动作】\n并用 [1][2] 标注来源，不要输出额外客套话。",
                    ),
                    (
                        "human",
                        "问题：{question}\n\n参考片段：\n{context}\n\n请按以下格式输出：\n【结论】\n用 2-4 句回答核心判断。\n\n【关键依据】\n逐条列出依据，并在句尾标注引用编号。\n\n【建议动作】\n给出 2-4 条现场建议动作。",
                    ),
                ]
            )
            # LangChain 风
            chain = prompt | _build_chat_model(temperature) | StrOutputParser()
            answer_text = _format_answer_text(
                str(chain.invoke({"question": prompt_question, "context": context})).strip()
            )

        return {
            "engine": "langchain_demo",
            "engine_label": "LangChain Demo",
            "model": _chat_model_name(),
            "question": prompt_question,
            "answer": answer_text,
            "sources": citations,
            "retrieval": {
                "strategy": "MMR",
                "top_k": top_k,
                "max_context_chunks": max_context_chunks,
                "fetch_k": max(top_k * 4, 12),
                "lambda_mult": lambda_mult,
                "why": "Use Max Marginal Relevance to reduce duplicated chunks from similar sections.",
            },
            "manifest": self._load_manifest(),
        }

    def is_ready(self) -> bool:
        return (
            (self.index_dir / self.MANIFEST_NAME).exists()
            and (self.index_dir / f"{self.INDEX_NAME}.faiss").exists()
            and (self.index_dir / f"{self.INDEX_NAME}.pkl").exists()
        )

    def _load_manifest(self) -> dict[str, Any]:
        path = self.index_dir / self.MANIFEST_NAME
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


def _build_embeddings() -> OpenAIEmbeddings:
    """创建 LangChain Embeddings 客户端。"""
    load_project_env()
    provider = os.getenv("PROCESS_ASSISTANT_MODEL_PROVIDER", "aliyun_dashscope").strip().lower()
    return OpenAIEmbeddings(
        model=_embed_model_name(),
        api_key=_api_key(),
        base_url=_base_url(),
        request_timeout=int(os.getenv("PROCESS_ASSISTANT_API_TIMEOUT_SEC", "60")),
        check_embedding_ctx_length=False,
        chunk_size=10 if provider == "aliyun_dashscope" else 128,
    )


def _build_chat_model(temperature: float) -> ChatOpenAI:
    """创建 LangChain Chat 模型客户端。"""
    load_project_env()
    return ChatOpenAI(
        model=_chat_model_name(),
        api_key=_api_key(),
        base_url=_base_url(),
        temperature=temperature,
        timeout=int(os.getenv("PROCESS_ASSISTANT_API_TIMEOUT_SEC", "60")),
    )


def _api_key() -> str:
    value = (
        os.getenv("PROCESS_ASSISTANT_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    if not value:
        raise RuntimeError("missing API key for LangChain demo")
    return value


def _base_url() -> str:
    configured = os.getenv("PROCESS_ASSISTANT_BASE_URL", "").strip()
    if configured:
        return configured
    provider = os.getenv("PROCESS_ASSISTANT_MODEL_PROVIDER", "aliyun_dashscope").strip().lower()
    if provider == "aliyun_dashscope":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "openai_compatible":
        return "https://api.openai.com/v1"
    raise RuntimeError(f"unsupported provider for LangChain demo: {provider}")


def _chat_model_name() -> str:
    provider = os.getenv("PROCESS_ASSISTANT_MODEL_PROVIDER", "aliyun_dashscope").strip().lower()
    default = "qwen-plus" if provider == "aliyun_dashscope" else "gpt-4o-mini"
    return os.getenv("PROCESS_ASSISTANT_CHAT_MODEL", default).strip()


def _embed_model_name() -> str:
    provider = os.getenv("PROCESS_ASSISTANT_MODEL_PROVIDER", "aliyun_dashscope").strip().lower()
    default = "text-embedding-v3" if provider == "aliyun_dashscope" else "text-embedding-3-small"
    return os.getenv("PROCESS_ASSISTANT_EMBED_MODEL", default).strip()


def _to_langchain_document(raw_doc: Any) -> LangChainDocument:
    """把项目内部文档对象转成 LangChain Document。"""
    metadata = {
        "doc_id": raw_doc.doc_id,
        "title": raw_doc.title,
        "source_path": raw_doc.source_path,
        "file_type": raw_doc.file_type,
        "last_modified": raw_doc.last_modified,
    }
    metadata.update(dict(raw_doc.metadata or {}))
    return LangChainDocument(page_content=_normalize_content(raw_doc.content), metadata=metadata)


def _normalize_content(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def _build_citations(documents: list[LangChainDocument]) -> list[dict[str, Any]]:
    """把召回片段整理成简洁的来源信息。"""
    citations: list[dict[str, Any]] = []
    for idx, doc in enumerate(documents, start=1):
        metadata = dict(doc.metadata or {})
        citations.append(
            {
                "ref_id": idx,
                "title": metadata.get("title") or Path(str(metadata.get("source_path", ""))).stem or "未命名文档",
                "source_path": metadata.get("source_path", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "chunk_index": metadata.get("chunk_index", idx),
                "quote": _clip(doc.page_content, 200),
            }
        )
    return citations


def _format_context(citations: list[dict[str, Any]]) -> str:
    """把来源片段拼成最终 Prompt 上下文。"""
    if not citations:
        return ""
    blocks: list[str] = []
    for item in citations:
        blocks.append(
            "\n".join(
                [
                    f"[{item['ref_id']}] {item['title']}",
                    f"来源路径: {item['source_path']}",
                    f"Chunk: {item['chunk_id'] or item['chunk_index']}",
                    item["quote"],
                ]
            )
        )
    return "\n\n".join(blocks)


def _clip(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _format_answer_text(text: str) -> str:
    """让 LangChain 版输出在展示层更稳定、易读。"""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    for marker in ("【结论】", "【关键依据】", "【建议动作】"):
        cleaned = cleaned.replace(marker, f"\n{marker}\n")
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    cleaned = "\n".join(line for line in cleaned.split("\n") if line.strip() or line == "")
    cleaned = cleaned.strip()
    if "【结论】" not in cleaned:
        cleaned = f"【结论】\n{cleaned}"
    return cleaned
