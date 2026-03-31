from __future__ import annotations

"""命令行入口。

这个文件是整个项目最直接的运行起点之一。用户在终端里输入的每个子命令，
都会先进入这里，再被分流到诊断、优化、自研 RAG 或 LangChain 对照版。

可以把它理解成“项目总路由表”：

1. 先加载 `.env`
2. 再注册所有子命令和参数
3. 根据 `args.cmd` 选择对应功能链路
4. 最后统一输出结果
"""

import argparse
import json
from pathlib import Path
from typing import Any

from .diagnosis_engine import DiagnosisEngine
from .document_loader import load_documents_from_dir
from .embedder import build_client_from_env
from .env_loader import load_project_env
from .knowledge_base import append_feedback
from .langchain_demo import LangChainRagDemo
from .models import FeedbackEvent
from .optimization_engine import OptimizationEngine
from .rag_evaluator import RagEvaluator
from .rag_pipeline import RagPipeline
from .text_splitter import TextSplitter
from .vector_index import LocalVectorIndex


def main() -> None:
    """CLI 总入口。
   只负责：收集参数、选择功能、调用对应引擎、输出 JSON 结果。不直接实现业务。
    """
    load_project_env()
    parser = argparse.ArgumentParser(description="Process knowledge driven decision assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_diag = sub.add_parser("diagnose", help="Run structured anomaly diagnosis")
    p_diag.add_argument("--input", required=True, help="Path to diagnosis request JSON")
    p_diag.add_argument("--output", help="Optional output JSON path")
    p_diag.add_argument("--kb", default=str(_default_kb_path()), help="Knowledge base JSON path")
    p_diag.add_argument("--feedback-log", default=str(_default_feedback_path()), help="Feedback JSONL path")

    p_opt = sub.add_parser("optimize", help="Run process time bottleneck analysis")
    p_opt.add_argument("--excel", nargs="+", required=True, help="Excel files with process time records")
    p_opt.add_argument("--output", help="Optional output JSON path")

    p_fb = sub.add_parser("feedback", help="Write feedback event")
    p_fb.add_argument("--input", required=True, help="Path to feedback event JSON")
    p_fb.add_argument("--feedback-log", default=str(_default_feedback_path()), help="Feedback JSONL path")

    p_rag_build = sub.add_parser("rag-build", help="Build local RAG vector index")
    p_rag_build.add_argument("--docs-dir", default=str(_default_rag_docs_path()), help="RAG docs directory")
    p_rag_build.add_argument("--index-dir", default=str(_default_rag_index_path()), help="RAG index directory")
    p_rag_build.add_argument("--chunk-size", type=int, default=700, help="Chunk size in characters")
    p_rag_build.add_argument("--chunk-overlap", type=int, default=100, help="Overlap size in characters")
    p_rag_build.add_argument("--min-chunk-size", type=int, default=120, help="Minimum chunk size")
    p_rag_build.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    p_rag_build.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recursively scan docs dir",
    )
    p_rag_build.add_argument("--output", help="Optional output JSON path")

    p_rag_ask = sub.add_parser("rag-ask", help="Ask question with indexed RAG knowledge")
    p_rag_ask.add_argument("--index-dir", default=str(_default_rag_index_path()), help="RAG index directory")
    p_rag_ask.add_argument("--question", help="Question text")
    p_rag_ask.add_argument("--question-file", help="JSON file containing question")
    p_rag_ask.add_argument("--top-k", type=int, default=4, help="Top-k retrieved chunks")
    p_rag_ask.add_argument("--max-context-chunks", type=int, default=4, help="Max chunks for LLM context")
    p_rag_ask.add_argument("--hybrid-alpha", type=float, help="Legacy vector weight, kept for compatibility")
    p_rag_ask.add_argument("--vector-weight", type=float, default=0.70, help="Vector score weight")
    p_rag_ask.add_argument("--lexical-weight", type=float, default=0.20, help="Lexical score weight")
    p_rag_ask.add_argument("--title-weight", type=float, default=0.10, help="Title/heading score weight")
    p_rag_ask.add_argument("--max-per-doc", type=int, default=2, help="Max retrieved chunks per document")
    p_rag_ask.add_argument("--min-final-score", type=float, default=0.42, help="Evidence gate final score threshold")
    p_rag_ask.add_argument("--min-lexical-score", type=float, default=0.02, help="Evidence gate lexical score threshold")
    p_rag_ask.add_argument("--min-title-score", type=float, default=0.02, help="Evidence gate title score threshold")
    p_rag_ask.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    p_rag_ask.add_argument("--output", help="Optional output JSON path")

    p_rag_eval = sub.add_parser("rag-eval", help="Evaluate RAG results on dataset")
    p_rag_eval.add_argument("--index-dir", default=str(_default_rag_index_path()), help="RAG index directory")
    p_rag_eval.add_argument(
        "--dataset",
        default=str(_default_rag_eval_dataset_path()),
        help="Evaluation dataset JSON",
    )
    p_rag_eval.add_argument("--top-k", type=int, default=4, help="Top-k retrieved chunks")
    p_rag_eval.add_argument("--max-context-chunks", type=int, default=4, help="Max chunks for LLM context")
    p_rag_eval.add_argument("--hybrid-alpha", type=float, help="Legacy vector weight, kept for compatibility")
    p_rag_eval.add_argument("--vector-weight", type=float, default=0.70, help="Vector score weight")
    p_rag_eval.add_argument("--lexical-weight", type=float, default=0.20, help="Lexical score weight")
    p_rag_eval.add_argument("--title-weight", type=float, default=0.10, help="Title/heading score weight")
    p_rag_eval.add_argument("--max-per-doc", type=int, default=2, help="Max retrieved chunks per document")
    p_rag_eval.add_argument("--min-final-score", type=float, default=0.42, help="Evidence gate final score threshold")
    p_rag_eval.add_argument("--min-lexical-score", type=float, default=0.02, help="Evidence gate lexical score threshold")
    p_rag_eval.add_argument("--min-title-score", type=float, default=0.02, help="Evidence gate title score threshold")
    p_rag_eval.add_argument("--output", help="Optional output JSON path")

    p_rag_build_lc = sub.add_parser("rag-build-lc", help="Build LangChain demo vector index")
    p_rag_build_lc.add_argument("--docs-dir", default=str(_default_rag_docs_path()), help="RAG docs directory")
    p_rag_build_lc.add_argument("--index-dir", default=str(_default_rag_lc_index_path()), help="LangChain demo index directory")
    p_rag_build_lc.add_argument("--chunk-size", type=int, default=700, help="Chunk size in characters")
    p_rag_build_lc.add_argument("--chunk-overlap", type=int, default=100, help="Overlap size in characters")
    p_rag_build_lc.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recursively scan docs dir",
    )
    p_rag_build_lc.add_argument("--output", help="Optional output JSON path")

    p_rag_ask_lc = sub.add_parser("rag-ask-lc", help="Ask question with LangChain demo RAG")
    p_rag_ask_lc.add_argument("--index-dir", default=str(_default_rag_lc_index_path()), help="LangChain demo index directory")
    p_rag_ask_lc.add_argument("--question", help="Question text")
    p_rag_ask_lc.add_argument("--question-file", help="JSON file containing question")
    p_rag_ask_lc.add_argument("--top-k", type=int, default=4, help="Top-k retrieved chunks")
    p_rag_ask_lc.add_argument("--max-context-chunks", type=int, default=4, help="Max chunks for LLM context")
    p_rag_ask_lc.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    p_rag_ask_lc.add_argument("--lambda-mult", type=float, default=0.55, help="MMR diversity factor")
    p_rag_ask_lc.add_argument("--output", help="Optional output JSON path")

    args = parser.parse_args()

    # 下面这组分支就是 CLI 的主运行路径。
    if args.cmd == "diagnose":
        payload = _load_json(args.input)
        engine = DiagnosisEngine(args.kb, args.feedback_log)
        result = engine.diagnose(payload)
        _emit(result, args.output)
        return

    if args.cmd == "optimize":
        engine = OptimizationEngine()
        result = engine.analyze(args.excel)
        _emit(result, args.output)
        return

    if args.cmd == "feedback":
        payload = _load_json(args.input)
        event = FeedbackEvent.from_payload(payload)
        append_feedback(args.feedback_log, event)
        _emit({"status": "ok", "written": event.__dict__}, None)
        return

    if args.cmd == "rag-build":
        # 自研 RAG 建库顺序：文档加载 -> 切分 -> 向量化 -> 落盘索引
        docs = load_documents_from_dir(args.docs_dir, recursive=not args.no_recursive)
        splitter = TextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_size=args.min_chunk_size,
        )
        chunks = splitter.split_documents(docs)
        client = build_client_from_env(require_chat_model=False)
        index = LocalVectorIndex.build(
            chunks=chunks,
            embedder=client,
            index_dir=args.index_dir,
            batch_size=args.batch_size,
            metadata={
                "docs_dir": str(Path(args.docs_dir).resolve()),
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "min_chunk_size": args.min_chunk_size,
            },
        )
        _emit(
            {
                "status": "ok",
                "command": "rag-build",
                "docs": len(docs),
                "chunks": len(chunks),
                "index_dir": str(Path(args.index_dir).resolve()),
                "manifest": index.manifest,
            },
            args.output,
        )
        return

    if args.cmd == "rag-ask":
        # 自研 RAG 问答顺序：取问题 -> 读索引 -> 检索/重排/门控 -> 生成答案
        question = _resolve_question(args.question, args.question_file)
        vector_weight, lexical_weight, title_weight = _resolve_retrieval_weights(
            args.vector_weight,
            args.lexical_weight,
            args.title_weight,
            args.hybrid_alpha,
        )
        client = build_client_from_env(require_chat_model=True)
        index = LocalVectorIndex.load(args.index_dir)
        pipeline = RagPipeline(index=index, client=client)
        answer = pipeline.ask(
            question,
            top_k=args.top_k,
            max_context_chunks=args.max_context_chunks,
            vector_weight=vector_weight,
            lexical_weight=lexical_weight,
            title_weight=title_weight,
            max_per_doc=args.max_per_doc,
            min_final_score=args.min_final_score,
            min_lexical_score=args.min_lexical_score,
            min_title_score=args.min_title_score,
            temperature=args.temperature,
        )
        _emit(RagPipeline.to_payload(answer), args.output)
        return

    if args.cmd == "rag-eval":
        # 评测复用自研 RAG 主链路，只是把问题集批量跑一遍并统计指标。
        dataset = _load_json(args.dataset)
        if not isinstance(dataset, list):
            raise ValueError("rag-eval dataset must be a JSON array")
        vector_weight, lexical_weight, title_weight = _resolve_retrieval_weights(
            args.vector_weight,
            args.lexical_weight,
            args.title_weight,
            args.hybrid_alpha,
        )
        client = build_client_from_env(require_chat_model=True)
        index = LocalVectorIndex.load(args.index_dir)
        pipeline = RagPipeline(index=index, client=client)
        evaluator = RagEvaluator(pipeline)
        report = evaluator.evaluate(
            dataset,
            top_k=args.top_k,
            max_context_chunks=args.max_context_chunks,
            vector_weight=vector_weight,
            lexical_weight=lexical_weight,
            title_weight=title_weight,
            max_per_doc=args.max_per_doc,
            min_final_score=args.min_final_score,
            min_lexical_score=args.min_lexical_score,
            min_title_score=args.min_title_score,
        )
        _emit(report, args.output)
        return

    if args.cmd == "rag-build-lc":
        # LangChain 对照版刻意独立建库，避免和主干 RAG 混在一起。
        manifest = LangChainRagDemo.build(
            args.docs_dir,
            args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            recursive=not args.no_recursive,
        )
        _emit(
            {
                "status": "ok",
                "command": "rag-build-lc",
                "index_dir": str(Path(args.index_dir).resolve()),
                "manifest": manifest,
            },
            args.output,
        )
        return

    if args.cmd == "rag-ask-lc":
        # LangChain 对照版问答：Retriever -> Prompt -> LLM
        question = _resolve_question(args.question, args.question_file)
        demo = LangChainRagDemo(args.index_dir)
        if not demo.is_ready():
            raise RuntimeError(f"LangChain demo index is not ready: {Path(args.index_dir).resolve()}")
        payload = demo.ask(
            question,
            top_k=args.top_k,
            max_context_chunks=args.max_context_chunks,
            temperature=args.temperature,
            lambda_mult=args.lambda_mult,
        )
        _emit(payload, args.output)
        return


def _load_json(path: str) -> Any:
    """统一读取 JSON，兼容 Windows 常见的 UTF-8 BOM 文件。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _resolve_question(raw_question: str | None, question_file: str | None) -> str:
    """统一解析问题输入。

    支持直接命令行传问题，也支持从 JSON 文件读取问题。
    """
    question = (raw_question or "").strip()
    if question:
        return question
    if question_file:
        payload = _load_json(question_file)
        if isinstance(payload, dict) and payload.get("question"):
            return str(payload["question"]).strip()
        if isinstance(payload, str):
            return payload.strip()
        raise ValueError("question file must contain JSON object with `question` or plain JSON string")
    raise ValueError("missing question. pass --question or --question-file")


def _resolve_retrieval_weights(
    vector_weight: float,
    lexical_weight: float,
    title_weight: float,
    hybrid_alpha: float | None,
) -> tuple[float, float, float]:
    """兼容旧版检索参数。

    早期只有一个 `hybrid-alpha`，现在拆成了更可解释的三个权重：
    vector / lexical / title。
    """
    if hybrid_alpha is None:
        return vector_weight, lexical_weight, title_weight
    alpha = float(hybrid_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("hybrid-alpha must be within [0, 1]")
    remain = 1.0 - alpha
    # 保持旧行为：非向量分共同瓜分剩余权重，只是现在显式拆成两类。
    return alpha, remain * 0.67, remain * 0.33


def _emit(payload: Any, output: str | None) -> None:
    """统一输出结果到终端或文件。

    这里故意做成单一出口，便于保持所有命令输出风格一致，
    也方便集中兼容 Windows 终端编码差异。
    """
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(rendered)
    except UnicodeEncodeError:
        # Windows 终端可能运行在 GBK 环境中，这里只降级显示，不影响真正写出的 UTF-8 文件。
        safe = rendered.encode("gbk", errors="replace").decode("gbk")
        print(safe)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8-sig") as f:
            f.write(rendered + "\n")


def _default_kb_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"


def _default_feedback_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "feedback_log.jsonl"


def _default_rag_docs_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "rag_docs"


def _default_rag_index_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "rag_index"


def _default_rag_lc_index_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "rag_index_langchain"


def _default_rag_eval_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "rag_eval_dataset.json"


if __name__ == "__main__":
    main()

