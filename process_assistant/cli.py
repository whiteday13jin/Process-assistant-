from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .diagnosis_engine import DiagnosisEngine
from .document_loader import load_documents_from_dir
from .embedder import build_client_from_env
from .env_loader import load_project_env
from .knowledge_base import append_feedback
from .models import FeedbackEvent
from .optimization_engine import OptimizationEngine
from .rag_evaluator import RagEvaluator
from .rag_pipeline import RagPipeline
from .text_splitter import TextSplitter
from .vector_index import LocalVectorIndex


def main() -> None:
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

    args = parser.parse_args()

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


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _resolve_question(raw_question: str | None, question_file: str | None) -> str:
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
    if hybrid_alpha is None:
        return vector_weight, lexical_weight, title_weight
    alpha = float(hybrid_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("hybrid-alpha must be within [0, 1]")
    remain = 1.0 - alpha
    # Keep backward behavior: all non-vector share remaining score.
    return alpha, remain * 0.67, remain * 0.33


def _emit(payload: Any, output: str | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        print(rendered)
    except UnicodeEncodeError:
        # Windows terminals may run in GBK; keep CLI usable by degrading display only.
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


def _default_rag_eval_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "rag_eval_dataset.json"


if __name__ == "__main__":
    main()

