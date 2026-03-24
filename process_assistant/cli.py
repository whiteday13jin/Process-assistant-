from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .diagnosis_engine import DiagnosisEngine
from .knowledge_base import append_feedback
from .models import FeedbackEvent
from .optimization_engine import OptimizationEngine


def main() -> None:
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


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _emit(payload: Dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8-sig") as f:
            f.write(rendered + "\n")


def _default_kb_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"


def _default_feedback_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "feedback_log.jsonl"


if __name__ == "__main__":
    main()

