from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .models import Case, Cause, FeedbackEvent, Process, Solution, Symptom


@dataclass(frozen=True)
class Rule:
    id: str
    process_ids: List[str]
    symptom_ids: List[str]
    cause_id: str
    weight: float


@dataclass
class KnowledgeBase:
    processes: Dict[str, Process]
    symptoms: Dict[str, Symptom]
    causes: Dict[str, Cause]
    solutions_by_cause: Dict[str, List[Solution]]
    cases: List[Case]
    rules: List[Rule]
    route: Dict[str, Any]


@dataclass(frozen=True)
class CauseEffectiveness:
    cause_id: str
    success: int
    fail: int

    @property
    def score(self) -> float:
        # Laplace smoothing keeps low-sample causes from dominating.
        return (self.success + 1) / (self.success + self.fail + 2)


def load_knowledge_base(path: str | Path) -> KnowledgeBase:
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    processes = {x["id"]: Process(**x) for x in raw["processes"]}
    symptoms = {x["id"]: Symptom(**x) for x in raw["symptoms"]}
    causes = {x["id"]: Cause(**x) for x in raw["causes"]}

    solutions_by_cause: Dict[str, List[Solution]] = defaultdict(list)
    for item in raw["solutions"]:
        sol = Solution(**item)
        solutions_by_cause[sol.cause_id].append(sol)

    cases = [Case(**x) for x in raw["cases"]]
    rules = [Rule(**x) for x in raw["rules"]]

    return KnowledgeBase(
        processes=processes,
        symptoms=symptoms,
        causes=causes,
        solutions_by_cause=dict(solutions_by_cause),
        cases=cases,
        rules=rules,
        route=raw.get("route", {}),
    )


def load_feedback_effectiveness(path: str | Path) -> Dict[str, CauseEffectiveness]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "fail": 0})
    p = Path(path)
    if not p.exists():
        return {}

    with p.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            event = FeedbackEvent.from_payload(payload)
            bucket = stats[event.cause_id]
            if event.result == "SUCCESS":
                bucket["success"] += 1
            else:
                bucket["fail"] += 1

    return {
        cause_id: CauseEffectiveness(cause_id=cause_id, success=v["success"], fail=v["fail"])
        for cause_id, v in stats.items()
    }


def append_feedback(path: str | Path, event: FeedbackEvent) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8-sig") as f:
        f.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")

