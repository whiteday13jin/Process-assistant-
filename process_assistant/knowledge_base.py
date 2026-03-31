from __future__ import annotations

"""知识库加载与反馈持久化层。

这个模块服务于原有结构化决策链路，负责把 JSON 知识库和反馈日志读成内存对象。
它本身不做诊断推理，而是给诊断引擎提供“规则、案例、原因、方案、反馈效果”这些基础材料。
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .models import Case, Cause, FeedbackEvent, Process, Solution, Symptom


@dataclass(frozen=True)
class Rule:
    """显式规则，适合表达“某工序 + 某症状 -> 某原因”的确定性知识。"""

    id: str
    process_ids: List[str]
    symptom_ids: List[str]
    cause_id: str
    weight: float


@dataclass
class KnowledgeBase:
    """结构化知识库聚合对象，是诊断引擎启动时加载的主数据。"""

    processes: Dict[str, Process]
    symptoms: Dict[str, Symptom]
    causes: Dict[str, Cause]
    solutions_by_cause: Dict[str, List[Solution]]
    cases: List[Case]
    rules: List[Rule]
    route: Dict[str, Any]


@dataclass(frozen=True)
class CauseEffectiveness:
    """某个原因在历史反馈中的效果统计。"""

    cause_id: str
    success: int
    fail: int

    @property
    def score(self) -> float:
        # 拉普拉斯平滑可以防止样本极少的原因因为一次成功就被过度放大。
        return (self.success + 1) / (self.success + self.fail + 2)


def load_knowledge_base(path: str | Path) -> KnowledgeBase:
    """读取主知识库 JSON，并拆成诊断引擎便于直接使用的映射结构。"""

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
    """从反馈日志中统计每个原因的成功/失败次数，用于诊断结果现实校正。"""

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
    """把一次现场反馈追加写入日志，供后续统计与回放。"""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8-sig") as f:
        f.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")
