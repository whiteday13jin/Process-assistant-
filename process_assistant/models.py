from __future__ import annotations

"""结构化决策层的数据模型。

这一组 dataclass 主要服务于原有的诊断、优化和反馈闭环，
作用是把 JSON、Excel 等松散输入收口成可校验、可推理的稳定对象。
如果把项目看成一条生产链，这里就是所有上游数据进入业务逻辑前的“统一语言”。
"""

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


class ValidationError(ValueError):
    """输入数据不符合要求时抛出，通常是缺字段、字段类型错误或值域错误。"""


@dataclass(frozen=True)
class Process:


    id: str
    route_id: str
    sequence: int
    name: str
    category: str
    stage: str


@dataclass(frozen=True)
class Symptom:
  

    id: str
    code: str
    name: str
    severity: int


@dataclass(frozen=True)
class Cause:


    id: str
    name: str
    category: str
    process_ids: List[str]
    symptom_ids: List[str]
    base_weight: float
    conditions: Dict[str, Dict[str, float | str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Solution:


    id: str
    cause_id: str
    actions: List[str]
    checkpoints: List[str]
    expected_minutes: int
    owner_role: str


@dataclass(frozen=True)
class Case:
    """历史案例，给诊断结果提供参考。"""

    id: str
    process_id: str
    symptom_ids: List[str]
    cause_id: str
    context: Dict[str, Any]
    success_score: float


@dataclass(frozen=True)
class ProcessTime:
    """从节拍分析表中提取出的单条工序统计记录。"""

    source_file: str
    sheet_name: str
    section: str
    process_id: str
    process_name: str
    sequence: Optional[int]
    ct_sec: Optional[float]
    defect_rate: Optional[float]
    extra_factor: Optional[float]
    ect_sec: Optional[float]
    hourly_capacity: Optional[float]
    takt_sec: Optional[float]
    manpower: Optional[float]
    daily_output: Optional[float]
    equipment: Optional[str]
    equipment_ratio: Optional[float]
    note: Optional[str]


@dataclass(frozen=True)
class FeedbackEvent:
    """现场执行后的反馈事件，用于校正知识库效果。"""

    request_id: str
    cause_id: str
    solution_id: str
    result: str
    created_at: str

    @staticmethod
    def from_payload(payload: Dict[str, Any]) -> "FeedbackEvent":
       

        required = {"request_id", "cause_id", "solution_id", "result"}
        allowed = required | {"created_at"}
        _validate_keys(payload, required, allowed, "feedback")
        result = str(payload["result"]).upper()
        if result not in {"SUCCESS", "FAIL"}:
            raise ValidationError("feedback.result must be SUCCESS or FAIL")
        return FeedbackEvent(
            request_id=str(payload["request_id"]),
            cause_id=str(payload["cause_id"]),
            solution_id=str(payload["solution_id"]),
            result=result,
            created_at=str(payload.get("created_at") or datetime.utcnow().isoformat()),
        )


@dataclass(frozen=True)
class DiagnosisRequest:
   

    request_id: str
    process_id: str
    symptom_ids: List[str]
    observed: Dict[str, Any]
    top_k: int = 3

    @staticmethod
    def from_payload(payload: Dict[str, Any]) -> "DiagnosisRequest":
        """校验并标准化诊断请求。"""

        required = {"request_id", "process_id", "symptom_ids", "observed"}
        allowed = required | {"top_k"}
        _validate_keys(payload, required, allowed, "diagnosis_request")

        symptom_ids = payload["symptom_ids"]
        if not isinstance(symptom_ids, list) or not symptom_ids:
            raise ValidationError("symptom_ids must be a non-empty list[str]")
        if any(not isinstance(x, str) for x in symptom_ids):
            raise ValidationError("symptom_ids must contain only strings")
        observed = payload["observed"]
        if not isinstance(observed, dict):
            raise ValidationError("observed must be an object")
        top_k = int(payload.get("top_k", 3))
        if top_k < 1 or top_k > 10:
            raise ValidationError("top_k must be in [1, 10]")

        return DiagnosisRequest(
            request_id=str(payload["request_id"]),
            process_id=str(payload["process_id"]),
            symptom_ids=symptom_ids,
            observed=observed,
            top_k=top_k,
        )


@dataclass(frozen=True)
class CauseCandidate:
    """诊断排序阶段的中间结果，记录一个候选原因及其证据轨迹。"""

    cause_id: str
    score: float
    traces: List[Dict[str, Any]]


@dataclass(frozen=True)
class ProcessStats:
    """面向优化分析的工序统计摘要。"""

    process_id: str
    process_name: str
    samples: int
    ct_avg: Optional[float]
    ct_max: Optional[float]
    ct_std: Optional[float]
    ect_avg: Optional[float]
    ect_max: Optional[float]
    ect_std: Optional[float]


def stats_from_records(process_id: str, process_name: str, records: List[ProcessTime]) -> ProcessStats:
    """把同一工序的多条时序记录汇总成统计特征。"""

    ct_values = [x.ct_sec for x in records if x.ct_sec is not None]
    ect_values = [x.ect_sec for x in records if x.ect_sec is not None]

    return ProcessStats(
        process_id=process_id,
        process_name=process_name,
        samples=len(records),
        ct_avg=mean(ct_values) if ct_values else None,
        ct_max=max(ct_values) if ct_values else None,
        ct_std=pstdev(ct_values) if len(ct_values) > 1 else 0.0 if ct_values else None,
        ect_avg=mean(ect_values) if ect_values else None,
        ect_max=max(ect_values) if ect_values else None,
        ect_std=pstdev(ect_values) if len(ect_values) > 1 else 0.0 if ect_values else None,
    )


def _validate_keys(payload: Dict[str, Any], required: set[str], allowed: set[str], obj_name: str) -> None:
    """统一检查缺字段和多字段，避免每个模型重复写样板校验。"""

    missing = required - payload.keys()
    if missing:
        raise ValidationError(f"{obj_name}: missing keys {sorted(missing)}")
    extra = payload.keys() - allowed
    if extra:
        raise ValidationError(f"{obj_name}: unknown keys {sorted(extra)}")
