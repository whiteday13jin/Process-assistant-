from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


class ValidationError(ValueError):
    """Raised when an input payload violates schema constraints."""


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
    id: str
    process_id: str
    symptom_ids: List[str]
    cause_id: str
    context: Dict[str, Any]
    success_score: float


@dataclass(frozen=True)
class ProcessTime:
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
    cause_id: str
    score: float
    traces: List[Dict[str, Any]]


@dataclass(frozen=True)
class ProcessStats:
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
    missing = required - payload.keys()
    if missing:
        raise ValidationError(f"{obj_name}: missing keys {sorted(missing)}")
    extra = payload.keys() - allowed
    if extra:
        raise ValidationError(f"{obj_name}: unknown keys {sorted(extra)}")
