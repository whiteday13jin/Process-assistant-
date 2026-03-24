from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from .knowledge_base import KnowledgeBase, load_feedback_effectiveness, load_knowledge_base
from .models import CauseCandidate, DiagnosisRequest, ValidationError


class DiagnosisEngine:
    def __init__(self, knowledge_base_path: str, feedback_log_path: str) -> None:
        self.kb: KnowledgeBase = load_knowledge_base(knowledge_base_path)
        self.feedback_log_path = feedback_log_path

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["request_id", "process_id", "symptom_ids", "observed"],
            "additionalProperties": False,
            "properties": {
                "request_id": {"type": "string"},
                "process_id": {"type": "string"},
                "symptom_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                "observed": {
                    "type": "object",
                    "additionalProperties": {
                        "oneOf": [{"type": "number"}, {"type": "string"}, {"type": "integer"}]
                    },
                },
            },
        }

    def diagnose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = DiagnosisRequest.from_payload(payload)

        if req.process_id not in self.kb.processes:
            raise ValidationError(f"unknown process_id: {req.process_id}")
        unknown_symptoms = [x for x in req.symptom_ids if x not in self.kb.symptoms]
        if unknown_symptoms:
            raise ValidationError(f"unknown symptom_ids: {unknown_symptoms}")

        cause_scores: Dict[str, float] = defaultdict(float)
        traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        feedback_effectiveness = load_feedback_effectiveness(self.feedback_log_path)

        self._score_by_rules(req, cause_scores, traces)
        self._score_by_cases(req, cause_scores, traces)
        self._score_by_cause_profile(req, cause_scores, traces)

        for cause_id, score in list(cause_scores.items()):
            base_weight = self.kb.causes[cause_id].base_weight
            effect = feedback_effectiveness.get(cause_id)
            effect_multiplier = effect.score if effect else 0.6
            cause_scores[cause_id] = score * base_weight * effect_multiplier
            traces[cause_id].append(
                {
                    "source": "feedback",
                    "effectiveness": round(effect_multiplier, 4),
                    "success": effect.success if effect else 0,
                    "fail": effect.fail if effect else 0,
                }
            )

        ranked = sorted(cause_scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: req.top_k]

        candidates = [
            CauseCandidate(cause_id=cause_id, score=score, traces=traces[cause_id])
            for cause_id, score in top
        ]

        action_plans: List[Dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            for sol in self.kb.solutions_by_cause.get(candidate.cause_id, []):
                action_plans.append(
                    {
                        "rank": rank,
                        "cause_id": candidate.cause_id,
                        "cause_name": self.kb.causes[candidate.cause_id].name,
                        "solution_id": sol.id,
                        "owner_role": sol.owner_role,
                        "expected_minutes": sol.expected_minutes,
                        "actions": sol.actions,
                        "checkpoints": sol.checkpoints,
                        "traceability": candidate.traces,
                        "score": round(candidate.score, 4),
                    }
                )

        return {
            "request_id": req.request_id,
            "engine": "diagnosis_rule_case_ranker_v1",
            "mode": "action_only",
            "input_schema": self.input_schema(),
            "recommended_actions": action_plans,
        }

    def _score_by_rules(
        self,
        req: DiagnosisRequest,
        cause_scores: Dict[str, float],
        traces: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        request_sym = set(req.symptom_ids)

        for rule in self.kb.rules:
            if req.process_id not in rule.process_ids:
                continue
            overlap = request_sym.intersection(rule.symptom_ids)
            if not overlap:
                continue

            symptom_match = len(overlap) / max(len(rule.symptom_ids), 1)
            score = rule.weight * (0.6 + 0.4 * symptom_match)

            cause_scores[rule.cause_id] += score
            traces[rule.cause_id].append(
                {
                    "source": "rule",
                    "rule_id": rule.id,
                    "matched_symptoms": sorted(overlap),
                    "score": round(score, 4),
                }
            )

    def _score_by_cases(
        self,
        req: DiagnosisRequest,
        cause_scores: Dict[str, float],
        traces: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        req_sym = set(req.symptom_ids)

        for case in self.kb.cases:
            symptom_similarity = self._jaccard(req_sym, set(case.symptom_ids))
            if symptom_similarity == 0:
                continue
            process_similarity = 1.0 if case.process_id == req.process_id else 0.0
            context_similarity = self._context_similarity(req.observed, case.context)

            score = (0.55 * symptom_similarity + 0.3 * process_similarity + 0.15 * context_similarity) * case.success_score
            if score < 0.25:
                continue

            cause_scores[case.cause_id] += score
            traces[case.cause_id].append(
                {
                    "source": "case",
                    "case_id": case.id,
                    "symptom_similarity": round(symptom_similarity, 4),
                    "process_similarity": round(process_similarity, 4),
                    "context_similarity": round(context_similarity, 4),
                    "score": round(score, 4),
                }
            )

    def _score_by_cause_profile(
        self,
        req: DiagnosisRequest,
        cause_scores: Dict[str, float],
        traces: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        req_sym = set(req.symptom_ids)

        for cause in self.kb.causes.values():
            if req.process_id not in cause.process_ids:
                continue
            overlap = req_sym.intersection(cause.symptom_ids)
            if not overlap:
                continue
            symptom_match = len(overlap) / len(cause.symptom_ids)
            condition_multiplier = self._evaluate_conditions(req.observed, cause.conditions)
            score = 0.65 * symptom_match * condition_multiplier

            cause_scores[cause.id] += score
            traces[cause.id].append(
                {
                    "source": "cause_profile",
                    "matched_symptoms": sorted(overlap),
                    "condition_multiplier": round(condition_multiplier, 4),
                    "score": round(score, 4),
                }
            )

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        inter = len(a.intersection(b))
        union = len(a.union(b))
        return inter / union if union else 0.0

    @staticmethod
    def _context_similarity(observed: Dict[str, Any], sample: Dict[str, Any]) -> float:
        if not sample:
            return 0.6
        scores: List[float] = []
        for key, sample_value in sample.items():
            if key not in observed:
                continue
            obs_value = observed[key]
            if isinstance(sample_value, (int, float)) and isinstance(obs_value, (int, float)):
                den = max(abs(float(sample_value)), 1.0)
                diff = abs(float(obs_value) - float(sample_value)) / den
                scores.append(max(0.0, 1.0 - min(diff, 1.0)))
            else:
                scores.append(1.0 if str(obs_value) == str(sample_value) else 0.0)
        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    @staticmethod
    def _evaluate_conditions(observed: Dict[str, Any], conditions: Dict[str, Dict[str, float | str]]) -> float:
        if not conditions:
            return 1.0

        total = 0.0
        for key, cond in conditions.items():
            op = cond.get("op")
            value = cond.get("value")
            if key not in observed:
                total += 0.8
                continue
            obs = observed[key]
            passed = _compare(obs, op, value)
            total += 1.15 if passed else 0.6
        return total / len(conditions)


def _compare(obs: Any, op: Any, value: Any) -> bool:
    if op not in {">", ">=", "<", "<=", "=="}:
        return False
    if isinstance(obs, (int, float)) and isinstance(value, (int, float)):
        lhs = float(obs)
        rhs = float(value)
    else:
        lhs = str(obs)
        rhs = str(value)

    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    return lhs == rhs
