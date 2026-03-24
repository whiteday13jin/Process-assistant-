from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from statistics import median
from typing import Any, Dict, List, Tuple

from .excel_loader import load_process_times_from_excel
from .models import ProcessTime, stats_from_records


class OptimizationEngine:
    def analyze(self, excel_paths: List[str]) -> Dict[str, Any]:
        records: List[ProcessTime] = []
        for path in excel_paths:
            records.extend(load_process_times_from_excel(path))

        analyzable_records = [r for r in records if _is_core_process(r)]
        if not analyzable_records:
            return {
                "engine": "process_path_optimizer_v1",
                "records": 0,
                "process_statistics": [],
                "bottlenecks": [],
                "optimization_actions": [],
            }

        stats = self._compute_process_stats(analyzable_records)
        bottlenecks = self._detect_bottlenecks(analyzable_records)
        actions = self._suggest_actions(analyzable_records, bottlenecks)

        return {
            "engine": "process_path_optimizer_v1",
            "records": len(analyzable_records),
            "process_statistics": stats,
            "bottlenecks": bottlenecks[:25],
            "optimization_actions": actions,
        }

    def _compute_process_stats(self, records: List[ProcessTime]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str], List[ProcessTime]] = defaultdict(list)
        for record in records:
            grouped[(record.process_id, record.process_name)].append(record)

        result = []
        for (process_id, process_name), items in grouped.items():
            s = stats_from_records(process_id, process_name, items)
            result.append(asdict(s))

        result.sort(key=lambda x: (x["process_id"], x["process_name"]))
        return result

    def _detect_bottlenecks(self, records: List[ProcessTime]) -> List[Dict[str, Any]]:
        by_section: Dict[str, List[ProcessTime]] = defaultdict(list)
        for r in records:
            by_section[r.section].append(r)

        findings: List[Dict[str, Any]] = []
        for section, items in by_section.items():
            valid_output = [x.daily_output for x in items if x.daily_output is not None]
            section_median_output = median(valid_output) if valid_output else None

            for r in items:
                load_ratio = None
                if r.ect_sec and r.takt_sec and r.takt_sec > 0:
                    load_ratio = r.ect_sec / r.takt_sec

                low_output = False
                if section_median_output and r.daily_output:
                    low_output = r.daily_output < 0.9 * section_median_output

                quality_risk = bool(r.defect_rate and r.defect_rate >= 0.08)

                if not any([
                    load_ratio is not None and load_ratio > 1.05,
                    low_output,
                    quality_risk,
                ]):
                    continue

                severity = 0.0
                if load_ratio is not None:
                    severity += min(load_ratio, 2.0)
                if low_output:
                    severity += 0.8
                if quality_risk:
                    severity += 0.6

                findings.append(
                    {
                        "section": section,
                        "process_id": r.process_id,
                        "process_name": r.process_name,
                        "sequence": r.sequence,
                        "load_ratio": round(load_ratio, 4) if load_ratio is not None else None,
                        "daily_output": r.daily_output,
                        "defect_rate": r.defect_rate,
                        "severity": round(severity, 4),
                        "evidence": {
                            "ect_sec": r.ect_sec,
                            "takt_sec": r.takt_sec,
                            "median_output_section": section_median_output,
                            "extra_factor": r.extra_factor,
                        },
                    }
                )

        findings.sort(key=lambda x: x["severity"], reverse=True)
        return findings

    def _suggest_actions(self, records: List[ProcessTime], bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        record_map = {(r.section, r.process_name): r for r in records}
        actions: List[Dict[str, Any]] = []

        for b in bottlenecks[:12]:
            r = record_map.get((b["section"], b["process_name"]))
            if not r:
                continue

            if b["load_ratio"] is not None and b["load_ratio"] > 1.1:
                target_manpower = _estimate_target_manpower(r)
                actions.append(
                    {
                        "type": "parallel",
                        "priority": "high",
                        "process_name": r.process_name,
                        "action": f"将并行工位人力调整为至少{target_manpower:.2f}人（当前{(r.manpower or 0):.2f}人），目标ECT<=节拍。",
                        "reason": "ECT高于节拍，当前节拍受限。",
                        "traceability": b,
                    }
                )

            if r.extra_factor and r.extra_factor >= 1.15:
                actions.append(
                    {
                        "type": "merge",
                        "priority": "medium",
                        "process_name": r.process_name,
                        "action": "将准备动作前置到上游工位，缩短非生产时间系数。",
                        "reason": "额外非生产时间系数偏高。",
                        "traceability": b,
                    }
                )

            if r.defect_rate and r.defect_rate >= 0.08:
                actions.append(
                    {
                        "type": "substitute",
                        "priority": "high",
                        "process_name": r.process_name,
                        "action": "引入治具防错或自动检测替代人工复判，先做小批验证。",
                        "reason": "不良率超过8%，质量返工拉低产能。",
                        "traceability": b,
                    }
                )

            if r.note and "人等机" in r.note:
                actions.append(
                    {
                        "type": "parallel",
                        "priority": "medium",
                        "process_name": r.process_name,
                        "action": "将等待窗口切分给相邻可并行工序，执行跨工位一人多岗。",
                        "reason": "备注显示等待设备，存在并行窗口。",
                        "traceability": b,
                    }
                )

        actions.extend(self._find_merge_opportunities(records))
        actions.sort(key=lambda x: _priority_rank(x["priority"]))

        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in actions:
            key = (item["type"], item["process_name"], item["action"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        return unique

    def _find_merge_opportunities(self, records: List[ProcessTime]) -> List[Dict[str, Any]]:
        by_section: Dict[str, List[ProcessTime]] = defaultdict(list)
        for r in records:
            if r.sequence is None:
                continue
            by_section[r.section].append(r)

        suggestions: List[Dict[str, Any]] = []
        for section, items in by_section.items():
            ordered = sorted(items, key=lambda x: x.sequence or 0)
            for prev, curr in zip(ordered, ordered[1:]):
                if prev.ct_sec is None or curr.ct_sec is None:
                    continue
                if prev.ct_sec > 25 or curr.ct_sec > 25:
                    continue
                if prev.process_name == curr.process_name:
                    continue
                if abs((curr.sequence or 0) - (prev.sequence or 0)) > 1:
                    continue

                suggestions.append(
                    {
                        "type": "merge",
                        "priority": "low",
                        "process_name": f"{prev.process_name}+{curr.process_name}",
                        "action": f"在{section}尝试将相邻短节拍工序合并为单工位作业。",
                        "reason": "相邻工序均为短节拍，可降低搬运和切换损耗。",
                        "traceability": {
                            "section": section,
                            "ct_prev": prev.ct_sec,
                            "ct_curr": curr.ct_sec,
                            "sequence_prev": prev.sequence,
                            "sequence_curr": curr.sequence,
                        },
                    }
                )

        return suggestions


def _estimate_target_manpower(record: ProcessTime) -> float:
    if record.ect_sec is None or record.takt_sec is None or record.takt_sec == 0:
        return 1.0
    theoretical = record.ect_sec / record.takt_sec
    current = record.manpower or 1.0
    return max(current, theoretical)


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 3)


def _is_core_process(record: ProcessTime) -> bool:
    if record.sequence is None or record.ct_sec is None:
        return False
    name = record.process_name.strip()
    noise_tokens = ["--", "员", "返工", "辅助", "物料", "出料", "刀模管理员", "前加工"]
    if any(tok in name for tok in noise_tokens):
        return False
    if name.startswith("*"):
        return False
    return True
