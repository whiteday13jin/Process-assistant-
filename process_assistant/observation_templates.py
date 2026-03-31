from __future__ import annotations

"""工序观测模板加载层。

这层的目标是把“某个工序该让用户填写哪些字段、可选哪些症状”
从页面实现里拆出来，做成可维护的配置。
这样 Web 表单就不再是一个写死的大杂烩，而是围绕工序模板动态生成。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ObservationField:
    """单个观测字段定义。"""

    key: str
    label: str
    input_type: str
    parser: str
    placeholder: str = ""
    help_text: str = ""
    unit: str = ""
    options: List[str] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "input_type": self.input_type,
            "parser": self.parser,
            "placeholder": self.placeholder,
            "help_text": self.help_text,
            "unit": self.unit,
            "options": list(self.options),
        }


@dataclass(frozen=True)
class ProcessObservationTemplate:
    """某个工序的表单模板。"""

    process_id: str
    summary: str
    symptom_ids: List[str]
    fields: List[ObservationField]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "summary": self.summary,
            "symptom_ids": list(self.symptom_ids),
            "fields": [field.to_payload() for field in self.fields],
        }


def load_process_observation_templates(path: str | Path) -> Dict[str, ProcessObservationTemplate]:
    """从 JSON 配置中读取工序观测模板。"""

    with Path(path).open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    templates: Dict[str, ProcessObservationTemplate] = {}
    for process_id, item in raw.items():
        fields = [
            ObservationField(
                key=str(field["key"]),
                label=str(field["label"]),
                input_type=str(field["input_type"]),
                parser=str(field["parser"]),
                placeholder=str(field.get("placeholder", "")),
                help_text=str(field.get("help_text", "")),
                unit=str(field.get("unit", "")),
                options=[str(x) for x in field.get("options", [])],
            )
            for field in item.get("fields", [])
        ]
        templates[process_id] = ProcessObservationTemplate(
            process_id=process_id,
            summary=str(item.get("summary", "")),
            symptom_ids=[str(x) for x in item.get("symptom_ids", [])],
            fields=fields,
        )
    return templates
