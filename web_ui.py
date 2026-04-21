from __future__ import annotations

"""本地 Web 入口。
负责：
1. 接收页面表单输入
2. 做少量现场友好的参数清洗
3. 调用诊断 / 优化 / RAG 对应链路
4. 把结果渲染成页面并写入历史报告
"""

import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, abort, redirect, render_template, request, session, url_for

from process_assistant.diagnosis_engine import DiagnosisEngine
from process_assistant.document_loader import load_documents_from_dir
from process_assistant.embedder import build_client_from_env
from process_assistant.env_loader import load_project_env
from process_assistant.langchain_demo import LangChainRagDemo
from process_assistant.observation_templates import load_process_observation_templates
from process_assistant.optimization_engine import OptimizationEngine
from process_assistant.rag_pipeline import RagPipeline
from process_assistant.text_splitter import TextSplitter
from process_assistant.vector_index import LocalVectorIndex

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
REFERENCE_DIR = BASE_DIR / "reference"
KB_PATH = DATA_DIR / "knowledge_base.json"
OBS_TEMPLATE_PATH = DATA_DIR / "process_observation_templates.json"
FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.jsonl"
RAG_DOCS_DIR = DATA_DIR / "rag_docs"
RAG_INDEX_DIR = DATA_DIR / "rag_index"
RAG_LC_INDEX_DIR = DATA_DIR / "rag_index_langchain"

LOGIN_USERNAME = "user"
LOGIN_PASSWORD = "123"

load_project_env()
app = Flask(__name__)
app.secret_key = os.getenv("PROCESS_ASSISTANT_SECRET_KEY", "dev-only-change-me")


@lru_cache(maxsize=1)
def _diagnosis_engine() -> DiagnosisEngine:
    """延迟创建诊断引擎，保持入口文件轻量。"""
    return DiagnosisEngine(str(KB_PATH), str(FEEDBACK_LOG_PATH))


def _optimization_engine() -> OptimizationEngine:
    return OptimizationEngine()


def _safe_float(raw: str | None) -> float | None:
    """把表单字符串稳妥地转成数字。"""
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() == "unknown":
        return None
    if value.endswith("%"):
        value = value[:-1]
        return float(value) / 100.0
    return float(value)


@lru_cache(maxsize=1)
def _observation_templates() -> dict[str, Any]:
    """加载工序观测模板。"""
    return load_process_observation_templates(OBS_TEMPLATE_PATH)


def _diagnosis_ui_assets() -> dict[str, Any]:
    """把知识库与工序模板组装成前端可直接消费的页面资产。"""
    kb = _diagnosis_engine().kb
    templates = _observation_templates()
    processes = []
    for process in sorted(kb.processes.values(), key=lambda x: x.sequence):
        if process.id not in templates:
            continue
        processes.append(
            {
                "id": process.id,
                "name": process.name,
                "stage": process.stage,
                "category": process.category,
            }
        )

    symptoms = {
        sid: {"id": sid, "name": symptom.name, "severity": symptom.severity}
        for sid, symptom in kb.symptoms.items()
    }
    template_payload = {pid: template.to_payload() for pid, template in templates.items()}
    return {"processes": processes, "symptoms": symptoms, "templates": template_payload}


def _default_diagnosis_form_state() -> dict[str, Any]:
    assets = _diagnosis_ui_assets()
    first_process = assets["processes"][0]["id"] if assets["processes"] else ""
    return {
        "request_id": "",
        "process_id": first_process,
        "symptom_ids": [],
        "top_k": 3,
        "observed": {},
    }


def _render_diagnosis_form(error: str | None = None, form_state: dict[str, Any] | None = None):
    """统一渲染诊断表单，便于 GET 与 POST 失败回显复用。"""
    state = _default_diagnosis_form_state()
    if form_state:
        state.update(form_state)
    return render_template(
        "diagnose_form.html",
        role=session.get("role", "leader"),
        username=session.get("username", "user"),
        error=error,
        diagnosis_assets=_diagnosis_ui_assets(),
        form_state=state,
    )


def _parse_delta(raw: str | None) -> float | None:
    """解析类似 `±10℃` 的现场输入。"""
    if raw is None:
        return None
    text = raw.replace("℃", "").replace("°C", "").replace(" ", "")
    if "+-" in text:
        text = text.split("+-", maxsplit=1)[1]
    elif "±" in text:
        text = text.split("±", maxsplit=1)[1]
    match = re.search(r"[-+]?\d+(\.\d+)?", text)
    if match:
        return float(match.group(0))
    return None


def _process_alias_to_id(raw: str) -> str:
    """把现场口语化工序名映射到内部工序 ID。"""
    text = raw.strip()
    direct = {f"P{i:02d}": f"P{i:02d}" for i in range(1, 13)}
    if text in direct:
        return text
    mapping = {
        "焊盘": "P10",
        "焊接": "P10",
        "焊线": "P10",
        "预焊": "P09",
        "模压": "P11",
        "耐压": "P12",
        "检测": "P12",
    }
    for key, value in mapping.items():
        if key in text:
            return value
    return "P10"


def _symptom_aliases_to_ids(raw: str) -> list[str]:
    """把自然语言症状转成结构化 symptom_id。"""
    if not raw.strip():
        return ["S07"]
    parts = [x.strip() for x in raw.replace("，", ",").split(",") if x.strip()]
    result: list[str] = []
    direct = {f"S{i:02d}": f"S{i:02d}" for i in range(1, 19)}
    fuzzy = {
        "开路": "S01",
        "短路": "S02",
        "阻值高": "S03",
        "阻值偏高": "S03",
        "阻值低": "S04",
        "阻值偏低": "S04",
        "起泡": "S05",
        "分层": "S06",
        "虚焊": "S07",
        "焊盘开口": "S07",
        "焊盘氧化": "S08",
        "耐压": "S09",
        "偏位": "S10",
        "线宽不均": "S11",
        "残铜": "S12",
        "毛刺": "S12",
        "翘曲": "S13",
        "污染": "S14",
        "孔位偏移": "S15",
        "附着力不足": "S16",
        "发黑": "S17",
        "复测不一致": "S18",
    }
    for part in parts:
        if part in direct:
            result.append(direct[part])
            continue
        for key, sid in fuzzy.items():
            if key in part:
                result.append(sid)
                break
    dedup: list[str] = []
    for sid in result:
        if sid not in dedup:
            dedup.append(sid)
    return dedup or ["S07"]


def _parse_observation_value(raw: str | None, parser: str) -> Any | None:
    """按字段定义解析表单值。"""
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() == "unknown":
        return None
    if parser in {"float", "percent_or_float"}:
        return _safe_float(text)
    if parser == "int":
        return int(float(text))
    if parser in {"text", "select"}:
        return text
    return text


def _find_reference_excel(user_input_name: str) -> Path:
    """根据输入名称模糊匹配本地参考 Excel。"""
    text = user_input_name.strip().replace(".xlsx", "")
    for file in REFERENCE_DIR.glob("*.xlsx"):
        if text in file.stem:
            return file
    files = sorted(REFERENCE_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No xlsx file found in reference directory")
    return files[0]


def _write_report(payload: dict[str, Any], prefix: str) -> Path:
    """把页面结果统一沉淀成 JSON 报告。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"{prefix}_{stamp}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _report_kind(name: str) -> str:
    """根据文件名前缀判断报告类型。"""
    if name.startswith("diagnosis"):
        return "diagnosis"
    if name.startswith("optimization"):
        return "optimization"
    if name.startswith("rag_lc"):
        return "rag_lc"
    if name.startswith("rag"):
        return "rag"
    return "other"


def _kind_label(kind: str) -> str:
    labels = {
        "diagnosis": "诊断",
        "optimization": "优化",
        "rag": "RAG",
        "rag_lc": "LangChain RAG",
        "other": "其他",
    }
    return labels.get(kind, kind)


def _load_report_payload(path: Path) -> dict[str, Any]:
    """读取单份报告内容，失败时返回可降级结果。"""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def _shorten(text: str, limit: int = 64) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _diagnosis_report_summary(payload: dict[str, Any], file_name: str) -> dict[str, Any]:
    summary = payload.get("input_summary") or {}
    actions = payload.get("recommended_actions") or []
    top = actions[0] if actions else {}
    symptom_items = summary.get("symptoms") or []
    observed_items = summary.get("observed") or []

    process_name = summary.get("process_name") or payload.get("process_id") or "未记录工序"
    symptoms_text = "、".join(item.get("name", "") for item in symptom_items if item.get("name")) or "未记录异常"
    observed_text = "；".join(
        f"{item.get('label', item.get('key', '参数'))}={item.get('value')}" for item in observed_items[:3]
    ) or "未填写额外观测参数"

    return {
        "title": f"{process_name}异常诊断",
        "subtitle": f"请求编号：{payload.get('request_id', file_name)}",
        "badges": [process_name, f"{len(actions)} 条建议"],
        "summary_lines": [
            {"label": "异常现象", "value": symptoms_text},
            {"label": "首要原因", "value": top.get("cause_name", "暂无推荐结果")},
            {"label": "责任角色", "value": top.get("owner_role", "未提供")},
            {"label": "关键参数", "value": _shorten(observed_text, 80)},
        ],
    }


def _optimization_report_summary(payload: dict[str, Any], file_name: str) -> dict[str, Any]:
    bottlenecks = payload.get("bottlenecks") or []
    actions = payload.get("optimization_actions") or []
    focus = payload.get("focus_filter") or {}
    top = bottlenecks[0] if bottlenecks else {}

    return {
        "title": "工艺路径优化分析",
        "subtitle": f"引擎：{payload.get('engine', 'optimization')}",
        "badges": [f"{len(bottlenecks)} 个瓶颈", f"{len(actions)} 条建议"],
        "summary_lines": [
            {"label": "主要瓶颈", "value": top.get("process_name", "暂无瓶颈结果")},
            {"label": "严重度", "value": str(top.get("severity", "未提供"))},
            {"label": "来源文件", "value": focus.get("source_excel", file_name)},
            {"label": "关注筛选", "value": focus.get("focus_stage", "未设置")},
        ],
    }


def _rag_report_summary(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    citations = payload.get("citations") or payload.get("sources") or []
    answer = payload.get("answer", "")
    title = "知识问答结果" if kind == "rag" else "LangChain 对照问答"
    engine_label = payload.get("engine_label") or payload.get("engine") or kind

    return {
        "title": title,
        "subtitle": f"引擎：{engine_label}",
        "badges": [f"{len(citations)} 条引用", payload.get("model", "未标注模型")],
        "summary_lines": [
            {"label": "问题", "value": _shorten(payload.get("question", "未记录问题"), 90)},
            {"label": "回答摘要", "value": _shorten(answer, 100) if answer else "未生成回答"},
            {"label": "检索策略", "value": str((payload.get("retrieval") or {}).get("strategy", "默认"))},
            {"label": "来源数", "value": str(len(citations))},
        ],
    }


def _other_report_summary(payload: dict[str, Any], file_name: str) -> dict[str, Any]:
    return {
        "title": "其他类型报告",
        "subtitle": file_name,
        "badges": ["调试记录"],
        "summary_lines": [
            {"label": "说明", "value": "当前报告未匹配到专用摘要模板"},
        ],
    }


def _report_summary(kind: str, payload: dict[str, Any], file_name: str) -> dict[str, Any]:
    if payload.get("_load_error"):
        return {
            "title": "报告读取失败",
            "subtitle": file_name,
            "badges": ["异常"],
            "summary_lines": [{"label": "错误", "value": payload["_load_error"]}],
        }
    if kind == "diagnosis":
        return _diagnosis_report_summary(payload, file_name)
    if kind == "optimization":
        return _optimization_report_summary(payload, file_name)
    if kind in {"rag", "rag_lc"}:
        return _rag_report_summary(payload, kind)
    return _other_report_summary(payload, file_name)


def _diagnosis_business_detail(payload: dict[str, Any], file_name: str) -> dict[str, Any]:
    summary = payload.get("input_summary") or {}
    actions = payload.get("recommended_actions") or []
    symptom_text = "、".join(item.get("name", "") for item in summary.get("symptoms", []) if item.get("name")) or "未记录"
    observed_lines = [
        {"label": item.get("label", item.get("key", "参数")), "value": item.get("value", "")}
        for item in summary.get("observed", [])
    ]
    recommendations = []
    for item in actions[:5]:
        recommendations.append(
            {
                "title": f"#{item.get('rank', '-') } {item.get('cause_name', '未命名原因')}",
                "meta": f"责任角色：{item.get('owner_role', '未提供')} · 预计耗时：{item.get('expected_minutes', '-') } 分钟 · 评分：{item.get('score', '-')}",
                "bullets": list(item.get("actions", []))[:4],
                "checks": list(item.get("checkpoints", []))[:3],
            }
        )
    return {
        "title": "诊断业务详情",
        "subtitle": file_name,
        "header_lines": [
            {"label": "请求编号", "value": payload.get("request_id", file_name)},
            {"label": "工序", "value": summary.get("process_name", "未记录工序")},
            {"label": "异常现象", "value": symptom_text},
            {"label": "推荐数量", "value": f"{len(actions)} 条"},
        ],
        "sections": [
            {"title": "现场观测参数", "pairs": observed_lines or [{"label": "说明", "value": "未填写额外观测参数"}]},
            {"title": "推荐动作", "cards": recommendations or [{"title": "暂无推荐", "meta": "当前报告未返回推荐动作", "bullets": [], "checks": []}]},
        ],
    }


def _optimization_business_detail(payload: dict[str, Any], file_name: str) -> dict[str, Any]:
    bottlenecks = payload.get("bottlenecks") or []
    actions = payload.get("optimization_actions") or []
    focus = payload.get("focus_filter") or {}
    bottleneck_cards = []
    for item in bottlenecks[:5]:
        bottleneck_cards.append(
            {
                "title": item.get("process_name", "未命名工序"),
                "meta": f"严重度：{item.get('severity', '-')} · 负载比：{item.get('load_ratio', '-')} · 不良率：{item.get('defect_rate', '-')}",
                "bullets": [
                    f"节段：{item.get('section', '未记录')}",
                    f"日产出：{item.get('daily_output', '未记录')}",
                    f"工序编号：{item.get('process_id', '未记录')}",
                ],
                "checks": [],
            }
        )
    action_cards = []
    for item in actions[:6]:
        action_cards.append(
            {
                "title": item.get("process_name", "未命名工序"),
                "meta": f"动作类型：{item.get('type', '未标注')} · 优先级：{item.get('priority', '未标注')}",
                "bullets": [item.get("action", "未提供动作"), item.get("reason", "未提供原因")],
                "checks": [],
            }
        )
    return {
        "title": "优化业务详情",
        "subtitle": file_name,
        "header_lines": [
            {"label": "分析引擎", "value": payload.get("engine", "optimization")},
            {"label": "瓶颈数量", "value": str(len(bottlenecks))},
            {"label": "建议数量", "value": str(len(actions))},
            {"label": "来源文件", "value": focus.get("source_excel", file_name)},
        ],
        "sections": [
            {"title": "主要瓶颈", "cards": bottleneck_cards or [{"title": "暂无瓶颈", "meta": "当前报告未返回瓶颈信息", "bullets": [], "checks": []}]},
            {"title": "建议动作", "cards": action_cards or [{"title": "暂无建议", "meta": "当前报告未返回优化动作", "bullets": [], "checks": []}]},
        ],
    }


def _rag_business_detail(payload: dict[str, Any], kind: str, file_name: str) -> dict[str, Any]:
    citations = payload.get("citations") or payload.get("sources") or []
    answer = payload.get("answer", "")
    citation_cards = []
    for item in citations[:6]:
        title = item.get("title") or item.get("source_path") or "未标注来源"
        quote = item.get("quote", "")
        citation_cards.append(
            {
                "title": title,
                "meta": f"来源：{item.get('source_path', '未记录')} · 片段：{item.get('chunk_index', item.get('ref_id', '-'))}",
                "bullets": [_shorten(quote, 140)] if quote else [],
                "checks": [],
            }
        )
    label = "RAG 业务详情" if kind == "rag" else "LangChain 问答业务详情"
    return {
        "title": label,
        "subtitle": file_name,
        "header_lines": [
            {"label": "问题", "value": payload.get("question", "未记录问题")},
            {"label": "引擎", "value": payload.get("engine_label") or payload.get("engine", kind)},
            {"label": "模型", "value": payload.get("model", "未标注模型")},
            {"label": "引用数量", "value": str(len(citations))},
        ],
        "sections": [
            {"title": "回答内容", "text": answer or "未生成回答"},
            {"title": "主要来源", "cards": citation_cards or [{"title": "暂无引用", "meta": "当前结果未返回来源信息", "bullets": [], "checks": []}]},
        ],
    }


def _other_business_detail(payload: dict[str, Any], file_name: str) -> dict[str, Any]:
    return {
        "title": "报告业务详情",
        "subtitle": file_name,
        "header_lines": [{"label": "说明", "value": "当前报告未匹配到专用业务详情模板"}],
        "sections": [],
    }


def _report_business_detail(kind: str, payload: dict[str, Any], file_name: str) -> dict[str, Any]:
    if kind == "diagnosis":
        return _diagnosis_business_detail(payload, file_name)
    if kind == "optimization":
        return _optimization_business_detail(payload, file_name)
    if kind in {"rag", "rag_lc"}:
        return _rag_business_detail(payload, kind, file_name)
    return _other_business_detail(payload, file_name)


def _scan_reports(kind: str, keyword: str) -> list[dict[str, Any]]:
    """为历史记录页准备摘要信息。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for file in sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        file_kind = _report_kind(file.name)
        if kind != "all" and kind != file_kind:
            continue
        payload = _load_report_payload(file)
        summary = _report_summary(file_kind, payload, file.name)
        keyword_text = " ".join(
            [file.name, summary.get("title", ""), summary.get("subtitle", "")]
            + [line.get("value", "") for line in summary.get("summary_lines", [])]
        )
        if keyword and keyword not in keyword_text:
            continue
        st = file.stat()
        items.append(
            {
                "name": file.name,
                "kind": file_kind,
                "kind_label": _kind_label(file_kind),
                "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(st.st_size / 1024.0, 1),
                "summary": summary,
            }
        )
    return items


def _require_login() -> bool:
    return bool(session.get("logged_in"))


def _rag_index_ready(index_dir: Path) -> bool:
    return all((index_dir / fn).exists() for fn in ("manifest.json", "chunks.jsonl", "vectors.npy"))


def _ensure_rag_index(index_dir: Path) -> None:
    """确保自研版 RAG 索引存在。

    这样用户第一次点进页面时，不需要先手工跑建库命令。
    """
    if _rag_index_ready(index_dir):
        return
    docs = load_documents_from_dir(RAG_DOCS_DIR)
    chunks = TextSplitter(chunk_size=700, chunk_overlap=100, min_chunk_size=120).split_documents(docs)
    client = build_client_from_env(require_chat_model=False)
    LocalVectorIndex.build(
        chunks=chunks,
        embedder=client,
        index_dir=index_dir,
        batch_size=10,
        metadata={"docs_dir": str(RAG_DOCS_DIR.resolve())},
    )


def _rag_pipeline() -> RagPipeline:
    """提供自研版 RAG 的页面调用入口。"""
    _ensure_rag_index(RAG_INDEX_DIR)
    index = LocalVectorIndex.load(RAG_INDEX_DIR)
    client = build_client_from_env(require_chat_model=True)
    return RagPipeline(index=index, client=client)


def _ensure_rag_lc_index(index_dir: Path) -> None:
    """确保 LangChain 对照版索引存在。"""
    demo = LangChainRagDemo(index_dir)
    if demo.is_ready():
        return
    LangChainRagDemo.build(
        RAG_DOCS_DIR,
        index_dir,
        chunk_size=700,
        chunk_overlap=100,
        recursive=True,
    )


def _rag_lc_demo() -> LangChainRagDemo:
    """提供 LangChain 对照版的页面调用入口。"""
    _ensure_rag_lc_index(RAG_LC_INDEX_DIR)
    return LangChainRagDemo(RAG_LC_INDEX_DIR)


@app.before_request
def _gate():
    """统一登录拦截，避免每个路由重复写鉴权。"""
    open_paths = {"/login", "/health"}
    if request.path.startswith("/static/"):
        return
    if request.path in open_paths:
        return
    if not _require_login():
        return redirect(url_for("login"))


@app.get("/login")
def login():
    return render_template("login.html", error=None)


@app.post("/login")
def login_submit():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
        session["logged_in"] = True
        session["username"] = username
        session["role"] = "leader"
        return redirect(url_for("role_page"))
    return render_template("login.html", error="账号或密码错误")


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
def index():
    return redirect(url_for("role_page"))


@app.get("/role")
def role_page():
    return render_template("role_page.html", role=session.get("role", "leader"), username=session.get("username", "user"))


@app.get("/role/switch/<role>")
def switch_role(role: str):
    if role not in {"leader", "engineer"}:
        abort(404)
    session["role"] = role
    if role == "leader":
        return redirect(url_for("diagnose_form"))
    return redirect(url_for("optimize_form"))


@app.get("/diagnose")
def diagnose_form():
    return _render_diagnosis_form()


@app.post("/diagnose")
def diagnose_submit():
    """诊断页提交逻辑。

    运行顺序：读取表单 -> 转成结构化请求 -> 调用诊断引擎 -> 写报告并渲染结果页。
    """
    request_id = request.form.get("request_id", "").strip() or f"REQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    process_id = request.form.get("process_id", "").strip()
    if not process_id:
        process_id = _process_alias_to_id(request.form.get("process", ""))

    templates = _observation_templates()
    if process_id not in templates:
        form_state = {
            "request_id": request_id,
            "process_id": process_id,
            "symptom_ids": request.form.getlist("symptom_ids"),
            "top_k": request.form.get("top_k", "3"),
            "observed": dict(request.form.items()),
        }
        return _render_diagnosis_form("请选择有效工序后再提交。", form_state)

    symptom_ids = [x for x in request.form.getlist("symptom_ids") if x]
    if not symptom_ids:
        symptom_ids = _symptom_aliases_to_ids(request.form.get("symptoms", ""))
    if not symptom_ids:
        form_state = {
            "request_id": request_id,
            "process_id": process_id,
            "symptom_ids": [],
            "top_k": request.form.get("top_k", "3"),
            "observed": dict(request.form.items()),
        }
        return _render_diagnosis_form("请至少选择一个异常现象。", form_state)

    observed: dict[str, Any] = {}
    for field in templates[process_id].fields:
        raw = request.form.get(field.key)
        try:
            value = _parse_observation_value(raw, field.parser)
        except ValueError:
            form_state = {
                "request_id": request_id,
                "process_id": process_id,
                "symptom_ids": symptom_ids,
                "top_k": request.form.get("top_k", "3"),
                "observed": dict(request.form.items()),
            }
            return _render_diagnosis_form(f"字段“{field.label}”格式不正确，请检查后重试。", form_state)
        if value is not None:
            observed[field.key] = value

    payload = {
        "request_id": request_id,
        "process_id": process_id,
        "symptom_ids": symptom_ids,
        "observed": observed,
        "top_k": int(request.form.get("top_k", "3")),
    }
    result = _diagnosis_engine().diagnose(payload)
    kb = _diagnosis_engine().kb
    field_label_map = {field.key: field.label for field in templates[process_id].fields}
    result["input_summary"] = {
        "process_id": process_id,
        "process_name": kb.processes[process_id].name,
        "symptoms": [
            {"id": sid, "name": kb.symptoms[sid].name}
            for sid in symptom_ids
            if sid in kb.symptoms
        ],
        "observed": [
            {"key": key, "label": field_label_map.get(key, key), "value": value}
            for key, value in observed.items()
        ],
    }
    report_path = _write_report(result, f"diagnosis_{request_id}")
    return render_template("diagnose_result.html", result=result, report_path=str(report_path))


@app.get("/optimize")
def optimize_form():
    excels = [x.name for x in sorted(REFERENCE_DIR.glob("*.xlsx"))]
    return render_template("optimize_form.html", excels=excels, role=session.get("role", "engineer"), username=session.get("username", "user"))


@app.post("/optimize")
def optimize_submit():
    """优化页提交逻辑。

    页面层会再做一层筛选，让结果更贴近现场关注点。
    """
    excel_input = request.form.get("excel_name", "").strip()
    focus_stage = request.form.get("focus_stage", "").strip()
    target_capacity = request.form.get("target_capacity", "").strip()
    priority = request.form.get("priority", "").strip()

    excel_path = _find_reference_excel(excel_input)
    result = _optimization_engine().analyze([str(excel_path)])

    filtered = []
    for item in result["bottlenecks"]:
        process_name = str(item.get("process_name", ""))
        if focus_stage and focus_stage not in process_name and focus_stage not in item.get("section", ""):
            keywords = ["焊", "模压", "耐压", "外观", "包装", "贴标签", "清洁焊盘", "预焊"]
            if focus_stage == "组装段" and not any(k in process_name for k in keywords):
                continue
        filtered.append(item)

    result["focus_filter"] = {
        "focus_stage": focus_stage,
        "target_capacity": target_capacity,
        "priority": priority,
        "source_excel": str(excel_path),
    }
    result["bottlenecks"] = filtered[:15]
    result["optimization_actions"] = result["optimization_actions"][:20]

    report_path = _write_report(result, "optimization_web")
    return render_template("optimize_result.html", result=result, report_path=str(report_path))


@app.get("/rag")
def rag_form():
    return render_template("rag_form.html", role=session.get("role", "leader"), username=session.get("username", "user"))


@app.post("/rag")
def rag_submit():
    """自研版 RAG 页面入口。"""
    question = request.form.get("question", "").strip()
    top_k = int(request.form.get("top_k", "4"))
    max_context_chunks = int(request.form.get("max_context_chunks", "4"))
    if not question:
        return render_template(
            "rag_form.html",
            role=session.get("role", "leader"),
            username=session.get("username", "user"),
            error="请输入问题后再提交。",
            question=question,
            top_k=top_k,
            max_context_chunks=max_context_chunks,
        )
    try:
        # 页面演示使用固定默认参数，避免先把用户暴露到底层调参复杂度上。
        answer = _rag_pipeline().ask(
            question,
            top_k=top_k,
            max_context_chunks=max_context_chunks,
            vector_weight=0.70,
            lexical_weight=0.20,
            title_weight=0.10,
            max_per_doc=2,
            min_final_score=0.42,
            min_lexical_score=0.02,
            min_title_score=0.02,
            temperature=0.1,
        )
        payload = RagPipeline.to_payload(answer)
        report_path = _write_report(payload, "rag_qa")
        return render_template("rag_result.html", result=payload, report_path=str(report_path))
    except Exception as exc:
        return render_template(
            "rag_form.html",
            role=session.get("role", "leader"),
            username=session.get("username", "user"),
            error=f"RAG 查询失败: {exc}",
            question=question,
            top_k=top_k,
            max_context_chunks=max_context_chunks,
        )


@app.get("/rag-lc")
def rag_lc_form():
    return render_template("rag_lc_form.html", role=session.get("role", "leader"), username=session.get("username", "user"))


@app.post("/rag-lc")
def rag_lc_submit():
    """LangChain 对照版页面入口。"""
    question = request.form.get("question", "").strip()
    top_k = int(request.form.get("top_k", "4"))
    max_context_chunks = int(request.form.get("max_context_chunks", "4"))
    if not question:
        return render_template(
            "rag_lc_form.html",
            role=session.get("role", "leader"),
            username=session.get("username", "user"),
            error="请输入问题后再提交。",
            question=question,
            top_k=top_k,
            max_context_chunks=max_context_chunks,
        )
    try:
        payload = _rag_lc_demo().ask(
            question,
            top_k=top_k,
            max_context_chunks=max_context_chunks,
            temperature=0.1,
            lambda_mult=0.55,
        )
        report_path = _write_report(payload, "rag_lc_qa")
        return render_template("rag_lc_result.html", result=payload, report_path=str(report_path))
    except Exception as exc:
        return render_template(
            "rag_lc_form.html",
            role=session.get("role", "leader"),
            username=session.get("username", "user"),
            error=f"LangChain RAG 查询失败: {exc}",
            question=question,
            top_k=top_k,
            max_context_chunks=max_context_chunks,
        )


@app.get("/history")
def history_page():
    kind = request.args.get("kind", "all")
    keyword = request.args.get("q", "").strip()
    reports = _scan_reports(kind, keyword)
    return render_template("history.html", reports=reports, kind=kind, keyword=keyword)


@app.get("/history/detail")
def history_detail():
    name = request.args.get("name", "").strip()
    if not name or "/" in name or "\\" in name:
        abort(400)
    path = REPORT_DIR / name
    if not path.exists() or path.suffix.lower() != ".json":
        abort(404)
    payload = _load_report_payload(path)
    kind = _report_kind(name)
    detail = _report_business_detail(kind, payload, name)
    return render_template(
        "history_detail.html",
        report_name=name,
        report_kind=_kind_label(kind),
        detail=detail,
    )


@app.get("/history/view")
def history_view():
    name = request.args.get("name", "").strip()
    if not name or "/" in name or "\\" in name:
        abort(400)
    path = REPORT_DIR / name
    if not path.exists() or path.suffix.lower() != ".json":
        abort(404)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    return render_template("history_view.html", report_name=name, content=rendered)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)

