from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, abort, redirect, render_template, request, session, url_for

from process_assistant.diagnosis_engine import DiagnosisEngine
from process_assistant.optimization_engine import OptimizationEngine

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
REFERENCE_DIR = BASE_DIR / "reference"
KB_PATH = DATA_DIR / "knowledge_base.json"
FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.jsonl"

LOGIN_USERNAME = "user"
LOGIN_PASSWORD = "123"

app = Flask(__name__)
app.secret_key = os.getenv("PROCESS_ASSISTANT_SECRET_KEY", "dev-only-change-me")


def _diagnosis_engine() -> DiagnosisEngine:
    return DiagnosisEngine(str(KB_PATH), str(FEEDBACK_LOG_PATH))


def _optimization_engine() -> OptimizationEngine:
    return OptimizationEngine()


def _safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() == "unknown":
        return None
    if value.endswith("%"):
        value = value[:-1]
        return float(value) / 100.0
    return float(value)


def _parse_delta(raw: str | None) -> float | None:
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
    if not raw.strip():
        return ["S07"]
    parts = [x.strip() for x in raw.replace("，", ",").split(",") if x.strip()]
    result: list[str] = []
    direct = {f"S{i:02d}": f"S{i:02d}" for i in range(1, 11)}
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


def _find_reference_excel(user_input_name: str) -> Path:
    text = user_input_name.strip().replace(".xlsx", "")
    for file in REFERENCE_DIR.glob("*.xlsx"):
        if text in file.stem:
            return file
    files = sorted(REFERENCE_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No xlsx file found in reference directory")
    return files[0]


def _write_report(payload: dict[str, Any], prefix: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"{prefix}_{stamp}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _report_kind(name: str) -> str:
    if name.startswith("diagnosis"):
        return "diagnosis"
    if name.startswith("optimization"):
        return "optimization"
    return "other"


def _scan_reports(kind: str, keyword: str) -> list[dict[str, Any]]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for file in sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        file_kind = _report_kind(file.name)
        if kind != "all" and kind != file_kind:
            continue
        if keyword and keyword not in file.name:
            continue
        st = file.stat()
        items.append(
            {
                "name": file.name,
                "kind": file_kind,
                "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(st.st_size / 1024.0, 1),
            }
        )
    return items


def _require_login() -> bool:
    return bool(session.get("logged_in"))


@app.before_request
def _gate():
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
    return render_template("diagnose_form.html", role=session.get("role", "leader"), username=session.get("username", "user"))


@app.post("/diagnose")
def diagnose_submit():
    request_id = request.form.get("request_id", "").strip() or f"REQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    process_id = _process_alias_to_id(request.form.get("process", ""))
    symptom_ids = _symptom_aliases_to_ids(request.form.get("symptoms", ""))
    observed: dict[str, Any] = {}

    defect_rate = _safe_float(request.form.get("defect_rate"))
    if defect_rate is not None:
        observed["defect_rate"] = defect_rate
    humidity = request.form.get("humidity", "").strip()
    if humidity:
        if humidity.lower() == "unknown":
            observed["humidity"] = "unknown"
        else:
            observed["humidity"] = float(humidity)
    flux_age_hours = _safe_float(request.form.get("flux_age_hours"))
    if flux_age_hours is not None:
        observed["flux_age_hours"] = flux_age_hours
    oven_temp_delta = _parse_delta(request.form.get("oven_temp_delta"))
    if oven_temp_delta is not None:
        observed["oven_temp_delta"] = oven_temp_delta

    payload = {
        "request_id": request_id,
        "process_id": process_id,
        "symptom_ids": symptom_ids,
        "observed": observed,
        "top_k": int(request.form.get("top_k", "3")),
    }
    result = _diagnosis_engine().diagnose(payload)
    report_path = _write_report(result, f"diagnosis_{request_id}")
    return render_template("diagnose_result.html", result=result, report_path=str(report_path))


@app.get("/optimize")
def optimize_form():
    excels = [x.name for x in sorted(REFERENCE_DIR.glob("*.xlsx"))]
    return render_template("optimize_form.html", excels=excels, role=session.get("role", "engineer"), username=session.get("username", "user"))


@app.post("/optimize")
def optimize_submit():
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


@app.get("/history")
def history_page():
    kind = request.args.get("kind", "all")
    keyword = request.args.get("q", "").strip()
    reports = _scan_reports(kind, keyword)
    return render_template("history.html", reports=reports, kind=kind, keyword=keyword)


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

