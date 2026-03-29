from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .rag_models import Document

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_documents_from_dir(docs_dir: str | Path, recursive: bool = True) -> List[Document]:
    root = Path(docs_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"docs directory not found: {root}")

    candidates = root.rglob("*") if recursive else root.glob("*")
    files = sorted(
        [p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda p: p.as_posix(),
    )
    if not files:
        raise ValueError(f"no supported documents found in {root}")

    docs: List[Document] = []
    for path in files:
        content = _load_content(path)
        content = content.strip()
        if not content:
            continue
        rel = path.relative_to(root).as_posix()
        doc_id = f"DOC-{_hash(rel)[:12]}"
        title = _infer_title(path, content)
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        docs.append(
            Document(
                doc_id=doc_id,
                source_path=rel,
                title=title,
                file_type=path.suffix.lower().lstrip("."),
                content=content,
                last_modified=mtime,
                metadata={"size_bytes": path.stat().st_size},
            )
        )

    if not docs:
        raise ValueError(f"all documents are empty under {root}")
    return docs


def _load_content(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _read_text(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"unsupported file type: {path}")


def _read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - exercised in runtime envs without pypdf
        raise RuntimeError("reading pdf requires pypdf. install with `pip install pypdf`.") from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _infer_title(path: Path, content: str) -> str:
    if path.suffix.lower() == ".md":
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or path.stem
    return path.stem


def _hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()
