from __future__ import annotations

"""RAG 文档接入层。

这一层的职责很单纯：标准化。把磁盘里的原始文件统一读成系统内部的 `Document` 对象。
主流程不会直接面向 md/txt/pdf 各种文件格式，而是先经过这里做一次标准化，
后面的切分器、索引器和问答链路都只和统一对象打交道。
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .rag_models import Document

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_documents_from_dir(docs_dir: str | Path, recursive: bool = True) -> List[Document]:
    """从目录中读取可支持的文档，并转成统一的 `Document` 列表。"""

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
        # 这里先按文件类型读取原始内容，再统一补齐标题、相对路径、更新时间等元数据。
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
    """按文件后缀分发到具体读取函数。"""

    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _read_text(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"unsupported file type: {path}")


def _read_text(path: Path) -> str:
    """优先尝试常见中文场景编码，尽量减少样本文档因编码问题无法接入。"""

    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    """抽取 PDF 纯文本。

    PDF 在真实项目里往往最不稳定，所以这里先做“尽力读出来”的策略，
    如果抽取质量不好，后面可以再替换成更强的版面解析方案。
    """

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
    """优先从 Markdown 一级标题推断标题，读不到时退回文件名。"""

    if path.suffix.lower() == ".md":
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or path.stem
    return path.stem


def _hash(value: str) -> str:
    """用稳定哈希生成文档 ID，避免文件名变化太大时主键不可控。"""

    return hashlib.sha1(value.encode("utf-8")).hexdigest()
