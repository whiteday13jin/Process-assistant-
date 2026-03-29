from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def load_project_env() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    dotenv_path = root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False, encoding="utf-8")
        return dotenv_path
    load_dotenv(override=False)
    return None

