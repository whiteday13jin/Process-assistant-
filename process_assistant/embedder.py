from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import requests

from .env_loader import load_project_env


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    base_url: str
    api_key: str
    chat_model: str
    embed_model: str
    timeout_sec: int = 60


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._session = requests.Session()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        effective_batch = min(batch_size, _max_embed_batch_size(self.config.provider))
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), effective_batch):
            batch = texts[i : i + effective_batch]
            payload = {"model": self.config.embed_model, "input": batch}
            data = self._post_json("/embeddings", payload)
            vectors = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            all_vectors.extend(vectors)
        return all_vectors

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        payload = {
            "model": self.config.chat_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = self._post_json("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("chat API returned no choices")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("chat API returned empty content")
        return str(content)

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        url = self.config.base_url.rstrip("/") + endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        resp = self._session.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.config.timeout_sec,
        )
        if resp.status_code >= 400:
            body = resp.text[:600]
            raise RuntimeError(f"API request failed: {resp.status_code} {body}")
        try:
            return resp.json()
        except ValueError as exc:
            raise RuntimeError(f"API returned invalid JSON: {resp.text[:300]}") from exc


def build_client_from_env(require_chat_model: bool = True) -> OpenAICompatibleClient:
    load_project_env()
    provider = os.getenv("PROCESS_ASSISTANT_MODEL_PROVIDER", "aliyun_dashscope").strip().lower()
    base_url = _resolve_base_url(provider)
    api_key = (
        os.getenv("PROCESS_ASSISTANT_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "missing API key. set PROCESS_ASSISTANT_API_KEY (or DASHSCOPE_API_KEY / OPENAI_API_KEY)."
        )

    chat_default = "qwen-plus"
    embed_default = "text-embedding-v3"
    if provider == "openai_compatible":
        chat_default = "gpt-4o-mini"
        embed_default = "text-embedding-3-small"

    chat_model = os.getenv("PROCESS_ASSISTANT_CHAT_MODEL", chat_default).strip()
    embed_model = os.getenv("PROCESS_ASSISTANT_EMBED_MODEL", embed_default).strip()
    if not embed_model:
        raise RuntimeError("missing embedding model. set PROCESS_ASSISTANT_EMBED_MODEL.")
    if require_chat_model and not chat_model:
        raise RuntimeError("missing chat model. set PROCESS_ASSISTANT_CHAT_MODEL.")

    config = ModelConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        chat_model=chat_model,
        embed_model=embed_model,
        timeout_sec=int(os.getenv("PROCESS_ASSISTANT_API_TIMEOUT_SEC", "60")),
    )
    return OpenAICompatibleClient(config)


def _resolve_base_url(provider: str) -> str:
    configured = os.getenv("PROCESS_ASSISTANT_BASE_URL", "").strip()
    if configured:
        return configured
    if provider == "aliyun_dashscope":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "openai_compatible":
        return "https://api.openai.com/v1"
    raise RuntimeError(f"unknown provider: {provider}")


def _max_embed_batch_size(provider: str) -> int:
    if provider == "aliyun_dashscope":
        return 10
    return 128
