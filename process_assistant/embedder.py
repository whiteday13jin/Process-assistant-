from __future__ import annotations

"""模型适配层。

这一层的目标不是“实现一个模型”，而是把外部兼容 OpenAI 协议的服务
统一包装成项目内部稳定可调用的客户端。当前默认对接阿里百炼，
但业务层只依赖这里暴露的 `embed_texts` 和 `chat`，后续替换供应商时影响最小。
"""

import json
import os
from dataclasses import dataclass
from typing import List

import requests

from .env_loader import load_project_env


@dataclass(frozen=True)
class ModelConfig:
    """模型调用配置。

    把 provider、地址、模型名和超时集中收口，避免业务层到处读环境变量。
    """

    provider: str
    base_url: str
    api_key: str
    chat_model: str
    embed_model: str
    timeout_sec: int = 60


class OpenAICompatibleClient:
    """面向 OpenAI-compatible 接口的轻量客户端。"""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._session = requests.Session()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量向量化文本。

        建索引和问题检索都会走这里，所以这里既要做空输入保护，
        也要兼容不同 provider 对 batch 大小的限制。
        """

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
        """调用聊天模型生成最终回答。"""

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
        """统一处理 HTTP 调用、报错抛出和 JSON 解析。"""

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

    # 这里放默认模型，是为了降低第一次跑通项目的门槛。
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
    """选模型，不同 provider 的 API 地址不一样，这里选的阿里千问"""

    configured = os.getenv("PROCESS_ASSISTANT_BASE_URL", "").strip()
    if configured:
        return configured
    if provider == "aliyun_dashscope":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "openai_compatible":
        return "https://api.openai.com/v1"
    raise RuntimeError(f"unknown provider: {provider}")


def _max_embed_batch_size(provider: str) -> int:
    """不同服务对 embedding 批量大小限制不同，这里做 provider 级兼容。"""

    if provider == "aliyun_dashscope":
        return 10
    #由于我所接的千问模型的 embedding 限制，故return 10，其他模型可以放宽。
    return 128
