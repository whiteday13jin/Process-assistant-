from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .embedder import OpenAICompatibleClient
from .rag_models import Chunk, RetrievedChunk


class LocalVectorIndex:
    def __init__(self, chunks: List[Chunk], vectors: np.ndarray, manifest: Dict[str, Any]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks length does not match vector count")
        self.chunks = chunks
        self.vectors = _to_float32(vectors)
        self.manifest = manifest
        self._norm_vectors = _normalize(self.vectors)

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        embedder: OpenAICompatibleClient,
        index_dir: str | Path,
        *,
        batch_size: int = 32,
        metadata: Dict[str, Any] | None = None,
    ) -> "LocalVectorIndex":
        if not chunks:
            raise ValueError("cannot build index from empty chunks")

        vectors = np.asarray(embedder.embed_texts([c.text for c in chunks], batch_size=batch_size))
        if vectors.ndim != 2:
            raise ValueError("embedding output must be a 2D matrix")

        manifest = {
            "engine": "local_rag_v1",
            "provider": embedder.config.provider,
            "base_url": embedder.config.base_url,
            "embedding_model": embedder.config.embed_model,
            "chat_model": embedder.config.chat_model,
            "vector_dim": int(vectors.shape[1]),
            "doc_count": len({c.doc_id for c in chunks}),
            "chunk_count": len(chunks),
            "built_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if metadata:
            manifest.update(metadata)

        index = cls(chunks=chunks, vectors=vectors, manifest=manifest)
        index.save(index_dir)
        return index

    @classmethod
    def load(cls, index_dir: str | Path) -> "LocalVectorIndex":
        root = Path(index_dir).resolve()
        manifest_path = root / "manifest.json"
        chunks_path = root / "chunks.jsonl"
        vectors_path = root / "vectors.npy"

        if not manifest_path.exists() or not chunks_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(
                f"index files missing in {root}. required: manifest.json, chunks.jsonl, vectors.npy"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        chunks: List[Chunk] = []
        with chunks_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(Chunk(**json.loads(line)))
        vectors = np.load(str(vectors_path))
        return cls(chunks=chunks, vectors=vectors, manifest=manifest)

    def save(self, index_dir: str | Path) -> None:
        root = Path(index_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

        (root / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (root / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        np.save(str(root / "vectors.npy"), self.vectors.astype(np.float32))

    def search(
        self,
        query_vector: Iterable[float],
        top_k: int = 4,
        *,
        query_text: str | None = None,
        hybrid_alpha: float = 0.82,
        max_per_doc: int | None = 2,
    ) -> List[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be within [0, 1]")
        q = np.asarray(list(query_vector), dtype=np.float32)
        if q.ndim != 1:
            raise ValueError("query vector must be 1D")
        if q.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"query vector dim mismatch: expected {self.vectors.shape[1]}, got {q.shape[0]}"
            )

        q = _normalize(q.reshape(1, -1))[0]
        vector_scores = np.dot(self._norm_vectors, q)
        final_scores = vector_scores.copy()
        lexical_scores = np.zeros_like(vector_scores)
        if query_text:
            query_terms = _terms(query_text)
            if query_terms:
                for i, chunk in enumerate(self.chunks):
                    lexical_scores[i] = _lexical_overlap(query_terms, _terms(chunk.text))
                final_scores = hybrid_alpha * vector_scores + (1.0 - hybrid_alpha) * lexical_scores

        # Retrieve a wider pool first, then apply diversity control.
        candidate_k = min(max(top_k * 4, 20), len(self.chunks))
        candidate_indices = np.argsort(-final_scores)[:candidate_k]
        top_indices: List[int] = []
        doc_counts: Dict[str, int] = {}
        for idx in candidate_indices:
            chunk = self.chunks[int(idx)]
            count = doc_counts.get(chunk.doc_id, 0)
            if max_per_doc is not None and count >= max_per_doc:
                continue
            doc_counts[chunk.doc_id] = count + 1
            top_indices.append(int(idx))
            if len(top_indices) >= min(top_k, len(self.chunks)):
                break

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            score = float(final_scores[int(idx)])
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_path=chunk.source_path,
                    title=chunk.title,
                    chunk_index=chunk.chunk_index,
                    score=round(score, 6),
                    text=chunk.text,
                    heading_path=list(chunk.heading_path),
                    metadata=dict(chunk.metadata),
                )
            )
        return results


def _to_float32(array: np.ndarray) -> np.ndarray:
    return array.astype(np.float32, copy=False)


def _normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return array / norms


def _terms(text: str) -> set[str]:
    text = " ".join(text.lower().split())
    tokens = {t for t in re.findall(r"[a-z0-9_]{2,}", text)}
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
    # Use bi-grams so Chinese lexical overlap stays meaningful.
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i : i + 2])
    return tokens


def _lexical_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    den = max(len(a), 1)
    return inter / den
