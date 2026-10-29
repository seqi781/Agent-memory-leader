"""Optional local cross-encoder reranker (ONNX via fastembed). Rerankers are unrestricted in the
academic division; the model runs on the service's own CPU, so it costs nothing per call."""
from __future__ import annotations

import threading

from . import config

_model = None
_lock = threading.Lock()


def enabled() -> bool:
    return bool(config.RERANK_MODEL)


def _load():
    global _model
    if _model is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        kwargs = dict(cache_dir=str(config.MODEL_DIR), threads=config.RERANK_THREADS)
        try:  # prefer the local copy (the production host cannot reach Hugging Face)
            _model = TextCrossEncoder(config.RERANK_MODEL, local_files_only=True, **kwargs)
        except Exception:
            _model = TextCrossEncoder(config.RERANK_MODEL, **kwargs)
    return _model


def scores(query: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    with _lock:  # one inference at a time per process keeps CPU use predictable
        model = _load()
        return [float(s) for s in model.rerank(query, docs, batch_size=config.RERANK_BATCH)]


def warmup() -> None:
    if enabled():
        scores("warmup", ["warmup"])
