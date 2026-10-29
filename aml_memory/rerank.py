"""Local cross-encoder reranker run directly with ONNX Runtime. Rerankers are unrestricted in the
academic division; the model runs on the service's own CPU, so it costs nothing per call.

ONNX Runtime's CPU memory arena is disabled and inputs are truncated to a fixed maximum length:
with the arena on, memory grew by gigabytes under varied input shapes.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import config

_session = None
_tokenizer = None
_inputs: set[str] = set()
_lock = threading.Lock()


def enabled() -> bool:
    return bool(config.RERANK_MODEL)


def _model_dir() -> Path:
    org_name = config.RERANK_MODEL.replace("/", "--")
    found = sorted(config.MODEL_DIR.glob(f"models--{org_name}/snapshots/*/onnx/model.onnx"))
    if found:
        return found[-1].parent.parent
    from huggingface_hub import snapshot_download  # first run on a machine without the files

    return Path(snapshot_download(
        config.RERANK_MODEL, cache_dir=str(config.MODEL_DIR),
        allow_patterns=["onnx/model.onnx", "tokenizer.json", "config.json"],
    ))


def _load() -> None:
    global _session, _tokenizer, _inputs
    if _session is not None:
        return
    import onnxruntime as ort
    from tokenizers import Tokenizer

    root = _model_dir()
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    tok.enable_truncation(max_length=config.RERANK_MAX_TOKENS)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.intra_op_num_threads = config.RERANK_THREADS
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(root / "onnx" / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    _inputs = {i.name for i in sess.get_inputs()}
    _tokenizer, _session = tok, sess


def scores(query: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    out: list[float] = []
    with _lock:  # one inference at a time per process keeps CPU and memory predictable
        _load()
        for i in range(0, len(docs), config.RERANK_BATCH):
            batch = _tokenizer.encode_batch([(query, d) for d in docs[i : i + config.RERANK_BATCH]])
            feed = {
                "input_ids": np.array([e.ids for e in batch], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in batch], dtype=np.int64),
            }
            if "token_type_ids" in _inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in batch], dtype=np.int64)
            logits = _session.run(None, feed)[0]
            out.extend(float(x) for x in np.asarray(logits).reshape(len(batch), -1)[:, 0])
    return out


def warmup() -> None:
    if enabled():
        scores("warmup", ["warmup"])
