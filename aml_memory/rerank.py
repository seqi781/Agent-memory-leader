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


_torch = None  # (tokenizer, model, device) when RERANK_BACKEND=torch


def _load_torch() -> None:
    """Hugging Face transformers backend, for larger cross-encoders on a GPU."""
    global _torch
    if _torch is not None:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = config.RERANK_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(config.RERANK_MODEL, cache_dir=str(config.MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        config.RERANK_MODEL, cache_dir=str(config.MODEL_DIR),
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device).eval()
    _torch = (tok, model, device)


def _scores_torch(query: str, docs: list[str]) -> list[float]:
    import torch

    _load_torch()
    tok, model, device = _torch
    out: list[float] = []
    for i in range(0, len(docs), config.RERANK_BATCH):
        enc = tok([query] * len(docs[i : i + config.RERANK_BATCH]), docs[i : i + config.RERANK_BATCH],
                  padding=True, truncation="only_second", max_length=config.RERANK_MAX_TOKENS, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        out.extend(float(x) for x in logits.float().reshape(logits.shape[0], -1)[:, 0].cpu())
    return out


def scores(query: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    if config.RERANK_BACKEND == "torch":
        with _lock:
            return _scores_torch(query, docs)
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
