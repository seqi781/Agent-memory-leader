"""AML Add / Search API service."""
from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import config
from .store import Piece, Store
from .text import chunk_text
from .timeresolve import annotate

log = logging.getLogger("aml_memory")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["store"] = Store()
    yield


app = FastAPI(title="AML Memory", lifespan=lifespan)


class BadRequest(Exception):
    pass


def _error(status: int, msg: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "detail": msg})


@app.middleware("http")
async def auth(request: Request, call_next):
    if config.SERVICE_KEY and request.url.path != "/health":
        header = request.headers.get("authorization", "")
        token = header.split(" ", 1)[1].strip() if " " in header else ""
        token = token or request.headers.get("x-api-key", "")
        if not hmac.compare_digest(token.encode(), config.SERVICE_KEY.encode()):
            return _error(401, "invalid credentials")
    return await call_next(request)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"service": "aml-memory", "endpoints": ["/v1/memories/add", "/v1/memories/search", "/health"]}


# ---------- parsing helpers ----------
def _req_str(body: dict, key: str) -> str:
    val = body.get(key)
    if not isinstance(val, str) or not val.strip():
        raise BadRequest(f"'{key}' must be a non-empty string")
    return val


def _content_text(content: Any) -> str:
    """Text of a message: plain string, or the text parts of a multimodal content array."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, dict) and part.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(parts)
    raise BadRequest("message 'content' must be a string or a content array")


def _timestamp(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise BadRequest("'timestamp' must be Unix milliseconds")
    if isinstance(value, (int, float)):
        ts = int(value)
        return ts * 1000 if ts < 10**11 else ts  # tolerate seconds
    if isinstance(value, str):
        if value.strip().lstrip("-").isdigit():
            return _timestamp(int(value))
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            return None
    return None


# ---------- endpoints ----------
@app.post("/v1/memories/add")
async def add(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")
        request_id = _req_str(body, "request_id")
        user_id = _req_str(body, "user_id")
        session_id = _req_str(body, "session_id")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BadRequest("'messages' must be a non-empty array")
        default_ts = _timestamp(body.get("timestamp"))
        pieces: list[Piece] = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise BadRequest("each message must be an object")
            role = str(msg.get("role") or "user")
            text = _content_text(msg.get("content")).strip()
            if not text:
                raise BadRequest(f"message {i} has empty content")
            ts = _timestamp(msg.get("timestamp"))
            ts = ts if ts is not None else default_ts
            if config.RESOLVE_TIME:
                text = annotate(text, ts)
            for j, chunk in enumerate(chunk_text(text, config.CHUNK_CHARS, config.CHUNK_OVERLAP)):
                pieces.append(Piece(msg_idx=i, chunk_idx=j, role=role, ts=ts, content=chunk))
    except BadRequest as e:
        return _error(422, str(e))
    except ValueError:
        return _error(400, "invalid JSON")

    store: Store = state["store"]
    await run_in_threadpool(store.add, request_id, user_id, session_id, pieces)  # idempotent on request_id
    return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}


@app.post("/v1/memories/search")
async def search(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")
        user_id = _req_str(body, "user_id")
        query = _content_text(body.get("query")).strip()
        if not query:
            raise BadRequest("'query' must not be empty")
        top_k = body.get("top_k", 100)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise BadRequest("'top_k' must be a non-negative integer")
        options = body.get("options") or []
        if not isinstance(options, list):
            raise BadRequest("'options' must be an array of strings")
        options = [str(o) for o in options if str(o).strip()]
    except BadRequest as e:
        return _error(422, str(e))
    except ValueError:
        return _error(400, "invalid JSON")

    data = await run_in_threadpool(state["store"].search, user_id, query, options, top_k)
    return {"data": data[:top_k]}
