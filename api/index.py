"""FastAPI app. Vercel runs this file as a Python function; locally:  uvicorn api.index:app --reload"""
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from rag.pipeline import RagService  # noqa: E402

app = FastAPI(title="Kitebox help bot (RAG demo)", docs_url="/api/docs", openapi_url="/api/openapi.json")
_svc: RagService | None = None


def svc() -> RagService:
    global _svc
    if _svc is None:
        _svc = RagService()
    return _svc


# A public demo spends someone's LLM credits, so cap it per visitor (per instance; fine for a demo).
LIMIT, WINDOW = int(os.getenv("RATE_LIMIT", "30")), 600
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= LIMIT:
        raise HTTPException(429, f"Demo limit: {LIMIT} questions per 10 minutes. Try again shortly.")
    q.append(now)


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=2000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=12)
    pipeline: str = Field("fixed", pattern="^(fixed|naive)$")


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    rate_limit(request)
    try:
        return svc().ask(body.message.strip(), [t.model_dump() for t in body.history], body.pipeline)
    except Exception as e:  # surface provider errors (quota, timeouts) as a readable message
        return JSONResponse({"error": f"{type(e).__name__}: {str(e)[:300]}"}, status_code=502)


@app.get("/api/meta")
def meta():
    return svc().meta()


@app.get("/api/eval")
def eval_results():
    path = ROOT / "eval" / "results.json"
    if not path.exists():
        raise HTTPException(404, "No eval results yet. Run: python -m eval.run_eval")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/kb/{name}", response_class=PlainTextResponse)
def kb_article(name: str):
    files = {p.name: p for p in (ROOT / "kb").glob("*.md")}
    if name not in files:  # whitelist: only the knowledge-base files themselves
        raise HTTPException(404, "No such article")
    return files[name].read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": True}


if not os.getenv("VERCEL"):  # on Vercel the CDN serves public/; locally FastAPI does
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=ROOT / "public", html=True), name="static")
