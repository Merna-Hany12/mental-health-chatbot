"""
FastAPI Application — Mental Health Chatbot
============================================
Endpoints:
    GET  /              → Serve chat UI (HTML)
    POST /api/chat      → Main chat endpoint
    GET  /api/health    → Health check
    GET  /api/modules   → Module status
    POST /api/index     → Trigger dataset indexing (admin)

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Mental Health Chatbot API",
    description = "RAG-powered empathetic mental health support chatbot.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve static files (CSS, JS, images)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Pydantic Models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:      str             = Field(..., min_length=1, max_length=2000,
                                          description="User's message")
    session_id:   Optional[str]   = Field(None, description="Optional session identifier")
    show_sources: bool            = Field(True,  description="Include RAG sources in response")


class ChatResponse(BaseModel):
    answer:     str
    language:   dict
    intent:     dict
    emotion:    Optional[dict]
    sources:    list
    used_rag:   bool
    latency_ms: float


class IndexRequest(BaseModel):
    secret: str = Field(..., description="Admin secret to authorize indexing")


# ── Orchestrator (singleton) ───────────────────────────────────────────────────
from orchestrator import Orchestrator

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(
            groq_api_key   = os.getenv("groq_api_key", ""),
            qdrant_url     = os.getenv("qdrant_url", ""),
            qdrant_api_key = os.getenv("qdrant_api_key", ""),
        )
    return _orchestrator


@app.on_event("startup")
def warm_up_orchestrator() -> None:
    """Load models and connect external services before the first chat request."""
    t0 = time.perf_counter()
    logger.info("Starting application warmup...")
    try:
        get_orchestrator().warm_up()
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"Application warmup complete in {elapsed} ms.")
    except Exception:
        logger.exception("Application warmup failed. The first chat request will retry initialization.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request) -> HTMLResponse:
    """Serve the chat UI."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found. Place index.html in templates/</h1>", status_code=404)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    Runs the full pipeline:
    Language Detection → Intent Classification →
    (Emotion Classification) → (RAG Answer)
    """
    t0 = time.perf_counter()
    try:
        logger.info(f"/api/chat received message ({len(req.message)} chars).")
        bot = get_orchestrator()
        logger.info("Calling orchestrator chat pipeline...")
        result = bot.chat(req.message)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"/api/chat pipeline finished in {latency} ms.")

        # Strip sources if not requested
        if not req.show_sources:
            result["sources"] = []

        return ChatResponse(
            answer     = result["answer"],
            language   = result["language"],
            intent     = result["intent"],
            emotion    = result.get("emotion"),
            sources    = result.get("sources", []),
            used_rag   = result["used_rag"],
            latency_ms = latency,
        )
    except Exception as e:
        import traceback
        logger.error(f"Chat endpoint error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health() -> JSONResponse:
    """Health check — returns module status."""
    try:
        bot    = get_orchestrator()
        status = bot.health_check()
        ok     = status["language_detector"] and status["intent_classifier"]
        return JSONResponse(
            content = {"status": "ok" if ok else "degraded", "modules": status},
            status_code = 200 if ok else 207,
        )
    except Exception as e:
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)


@app.get("/api/modules")
async def modules_info() -> JSONResponse:
    """Detailed module information."""
    return JSONResponse({
        "modules": [
            {
                "id": 1,
                "name": "Language Detector",
                "tech": "TF-IDF (char n-grams) + Logistic Regression",
                "languages": ["English", "Arabic", "French", "Spanish", "German",
                              "Italian", "Portuguese", "Russian", "Turkish", "Hindi"],
            },
            {
                "id": 2,
                "name": "Emotion Classifier",
                "tech": "DistilBERT fine-tuned on dair-ai/emotion",
                "emotions": ["sadness", "joy", "love", "anger", "fear", "surprise"],
            },
            {
                "id": 3,
                "name": "Intent Classifier",
                "tech": "Few-shot prompting via Groq LLM",
                "intents": ["greeting", "goodbye", "gratitude",
                            "asking_mental_health_question", "out_of_scope"],
            },
            {
                "id": 4,
                "name": "RAG Pipeline",
                "tech": "LangChain + Qdrant Cloud + SentenceTransformer + Groq",
                "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
                "llm": "openai/gpt-oss-120b",
                "vector_db": "Qdrant Cloud",
            },
        ]
    })


@app.post("/api/index")
async def trigger_indexing(req: IndexRequest) -> JSONResponse:
    """
    Admin endpoint: triggers one-time dataset indexing into Qdrant.
    Protect with a secret in production.
    """
    admin_secret = os.getenv("ADMIN_SECRET", "change-me-in-production")
    if req.secret != admin_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret.")

    qdrant_url     = os.getenv("qdrant_url", "")
    qdrant_api_key = os.getenv("qdrant_api_key", "")

    if not qdrant_url or not qdrant_api_key:
        raise HTTPException(status_code=400, detail="QDRANT_URL / QDRANT_API_KEY not set.")

    try:
        from modules.rag_pipeline import RAGPipeline
        # Run indexing in background (fire-and-forget via asyncio)
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None,
            RAGPipeline.index_dataset,
            qdrant_url,
            qdrant_api_key,
        )
        return JSONResponse({"status": "indexing_started", "message": "Check server logs."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = os.getenv("APP_HOST", "0.0.0.0"),
        port    = int(os.getenv("APP_PORT", 8000)),
        reload  = os.getenv("APP_DEBUG", "false").lower() == "true",
        workers = 1,
    )
