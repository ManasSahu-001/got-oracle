# =============================================================================
# GOT ORACLE — APP.PY
# FastAPI server — connects UI to RAG pipeline
# Run with: uvicorn app:app --reload
# Open: http://localhost:8000
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
import rag_engine_pipeline as rag


# ── Lifespan Event (Startup + Shutdown) ──────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("🚀 Loading RAG system...")

    # load FAISS + embeddings
    rag.initialize("docs")

    print("✅ RAG system ready")

    yield

    print("🛑 Server shutting down")

# ── Initialize FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="GOT Oracle — RAG Chatbot",
    lifespan=lifespan
)


# ── Allow frontend to talk to backend ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Serve static files (HTML, CSS, JS) ───────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request Model ────────────────────────────────────────────────────────────
class Question(BaseModel):
    question: str
    length: str = "long"   # short | medium | long | detailed


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    """Serve the main chat UI"""
    return FileResponse("static/index.html")


@app.post("/ask")
def ask(payload: Question):
    """Ask a question to the RAG system"""

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = rag.ask(payload.question, length=payload.length)

    except Exception as e:

        if "rate_limit" in str(e).lower() or "429" in str(e):
            return {
                "answer": "⚠️ The Oracle is resting — daily limit reached. Please try again later."
            }

        raise HTTPException(status_code=500, detail=str(e))

    if answer is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    return {"answer": answer}


@app.post("/clear")
def clear():
    """Clear chat history"""
    rag.clear_history()
    return {"status": "History cleared."}


@app.get("/health")
def health():
    """Check if RAG system is ready"""
    return {
        "status": "ok",
        "initialized": rag._retriever is not None
    }