"""
RAG Pipeline — Large PDF Edition
=================================
Optimised for single or multiple large PDFs (up to 10,000 pages).

How it works:
  First run  → PDFs are chunked → embedded → saved to FAISS index on disk
  Every run  → Index loaded from disk instantly — no re-embedding
  Chatting   → Full conversation history saved to chat_history.json

Usage:
  1. Set your Groq API key in the .env file or as an environment variable:
         GROQ_API_KEY=gsk_your_groq_key_here

  2. Run:  python rag_pipeline.py

  3. Inside the interactive loop:
         initialize("your_file.pdf")          # run once
         ask("your question")                  # chat freely
         ask("question", length="detailed")    # control answer size
         add_pdfs(["new.pdf"])                 # add more PDFs later
         list_indexed_files()                  # see what's indexed
         show_history()                        # view conversation
         clear_history()                       # wipe conversation history
"""

import os
import json
import glob
from datetime import datetime

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ── Configuration ──────────────────────────────────────────────────────────────
INDEX_DIR      = "faiss_index"
REGISTRY_FILE  = "indexed_files.json"
HISTORY_FILE   = "chat_history.json"
CHUNK_SIZE     = 1500
CHUNK_OVERLAP  = 200
RETRIEVER_K    = 12
HISTORY_WINDOW = 5

# ── Answer-length presets ──────────────────────────────────────────────────────
LENGTH_PRESETS = {
    "short":    "Answer in 2-3 sentences. Be concise and direct.",
    "medium":   "Answer in 1-2 paragraphs covering the key points.",
    "long":     "Answer thoroughly, covering all relevant points in detail.",
    "detailed": "Answer exhaustively with examples, explanations, and all relevant details from the context.",
}
DEFAULT_LENGTH = "long"

# ── Prompt template ────────────────────────────────────────────────────────────
_PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant that answers questions strictly from the provided documents.\n"
        "\n"
        "RULES — follow every one of these without exception:\n"
        "1. Answer ONLY from the Relevant Document Context below.\n"
        "2. Do NOT use any prior knowledge, even if you believe it to be relevant or correct.\n"
        "3. Cite every factual claim inline as [filename | Page X].\n"
        "4. If multiple documents contradict each other, note the discrepancy and cite both.\n"
        "5. CRITICAL SOURCE RULE — you MUST check every source tag before answering:\n"
        "   - If ALL retrieved chunks are tagged [TV-SHOW/WIKI SOURCE], you MUST begin\n"
        "     your answer with this exact phrase on its own line:\n"
        "     '⚠️ This is not in the books yet — the following is from the TV show only:\n'\n"
        "   - If chunks come from BOTH [BOOK SOURCE] and [TV-SHOW/WIKI SOURCE], begin with:\n"
        "     '📖 The books say: ...' then '📺 The TV show adds: ...'\n"
        "   - If ALL chunks are [BOOK SOURCE], answer normally with no prefix.\n"
"   - NEVER answer a TV-show-only question without the ⚠️ warning prefix.\n"
        "6. If the answer is in neither source, say exactly:\n"
        "   'I cannot find this in the documents.' — do not add anything further.\n"
        "7. Length instruction: {length_instruction}\n"
        "\n"
        "=== Conversation History ===\n"
        "{history}\n"
        "\n"
        "=== Relevant Document Context ===\n"
        "{context}\n"
        "\n"
        "Human: {question}\n"
        "Assistant:"
    ),
    input_variables=["context", "question", "history", "length_instruction"],
)

# ── Global state ───────────────────────────────────────────────────────────────
_embedding_model = None
_db              = None
_retriever       = None
_llm             = None
_chain_cache     = {}
_chat_history    = []


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {}

def save_registry(registry: dict) -> None:
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 1. EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════════════
def get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    import logging, transformers
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    transformers.logging.set_verbosity_error()

    print("  Loading embedding model (BAAI/bge-base-en-v1.5)...")
    _embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="Represent this sentence for searching relevant passages: ",
    )
    print("  ✅ Embedding model ready.")
    return _embedding_model


# ══════════════════════════════════════════════════════════════════════════════
# 2. VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
def get_vector_store(file_paths: list, embeddings) -> FAISS:
    global _db

    registry = load_registry()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    new_paths = []
    for p in file_paths:
        abs_p = os.path.abspath(p)
        if abs_p not in registry:
            new_paths.append(abs_p)
        else:
            print(f"  ⏭  Already indexed — skipping: {os.path.basename(p)}")

    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        print(f"  Loading existing FAISS index from '{INDEX_DIR}'...")
        _db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        print("  ✅ Index loaded.")
    else:
        _db = None

    if not new_paths:
        print("  ✅ All files already indexed. Nothing new to embed.")
        return _db

    for file_path in new_paths:
        name = os.path.basename(file_path)
        print(f"\n  📄 Embedding new file: {name}")
        try:
            # ── Plain text files ──────────────────────────────────────────
            if file_path.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                docs = [Document(
                    page_content=text,
                    metadata={"source": file_path, "page": 1}
                )]

            # ── PDF files ─────────────────────────────────────────────────
            else:
                reader = PdfReader(file_path, strict=False)
                docs = [
                    Document(
                        page_content=page.extract_text() or "",
                        metadata={"source": file_path, "page": i + 1},
                    )
                    for i, page in enumerate(reader.pages)
                    if (page.extract_text() or "").strip()
                ]

            if not docs:
                print(f"     ⚠️  No text extracted from {name} — skipping.")
                continue

            print(f"     Loaded {len(docs)} page(s).")
            chunks = splitter.split_documents(docs)
            print(f"     Created {len(chunks)} chunks. Embedding...")

            new_db = FAISS.from_documents(chunks, embeddings)
            _db = new_db if _db is None else (_db.merge_from(new_db) or _db)

            registry[file_path] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_registry(registry)
            print(f"     ✅ Done — {name} added to index.")

        except Exception as e:
            print(f"     ❌ Failed to index {name}: {e}")

    if _db is None:
        raise RuntimeError(
            "No files were successfully indexed. "
            "Check that your files are valid and not corrupted."
        )

    _db.save_local(INDEX_DIR)
    print(f"\n  ✅ FAISS index updated and saved to '{INDEX_DIR}'.")
    return _db


# ══════════════════════════════════════════════════════════════════════════════
# 3. LLM
# ══════════════════════════════════════════════════════════════════════════════
def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    print("  Connecting to Groq (llama-3.3-70b-versatile)...")
    _llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )
    print("  ✅ LLM ready.")
    return _llm


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHAT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
def load_history() -> None:
    global _chat_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            _chat_history = json.load(f)
        print(f"  ✅ Loaded {len(_chat_history)} messages from previous session.")
    else:
        _chat_history = []
        print("  No previous history found. Starting fresh.")

def save_history() -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(_chat_history, f, indent=2)

def format_history_for_prompt() -> str:
    recent = _chat_history[-(HISTORY_WINDOW * 2):]
    if not recent:
        return "No previous conversation."
    lines = [
        f"{'Human' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    ]
    return "\n".join(lines)

def show_history(last_n: int = 5) -> None:
    pairs = _chat_history[-(last_n * 2):]
    if not pairs:
        print("No history yet.")
        return
    for msg in pairs:
        icon = "🙋" if msg["role"] == "user" else "🤖"
        print(f"{icon} [{msg.get('time', '')}]\n{msg['content']}\n")

def clear_history() -> None:
    global _chat_history
    _chat_history = []
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    print("✅ Chat history cleared.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CHAIN
# ══════════════════════════════════════════════════════════════════════════════
def _format_docs(docs) -> str:
    parts = []
    for d in docs:
        name = os.path.basename(d.metadata.get('source', 'doc'))
        page = d.metadata.get('page', '?')
        # Tag the source type so the LLM can distinguish books from wiki
        if name.lower().endswith('.txt') or 'wiki' in name.lower():
            source_type = "TV-SHOW/WIKI SOURCE"
        else:
            source_type = "BOOK SOURCE"
        parts.append(
            f"[{source_type} | {name} | Page {page}]\n{d.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def get_chain(length_key: str):
    if length_key in _chain_cache:
        return _chain_cache[length_key]

    length_instruction = LENGTH_PRESETS[length_key]
    chain = (
        RunnableParallel({
            "context":            _retriever | RunnableLambda(_format_docs),
            "question":           RunnablePassthrough(),
            "history":            RunnableLambda(lambda _: format_history_for_prompt()),
            "length_instruction": RunnableLambda(lambda _: length_instruction),
        })
        | _PROMPT
        | get_llm()
        | StrOutputParser()
    )
    _chain_cache[length_key] = chain
    return chain

def _invalidate_chain_cache() -> None:
    _chain_cache.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 6. INITIALIZE
# ══════════════════════════════════════════════════════════════════════════════
def initialize(pdfs) -> None:
    global _retriever

    file_paths = _resolve_file_paths(pdfs)

    print("\n🔧 INITIALIZING RAG SYSTEM")
    print("=" * 50)
    print(f"  Total files provided : {len(file_paths)}")

    embeddings = get_embedding_model()
    db         = get_vector_store(file_paths, embeddings)

    _retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    _invalidate_chain_cache()
    get_llm()
    load_history()

    print("\n" + "=" * 50)
    print("✅ System ready!")
    print("   ask('question')                    — default length (long)")
    print("   ask('question', length='short')    — short / medium / long / detailed")
    print("   add_pdfs([...])                    — add more files")
    print("   list_indexed_files()               — see all indexed files")
    print("   show_history()                     — view conversation")
    print("   clear_history()                    — wipe conversation history")
    print("=" * 50 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7. ADD MORE FILES
# ══════════════════════════════════════════════════════════════════════════════
def add_pdfs(pdfs) -> None:
    global _retriever

    if _retriever is None:
        print("❌ Run initialize() first before adding more files.")
        return

    file_paths = _resolve_file_paths(pdfs)
    embeddings = get_embedding_model()

    print("\n➕ ADDING NEW FILES TO INDEX")
    print("=" * 50)
    db = get_vector_store(file_paths, embeddings)

    _retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    _invalidate_chain_cache()
    print("✅ Index updated. You can now query the new files.")


# ══════════════════════════════════════════════════════════════════════════════
# 8. UTILITY
# ══════════════════════════════════════════════════════════════════════════════
def list_indexed_files() -> None:
    registry = load_registry()
    if not registry:
        print("No files indexed yet.")
        return
    print(f"📚 {len(registry)} file(s) in index:")
    for path, ts in registry.items():
        print(f"   • {os.path.basename(path)}  (indexed: {ts})")

def _resolve_file_paths(pdfs) -> list:
    """Normalise input to a flat list of .pdf and .txt file paths."""
    if isinstance(pdfs, str):
        if os.path.isdir(pdfs):
            pdf_files = sorted(glob.glob(os.path.join(pdfs, "**/*.pdf"), recursive=True))
            txt_files = sorted(glob.glob(os.path.join(pdfs, "**/*.txt"), recursive=True))
            paths = pdf_files + txt_files
            if not paths:
                raise ValueError(f"No PDF or TXT files found in directory: {pdfs}")
            print(f"  Found {len(paths)} file(s) in '{pdfs}' ({len(pdf_files)} PDF, {len(txt_files)} TXT).")
            return paths
        return [pdfs]
    return list(pdfs)


# ══════════════════════════════════════════════════════════════════════════════
# 9. ASK
# ══════════════════════════════════════════════════════════════════════════════
def ask(query: str, length: str = None) -> str | None:
    if _retriever is None:
        print(
            "❌ System not initialized.\n"
            "   Run initialize('docs') first.\n"
            "   If you've already indexed your files, initialize() loads from disk instantly."
        )
        return None

    length_key = (length or DEFAULT_LENGTH).lower()
    if length_key not in LENGTH_PRESETS:
        print(f"❌ Invalid length '{length_key}'. Choose from: {list(LENGTH_PRESETS.keys())}")
        return None

    print(f"\n🙋 You: {query}")
    print(f"📏 Length: {length_key}")
    print("🤖 Thinking...\n")

    chain    = get_chain(length_key)
    response = chain.invoke(query).strip()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _chat_history.append({"role": "user",      "content": query,    "time": ts, "length": length_key})
    _chat_history.append({"role": "assistant", "content": response, "time": ts})
    save_history()

    print("=" * 60)
    print(f"🤖 Assistant [{length_key.upper()}]:")
    print(response)
    print("=" * 60)
    return response


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — Interactive CLI
# ══════════════════════════════════════════════════════════════════════════════
def _load_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["GROQ_API_KEY"] = key
                    return key
    raise ValueError(
        "Groq API key not found!\n"
        "Set it in a .env file:  GROQ_API_KEY=gsk_your_key_here\n"
        "Or export it:           export GROQ_API_KEY=gsk_your_key_here"
    )

if __name__ == "__main__":
    _load_api_key()
    print("✅ Groq API key loaded.")

    print("\n📖 RAG Pipeline — Interactive Mode")
    print("Commands:")
    print("  initialize <path_or_folder>   — index PDFs and TXT files")
    print("  ask <question>                — ask a question (default: long)")
    print("  ask short: <question>         — short answer")
    print("  ask medium: <question>        — medium answer")
    print("  ask detailed: <question>      — detailed answer")
    print("  add <path_or_folder>          — add more files")
    print("  list                          — list indexed files")
    print("  history                       — show last 5 exchanges")
    print("  clear                         — clear history")
    print("  exit                          — quit\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower == "exit":
            print("Goodbye!")
            break
        elif lower == "initialize" or lower.startswith("initialize "):
            parts = user_input.split(None, 1)
            if len(parts) < 2:
                print("❌ Usage: initialize <path_or_folder>")
                print("   Example: initialize docs")
                print("            initialize report.pdf")
            else:
                initialize(parts[1].strip())
        elif lower.startswith("add "):
            add_pdfs(user_input[len("add "):].strip())
        elif lower == "list":
            list_indexed_files()
        elif lower == "history":
            show_history()
        elif lower == "clear":
            clear_history()
        elif lower == "ask" or lower.startswith("ask "):
            rest = user_input[3:].strip()
            if not rest:
                print("❌ Usage: ask <question>  or  ask short: <question>")
            else:
                length = None
                for preset in LENGTH_PRESETS:
                    if rest.lower().startswith(f"{preset}:"):
                        length = preset
                        rest   = rest[len(preset) + 1:].strip()
                        break
                ask(rest, length=length)
        else:
            ask(user_input)