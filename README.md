# ⚔️ Voice of Westeros
### An AI-Powered RAG Chatbot for Game of Thrones & A Song of Ice and Fire

---

Voice of Westeros is an AI-powered chatbot that answers questions about the world of Game of Thrones and A Song of Ice and Fire. Every answer is grounded in actual source material — the books themselves — rather than the language model's internal memory. If something cannot be found in the indexed documents, Voice of Westeros says so rather than guessing.

Its defining feature is **source awareness**: it knows the difference between book canon and TV show content. Questions that can only be answered from the show are always flagged with a clear warning, so book-only events are never silently mixed with show-only events.

---

## 📚 Knowledge Base

| File | Type | Description |
|------|------|-------------|
| `A Game of Thrones.pdf` | Book | ASOIAF — Book 1 |
| `A Clash of Kings.pdf` | Book | ASOIAF — Book 2 |
| `A Storm of Swords.pdf` | Book | ASOIAF — Book 3 |
| `A Feast for Crows.pdf` | Book | ASOIAF — Book 4 |
| `A Dance with Dragons.pdf` | Book | ASOIAF — Book 5 |
| `Fire and Blood.pdf` | Book | Targaryen history by George R. R. Martin |
| `The World of Ice and Fire.pdf` | Book | In-universe illustrated encyclopaedia |
| `A Knight of the Seven Kingdoms.pdf` | Book | Tales of Ser Duncan the Tall & Egg |
| `wiki.txt` | TV/Wiki | Fan wiki — TV show derived content |

> **Note:** PDFs = Book canon. `wiki.txt` = TV/wiki content. The source tagging system treats these two groups differently when answering.

---

## 🛠️ Technology Stack

| Tool | Role | Why Used |
|------|------|----------|
| **LangChain** | Pipeline framework | Chains retrieval, prompts, and LLM together cleanly |
| **FAISS** | Vector database | Fast local similarity search; index saved to disk |
| **BGE Embeddings (BAAI)** | Text → vector model | High-quality open-source embeddings, runs on CPU |
| **Groq + LLaMA 3.3 70B Versatile** | Language model | Fast inference; powerful open-weight model, free tier |
| **pypdf** | PDF reader | Reliable text extraction from large multi-page PDFs |
| **RecursiveCharSplitter** | Chunking utility | Splits at paragraph/sentence boundaries intelligently |

---

## 🏗️ RAG Architecture & Pipeline

Voice of Westeros uses a **Retrieval-Augmented Generation (RAG)** pipeline. Instead of asking the LLM to recall information from its training, the system retrieves the most relevant passages from the books first, then uses those passages as the sole basis for the answer. The LLM's job is only to synthesise and format — never to invent.

### Phase 1 — Indexing *(Runs Once)*

1. PDFs and TXT files are loaded; each page is extracted as a LangChain `Document` with source + page metadata.
2. Pages are split into **1,500-character chunks** with **200-character overlaps** using `RecursiveCharacterTextSplitter`.
3. Each chunk is converted to a **768-dimensional vector** by the BGE embedding model (runs locally on CPU).
4. All vectors are stored in a **FAISS index** and saved to disk. A registry file records indexed files so they are never re-embedded.

### Phase 2 — Query *(Every Question)*

1. Your question is embedded into the same 768-dimensional vector space.
2. FAISS performs cosine similarity search and returns the **12 most semantically similar chunks** from both the Books Index and the Wiki/Show Index simultaneously.
3. Each chunk is tagged: PDFs → `[BOOK SOURCE]`, wiki.txt → `[TV-SHOW/WIKI SOURCE]`.
4. A structured prompt is assembled: tagged chunks + last 5 conversation exchanges + length instruction + user question.
5. Groq's **LLaMA 3.3 70B Versatile** generates a cited answer, applying the source-warning rules based on which tags are present.

```
Question → BGE embed → FAISS search (Books + Wiki index) → Tag chunks
→ Assemble prompt (chunks + history + rules) → Groq LLaMA 3.3 70B Versatile → Cited answer
```

---

## 📖 vs 📺 Source Tagging

When FAISS retrieves chunks, each one is examined by `_format_docs()`. The source filename determines the tag it receives. The LLM prompt contains strict instructions on how to behave based on those tags:

| Condition | Response Format |
|-----------|----------------|
| All chunks from book PDFs only | Answers normally with `[filename \| Page X]` citations |
| All chunks from `wiki.txt` only | ⚠️ Prefixes: *"This is not in the books yet — TV show only"* |
| Chunks from both PDFs and wiki.txt | 📖 Books say: ... then 📺 TV show adds: ... |
| Topic absent from all retrieved chunks | *"I cannot find this in the documents."* — nothing else |

---

## 💬 How to Use

### Asking Questions

```bash
ask who is Jon Snow                  # Default long answer
ask short: who is Jon Snow           # 2–3 sentence concise answer
ask medium: who is Jon Snow          # 1–2 paragraph answer
ask detailed: who is Jon Snow        # Exhaustive answer with all details
```

### System Commands

```bash
initialize <path>    # Index PDFs/TXT on first run — loads from disk on repeat runs
add <path>           # Add new files to an existing index without re-indexing
list                 # Show all indexed files and their indexing timestamps
history              # Display the last 5 question-answer exchanges
clear                # Wipe conversation history (FAISS index is not affected)
exit                 # Quit Voice of Westeros
```

---

## 🧪 Example Q&A

**Book-only question:**
```
ASK ▶  What is Valyrian steel?

📖 FROM THE BOOKS
Valyrian steel is a legendary metal forged in ancient Valyria using dragonfire
and lost spells. Blades made from it hold a supernatural edge indefinitely.
The forging knowledge was lost in the Doom of Valyria, making existing blades
extraordinarily rare and valuable. It is one of the few known materials capable
of harming the Others. [A Game of Thrones.pdf | Page 312]
```

**TV/wiki-only question:**
```
ASK ▶  Who killed the Night King?

⚠️ TV / WIKI ONLY
The Night King is slain by Arya Stark during the Battle of Winterfell in Season 8.
She catches him from behind with a Valyrian steel dagger just as he is about to
destroy Bran Stark. [wiki.txt]
```

**Mixed source question:**
```
ASK ▶  What is the fate of Jon Snow?

📖📺 MIXED SOURCES
📖 Books: Jon Snow is stabbed repeatedly by Night's Watch brothers at the end of
A Dance with Dragons. His fate is unresolved. [A Dance with Dragons.pdf | Page 900]

📺 TV show: Jon is resurrected by Melisandre, kills Daenerys, and is exiled
beyond the Wall where he lives among the Free Folk.
```

---

## 🔧 Potential Improvements

| # | Improvement | Description |
|---|-------------|-------------|
| 01 | **Hybrid Search (BM25 + Semantic)** | Adding BM25 keyword search alongside semantic search would improve recall for exact proper nouns — common in ASOIAF. |
| 02 | **Cross-Encoder Re-Ranking** | A cross-encoder model could re-score and re-rank the 12 retrieved chunks by relevance, pushing the most directly relevant passages to the top. |
| 03 | **Streaming Responses** | ChatGroq supports streaming output — enabling it would let answers appear word by word in real time rather than waiting for the full response. |
| 04 | **Web UI (Streamlit / Gradio)** | A web frontend would make Voice of Westeros accessible to non-technical users with a chat window, source-tag indicators, and PDF upload. |
| 05 | **Config-Based Source Tagging** | A config file mapping each filename to a source type would make tagging more robust and prevent misclassification if files are renamed. |
| 06 | **MMR Retrieval** | Switching FAISS retriever to Maximal Marginal Relevance would ensure retrieved chunks cover diverse parts of a topic rather than near-duplicates. |
| 07 | **GPU Acceleration** | Changing `device='cpu'` to `device='cuda'` in the embedding model would drastically reduce indexing time for large PDF collections. |

---

## ⚠️ Limitations

- **Text extraction quality** — Scanned or image-based PDF pages yield no text and will be absent from the index.
- **Top-K ceiling at 12** — Only 12 chunks are retrieved per query. Broad questions may miss relevant passages that rank outside the top 12.
- **Conversation memory** — Limited to the last 5 exchanges. Earlier context in long sessions is not included in the prompt.
- **Filename-based source tagging** — Renaming a book PDF to include `wiki` in its name would misclassify it as TV-show content.
- **Books 6 & 7 absent** — *The Winds of Winter* and *A Dream of Spring* are unpublished and not in the knowledge base.
- **No internet access** — All knowledge is strictly limited to indexed files.

---

## 📝 Conclusion

Voice of Westeros demonstrates that a well-engineered RAG pipeline can turn a static collection of books into a living, conversational knowledge base. By keeping retrieval and generation strictly separated — the books supply the facts, the LLM only synthesises them — the system avoids the hallucinations and outdated knowledge that plague general-purpose AI tools when applied to niche domains like A Song of Ice and Fire.

The source-tagging system is the feature that sets this project apart. In a universe where the show diverged heavily from the books, silently mixing the two sources is not just inaccurate — it is misleading to anyone who cares about what George R. R. Martin actually wrote. Voice of Westeros treats that distinction as a first-class concern, not an afterthought.

> *"A voice in Westeros carries one message, faithfully, from sender to receiver — no interpretation, no embellishment. Voice of Westeros carries that same philosophy into every answer it gives: the books are the source of truth, and its only job is to deliver that truth clearly."*

---

*Voice of Westeros · Game of Thrones RAG Chatbot · Built with LangChain · FAISS · Groq · LLaMA 3.3 70B Versatile*

