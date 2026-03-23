# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Package manager:** If `uv.lock` is present, use `uv` as the package manager. If `poetry.lock` is present, use `poetry` as the package manager.

**Install dependencies:**
```bash
uv sync
```

**Run the application:**
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Access:**
- Web UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

**Environment setup:** Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for course materials. FastAPI serves both the API and the static frontend.

### Request Flow

1. Frontend POSTs `{query, session_id}` to `/api/query`
2. `RAGSystem` retrieves conversation history from `SessionManager` and calls `AIGenerator`
3. Claude (claude-sonnet-4-20250514) decides to invoke the `search_course_content` tool via `ToolManager`
4. `CourseSearchTool` queries `VectorStore` (ChromaDB) using sentence-transformer embeddings
5. Claude generates a final answer using the retrieved chunks
6. Response `{answer, sources, session_id}` is returned and session history updated

### Key Backend Components (`backend/`)

| File | Role |
|------|------|
| `app.py` | FastAPI app, routes, startup event (indexes docs) |
| `rag_system.py` | Main orchestrator — coordinates all components |
| `document_processor.py` | Parses `docs/course*.txt` into `Course`/`CourseChunk` objects |
| `vector_store.py` | ChromaDB wrapper — two collections: `course_catalog` and `course_content` |
| `ai_generator.py` | Claude API calls with tool use and conversation history |
| `search_tools.py` | Tool definitions and `CourseSearchTool` execution |
| `session_manager.py` | In-memory conversation sessions |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |
| `config.py` | All configuration (API key, model name, chunk sizes, ChromaDB path) |

### Document Format (`docs/course*.txt`)

Course files must follow this structure for `DocumentProcessor` to parse them:
```
Course Title: [Name]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Title]
Lesson Link: [URL]
[Content...]

Lesson 1: [Title]
...
```

### Configuration (`config.py`)

Key tunables:
- `CHUNK_SIZE` = 800 chars, `CHUNK_OVERLAP` = 100 chars
- `MAX_RESULTS` = 5 (semantic search hits returned to Claude)
- `MAX_HISTORY` = 2 (conversation turns kept in context)
- `CHROMA_PATH` = `./chroma_db` (persistent vector DB; delete to force re-indexing)
- `EMBEDDING_MODEL` = `all-MiniLM-L6-v2`
- `ANTHROPIC_MODEL` = `claude-sonnet-4-20250514`

### Frontend (`frontend/`)

Vanilla JS/HTML/CSS — no build step. Served as static files by FastAPI. `script.js` manages session IDs, API calls, and markdown rendering via Marked.js.
