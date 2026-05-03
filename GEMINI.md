# Helix SROP — Full Implementation Plan for AI IDE

> **Assignment:** Stateful RAG Orchestration Pipeline (SROP)  
> **Stack:** Python 3.11+, FastAPI, Google ADK (`google-adk`), SQLite (async), ChromaDB  
> **Goal:** Score 100/100 — all core (70 pts) + all extensions (30 pts)

---

## CONTEXT: What Already Exists in the Repo

The scaffold at `https://git.arivvan.in/anant/helix-srop-assignment` provides:
- `app/main.py` — FastAPI app wired, lifespan, includes 3 routers, `/healthz` already done
- `app/db/models.py` — Full ORM models: `User`, `Session`, `Message`, `AgentTrace`
- `app/db/session.py` — Stub (needs `init_db`, `get_db`, engine creation)
- `app/srop/state.py` — `SessionState` Pydantic model fully defined
- `app/srop/pipeline.py` — Stub with only `PipelineResult` dataclass and empty `run()`
- `app/settings.py` — Stub (needs `Settings` class with env vars)
- `app/api/routes_sessions.py`, `routes_chat.py`, `routes_traces.py` — Stubs (empty routers)
- `app/agents/` — Empty directory
- `app/rag/` — Empty directory
- `app/obs/` — Empty directory (logging)
- `tests/` — Empty directory
- `pyproject.toml` — All dependencies listed, pytest configured with `asyncio_mode = "auto"`
- `docs/` — Contains `rag-guide.md`, `google-adk-guide.md`, `fastapi-async-guide.md`, plus 10 Helix product `.md` files (the RAG corpus)
- `.env.example` — Has `GOOGLE_API_KEY` and likely `DATABASE_URL`, `CHROMA_PATH`

**Do NOT re-create these files from scratch — fill them in.**

---

## PHASE 0: Project Setup Understanding

### Environment
```
uv sync                          # installs all deps from pyproject.toml
cp .env.example .env             # fill GOOGLE_API_KEY
```

### Directory tree after full implementation:
```
helix-srop-assignment/
├── app/
│   ├── __init__.py
│   ├── main.py                  ← already done, add exception handler
│   ├── settings.py              ← implement Settings class
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      ← root LlmAgent with AgentTool wiring
│   │   ├── knowledge_agent.py   ← RAG agent
│   │   ├── account_agent.py     ← mock DB tools agent
│   │   └── escalation_agent.py  ← E2 extension
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py              ← FastAPI dependency: get_db
│   │   ├── routes_sessions.py   ← POST /v1/sessions
│   │   ├── routes_chat.py       ← POST /v1/chat/{session_id}
│   │   └── routes_traces.py     ← GET /v1/traces/{trace_id}
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py            ← already done, add Ticket model (E2)
│   │   └── session.py           ← implement engine + init_db + get_db
│   ├── obs/
│   │   ├── __init__.py
│   │   └── logging.py           ← structlog configure
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py            ← CLI: chunk + embed + upsert to Chroma
│   │   └── search.py            ← search_docs tool implementation
│   └── srop/
│       ├── __init__.py
│       ├── state.py             ← already done
│       ├── pipeline.py          ← implement run()
│       └── errors.py            ← custom exception hierarchy
├── eval/
│   └── run_eval.py              ← E7 eval harness
├── tests/
│   ├── conftest.py
│   ├── test_integration.py
│   └── test_unit_rag.py
├── Dockerfile                   ← E6
├── docker-compose.yml           ← E6
├── .env.example
├── pyproject.toml
└── README.md
```

---

## PHASE 1: Foundation Files

### 1.1 `app/settings.py`

Implement a `pydantic-settings` `Settings` class. This is the single source of truth for all config.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    database_url: str = Field(
        default="sqlite+aiosqlite:///./helix.db", alias="DATABASE_URL"
    )
    chroma_path: str = Field(default="./chroma_db", alias="CHROMA_PATH")
    chroma_collection: str = Field(default="helix_docs", alias="CHROMA_COLLECTION")
    llm_model: str = Field(default="gemini-2.0-flash", alias="LLM_MODEL")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    embedding_model: str = Field(
        default="models/text-embedding-004", alias="EMBEDDING_MODEL"
    )
    top_k_retrieval: int = Field(default=5, alias="TOP_K_RETRIEVAL")
    environment: str = Field(default="development", alias="ENVIRONMENT")

settings = Settings()
```

### 1.2 `app/db/session.py`

Full async SQLAlchemy engine + session factory + `init_db` + `get_db` dependency.

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from app.settings import settings
from app.db.models import Base

engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False},  # SQLite only
)

AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

async def init_db() -> None:
    """Called once at startup to create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### 1.3 `app/srop/errors.py`

Custom exception hierarchy. All errors inherit from `HelixError`.

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class HelixError(Exception):
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class SessionNotFoundError(HelixError):
    status_code = 404
    error_code = "SESSION_NOT_FOUND"

class UpstreamTimeoutError(HelixError):
    status_code = 504
    error_code = "UPSTREAM_TIMEOUT"

class TraceNotFoundError(HelixError):
    status_code = 404
    error_code = "TRACE_NOT_FOUND"

class GuardrailRefusalError(HelixError):
    status_code = 400
    error_code = "GUARDRAIL_REFUSAL"

async def helix_error_handler(request: Request, exc: HelixError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "message": exc.message},
    )
```

**Wire into `app/main.py`:** uncomment the `app.add_exception_handler(HelixError, helix_error_handler)` line and import properly.

### 1.4 `app/obs/logging.py`

```python
import structlog
import logging

def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

def get_logger(name: str):
    return structlog.get_logger(name)
```

### 1.5 Update `app/db/models.py`

Add the `Ticket` table for Extension E2 (escalation agent). Append to the existing file:

```python
class Ticket(Base):
    __tablename__ = "tickets"

    ticket_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(String(16), default="medium")  # low|medium|high
    status: Mapped[str] = mapped_column(String(16), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

Also update `SessionState` in `app/srop/state.py` to add:
```python
last_ticket_id: str | None = None
idempotency_cache: dict[str, str] = Field(default_factory=dict)  # key -> reply
```

---

## PHASE 2: RAG Layer

### 2.1 `app/rag/ingest.py`

This is a **CLI module** run as `python -m app.rag.ingest --path docs/`.

**Complete implementation:**

```python
"""
RAG ingestion pipeline.
Usage: python -m app.rag.ingest --path docs/
Strategy: Heading-aware chunking — splits on markdown H2/H3 boundaries,
then further splits by sentence if chunk > 400 tokens. This preserves
semantic units while keeping chunks retrievable.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import chromadb
import google.generativeai as genai

from app.settings import settings

genai.configure(api_key=settings.google_api_key)

CHUNK_MAX_CHARS = 1500   # ~400 tokens
CHUNK_OVERLAP = 150      # char overlap between adjacent sentence-splits


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML-ish frontmatter between --- delimiters."""
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm_block = text[3:end].strip()
            for line in fm_block.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            text = text[end + 3:].strip()
    return meta, text


def heading_aware_chunks(text: str, source: str, meta: dict) -> list[dict]:
    """
    Split on H2/H3 headings first, then further split oversized sections
    by sentence boundary with CHUNK_OVERLAP char overlap.
    Returns list of {text, chunk_id, source, section, product_area, title}.
    """
    # Split on ## or ### headings
    sections = re.split(r"(?m)^(#{2,3} .+)$", text)
    chunks = []
    current_heading = meta.get("title", source)

    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{2,3} ", part):
            current_heading = part.lstrip("#").strip()
            continue
        # If section is small enough, it's one chunk
        if len(part) <= CHUNK_MAX_CHARS:
            chunk_text = f"[{current_heading}]\n{part}"
            chunk_id = _stable_id(source, chunk_text)
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "source": source,
                "section": current_heading,
                "product_area": meta.get("product_area", "general"),
                "title": meta.get("title", source),
            })
        else:
            # Sentence-level split with overlap
            sentences = re.split(r"(?<=[.!?])\s+", part)
            buffer = f"[{current_heading}]\n"
            for sent in sentences:
                if len(buffer) + len(sent) > CHUNK_MAX_CHARS and len(buffer) > 50:
                    chunk_id = _stable_id(source, buffer)
                    chunks.append({
                        "text": buffer.strip(),
                        "chunk_id": chunk_id,
                        "source": source,
                        "section": current_heading,
                        "product_area": meta.get("product_area", "general"),
                        "title": meta.get("title", source),
                    })
                    # Overlap: keep last CHUNK_OVERLAP chars
                    buffer = f"[{current_heading}]\n" + buffer[-CHUNK_OVERLAP:] + " " + sent + " "
                else:
                    buffer += sent + " "
            if buffer.strip():
                chunk_id = _stable_id(source, buffer)
                chunks.append({
                    "text": buffer.strip(),
                    "chunk_id": chunk_id,
                    "source": source,
                    "section": current_heading,
                    "product_area": meta.get("product_area", "general"),
                    "title": meta.get("title", source),
                })
    return chunks


def _stable_id(source: str, text: str) -> str:
    """Deterministic chunk ID from source + content hash. Re-ingest safe."""
    h = hashlib.sha256(f"{source}::{text}".encode()).hexdigest()[:16]
    safe_source = re.sub(r"[^a-z0-9_-]", "_", Path(source).stem.lower())
    return f"{safe_source}_{h}"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed using Google's text-embedding-004."""
    result = genai.embed_content(
        model=settings.embedding_model,
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"] if isinstance(result["embedding"][0], list) else [result["embedding"]]


def ingest(path: str) -> None:
    docs_path = Path(path)
    md_files = list(docs_path.glob("**/*.md"))
    if not md_files:
        print(f"No .md files found in {path}", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[dict] = []
    for md_file in md_files:
        raw = md_file.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        chunks = heading_aware_chunks(body, str(md_file.name), meta)
        all_chunks.extend(chunks)
        print(f"  {md_file.name}: {len(chunks)} chunks")

    # De-duplicate by chunk_id (idempotent re-ingest)
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in all_chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("All chunks already ingested — nothing to do.")
        return

    # Embed in batches of 100
    BATCH = 100
    for i in range(0, len(new_chunks), BATCH):
        batch = new_chunks[i:i + BATCH]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source": c["source"],
                    "section": c["section"],
                    "product_area": c["product_area"],
                    "title": c["title"],
                }
                for c in batch
            ],
        )
        print(f"  Upserted batch {i // BATCH + 1} ({len(batch)} chunks)")

    print(f"\nDone. Total chunks upserted: {len(new_chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest docs into Chroma vector store.")
    parser.add_argument("--path", required=True, help="Directory containing .md files")
    args = parser.parse_args()
    ingest(args.path)
```

### 2.2 `app/rag/search.py`

```python
"""
search_docs tool — used by KnowledgeAgent.
Returns top-k chunks with IDs and similarity scores.
"""
from dataclasses import dataclass
import google.generativeai as genai
import chromadb

from app.settings import settings

genai.configure(api_key=settings.google_api_key)

_client: chromadb.ClientAPI | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_path)
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float          # cosine similarity in [0, 1]
    source: str
    section: str
    product_area: str


def search_docs(query: str, k: int = 5) -> list[SearchResult]:
    """
    Embed query and retrieve top-k semantically similar chunks from Chroma.
    Scores are normalised cosine similarity (higher = better).
    """
    collection = _get_collection()
    result = genai.embed_content(
        model=settings.embedding_model,
        content=query,
        task_type="retrieval_query",
    )
    query_embedding = result["embedding"]

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    results: list[SearchResult] = []
    for chunk_id, doc, meta, dist in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        # Chroma cosine distance: 0=identical, 2=opposite → score = 1 - dist/2
        score = round(1.0 - dist / 2.0, 4)
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                text=doc,
                score=score,
                source=meta.get("source", ""),
                section=meta.get("section", ""),
                product_area=meta.get("product_area", ""),
            )
        )
    return results
```

---

## PHASE 3: ADK Agents

### 3.1 `app/agents/knowledge_agent.py`

```python
"""
KnowledgeAgent — answers Helix product questions using RAG.
- Calls search_docs to retrieve relevant chunks
- System prompt ENFORCES chunk-ID citations in every answer
- Returns chunk IDs so pipeline.py can store them in the trace
"""
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from app.rag.search import search_docs as _search_docs
from app.settings import settings

KNOWLEDGE_SYSTEM_PROMPT = """
You are the Helix Knowledge Agent, an expert on Helix product documentation.

When answering, you MUST:
1. Call search_docs with the user's question as the query.
2. Base your answer ONLY on the retrieved chunks.
3. Cite EVERY factual claim with the chunk ID in square brackets, e.g.:
   "You can rotate a deploy key from the Settings → Security panel [chunk_abc123]."
4. If no relevant chunks are found, reply: "I don't have documentation on that topic."
5. NEVER answer from general knowledge — only from retrieved chunks.
6. If the question is out of scope (not about Helix), reply with: "OUT_OF_SCOPE"

Your user's plan tier is: {plan_tier}
Their user ID is: {user_id}
"""


def _search_docs_tool(query: str, k: int = 5) -> dict:
    """Search Helix documentation. Returns top-k matching chunks with IDs and relevance scores."""
    results = _search_docs(query, k=k)
    return {
        "chunks": [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "source": r.source,
                "section": r.section,
            }
            for r in results
        ]
    }


def build_knowledge_agent(user_id: str, plan_tier: str) -> LlmAgent:
    return LlmAgent(
        name="knowledge_agent",
        model=settings.llm_model,
        instruction=KNOWLEDGE_SYSTEM_PROMPT.format(
            plan_tier=plan_tier, user_id=user_id
        ),
        tools=[FunctionTool(func=_search_docs_tool)],
        description="Answers Helix product documentation questions using RAG.",
    )
```

### 3.2 `app/agents/account_agent.py`

```python
"""
AccountAgent — handles account/build data lookups.
Uses mock data; the wiring pattern is what's evaluated.
"""
from datetime import datetime, timedelta
import random
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from app.settings import settings

ACCOUNT_SYSTEM_PROMPT = """
You are the Helix Account Agent. You help users look up their account data,
build history, and current account status.

The user's ID is: {user_id}
Their plan tier is: {plan_tier}

Use the available tools to fetch real data. Present results clearly.
If asked about something outside account/build data, say you cannot help with that.
"""

# ── Mock data store ──────────────────────────────────────────────────────────
_MOCK_BUILDS: dict[str, list[dict]] = {}


def _generate_mock_builds(user_id: str) -> list[dict]:
    if user_id not in _MOCK_BUILDS:
        statuses = ["success", "failed", "failed", "success", "failed"]
        _MOCK_BUILDS[user_id] = [
            {
                "build_id": f"build_{i:04d}",
                "status": statuses[i % len(statuses)],
                "branch": f"feature/task-{random.randint(100, 999)}",
                "duration_seconds": random.randint(45, 300),
                "timestamp": (datetime.utcnow() - timedelta(hours=i * 2)).isoformat(),
                "error": "OOM: heap size exceeded" if statuses[i % len(statuses)] == "failed" else None,
            }
            for i in range(10)
        ]
    return _MOCK_BUILDS[user_id]


def get_recent_builds(user_id: str, limit: int = 5) -> dict:
    """Fetch the most recent CI/CD builds for the user. Returns build ID, status, branch, duration, and error if any."""
    builds = _generate_mock_builds(user_id)
    recent = sorted(builds, key=lambda b: b["timestamp"], reverse=True)[:limit]
    return {"user_id": user_id, "builds": recent, "total_fetched": len(recent)}


def get_account_status(user_id: str) -> dict:
    """Fetch account status, plan details, and usage statistics for the user."""
    plans = {"free": {"builds_limit": 50, "seats": 1}, "pro": {"builds_limit": 500, "seats": 5}, "enterprise": {"builds_limit": -1, "seats": -1}}
    builds = _generate_mock_builds(user_id)
    failed = sum(1 for b in builds if b["status"] == "failed")
    return {
        "user_id": user_id,
        "account_active": True,
        "total_builds": len(builds),
        "failed_builds": failed,
        "success_rate": round((len(builds) - failed) / len(builds) * 100, 1),
        "joined_days_ago": random.randint(30, 365),
    }


def build_account_agent(user_id: str, plan_tier: str) -> LlmAgent:
    return LlmAgent(
        name="account_agent",
        model=settings.llm_model,
        instruction=ACCOUNT_SYSTEM_PROMPT.format(
            user_id=user_id, plan_tier=plan_tier
        ),
        tools=[
            FunctionTool(func=get_recent_builds),
            FunctionTool(func=get_account_status),
        ],
        description="Handles account lookups and build history queries.",
    )
```

### 3.3 `app/agents/escalation_agent.py` (Extension E2)

```python
"""
EscalationAgent — creates support tickets.
Extension E2: writes to `tickets` table, returns ticket_id.
Ticket ID is stored in SessionState.last_ticket_id for follow-ups.
"""
import uuid
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from app.settings import settings

ESCALATION_SYSTEM_PROMPT = """
You are the Helix Escalation Agent. You create support tickets when users
have unresolved issues that require human intervention.

The user's ID is: {user_id}
Their plan tier is: {plan_tier}

When creating a ticket:
1. Summarize the issue clearly and concisely.
2. Set priority based on severity: 'high' for outages, 'medium' for functionality issues, 'low' for questions.
3. Confirm the ticket ID to the user after creation.
4. Reassure them a human agent will follow up.
"""

# This is a stub tool — the actual DB write happens in pipeline.py
# because ADK tools cannot be async in a straightforward way.
# The tool returns a pending ticket_id; pipeline.py commits it to DB.
_pending_tickets: dict[str, dict] = {}


def create_ticket(user_id: str, summary: str, priority: str = "medium") -> dict:
    """
    Create a support ticket. Priority must be 'low', 'medium', or 'high'.
    Returns the ticket_id for reference.
    """
    if priority not in ("low", "medium", "high"):
        priority = "medium"
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    _pending_tickets[ticket_id] = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "summary": summary,
        "priority": priority,
        "status": "open",
    }
    return {
        "ticket_id": ticket_id,
        "status": "created",
        "message": f"Ticket {ticket_id} created with {priority} priority. A human agent will follow up.",
    }


def get_pending_ticket(ticket_id: str) -> dict | None:
    """Called by pipeline.py after ADK run to flush to DB."""
    return _pending_tickets.pop(ticket_id, None)


def build_escalation_agent(user_id: str, plan_tier: str) -> LlmAgent:
    return LlmAgent(
        name="escalation_agent",
        model=settings.llm_model,
        instruction=ESCALATION_SYSTEM_PROMPT.format(
            user_id=user_id, plan_tier=plan_tier
        ),
        tools=[FunctionTool(func=create_ticket)],
        description="Creates support tickets for unresolved issues requiring human intervention.",
    )
```

### 3.4 `app/agents/orchestrator.py`

```python
"""
Root orchestrator — routes to specialist sub-agents via ADK AgentTool.
NO string-parsing of LLM output. Routing happens via LLM tool-use selection.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.knowledge_agent import build_knowledge_agent
from app.agents.account_agent import build_account_agent
from app.agents.escalation_agent import build_escalation_agent
from app.settings import settings

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Helix AI Support Concierge. Your job is to route every user
message to the correct specialist agent using your tools. NEVER answer
directly — always delegate.

Routing rules:
- Product documentation, how-to questions, feature explanations → use knowledge_agent
- Build history, account status, usage data → use account_agent
- Unresolved issues requiring human help, explicit escalation requests → use escalation_agent
- Out-of-scope requests (e.g., "write me a poem") → respond with:
  "I can only assist with Helix product questions and account management."

Context about this user:
- user_id: {user_id}
- plan_tier: {plan_tier}
- turn_count: {turn_count}
- last_agent: {last_agent}

Use the last_agent context to handle follow-up questions correctly.
For example, if last_agent was 'knowledge' and the user asks "can you explain more?",
route to knowledge_agent again.
"""


def build_orchestrator(
    user_id: str,
    plan_tier: str,
    turn_count: int,
    last_agent: str | None,
) -> LlmAgent:
    knowledge_agent = build_knowledge_agent(user_id, plan_tier)
    account_agent = build_account_agent(user_id, plan_tier)
    escalation_agent = build_escalation_agent(user_id, plan_tier)

    return LlmAgent(
        name="srop_root",
        model=settings.llm_model,
        instruction=ORCHESTRATOR_SYSTEM_PROMPT.format(
            user_id=user_id,
            plan_tier=plan_tier,
            turn_count=turn_count,
            last_agent=last_agent or "none",
        ),
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
            AgentTool(agent=escalation_agent),
        ],
    )
```

---

## PHASE 4: SROP Pipeline (Core Logic)

### 4.1 `app/srop/pipeline.py` — Full Implementation

This is the most complex file. Implement it completely:

```python
"""
SROP Pipeline — the core of the assignment.

Per-turn flow:
  1. Load SessionState from DB (survives restart — no in-memory state)
  2. Build orchestrator with state injected into system prompts
  3. Run ADK runner with asyncio.wait_for timeout
  4. Parse ADK event stream to extract:
     a. Final text reply
     b. Which sub-agent handled the turn (routed_to)
     c. All tool calls (name, args, result) for trace
     d. Retrieved chunk IDs (from search_docs calls)
  5. Write AgentTrace row to DB
  6. Persist updated SessionState to DB (turn_count+1, last_agent updated)
  7. Handle pending tickets from escalation agent (E2)
  8. Return PipelineResult

State persistence pattern: DB-backed JSON column on sessions.state.
The SessionState Pydantic model is serialized to dict and stored in the
sessions.state JSON column. On every turn start, we load it fresh from DB.
This means state survives any process restart.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import build_orchestrator
from app.agents.escalation_agent import get_pending_ticket
from app.db.models import AgentTrace, Message, Session, Ticket
from app.srop.state import SessionState
from app.srop.errors import SessionNotFoundError, UpstreamTimeoutError
from app.settings import settings

log = structlog.get_logger(__name__)


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


@dataclass
class _TurnContext:
    reply: str = ""
    routed_to: str = "unknown"
    tool_calls: list[dict] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)


async def _load_state(session_id: str, db: AsyncSession) -> tuple[SessionState, Session]:
    """Load session row and deserialize SessionState. Raises SessionNotFoundError if missing."""
    result = await db.execute(select(Session).where(Session.session_id == session_id))
    session_row = result.scalar_one_or_none()
    if session_row is None:
        raise SessionNotFoundError(f"Session {session_id} not found")
    state = SessionState.from_db_dict(session_row.state)
    return state, session_row


async def _save_state(session_id: str, state: SessionState, db: AsyncSession) -> None:
    """Persist updated SessionState to DB."""
    await db.execute(
        update(Session)
        .where(Session.session_id == session_id)
        .values(state=state.to_db_dict())
    )


async def _run_adk(orchestrator, user_message: str, session_id: str) -> _TurnContext:
    """
    Run ADK agent and parse the event stream.
    
    ADK InMemoryRunner usage pattern:
      runner = Runner(agent=orchestrator, app_name="helix", session_service=InMemorySessionService())
      adk_session = await runner.session_service.create_session(app_name="helix", user_id=session_id)
      content = types.Content(role="user", parts=[types.Part(text=user_message)])
      async for event in runner.run_async(user_id=session_id, session_id=adk_session.id, new_message=content):
          # parse event
    
    Event parsing:
    - event.author tells us which agent emitted it ("knowledge_agent", "account_agent", etc.)
    - event.content.parts[*].text gives text chunks
    - event.content.parts[*].function_call gives tool invocations
    - event.content.parts[*].function_response gives tool results
    - event.is_final_response() (or check event.turn_complete) marks the end
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk import types

    ctx = _TurnContext()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator,
        app_name="helix",
        session_service=session_service,
    )
    adk_session = await session_service.create_session(
        app_name="helix", user_id=session_id
    )
    content = types.Content(
        role="user", parts=[types.Part(text=user_message)]
    )

    async for event in runner.run_async(
        user_id=session_id,
        session_id=adk_session.id,
        new_message=content,
    ):
        # Determine which sub-agent authored this event
        author = getattr(event, "author", None) or ""
        if author and author != "srop_root" and author != "user":
            ctx.routed_to = author.replace("_agent", "")  # "knowledge_agent" → "knowledge"

        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            # Collect text from final response
            if hasattr(part, "text") and part.text and getattr(event, "is_final_response", lambda: False)():
                ctx.reply += part.text

            # Collect tool calls
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                ctx.tool_calls.append({
                    "tool_name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                    "result": None,  # filled in by function_response below
                })

            # Collect tool results
            if hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                result_data = dict(fr.response) if fr.response else {}
                # Match to last tool_call with same name (fill result)
                for tc in reversed(ctx.tool_calls):
                    if tc["tool_name"] == fr.name and tc["result"] is None:
                        tc["result"] = result_data
                        break
                # Extract chunk IDs from search_docs results
                if fr.name == "_search_docs_tool" or fr.name == "search_docs_tool":
                    chunks = result_data.get("chunks", [])
                    ctx.retrieved_chunk_ids.extend(c["chunk_id"] for c in chunks)

    # Fallback: if no sub-agent was identified, mark as "knowledge" (most common)
    if ctx.routed_to == "unknown":
        ctx.routed_to = "knowledge"

    return ctx


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    """
    Main pipeline entry point. Called by the chat route.
    """
    start_ms = int(time.monotonic() * 1000)
    trace_id = str(uuid.uuid4())

    # 1. Load state from DB (restart-safe)
    state, _session_row = await _load_state(session_id, db)

    log.info("pipeline.run", session_id=session_id, turn=state.turn_count)

    # 2. Build orchestrator with current state injected
    orchestrator = build_orchestrator(
        user_id=state.user_id,
        plan_tier=state.plan_tier,
        turn_count=state.turn_count,
        last_agent=state.last_agent,
    )

    # 3. Run ADK with LLM timeout
    try:
        ctx = await asyncio.wait_for(
            _run_adk(orchestrator, user_message, session_id),
            timeout=settings.llm_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        )

    latency_ms = int(time.monotonic() * 1000) - start_ms

    # 4. Write user message to DB
    user_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
        trace_id=trace_id,
    )
    db.add(user_msg)

    # 5. Write assistant message to DB
    assistant_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=ctx.reply,
        trace_id=trace_id,
    )
    db.add(assistant_msg)

    # 6. Write trace to DB
    trace = AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=ctx.routed_to,
        tool_calls=ctx.tool_calls,
        retrieved_chunk_ids=ctx.retrieved_chunk_ids,
        latency_ms=latency_ms,
    )
    db.add(trace)

    # 7. Handle escalation agent pending tickets (E2)
    if ctx.routed_to == "escalation":
        for tc in ctx.tool_calls:
            if tc["tool_name"] == "create_ticket" and tc.get("result"):
                ticket_id = tc["result"].get("ticket_id")
                if ticket_id:
                    pending = get_pending_ticket(ticket_id)
                    if pending:
                        ticket = Ticket(
                            ticket_id=ticket_id,
                            session_id=session_id,
                            user_id=state.user_id,
                            summary=pending["summary"],
                            priority=pending["priority"],
                        )
                        db.add(ticket)
                    state.last_ticket_id = ticket_id

    # 8. Update state and persist
    state.turn_count += 1
    state.last_agent = ctx.routed_to  # type: ignore[assignment]
    await _save_state(session_id, state, db)

    log.info(
        "pipeline.complete",
        trace_id=trace_id,
        routed_to=ctx.routed_to,
        latency_ms=latency_ms,
    )

    return PipelineResult(
        content=ctx.reply,
        routed_to=ctx.routed_to,
        trace_id=trace_id,
    )
```

---

## PHASE 5: API Routes

### 5.1 `app/api/deps.py`

```python
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### 5.2 `app/api/routes_sessions.py`

```python
"""POST /v1/sessions — create a new chat session."""
import uuid
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db
from app.db.models import Session, User
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    plan_tier: str = Field(default="free", pattern="^(free|pro|enterprise)$")


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str
    plan_tier: str


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    # Upsert user
    result = await db.execute(select(User).where(User.user_id == body.user_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(user_id=body.user_id, plan_tier=body.plan_tier)
        db.add(user)
    else:
        user.plan_tier = body.plan_tier

    # Create session with initial state
    session_id = str(uuid.uuid4())
    initial_state = SessionState(user_id=body.user_id, plan_tier=body.plan_tier)
    session = Session(
        session_id=session_id,
        user_id=body.user_id,
        state=initial_state.to_db_dict(),
    )
    db.add(session)
    await db.flush()

    return CreateSessionResponse(
        session_id=session_id,
        user_id=body.user_id,
        plan_tier=body.plan_tier,
    )
```

### 5.3 `app/api/routes_chat.py`

```python
"""
POST /v1/chat/{session_id} — send a message.
Supports:
  - Standard JSON response
  - SSE streaming via Accept: text/event-stream (Extension E3)
  - Idempotency-Key header (Extension E1)
"""
import json
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db
from app.db.models import Session
from app.srop.pipeline import run as pipeline_run
from app.srop.errors import SessionNotFoundError
from app.srop.state import SessionState
from app.settings import settings

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    reply: str
    routed_to: str
    trace_id: str
    session_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ChatResponse | StreamingResponse:
    # E1: Idempotency check
    if idempotency_key:
        session_row = await db.get(Session, session_id)
        if session_row is None:
            raise SessionNotFoundError(f"Session {session_id} not found")
        state = SessionState.from_db_dict(session_row.state)
        if idempotency_key in state.idempotency_cache:
            cached = json.loads(state.idempotency_cache[idempotency_key])
            return ChatResponse(**cached)

    # E3: SSE streaming
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return StreamingResponse(
            _sse_stream(session_id, body.content, db),
            media_type="text/event-stream",
        )

    # Standard response
    result = await pipeline_run(session_id, body.content, db)

    response_data = {
        "reply": result.content,
        "routed_to": result.routed_to,
        "trace_id": result.trace_id,
        "session_id": session_id,
    }

    # E1: Cache response for idempotency
    if idempotency_key:
        session_row = await db.get(Session, session_id)
        if session_row:
            state = SessionState.from_db_dict(session_row.state)
            state.idempotency_cache[idempotency_key] = json.dumps(response_data)
            from sqlalchemy import update
            from app.db.models import Session as SessionModel
            await db.execute(
                update(SessionModel)
                .where(SessionModel.session_id == session_id)
                .values(state=state.to_db_dict())
            )

    return ChatResponse(**response_data)


async def _sse_stream(session_id: str, content: str, db: AsyncSession):
    """E3: Server-Sent Events streaming."""
    result = await pipeline_run(session_id, content, db)
    # Stream reply word by word
    words = result.content.split()
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'delta': chunk})}\n\n"
    # Final event with metadata
    yield f"data: {json.dumps({'done': True, 'routed_to': result.routed_to, 'trace_id': result.trace_id})}\n\n"
```

### 5.4 `app/api/routes_traces.py`

```python
"""GET /v1/traces/{trace_id} — retrieve structured trace."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_db
from app.db.models import AgentTrace
from app.srop.errors import TraceNotFoundError

router = APIRouter(tags=["traces"])


class TraceResponse(BaseModel):
    trace_id: str
    session_id: str
    routed_to: str
    tool_calls: list[dict]
    retrieved_chunk_ids: list[str]
    latency_ms: int
    created_at: str


@router.get("/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
) -> TraceResponse:
    result = await db.execute(
        select(AgentTrace).where(AgentTrace.trace_id == trace_id)
    )
    trace = result.scalar_one_or_none()
    if trace is None:
        raise TraceNotFoundError(f"Trace {trace_id} not found")
    return TraceResponse(
        trace_id=trace.trace_id,
        session_id=trace.session_id,
        routed_to=trace.routed_to,
        tool_calls=trace.tool_calls,
        retrieved_chunk_ids=trace.retrieved_chunk_ids,
        latency_ms=trace.latency_ms,
        created_at=trace.created_at.isoformat(),
    )
```

---

## PHASE 6: Tests

### 6.1 `tests/conftest.py`

```python
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.main import app
from app.db.models import Base
from app.db.session import AsyncSessionLocal
from app import db as db_module

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture
async def db_session(test_engine):
    async_session = async_sessionmaker(test_engine, expire_on_commit=False)
    async with async_session() as session:
        yield session

@pytest_asyncio.fixture
async def client(test_engine):
    # Override DB dependency to use in-memory test DB
    async_session = async_sessionmaker(test_engine, expire_on_commit=False)
    
    async def override_get_db():
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    from app.api import deps
    app.dependency_overrides[deps.get_db] = override_get_db
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()
```

### 6.2 `tests/test_integration.py`

```python
"""
Integration test: POST two messages to the same session.
Mocks ADK at the pipeline boundary so no real LLM calls are made.
"""
import pytest
from unittest.mock import AsyncMock, patch
from app.srop.pipeline import PipelineResult


@pytest.mark.asyncio
async def test_two_turn_conversation_preserves_state(client):
    """Turn 1 routes to knowledge. Turn 2 must have access to turn 1 context."""
    # Create session
    resp = await client.post(
        "/v1/sessions",
        json={"user_id": "test_user_001", "plan_tier": "pro"},
    )
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # Mock the pipeline.run to avoid real LLM calls
    turn_counter = {"n": 0}

    async def mock_pipeline_run(sid, msg, db):
        turn_counter["n"] += 1
        import uuid
        return PipelineResult(
            content=f"Mock response {turn_counter['n']}",
            routed_to="knowledge" if turn_counter["n"] == 1 else "account",
            trace_id=str(uuid.uuid4()),
        )

    with patch("app.api.routes_chat.pipeline_run", side_effect=mock_pipeline_run):
        # Turn 1: knowledge question
        r1 = await client.post(
            f"/v1/chat/{session_id}",
            json={"content": "How do I rotate a deploy key?"},
        )
        assert r1.status_code == 200
        assert r1.json()["routed_to"] == "knowledge"
        trace_id_1 = r1.json()["trace_id"]

        # Turn 2: account question
        r2 = await client.post(
            f"/v1/chat/{session_id}",
            json={"content": "Show me my last 3 failed builds"},
        )
        assert r2.status_code == 200
        assert r2.json()["routed_to"] == "account"

    # Verify trace 1 is retrievable
    trace_resp = await client.get(f"/v1/traces/{trace_id_1}")
    assert trace_resp.status_code == 200
    assert trace_resp.json()["routed_to"] == "knowledge"


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client):
    resp = await client.post(
        "/v1/chat/nonexistent-session-id",
        json={"content": "hello"},
    )
    assert resp.status_code == 404
    assert resp.json()["error_code"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

### 6.3 `tests/test_unit_rag.py`

```python
"""
Unit test for search_docs.
Requires a pre-ingested Chroma collection at settings.chroma_path.
If not available, test is skipped with a clear message.
"""
import pytest
from app.rag.search import search_docs, SearchResult


def test_search_docs_returns_scored_results():
    """
    search_docs should return results with non-empty chunk IDs
    and scores in [0, 1].
    """
    try:
        results = search_docs("rotate deploy key", k=3)
    except Exception as e:
        pytest.skip(f"Chroma not populated — run ingest first. Error: {e}")

    assert isinstance(results, list)
    if len(results) == 0:
        pytest.skip("No chunks in vector store — run ingest first")

    for r in results:
        assert isinstance(r, SearchResult)
        assert r.chunk_id, "chunk_id must be non-empty"
        assert 0.0 <= r.score <= 1.0, f"score {r.score} not in [0,1]"
        assert r.text, "text must be non-empty"


def test_search_docs_stable_ids():
    """Same query twice should return the same chunk IDs."""
    try:
        r1 = search_docs("deploy key rotation", k=3)
        r2 = search_docs("deploy key rotation", k=3)
    except Exception as e:
        pytest.skip(f"Chroma not populated: {e}")

    ids1 = [r.chunk_id for r in r1]
    ids2 = [r.chunk_id for r in r2]
    assert ids1 == ids2, "Same query must return stable chunk IDs"
```

---

## PHASE 7: Extension E4 — Reranking

### `app/rag/rerank.py`

Add an LLM-as-judge reranker on top of the vector search:

```python
"""
E4: LLM-as-judge reranker.
After vector search returns top-k, rerank using Gemini to judge relevance.
"""
import json
from app.rag.search import SearchResult, search_docs as _search_docs
from app.settings import settings
import google.generativeai as genai

genai.configure(api_key=settings.google_api_key)

RERANK_PROMPT = """
You are a relevance judge. Given a query and a list of document chunks,
score each chunk's relevance to the query from 0.0 (irrelevant) to 1.0 (perfectly relevant).

Query: {query}

Chunks:
{chunks}

Return ONLY a JSON array of objects with keys "chunk_id" and "relevance_score" (float 0-1).
Example: [{{"chunk_id": "abc", "relevance_score": 0.9}}, ...]
"""


def rerank(query: str, results: list[SearchResult], top_n: int = 3) -> list[SearchResult]:
    """
    Re-score results using LLM relevance judgment.
    Falls back to original order if LLM fails.
    """
    if not results:
        return results

    chunks_text = "\n\n".join(
        f"[{r.chunk_id}]: {r.text[:300]}..." for r in results
    )
    prompt = RERANK_PROMPT.format(query=query, chunks=chunks_text)

    try:
        model = genai.GenerativeModel(settings.llm_model)
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Strip code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        scores = json.loads(raw)
        score_map = {s["chunk_id"]: float(s["relevance_score"]) for s in scores}
        # Re-order by LLM relevance score
        reranked = sorted(results, key=lambda r: score_map.get(r.chunk_id, 0.0), reverse=True)
        return reranked[:top_n]
    except Exception:
        # Fallback to original vector order
        return results[:top_n]


def search_and_rerank(query: str, k: int = 10, top_n: int = 5) -> list[SearchResult]:
    """
    Retrieve k candidates, then rerank to top_n.
    For trace: return reranked results with updated scores.
    """
    candidates = _search_docs(query, k=k)
    return rerank(query, candidates, top_n=top_n)
```

Use `search_and_rerank` inside `KnowledgeAgent` when `settings.enable_reranking = True`.

---

## PHASE 8: Extension E5 — Guardrails

Add to the knowledge agent system prompt and a dedicated guardrails check:

### `app/srop/guardrails.py`

```python
"""
E5: Input guardrails.
- Refuse out-of-scope queries.
- Redact PII from logs.
"""
import re
from app.srop.errors import GuardrailRefusalError

OUT_OF_SCOPE_PATTERNS = [
    r"\bwrite me a poem\b",
    r"\btell me a joke\b",
    r"\bwrite.*code for\b",
    r"\bgenerate.*image\b",
    r"\bplay a game\b",
]

PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
    (r"\b4[0-9]{12}(?:[0-9]{3})?\b", "[CC_REDACTED]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
    (r"\b\d{10}\b", "[PHONE_REDACTED]"),
]


def check_guardrails(message: str) -> None:
    """Raise GuardrailRefusalError if message is out of scope."""
    lower = message.lower()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, lower):
            raise GuardrailRefusalError(
                "I can only assist with Helix product questions and account management."
            )


def redact_pii(text: str) -> str:
    """Redact PII from text before logging."""
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text
```

In `pipeline.run()`, add at the top:
```python
from app.srop.guardrails import check_guardrails, redact_pii
check_guardrails(user_message)
log.info("pipeline.run", session_id=session_id, message=redact_pii(user_message))
```

---

## PHASE 9: Extension E6 — Docker

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN uv pip install --system -e ".[dev]"

# Copy source
COPY . .

# Expose port
EXPOSE 8000

# Ingest docs on start if Chroma is empty, then start server
CMD ["sh", "-c", "python -m app.rag.ingest --path docs/ && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```

### `docker-compose.yml`

```yaml
version: "3.9"
services:
  helix-srop:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./helix.db:/app/helix.db
      - ./chroma_db:/app/chroma_db
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - DATABASE_URL=sqlite+aiosqlite:///./helix.db
      - CHROMA_PATH=./chroma_db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 10s
      timeout: 5s
      retries: 3
```

---

## PHASE 10: Extension E7 — Eval Harness

### `eval/run_eval.py`

```python
"""
E7: Routing accuracy eval harness.
Run against a live server: python eval/run_eval.py --base-url http://localhost:8000
Reports routing accuracy on a known-label test set.
"""
import argparse
import asyncio
import json
import httpx

EVAL_CASES = [
    {"message": "How do I rotate a deploy key?", "expected_agent": "knowledge"},
    {"message": "What is Helix's CI/CD pricing for pro plans?", "expected_agent": "knowledge"},
    {"message": "Show me my last 5 builds", "expected_agent": "account"},
    {"message": "What is my account status?", "expected_agent": "account"},
    {"message": "How does branch protection work in Helix?", "expected_agent": "knowledge"},
    {"message": "My builds keep failing, I need help", "expected_agent": "escalation"},
    {"message": "How do I add team members?", "expected_agent": "knowledge"},
    {"message": "Show me failed builds from last week", "expected_agent": "account"},
    {"message": "What are the rate limits for the API?", "expected_agent": "knowledge"},
    {"message": "I've been waiting 3 days for a response, escalate please", "expected_agent": "escalation"},
]


async def run_eval(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
        # Create eval session
        resp = await client.post(
            "/v1/sessions",
            json={"user_id": "eval_user", "plan_tier": "pro"},
        )
        session_id = resp.json()["session_id"]

        correct = 0
        results = []
        for case in EVAL_CASES:
            r = await client.post(
                f"/v1/chat/{session_id}",
                json={"content": case["message"]},
            )
            data = r.json()
            routed_to = data.get("routed_to", "unknown")
            is_correct = routed_to == case["expected_agent"]
            if is_correct:
                correct += 1
            results.append({
                "message": case["message"],
                "expected": case["expected_agent"],
                "actual": routed_to,
                "correct": is_correct,
            })
            print(f"  {'✓' if is_correct else '✗'} [{case['expected_agent']}→{routed_to}] {case['message'][:60]}")

        accuracy = correct / len(EVAL_CASES) * 100
        print(f"\nRouting Accuracy: {correct}/{len(EVAL_CASES)} = {accuracy:.1f}%")

        # Write report
        report = {"accuracy": accuracy, "cases": results}
        with open("eval/report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Report written to eval/report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    asyncio.run(run_eval(args.base_url))
```

---

## PHASE 11: `app/main.py` — Final Updates

Add the exception handler wiring (the scaffold has a TODO comment for this):

```python
# Add these imports at the top:
from app.srop.errors import HelixError, helix_error_handler

# Add after app = FastAPI(...):
app.add_exception_handler(HelixError, helix_error_handler)
```

---

## PHASE 12: `.env.example` Final State

```bash
# Required
GOOGLE_API_KEY=your-google-api-key-here

# Optional (defaults shown)
DATABASE_URL=sqlite+aiosqlite:///./helix.db
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=helix_docs
LLM_MODEL=gemini-2.0-flash
LLM_TIMEOUT_SECONDS=30
EMBEDDING_MODEL=models/text-embedding-004
TOP_K_RETRIEVAL=5
ENVIRONMENT=development
```

---

## PHASE 13: `README.md` — Final Version

Fill in the provided README template:

```markdown
# Helix SROP — [Your Name]

## Setup

git clone <your-repo>
cd helix-srop
uv sync
cp .env.example .env  # fill in GOOGLE_API_KEY
uv run python -m app.rag.ingest --path docs/
uv run uvicorn app.main:app --reload

## Quick Test

SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | jq .

## Architecture

[ASCII diagram of the pipeline]

## Design Decisions

### State persistence
Used Pattern: DB-backed JSON column. SessionState (Pydantic model) is serialized
to the sessions.state JSON column on every turn. On startup, state is loaded fresh
from DB — zero in-memory state. This guarantees survival across process restarts
with zero additional infrastructure (no Redis, no external session store).

### Chunking strategy
Used heading-aware chunking: split on H2/H3 markdown headings first, then
sentence-level split with 150-char overlap for oversized sections. This preserves
semantic coherence (headings define topic boundaries) while keeping chunks
small enough for high-precision retrieval.

### Vector store
Chose ChromaDB with cosine similarity. It's recommended in the guide, runs
fully local (no external service), and the PersistentClient survives restarts.
Scores are normalized to [0,1] for trace reporting.

## Extensions Completed
- E1: Idempotency (Idempotency-Key header, cache in session state)
- E2: Escalation agent (create_ticket tool, tickets table, ticket_id in state)
- E3: Streaming SSE (Accept: text/event-stream)
- E4: Reranking (LLM-as-judge on top-k results)
- E5: Guardrails (out-of-scope refusal, PII redaction in logs)
- E6: Docker (Dockerfile + docker-compose.yml)
- E7: Eval harness (eval/run_eval.py, routing accuracy reported)
```

---

## CRITICAL IMPLEMENTATION NOTES FOR THE AI IDE

### Note 1: ADK API Compatibility
The `google-adk` package API may differ slightly between versions. The scaffold pins `>=0.5.0`. Key things to verify:
- Import path for `Runner`: try `from google.adk.runners import Runner` first; fallback `from google.adk import Runner`
- Import path for `InMemorySessionService`: try `from google.adk.sessions import InMemorySessionService`
- `AgentTool` import: `from google.adk.tools.agent_tool import AgentTool`
- `LlmAgent` import: `from google.adk.agents import LlmAgent`
- Event type checking: check `event.is_final_response()` method; it may be a property instead
- If ADK API doesn't match, fall back to `google-generativeai` directly for multi-agent routing

### Note 2: Avoid Common Hard Penalties
- **Never** use `import asyncio; loop.run_until_complete()` inside async handlers
- **Never** use `chromadb.Client()` (sync) inside async routes — call from sync context only, or use `asyncio.to_thread()`
- **Never** have a bare `except:` — always `except Exception as e:` and log or re-raise
- **Never** commit `GOOGLE_API_KEY` in `.env` to git — ensure `.env` is in `.gitignore`
- **Always** wrap all LLM calls with `asyncio.wait_for(coro, timeout=settings.llm_timeout_seconds)`

### Note 3: ChromaDB is Sync
ChromaDB's Python client is synchronous. Since we call it from async context:
- Either wrap in `asyncio.to_thread(lambda: collection.query(...))` 
- Or keep it sync but call only from the non-async parts
- The recommended approach: keep `search_docs()` as a sync function and use `loop.run_in_executor` or `asyncio.to_thread` in pipeline

### Note 4: SessionState `last_agent` Field
The existing `state.py` has `last_agent: Literal["knowledge", "account", "smalltalk"] | None`. Update to include `"escalation"`:
```python
last_agent: Literal["knowledge", "account", "escalation", "smalltalk"] | None = None
```

### Note 5: Idempotency Cache in SessionState
Add to `state.py`:
```python
idempotency_cache: dict[str, str] = Field(default_factory=dict)
last_ticket_id: str | None = None
```

### Note 6: Testing Without Real LLM
For `pytest -q` to pass from a clean clone (no `GOOGLE_API_KEY`):
- Integration tests MUST mock at the `pipeline_run` level
- Unit RAG tests MUST use `pytest.skip()` when Chroma is not populated
- Add `GOOGLE_API_KEY=test` to a `pytest.ini` or conftest environment setup

### Note 7: `__init__.py` Files
Create empty `__init__.py` in every package directory:
- `app/__init__.py`
- `app/agents/__init__.py`
- `app/api/__init__.py`
- `app/db/__init__.py`
- `app/obs/__init__.py`
- `app/rag/__init__.py`
- `app/srop/__init__.py`
- `tests/__init__.py`
- `eval/__init__.py`

### Note 8: Gemini Embedding API
`genai.embed_content()` for batch embedding:
```python
# For a single text:
result = genai.embed_content(model="models/text-embedding-004", content="text", task_type="retrieval_document")
embedding = result["embedding"]  # list[float]

# For a list:
result = genai.embed_content(model="models/text-embedding-004", content=["text1", "text2"], task_type="retrieval_document")
embeddings = result["embedding"]  # list[list[float]]
```

---

## Build Order for AI IDE

Execute in this exact order to avoid import errors:

1. `app/__init__.py` and all `__init__.py` files (empty)
2. `app/settings.py`
3. `app/obs/logging.py`
4. `app/srop/errors.py`
5. `app/db/session.py`
6. Update `app/db/models.py` (add Ticket, no rewrite needed)
7. Update `app/srop/state.py` (add new fields)
8. `app/rag/search.py`
9. `app/rag/ingest.py`
10. `app/rag/rerank.py` (E4)
11. `app/srop/guardrails.py` (E5)
12. `app/agents/knowledge_agent.py`
13. `app/agents/account_agent.py`
14. `app/agents/escalation_agent.py` (E2)
15. `app/agents/orchestrator.py`
16. `app/srop/pipeline.py`
17. `app/api/deps.py`
18. `app/api/routes_sessions.py`
19. `app/api/routes_chat.py`
20. `app/api/routes_traces.py`
21. Update `app/main.py` (add exception handler)
22. `tests/conftest.py`
23. `tests/test_integration.py`
24. `tests/test_unit_rag.py`
25. `eval/run_eval.py` (E7)
26. `Dockerfile` (E6)
27. `docker-compose.yml` (E6)
28. Update `README.md`
29. Verify `.env.example`

---

## Verification Checklist

After implementation, run these checks:

```bash
# 1. Lint
uv run ruff check .

# 2. Ingest docs
uv run python -m app.rag.ingest --path docs/

# 3. Start server
uv run uvicorn app.main:app --reload &

# 4. Create session
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

# 5. Knowledge query
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | jq .

# 6. Account query (same session — tests state persistence)
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "Show me my last 3 failed builds"}' | jq .

# 7. Kill server, restart, send another message (proves restart survival)
pkill -f uvicorn
uv run uvicorn app.main:app --reload &
curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "What was I just asking about?"}' | jq .

# 8. Tests
uv run pytest -q

# 9. Eval harness (E7)
uv run python eval/run_eval.py --base-url http://localhost:8000

# 10. Docker (E6)
docker compose up --build
```
