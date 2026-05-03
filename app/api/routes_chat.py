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

from app.api.deps import get_db
from app.db.models import Session
from app.srop.errors import SessionNotFoundError
from app.srop.pipeline import run as pipeline_run
from app.srop.state import SessionState

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
