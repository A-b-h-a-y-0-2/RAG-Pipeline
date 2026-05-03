"""GET /v1/traces/{trace_id} — retrieve structured trace."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
