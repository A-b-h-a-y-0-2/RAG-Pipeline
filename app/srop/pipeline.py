"""
SROP Pipeline — the core of the assignment.

Per-turn flow:
  1. Load SessionState from DB (survives restart — no in-memory state)
  2. Fast-route intent (keyword heuristic — zero LLM calls, ~0ms)
  3. Build ONLY the specialist agent needed for this turn
  4. Run ADK runner with asyncio.wait_for timeout
  5. Parse ADK event stream to extract:
     a. Final text reply
     b. Which sub-agent handled the turn (routed_to)
     c. All tool calls (name, args, result) for trace
     d. Retrieved chunk IDs (from search_docs calls)
  6. Write AgentTrace row to DB
  7. Persist updated SessionState to DB (turn_count+1, last_agent updated)
  8. Handle pending tickets from escalation agent (E2)
  9. Return PipelineResult

Latency optimisation:
  Previous: Orchestrator LLM (hop 1) → AgentTool → Specialist LLM (hop 2)
            → tool → Orchestrator LLM summarises (hop 3) ≈ 12-15s
  Current:  Keyword router (0ms) → Specialist LLM (hop 1) → tool ≈ 4-6s

State persistence pattern: DB-backed JSON column on sessions.state.
The SessionState Pydantic model is serialized to dict and stored in the
sessions.state JSON column. On every turn start, we load it fresh from DB.
This means state survives any process restart.
"""
import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.escalation_agent import build_escalation_agent, get_pending_ticket
from app.agents.knowledge_agent import build_knowledge_agent
from app.agents.account_agent import build_account_agent
from app.agents.router import fast_route
from app.db.models import AgentTrace, Message, Session, Ticket
from app.settings import settings
from app.srop.errors import SessionNotFoundError, UpstreamTimeoutError
from app.srop.guardrails import check_guardrails, redact_pii
from app.srop.state import SessionState

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


async def _run_specialist(specialist_agent, user_message: str, session_id: str) -> _TurnContext:
    """
    Run a single specialist agent directly — no orchestrator LLM hop.
    
    This eliminates 2 of the 3 sequential LLM calls in the old architecture,
    cutting expected latency from ~12-15s to ~4-6s.
    """
    try:
        from google.adk.runners import Runner
    except ImportError:
        from google.adk import Runner
    try:
        from google.adk.sessions import InMemorySessionService
    except ImportError:
        from google.adk.sessions.in_memory_session_service import InMemorySessionService

    ctx = _TurnContext()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=specialist_agent,
        app_name="helix",
        session_service=session_service,
    )
    adk_session = await session_service.create_session(
        app_name="helix", user_id=session_id
    )
    from google.genai import types
    content = types.Content(
        role="user", parts=[types.Part(text=user_message)]
    )

    async for event in runner.run_async(
        user_id=session_id,
        session_id=adk_session.id,
        new_message=content,
    ):
        author = getattr(event, "author", None) or ""

        if not getattr(event, "content", None) or not getattr(event.content, "parts", None):
            continue

        for part in event.content.parts:
            # Accumulate text from the agent (not user echoes)
            if getattr(part, "text", None) and author != "user":
                ctx.reply += part.text

            # Collect tool calls
            if getattr(part, "function_call", None):
                fc = part.function_call
                ctx.tool_calls.append({
                    "tool_name": fc.name,
                    "args": dict(fc.args) if getattr(fc, "args", None) else {},
                    "result": None,
                })

            # Collect tool results
            if getattr(part, "function_response", None):
                fr = part.function_response
                result_data = dict(fr.response) if getattr(fr, "response", None) else {}
                # Match to last tool_call with same name
                for tc in reversed(ctx.tool_calls):
                    if tc["tool_name"] == fr.name and tc["result"] is None:
                        tc["result"] = result_data
                        break
                # Extract chunk IDs from search_docs results
                if fr.name == "search_docs":
                    chunks = result_data.get("chunks", [])
                    ctx.retrieved_chunk_ids.extend(c["chunk_id"] for c in chunks)

    # Fallback: extract chunk IDs from reply text using regex (for AgentTool encapsulation)
    chunk_ids = re.findall(r"\[([a-z0-9_-]+_[a-f0-9]{16})\]", ctx.reply)
    if chunk_ids:
        ctx.retrieved_chunk_ids = list(set(ctx.retrieved_chunk_ids + chunk_ids))

    return ctx


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    """
    Main pipeline entry point. Called by the chat route.
    """
    check_guardrails(user_message)
    log.info("pipeline.run", session_id=session_id, message=redact_pii(user_message))

    start_ms = int(time.monotonic() * 1000)
    trace_id = str(uuid.uuid4())

    # 1. Load state from DB (restart-safe)
    state, _session_row = await _load_state(session_id, db)
    log.info("pipeline.run", session_id=session_id, turn=state.turn_count)

    # 2. Fast route — keyword heuristic, 0ms, no LLM call
    route = fast_route(user_message)
    log.info("pipeline.route", session_id=session_id, route=route, method="heuristic")

    # 3. Build ONLY the specialist agent required
    if route == "account":
        specialist = build_account_agent(state.user_id, state.plan_tier)
    elif route == "escalation":
        specialist = build_escalation_agent(state.user_id, state.plan_tier)
    else:
        specialist = build_knowledge_agent(state.user_id, state.plan_tier)

    # 4. Run specialist with LLM timeout
    try:
        ctx = await asyncio.wait_for(
            _run_specialist(specialist, user_message, session_id),
            timeout=settings.llm_timeout_seconds,
        )
    except TimeoutError:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        )

    ctx.routed_to = route
    latency_ms = int(time.monotonic() * 1000) - start_ms

    # 5. Write user message to DB
    user_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
        trace_id=trace_id,
    )
    db.add(user_msg)

    # 6. Write assistant message to DB
    assistant_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=ctx.reply,
        trace_id=trace_id,
    )
    db.add(assistant_msg)

    # 7. Write trace to DB
    trace = AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=ctx.routed_to,
        tool_calls=ctx.tool_calls,
        retrieved_chunk_ids=ctx.retrieved_chunk_ids,
        latency_ms=latency_ms,
    )
    db.add(trace)

    # 8. Handle escalation agent pending tickets (E2)
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

    # 9. Update state and persist
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
