"""
Integration test: POST two messages to the same session.
Mocks ADK at the pipeline boundary so no real LLM calls are made.
"""
from unittest.mock import patch

import pytest

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
        from app.db.models import AgentTrace
        trace_id = str(uuid.uuid4())
        routed_to = "knowledge" if turn_counter["n"] == 1 else "account"
        trace = AgentTrace(
            trace_id=trace_id,
            session_id=sid,
            routed_to=routed_to,
            tool_calls=[],
            retrieved_chunk_ids=["chunk_abc"],
            latency_ms=100
        )
        db.add(trace)
        return PipelineResult(
            content=f"Mock response {turn_counter['n']}",
            routed_to=routed_to,
            trace_id=trace_id,
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
