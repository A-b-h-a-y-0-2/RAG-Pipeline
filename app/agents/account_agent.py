"""
AccountAgent — handles account/build data lookups.
Uses mock data; the wiring pattern is what's evaluated.
"""
import random
from datetime import datetime, timedelta

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
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
        "success_rate": round((len(builds) - failed) / len(builds) * 100, 1) if builds else 100.0,
        "joined_days_ago": random.randint(30, 365),
    }


def build_account_agent(user_id: str, plan_tier: str) -> LlmAgent:
    return LlmAgent(
        name="account_agent",
        model=LiteLlm(model=settings.llm_model),
        instruction=ACCOUNT_SYSTEM_PROMPT.format(
            user_id=user_id, plan_tier=plan_tier
        ),
        tools=[
            FunctionTool(func=get_recent_builds),
            FunctionTool(func=get_account_status),
        ],
        description="Handles account lookups and build history queries.",
    )
