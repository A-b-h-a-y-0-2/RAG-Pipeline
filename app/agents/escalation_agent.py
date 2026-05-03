"""
EscalationAgent — creates support tickets.
Extension E2: writes to `tickets` table, returns ticket_id.
Ticket ID is stored in SessionState.last_ticket_id for follow-ups.
"""
import uuid
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
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
        model=LiteLlm(model=settings.llm_model),
        instruction=ESCALATION_SYSTEM_PROMPT.format(
            user_id=user_id, plan_tier=plan_tier
        ),
        tools=[FunctionTool(func=create_ticket)],
        description="Creates support tickets for unresolved issues requiring human intervention.",
    )
