"""
Root orchestrator — routes to specialist sub-agents via ADK AgentTool.
NO string-parsing of LLM output. Routing happens via LLM tool-use selection.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm

from app.agents.knowledge_agent import build_knowledge_agent
from app.agents.account_agent import build_account_agent
from app.agents.escalation_agent import build_escalation_agent
from app.settings import settings

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Helix AI Support Concierge. Your job is to route every user
message to the correct specialist agent using your tools.

Routing rules:
- Product documentation, how-to questions, feature explanations → use knowledge_agent
- Build history, account status, usage data → use account_agent
- Unresolved issues requiring human help, explicit escalation requests → use escalation_agent
- Out-of-scope requests (e.g., "write me a poem") → respond with:
  "I can only assist with Helix product questions and account management."

CRITICAL: When a specialist agent responds, you MUST repeat their response
word-for-word to the user. Do not summarize, paraphrase, or omit any details
(especially chunk IDs like [chunk_abc123]).

Context about this user:
- user_id: {user_id}
- plan_tier: {plan_tier}
- turn_count: {turn_count}
- last_agent: {last_agent}

Use the last_agent context to handle follow-up questions correctly.
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
        model=LiteLlm(model=settings.llm_model),
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
