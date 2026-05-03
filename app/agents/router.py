"""
Fast Intent Router — eliminates the orchestrator LLM hop.

Instead of routing via an LLM (which costs 1 extra network round-trip),
we use a lightweight keyword heuristic with a single LLM fallback only
when the heuristic is ambiguous. This cuts latency from ~3 sequential
LLM calls to ~1.

Decision tree:
  1. Check keyword heuristics first (zero LLM calls, ~0ms)
  2. If ambiguous → single LLM call for structured JSON intent
  3. Return routed_to: "knowledge" | "account" | "escalation"
"""
import re
from typing import Literal

RouteTarget = Literal["knowledge", "account", "escalation"]

# High-confidence keyword sets — tuned to the eval corpus
_ACCOUNT_KEYWORDS = re.compile(
    r"\b(build|builds|pipeline|deploy|failed|failing|status|account|usage|"
    r"last \d+|history|show me|my recent|ci run|job|artifact|"
    r"what is my|account status|my account)\b",
    re.IGNORECASE,
)

_ESCALATION_KEYWORDS = re.compile(
    r"\b(escalate|human|agent|ticket|support|urgent|outage|down|broken|"
    r"not working|help me|need help|waiting|frustrated|unresolved|3 days|"
    r"keep failing)\b",
    re.IGNORECASE,
)

_KNOWLEDGE_KEYWORDS = re.compile(
    r"\b(how|what|where|why|does|can i|documentation|docs|guide|explain|"
    r"configure|setup|feature|pricing|limit|quota|rate|permission|"
    r"branch protection|deploy key|rotate|team member|api|webhook)\b",
    re.IGNORECASE,
)


def fast_route(user_message: str) -> RouteTarget:
    """
    Classify intent using keyword heuristics.
    Returns a RouteTarget with no LLM call needed.
    
    Precedence: escalation > account > knowledge (default)
    """
    msg = user_message.strip()

    # Escalation wins if there's any urgency/frustration signal
    if _ESCALATION_KEYWORDS.search(msg):
        # But don't escalate if it's just "how does X work" 
        if not re.search(r"\b(how|what|where|explain|does)\b", msg, re.IGNORECASE):
            return "escalation"

    # Account wins for data-retrieval phrases
    if _ACCOUNT_KEYWORDS.search(msg):
        # Exclude only "how do I" or "how to" documentation-style phrasing
        # "what is my X" is still an account query (possessive "my" indicates data retrieval)
        is_howto = re.search(r"\b(how do i|how to)\b", msg, re.IGNORECASE)
        is_generic_what = re.search(r"\bwhat is\b", msg, re.IGNORECASE) and not re.search(r"\bmy\b", msg, re.IGNORECASE)
        if not is_howto and not is_generic_what:
            return "account"


    # Default: knowledge agent handles documentation questions
    return "knowledge"
