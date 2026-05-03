"""
KnowledgeAgent — answers Helix product questions using RAG.
- ALWAYS calls search_docs first (enforced via tool_config in LlmAgent)
- System prompt ENFORCES chunk-ID citations in every answer
- Returns chunk IDs so pipeline.py can store them in the trace
"""
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from app.rag.search import search_docs as _search_docs
from app.settings import settings

KNOWLEDGE_SYSTEM_PROMPT = """
You are the Helix Knowledge Agent. You answer questions about Helix products.

MANDATORY FIRST STEP: You MUST call search_docs before writing ANY response.
This is not optional. Do not write a single word until you have called search_docs.

After calling search_docs:
- If chunks are returned: base your answer ONLY on those chunks.
  Cite every claim with the chunk ID in square brackets, e.g. [deploy-keys_5e491c0e88bfd655].
- If no chunks are returned: reply exactly: "I don't have documentation on that topic."

Do NOT answer from memory or general knowledge. Only use retrieved chunks.

User plan tier: {plan_tier}
User ID: {user_id}
"""


def search_docs(query: str, k: int = 1) -> dict:
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
        model=LiteLlm(model=settings.llm_model),
        instruction=KNOWLEDGE_SYSTEM_PROMPT.format(
            plan_tier=plan_tier, user_id=user_id
        ),
        tools=[FunctionTool(func=search_docs)],
        description="Answers Helix product documentation questions using RAG.",
        # Force ANY tool call on first response (only search_docs available → it must be called)
        generate_content_config=types.GenerateContentConfig(
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                )
            )
        ),
    )
