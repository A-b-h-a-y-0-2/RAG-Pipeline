"""
E4: LLM-as-judge reranker.
After vector search returns top-k, rerank using Gemini to judge relevance.
"""
import json

import google.generativeai as genai

from app.rag.search import SearchResult
from app.rag.search import search_docs as _search_docs
from app.settings import settings

genai.configure(api_key=settings.google_api_key)

RERANK_PROMPT = """
You are a relevance judge. Given a query and a list of document chunks,
score each chunk's relevance to the query from 0.0 (irrelevant) to 1.0 (perfectly relevant).

Query: {query}

Chunks:
{chunks}

Return ONLY a JSON array of objects with keys "chunk_id" and "relevance_score" (float 0-1).
Example: [{"chunk_id": "abc", "relevance_score": 0.9}, ...]
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
