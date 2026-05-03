"""
Unit test for search_docs.
Requires a pre-ingested Chroma collection at settings.chroma_path.
If not available, test is skipped with a clear message.
"""
import pytest

from app.rag.search import SearchResult, search_docs


def test_search_docs_returns_scored_results():
    """
    search_docs should return results with non-empty chunk IDs
    and scores in [0, 1].
    """
    try:
        results = search_docs("rotate deploy key", k=1)
    except Exception as e:
        pytest.skip(f"Chroma not populated — run ingest first. Error: {e}")

    assert isinstance(results, list)
    if len(results) == 0:
        pytest.skip("No chunks in vector store — run ingest first")

    for r in results:
        assert isinstance(r, SearchResult)
        assert r.chunk_id, "chunk_id must be non-empty"
        assert 0.0 <= r.score <= 1.0, f"score {r.score} not in [0,1]"
        assert r.text, "text must be non-empty"


def test_search_docs_stable_ids():
    """Same query twice should return the same chunk IDs."""
    try:
        r1 = search_docs("deploy key rotation", k=1)
        r2 = search_docs("deploy key rotation", k=1)
    except Exception as e:
        pytest.skip(f"Chroma not populated: {e}")

    ids1 = [r.chunk_id for r in r1]
    ids2 = [r.chunk_id for r in r2]
    assert ids1 == ids2, "Same query must return stable chunk IDs"
