"""
search_docs tool — used by KnowledgeAgent.
Returns top-k chunks with IDs and similarity scores.
Now using LiteLLM for OpenRouter compatibility.
"""
from dataclasses import dataclass
import litellm
import chromadb
import os

from app.settings import settings

# Configure OpenRouter for LiteLLM if key is present
if settings.openrouter_api_key:
    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key

_client: chromadb.ClientAPI | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_path)
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float          # cosine similarity in [0, 1]
    source: str
    section: str
    product_area: str


def search_docs(query: str, k: int = 5) -> list[SearchResult]:
    """
    Embed query and retrieve top-k semantically similar chunks from Chroma.
    Scores are normalised cosine similarity (higher = better).
    """
    collection = _get_collection()
    
    # Use LiteLLM for embedding
    response = litellm.embedding(
        model=settings.embedding_model,
        input=[query],
    )
    query_embedding = response.data[0]["embedding"]

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count() if collection.count() > 0 else 1),
        include=["documents", "metadatas", "distances"],
    )

    results: list[SearchResult] = []
    if not raw["ids"] or not raw["ids"][0]:
        return results

    for chunk_id, doc, meta, dist in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        # Chroma cosine distance: 0=identical, 2=opposite → score = 1 - dist/2
        score = round(1.0 - dist / 2.0, 4)
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                text=doc,
                score=score,
                source=meta.get("source", ""),
                section=meta.get("section", ""),
                product_area=meta.get("product_area", ""),
            )
        )
    return results
