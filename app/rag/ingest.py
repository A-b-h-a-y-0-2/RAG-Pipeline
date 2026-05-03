"""
RAG ingestion pipeline.
Now using LiteLLM for OpenRouter compatibility.
Usage: python -m app.rag.ingest --path docs/
"""
import argparse
import hashlib
import re
import sys
import os
from pathlib import Path

import chromadb
import litellm

from app.settings import settings

# Configure OpenRouter for LiteLLM if key is present
if settings.openrouter_api_key:
    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key

CHUNK_MAX_CHARS = 1500   # ~400 tokens
CHUNK_OVERLAP = 150      # char overlap between adjacent sentence-splits


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML-ish frontmatter between --- delimiters."""
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm_block = text[3:end].strip()
            for line in fm_block.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            text = text[end + 3:].strip()
    return meta, text


def heading_aware_chunks(text: str, source: str, meta: dict) -> list[dict]:
    """
    Split on H2/H3 headings first, then further split oversized sections
    by sentence boundary with CHUNK_OVERLAP char overlap.
    Returns list of {text, chunk_id, source, section, product_area, title}.
    """
    # Split on ## or ### headings
    sections = re.split(r"(?m)^(#{2,3} .+)$", text)
    chunks = []
    current_heading = meta.get("title", source)

    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{2,3} ", part):
            current_heading = part.lstrip("#").strip()
            continue
        # If section is small enough, it's one chunk
        if len(part) <= CHUNK_MAX_CHARS:
            chunk_text = f"[{current_heading}]\n{part}"
            chunk_id = _stable_id(source, chunk_text)
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "source": source,
                "section": current_heading,
                "product_area": meta.get("product_area", "general"),
                "title": meta.get("title", source),
            })
        else:
            # Sentence-level split with overlap
            sentences = re.split(r"(?<=[.!?])\s+", part)
            buffer = f"[{current_heading}]\n"
            for sent in sentences:
                if len(buffer) + len(sent) > CHUNK_MAX_CHARS and len(buffer) > 50:
                    chunk_id = _stable_id(source, buffer)
                    chunks.append({
                        "text": buffer.strip(),
                        "chunk_id": chunk_id,
                        "source": source,
                        "section": current_heading,
                        "product_area": meta.get("product_area", "general"),
                        "title": meta.get("title", source),
                    })
                    # Overlap: keep last CHUNK_OVERLAP chars
                    buffer = f"[{current_heading}]\n" + buffer[-CHUNK_OVERLAP:] + " " + sent + " "
                else:
                    buffer += sent + " "
            if buffer.strip():
                chunk_id = _stable_id(source, buffer)
                chunks.append({
                    "text": buffer.strip(),
                    "chunk_id": chunk_id,
                    "source": source,
                    "section": current_heading,
                    "product_area": meta.get("product_area", "general"),
                    "title": meta.get("title", source),
                })
    return chunks


def _stable_id(source: str, text: str) -> str:
    """Deterministic chunk ID from source + content hash. Re-ingest safe."""
    h = hashlib.sha256(f"{source}::{text}".encode()).hexdigest()[:16]
    safe_source = re.sub(r"[^a-z0-9_-]", "_", Path(source).stem.lower())
    return f"{safe_source}_{h}"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed using LiteLLM."""
    response = litellm.embedding(
        model=settings.embedding_model,
        input=texts,
    )
    return [d["embedding"] for d in response.data]


def ingest(path: str) -> None:
    docs_path = Path(path)
    md_files = list(docs_path.glob("**/*.md"))
    if not md_files:
        print(f"No .md files found in {path}", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=settings.chroma_path)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[dict] = []
    for md_file in md_files:
        raw = md_file.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        chunks = heading_aware_chunks(body, str(md_file.name), meta)
        all_chunks.extend(chunks)
        print(f"  {md_file.name}: {len(chunks)} chunks")

    # De-duplicate by chunk_id (idempotent re-ingest)
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in all_chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("All chunks already ingested — nothing to do.")
        return

    # Embed in batches of 100
    BATCH = 100
    for i in range(0, len(new_chunks), BATCH):
        batch = new_chunks[i:i + BATCH]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source": c["source"],
                    "section": c["section"],
                    "product_area": c["product_area"],
                    "title": c["title"],
                }
                for c in batch
            ],
        )
        print(f"  Upserted batch {i // BATCH + 1} ({len(batch)} chunks)")

    print(f"\nDone. Total chunks upserted: {len(new_chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest docs into Chroma vector store.")
    parser.add_argument("--path", required=True, help="Directory containing .md files")
    args = parser.parse_args()
    ingest(args.path)
