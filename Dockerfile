FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN uv pip install --system -e ".[dev]"

# Copy source
COPY . .

# Expose port
EXPOSE 8000

# Ingest docs on start if Chroma is empty, then start server
CMD ["sh", "-c", "python -m app.rag.ingest --path docs/ && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
