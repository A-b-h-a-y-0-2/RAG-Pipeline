from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    database_url: str = Field(
        default="sqlite+aiosqlite:///./helix.db", alias="DATABASE_URL"
    )
    chroma_path: str = Field(default="./chroma_db", alias="CHROMA_PATH")
    chroma_collection: str = Field(default="helix_docs", alias="CHROMA_COLLECTION")
    llm_model: str = Field(default="openrouter/google/gemini-2.0-flash-001", alias="LLM_MODEL")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    embedding_model: str = Field(
        default="openrouter/openai/text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    top_k_retrieval: int = Field(default=5, alias="TOP_K_RETRIEVAL")
    environment: str = Field(default="development", alias="ENVIRONMENT")

settings = Settings()
