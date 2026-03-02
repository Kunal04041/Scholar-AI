from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # LLM
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Qdrant
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "scholar_ai_docs"

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    MAX_RETRIEVED_DOCS: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
