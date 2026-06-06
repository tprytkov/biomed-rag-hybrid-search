 
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Vector DB
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    VECTOR_COLLECTION_NAME: str = "biomed_corpus"

    # Models
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    LLM_MODEL_NAME: str = "meta-llama/Llama-3-8B-Instruct"
    OPENAI_API_KEY: str = ""

    # Look for a local .env file first
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Global configuration instance
settings = Settings()
