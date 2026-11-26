# settings = Settings()

from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    # LLM configuration

    api_key: str = Field(
        min_length=1,
        repr=False,
    )

    llm_base_url: str

    llm_model: str = "Qwen/Qwen3-8B"

    llm_ssl_cert_path: Path


    # Embedding configuration

    embeddings_key: str = Field(
        min_length=1,
        repr=False,
    )

    embedding_base_url: str

    embedding_model: str = (
        "BAAI/bge-small-en-v1.5"
    )

    embedding_ssl_cert_path: Path


    # Environment configuration

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()