"""api.config - 애플리케이션 설정 (pydantic-settings)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수 기반 설정."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = (
        "postgresql+asyncpg://user:password@localhost:5432/insurance_qa"
    )

    # Ollama (Local LLM)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    embedding_model: str = "nomic-embed-text"

    # App
    app_env: str = "development"
    log_level: str = "DEBUG"


settings = Settings()
