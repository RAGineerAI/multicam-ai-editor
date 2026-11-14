"""
Настройки приложения.

Загружает переменные из .env.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Настройки multicam-ai-editor."""

    #Папки
    upload_dir: str = "uploads"
    output_dir: str = "output"
    model_dir: str = "models"

    # Redis (Celery)
    redis_url: str = "redis://localhost:6379/0"

    #Whisper
    whisper_model: str = "small"
    whisper_device: str = "cuda"

    # Hugging Face
    hf_token: str = ""

    # Отладка
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
