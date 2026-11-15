"""
Настройки приложения.

Загружает переменные из .env через pydantic-settings.

Использование:
    from app.core.config import settings
    print(settings.diarization_model_short)

Правила:
- snake_case в Python, UPPER_CASE в .env. Pydantic-settings матчит автоматически.
- Дефолты — рабочие значения «из коробки»; .env переопределяет.
- extra="ignore" — лишние переменные в .env не ломают загрузку.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки multicam-ai-editor."""

    # ------------------------------------------------------------------
    # Папки
    # ------------------------------------------------------------------
    upload_dir: str = "uploads"
    output_dir: str = "output"
    model_dir: str = "models"
    cache_dir: str = "output/cache"

    # ------------------------------------------------------------------
    # Redis (Celery, v0.9)
    # ------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------------
    # Whisper (v0.6)
    # ------------------------------------------------------------------
    whisper_model: str = "small"
    whisper_device: str = "cuda"

    # ------------------------------------------------------------------
    # Hugging Face
    # ------------------------------------------------------------------
    hf_token: str = ""
    # 1 = не скачивать модели, использовать локальный кэш (после первого запуска)
    hf_hub_offline: bool = False

    # ------------------------------------------------------------------
    # Диаризация (v0.3)
    # ------------------------------------------------------------------
    # Короткие файлы (< diarization_long_threshold_sec) → модель _short
    # Длинные (>= порога) → модель _long (лучше на длинных записях)
    diarization_model_short: str = "pyannote/speaker-diarization-3.1"
    diarization_model_long: str = "pyannote/speaker-diarization-community-1"
    diarization_long_threshold_sec: int = 1800  # 30 минут
    # auto → cuda если доступна, иначе cpu
    diarization_device: str = "auto"
    # Сегменты короче этого не попадут в эмбеддинги (v0.4)
    min_segment_duration_sec: float = 1.0

    # ------------------------------------------------------------------
    # Отладка
    # ------------------------------------------------------------------
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()