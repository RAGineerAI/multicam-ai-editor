"""
Диаризация (speaker diarization) на базе pyannote.audio.

Физика:
- Голос каждого человека имеет характерный спектральный «отпечаток»
  (форманты, тембр, ритм). Нейросеть pyannote превращает короткие окна
  аудио в эмбеддинги (векторы), похожие эмбеддинги кластеризуются
  в одного «спикера».
- VAD (voice activity detection) → segmentation → embedding → clustering.

Совместимость:
- pyannote.audio 3.x: pipeline() возвращает Annotation напрямую.
- pyannote.audio 4.x: pipeline() возвращает DiarizeOutput с атрибутом
  .speaker_diarization (внутри — Annotation).
- Мы используем getattr(output, "speaker_diarization", output) —
  работает в обеих версиях.

Обход torchcodec:
- pyannote 4.x по умолчанию читает аудио через torchcodec, который
  капризен к CUDA/FFmpeg-зависимостям (libnppicc.so.12 и т.п.).
- Мы грузим аудио через torchaudio.load() и передаём словарь
  {"waveform": ..., "sample_rate": ...} — torchcodec не вызывается.

Ограничения v0.3:
- Без чанкинга длинных файлов (v0.4).
- Без склейки спикеров между файлами (v0.4, когда появятся эмбеддинги).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from pyannote.audio import Pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)

# Singleton-кэш пайплайнов: имя модели → загруженный Pipeline.
# Нужен, чтобы повторные вызовы в одном процессе не грузили модель заново
# (загрузка ~10-15 сек + ~1.6 GB VRAM). Критично для FastAPI (v0.9).
_PIPELINES: dict[str, Pipeline] = {}


@dataclass(frozen=True)
class SpeakerSegment:
    """Один интервал речи одного спикера (в секундах от начала аудио)."""

    start: float
    end: float
    speaker: str  # "SPEAKER_00", "SPEAKER_01", ...

    def __post_init__(self) -> None:
        """Валидация: start < end, оба >= 0."""
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")

def format_timestamp(seconds: float) -> str:
    """Форматирует секунды как MM:SS.mmm (или HH:MM:SS.mmm для >1 часа)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"

def _resolve_device(device: str | None) -> torch.device:
    """
    Определяет устройство для pyannote.

    'auto' (или None) → cuda если доступна, иначе cpu.
    'cuda' → cuda (упадёт, если нет).
    'cpu' → cpu.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_pipeline(model_name: str, device: str | None = None) -> Pipeline:
    """
    Загружает pyannote Pipeline (singleton по имени модели).

    Args:
        model_name: "pyannote/speaker-diarization-3.1" или
            "pyannote/speaker-diarization-community-1".
        device: "auto" | "cuda" | "cpu" | None (= auto).

    Returns:
        Pipeline, перемещённый на device.

    Note:
        Первый вызов скачивает модель (~1-2 GB) в HF-кэш
        (settings.model_dir через HF_HOME). Последующие — из кэша.
    """
    if model_name in _PIPELINES:
        return _PIPELINES[model_name]

    token = settings.hf_token
    if not token:
        raise ValueError(
            "HF_TOKEN не задан. Установите в .env: HF_TOKEN=hf_..."
        )

    logger.info("Loading pyannote pipeline: %s (device=%s)", model_name, device)

    # pyannote 4.x использует token=, 3.x — use_auth_token=.
    # Пробуем token= (4.x), при TypeError — use_auth_token= (3.x).
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)

    resolved = _resolve_device(device or settings.diarization_device)
    pipeline.to(resolved)
    logger.info("Pipeline loaded on %s", resolved)

    _PIPELINES[model_name] = pipeline
    return pipeline


def _pick_model(audio_duration_sec: float) -> str:
    """
    Выбирает модель по длительности аудио.

    < threshold → speaker-diarization-3.1 (быстрее, стабильнее)
    >= threshold → speaker-diarization-community-1 (лучше на длинных)
    """
    if audio_duration_sec >= settings.diarization_long_threshold_sec:
        return settings.diarization_model_long
    return settings.diarization_model_short


def _audio_duration_sec(audio_path: Path) -> float:
    """
    Длительность аудио в секундах. Читает WAV-заголовок (без декодирования).
    """
    import wave

    with wave.open(str(audio_path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def diarize(
    audio_path: Path | str,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str | None = None,
    model_name: str | None = None,
) -> list[SpeakerSegment]:
    """
    Запускает диаризацию одного аудиофайла.

    Args:
        audio_path: путь к WAV (16 kHz mono рекомендуется).
        num_speakers: точное число спикеров (сильно повышает точность).
        min_speakers / max_speakers: границы, если число неизвестно.
        device: "auto" | "cuda" | "cpu" | None (= из settings).
        model_name: переопределить выбор модели (для тестов).

    Returns:
        Список сегментов, отсортированный по start.

    Raises:
        FileNotFoundError: если audio_path не существует.
        ValueError: если HF_TOKEN не задан или параметры некорректны.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    duration = _audio_duration_sec(audio_path)
    model = model_name or _pick_model(duration)
    logger.info(
        "Diarization: %s (%.1f sec) → model=%s",
        audio_path.name, duration, model,
    )

    pipeline = load_pipeline(model, device=device)

    # Параметры кластеризации.
    kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    # Обход torchcodec: грузим аудио через torchaudio и передаём словарь.
    # pyannote 4.x без этого падает на CUDA-зависимостях torchcodec.
    waveform, sample_rate = torchaudio.load(str(audio_path))
    output = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        **kwargs,
    )

    # pyannote 4.x: DiarizeOutput с .speaker_diarization
    # pyannote 3.x: Annotation напрямую
    annotation = getattr(output, "speaker_diarization", output)

    segments = [
        SpeakerSegment(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s.start)

    logger.info(
        "Diarization done: %d segments, %d speakers",
        len(segments), len({s.speaker for s in segments}),
    )
    return segments


def save_segments(
    segments: list[SpeakerSegment],
    path: Path | str,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """
    Сохраняет сегменты в JSON с метаданными и версией формата.

    Формат:
        {
          "version": 1,
          "created_at": "...",
          "num_speakers": 2,
          "num_segments": 47,
          "segments": [{"start": 0.5, "end": 3.2, "speaker": "SPEAKER_00"}, ...],
          ...meta
        }
    """
    from datetime import datetime, timezone

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_speakers": len({s.speaker for s in segments}),
        "num_segments": len(segments),
        "segments": [asdict(s) for s in segments],
    }
    if meta:
        payload.update(meta)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_segments(path: Path | str) -> list[SpeakerSegment]:
    """
    Загружает сегменты из JSON (обратная операция к save_segments).

    Raises:
        ValueError: если version не поддерживается.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    version = payload.get("version")
    if version != 1:
        raise ValueError(f"Unsupported format version: {version}")

    return [
        SpeakerSegment(start=s["start"], end=s["end"], speaker=s["speaker"])
        for s in payload["segments"]
    ]