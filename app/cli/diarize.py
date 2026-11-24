"""
CLI: диаризация аудиофайла.

Использование:
    multicam-diarize --audio output/audio_0.wav
    multicam-diarize --audio output/audio_0.wav --num-speakers 3
    multicam-diarize --audio output/audio_0.wav --force

Логика:
1. SHA256(audio) → cache_dir/<hash>/diarization.json
2. Если кэш есть и нет --force → загрузить, вывести.
3. Если нет → diarize() → сохранить → вывести.
"""

from __future__ import annotations

import warnings
# Подавляем warning про torchcodec: мы обходим его через torchaudio.load()
# в diarize(). См. RESEARCH.md (v0.3, "torchcodec bypass").
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
    

import argparse
import logging
import sys
from pathlib import Path

from app.core.config import settings
from app.services.diarization import (
    diarize,
    format_timestamp,
    load_segments,
    save_segments,
)
from app.utils.cache import audio_cache_dir, cache_file

logger = logging.getLogger(__name__)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multicam-diarize",
        description="Speaker diarization: кто когда говорит.",
    )
    parser.add_argument(
        "--audio", required=True, type=Path,
        help="Путь к WAV (16 kHz mono рекомендуется)."
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Куда сохранить JSON (по умолчанию - в кэш).",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="Точное число спикеров (сильно повышает точность)."
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None,
        help="Нижняя граница числа спикеров (если точное неизвестно)."
    )

    parser.add_argument(
        "--max-speakers", type=int, default=None,
        help="Верхняя граница числа спикеров.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Переопределить модель (по умолчанию — авто по длине).",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["auto", "cuda", "cpu"],
        help="Устройство (по умолчанию — из settings).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Игнорировать кэш, пересчитать.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Не печатать сегменты (только сводку).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Возвращает exit code (0 = успех)."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.audio.exists():
        print(f"ERROR: audio not found: {args.audio}", file=sys.stderr)
        return 1

    # 1. Кэш
    cache_root = Path(settings.cache_dir)
    cached_json = cache_file(cache_root, args.audio, "diarization.json")

    if cached_json.exists() and not args.force:
        print(f"Cache hit: {cached_json}")
        segments = load_segments(cached_json)
    else:
        print(f"Diarizing: {args.audio}")
        segments = diarize(
            args.audio,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            device=args.device,
            model_name=args.model,
        )
        # Сохраняем в кэш
        out_path = args.out or cached_json
        save_segments(
            segments, out_path,
            meta={
                "model": args.model or "auto",
                "audio": str(args.audio),
                "num_speakers_requested": args.num_speakers,
            },
        )
        print(f"Saved: {out_path}")

    # 2. Сводка
    speakers: dict[str, float] = {}
    for s in segments:
        speakers[s.speaker] = speakers.get(s.speaker, 0.0) + (s.end - s.start)

    print(f"\nСегментов: {len(segments)}")
    print("Спикеры (суммарная длительность):")
    for spk, dur in sorted(speakers.items()):
        print(f"  {spk}: {format_timestamp(dur)}")

    # 3. Сегменты (если не --quiet)
    if not args.quiet:
        print()
        for s in segments:
            print(f"{format_timestamp(s.start)} - {format_timestamp(s.end)}  {s.speaker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())