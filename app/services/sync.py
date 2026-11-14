"""
Синхронизация видео по аудио.

Задача: выровнять N видео по общему звуку.
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm
import numpy as np
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, audio_path: str, sample_rate: int = 16000) -> None:
    """Извлекает аудио из видео через FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Аудио извлечено: {audio_path}")


def load_audio_short(audio_path: str, max_seconds: int = 300) -> np.ndarray:
    """Загружает ТОЛЬКО первые N секунд WAV."""
    import wave
    with wave.open(audio_path, "rb") as w:
        frames_to_read = min(w.getnframes(), w.getframerate() * max_seconds)
        frames = w.readframes(frames_to_read)
        audio = np.frombuffer(frames, dtype=np.int16)
    return audio.astype(np.float32)


def find_shift(audio1: np.ndarray, audio2: np.ndarray) -> int:
    """Находит сдвиг audio2 относительно audio1 через FFT."""
    if len(audio1) == 0 or len(audio2) == 0:
        return 0
    corr = fftconvolve(audio1, audio2[::-1], mode="full")
    shift = int(np.argmax(corr) - len(audio2) + 1)
    return shift


def sync_videos(video_paths: List[str], output_dir: str = "output") -> Dict:
    """Синхронизирует N видео, возвращает сдвиги и метаданные."""
    Path(output_dir).mkdir(exist_ok=True)

    audio_paths = []
    for i, video in enumerate(tqdm(video_paths, desc="Извлечение аудио")):
        audio_path = f"{output_dir}/audio_{i}.wav"
        extract_audio(video, audio_path)
        audio_paths.append(audio_path)

    audios = [load_audio_short(p, max_seconds=300) for p in tqdm(audio_paths, desc="Загрузка аудио")]

    shifts = [0]
    for i, audio in enumerate(tqdm(audios[1:], desc="Синхронизация", total=len(audios)-1), start=1):
        shift = find_shift(audios[0], audio)
        shifts.append(shift)

    result = {
        "shifts_samples": shifts,
        "shifts_seconds": [round(s / 16000, 3) for s in shifts],
        "sample_rate": 16000,
        "num_videos": len(video_paths)
    }
    
    logger.info(f"Результаты: {result}")

    with open(f"{output_dir}/sync_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def align_videos(video_paths: List[str], shifts_seconds: List[float], output_dir: str = "output") -> List[str]:
    """Выравнивает видео (обрезает начало для синхронизации)."""
    Path(output_dir).mkdir(exist_ok=True)
    aligned_paths = []

    max_shift = max(shifts_seconds)
    logger.info(f"Макс. сдвиг: {max_shift:.3f} сек")

    for i, (video, shift) in enumerate(zip(video_paths, shifts_seconds)):
        output_path = f"{output_dir}/aligned_cam_{i}.mp4"
        trim_start = max_shift - shift

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(trim_start),
            "-i", video,
            "-c", "copy",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            aligned_paths.append(output_path)
            logger.info(f"Выровнено (обрезка {trim_start:.3f}с): {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка: {e}")
    
    return aligned_paths