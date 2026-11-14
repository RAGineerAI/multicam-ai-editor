"""
Синхронизация видео по аудио.

Задача: выровнять N видео по общему звуку.
"""

import logging
import subprocess
from pathlib import Path
from typing import List
from tqdm import tqdm

import numpy as np

logger = logging.getLogger(__name__)

def extract_audio(video_path: str, audio_path: str, sample_rate: int = 16000) -> None:
    """Извлекает аудио из видео через FFmpeg."""
    cmd = [
        "ffmpeg", "-y", #Запускаем утилиту FFmpeg
        "-i", video_path, #Путь к видеофайлу
        "-vn", #Отбросить видеодорожку. Нам нужен только звук.
        "-acodec", "pcm_s16le",  #Формат аудио: PCM signed 16-bit little-endian. Это сырой, несжатый звук
        "-ar", str(sample_rate), #Частота дискретизации.
        "-ac", "1", #1 = моно "2" — стерео
        audio_path #Путь, куда сохранить извлеченное аудио
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Аудио извлечено: {audio_path}")

def load_audio(audio_path: str) -> np.ndarray:
    """Загружает WAV в numpy-массив."""
    import wave

    with wave.open(audio_path, "rb") as w:
        frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
    
    return audio.astype(np.float32)

def find_shift(audio1: np.ndarray, audio2: np.ndarray, max_seconds: int = 300) -> int:
    """Находит сдвиг audio2 относительно audio1 (по первым N секундам)."""
    max_samples = 16000 * max_seconds
    a1 = audio1[:max_samples]
    a2 = audio2[:max_samples]
    
    corr = np.correlate(a1, a2, mode="full")
    shift = int(np.argmax(corr) - len(a2) + 1)
    return shift

def sync_videos(video_paths: List[str], output_dir: str = "output") -> List[int]:
    """Синхронизирует N видео, возвращает сдвиги."""
    Path(output_dir).mkdir(exist_ok=True)

    # 1. Извлечь аудио
    audio_paths = []
    for i, video in enumerate(tqdm(video_paths, desc="Извлечение аудио")):
        audio_path = f"{output_dir}/audio_{i}.wav"
        extract_audio(video, audio_path)
        audio_paths.append(audio_path)

    # 2. Загрузить аудио
    audios = [load_audio(p) for p in tqdm(audio_paths, desc="Загрузка аудио")]

    # 3. Сдвиги
    shifts = [0]
    for i, audio in enumerate(tqdm(audios[1:], desc="Синхронизация", total=len(audios)-1), start=1):
        shift = find_shift(audios[0], audio)
        shifts.append(shift)
    logger.info(f"Сдвиги: {shifts}")
    return shifts



