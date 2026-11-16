"""
Кэш артефактов по SHA256 содержимого входного файла.

Структура:
    <cache_root>/<sha256>/<name>

Пример:
    output/cache/abc123.../diarization.json
    output/cache/abc123.../embeddings.npy   (v0.4)

Физика:
- SHA256 — криптографический хэш. Один и тот же файл → один хэш,
  независимо от имени, пути, mtime.
- Чтение чанками (1 MB) — константная память, O(размер файла) время.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Размер чанка для чтения файла при хэшировании.
# 1 MB = 2^20 — оптимум между системными вызовами и памятью.

DEFAULT_CHUNK_SIZE = 1 << 20


def sha256_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """
    Возвращает SHA256 хэш содержимого файла (hex-строка, 64 символа).

    Args:
        path: путь к файлу. Должен существовать.
        chunk_size: размер чанка чтения в байтах.
            Для больших файлов не увеличивай сильно (память).
            Для тестов можно уменьшить (например, 4 — проверить чанкинг).

    Returns:
        Hex-строка SHA256 (например, "e3b0c442...b855").

    Raises:
        FileNotFoundError: если файла нет.
    """
    hasher = hashlib.sha256()                          # создаём объект хеша SHA-256
    with path.open("rb") as f:                         # открываем файл в бинарном режиме
        for chunk in iter(lambda: f.read(chunk_size), b""):  # читаем по кускам до конца файла
            hasher.update(chunk)                       # подкармливаем хеш очередным куском
    return hasher.hexdigest()                          # возвращаем итоговый хеш строкой

def audio_cache_dir(cache_root: Path, audio_path: Path) -> Path:
    """
    Возвращает папку кэша для конкретного аудиофайла, СОЗДАЁТ её.

    Структура: cache_root/<sha256(audio_path)>/

    Используется, когда ПИШЕМ результат. Для проверки «есть ли кэш»
    используй cache_file() — она не создаёт папок.

    Args:
        cache_root: корень кэша (например, output/cache).
        audio_path: путь к аудиофайлу.

    Returns:
        Путь к созданной папке кэша.
    """
    key = sha256_file(audio_path)              # хеш файла → уникальная строка-отпечаток
    cache_dir = cache_root / key               # папка кэша с именем = этот хеш
    cache_dir.mkdir(parents=True, exist_ok=True)  # создаём папку (и родителей), не падаем если уже есть
    return cache_dir                           # возвращаем путь к папке кэша

def cache_file(cache_root: Path, audio_path: Path, name: str) -> Path:
    """
    Возвращает путь к файлу внутри кэш-папки БЕЗ создания папок и файлов.

    Структура: cache_root/<sha256(audio_path)>/<name>

    Используется, когда ПРОВЕРЯЕМ «есть ли кэш» (через .exists()).
    Не создаёт пустых папок — это важно, чтобы output/cache/
    не засорялось при каждом запросе.

    Args:
        cache_root: корень кэша.
        audio_path: путь к аудиофайлу.
        name: имя файла внутри кэша (например, "diarization.json").

    Returns:
        Путь (не созданный).
    """
    key = sha256_file(audio_path)
    return cache_root / key / name