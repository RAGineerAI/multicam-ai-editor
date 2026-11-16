"""
Тесты для app.utils.cache.

Проверяем:
- Детерминизм SHA256 (один файл → один хэш).
- Разные файлы → разные хэши.
- Пустой файл → известный хэш (e3b0c442... — SHA256 пустой строки).
- audio_cache_dir() создаёт папку.
- cache_file() НЕ создаёт папок (важно: без side effect).
- Структура путей правильная.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.cache import audio_cache_dir, cache_file, sha256_file

# SHA256 от пустой строки. Известное значение —
# используется как «эталон» для проверки, что функция работает.
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Временный файл с содержимым 'hello world'."""
    p = tmp_path / "sample.wav"
    p.write_bytes(b"hello world")
    return p


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """Временный пустой файл."""
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    return p


def test_sha256_deterministic(sample_file: Path) -> None:
    """Один и тот же файл → один и тот же хэш при повторных вызовах."""
    h1 = sha256_file(sample_file)
    h2 = sha256_file(sample_file)
    assert h1 == h2
    assert len(h1) == 64  # hex SHA256 — всегда 64 символа


def test_sha256_different_files(tmp_path: Path) -> None:
    """Разные файлы → разные хэши."""
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"content A")
    b.write_bytes(b"content B")
    assert sha256_file(a) != sha256_file(b)


def test_sha256_empty_file(empty_file: Path) -> None:
    """Пустой файл → известный SHA256."""
    assert sha256_file(empty_file) == EMPTY_SHA256


def test_sha256_chunk_size_irrelevant(sample_file: Path) -> None:
    """Размер чанка не влияет на результат — только на скорость."""
    h_1mb = sha256_file(sample_file, chunk_size=1 << 20)
    h_4b = sha256_file(sample_file, chunk_size=4)
    assert h_1mb == h_4b


def test_sha256_missing_file(tmp_path: Path) -> None:
    """Несуществующий файл → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "nope.wav")


def test_audio_cache_dir_creates(sample_file: Path, tmp_path: Path) -> None:
    """audio_cache_dir создаёт папку <cache_root>/<sha256>/."""
    cache_root = tmp_path / "cache"
    d = audio_cache_dir(cache_root, sample_file)

    assert d.exists()
    assert d.is_dir()
    assert d.parent == cache_root
    # Имя папки — 64-символьный hex SHA256
    assert len(d.name) == 64


def test_audio_cache_dir_idempotent(sample_file: Path, tmp_path: Path) -> None:
    """Повторный вызов не падает (exist_ok=True)."""
    cache_root = tmp_path / "cache"
    d1 = audio_cache_dir(cache_root, sample_file)
    d2 = audio_cache_dir(cache_root, sample_file)
    assert d1 == d2


def test_cache_file_no_side_effect(sample_file: Path, tmp_path: Path) -> None:
    """cache_file НЕ создаёт папок — это важно для проверки кэша."""
    cache_root = tmp_path / "cache"

    path = cache_file(cache_root, sample_file, "diarization.json")

    # Путь правильный
    assert path.name == "diarization.json"
    assert path.parent.parent == cache_root
    assert len(path.parent.name) == 64  # папка = sha256

    # НО: ни папки, ни файла не должно быть
    assert not path.parent.exists()
    assert not path.exists()
    assert not cache_root.exists()


def test_cache_file_structure(sample_file: Path, tmp_path: Path) -> None:
    """Структура: <cache_root>/<sha256>/<name>."""
    cache_root = tmp_path / "cache"
    expected_hash = sha256_file(sample_file)

    path = cache_file(cache_root, sample_file, "diarization.json")

    assert path == cache_root / expected_hash / "diarization.json"


def test_cache_file_same_audio_same_path(tmp_path: Path) -> None:
    """Один и тот же контент → один и тот же путь кэша (даже если имя разное)."""
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"  # другое имя, то же содержимое
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")

    cache_root = tmp_path / "cache"
    p1 = cache_file(cache_root, a, "x.json")
    p2 = cache_file(cache_root, b, "x.json")
    assert p1 == p2  # одинаковый хэш → одинаковый путь


def test_cache_file_different_audio_different_path(tmp_path: Path) -> None:
    """Разный контент → разный путь кэша."""
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"content A")
    b.write_bytes(b"content B")

    cache_root = tmp_path / "cache"
    p1 = cache_file(cache_root, a, "x.json")
    p2 = cache_file(cache_root, b, "x.json")
    assert p1 != p2