"""
Визуализация overlap в диаризации.

Главный график: zoom на участок с overlap (видно, что два спикера
говорят одновременно — система это распознала).

Дополнительно: контекстный timeline сверху (где этот участок в общей записи).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def load_diarization(path: Path) -> tuple[list[dict], dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["segments"], {
        "num_speakers": payload.get("num_speakers"),
        "num_segments": payload.get("num_segments"),
        "model": payload.get("model", "unknown"),
        "audio": payload.get("audio", "unknown"),
    }


def find_overlaps(segments: list[dict]) -> list[tuple[float, float]]:
    """Находит интервалы, где пересекаются сегменты разных спикеров."""
    overlaps = []
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    for i, s1 in enumerate(sorted_segs):
        for s2 in sorted_segs[i + 1:]:
            if s2["start"] >= s1["end"]:
                break
            if s1["speaker"] != s2["speaker"]:
                overlap_start = max(s1["start"], s2["start"])
                overlap_end = min(s1["end"], s2["end"])
                if overlap_end > overlap_start:
                    overlaps.append((overlap_start, overlap_end))
    # Merge пересекающиеся интервалы
    if not overlaps:
        return []
    overlaps.sort()
    merged = [overlaps[0]]
    for start, end in overlaps[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def plot_overlap_zoom(
    segments: list[dict],
    overlaps: list[tuple[float, float]],
    ax: plt.Axes,
    zoom_start: float,
    zoom_end: float,
) -> None:
    """Zoom на участок с overlap: видно перекрытие блоков."""
    speakers = sorted({s["speaker"] for s in segments})
    speaker_y = {spk: i for i, spk in enumerate(speakers)}
    colors = {"SPEAKER_00": "#1f77b4", "SPEAKER_01": "#ff7f0e"}
    for i, spk in enumerate(speakers):
        colors.setdefault(spk, plt.cm.tab10(i))

    # Рисуем сегменты, попавшие в окно
    for s in segments:
        if s["end"] < zoom_start or s["start"] > zoom_end:
            continue
        y = speaker_y[s["speaker"]]
        x_start = max(s["start"], zoom_start)
        x_end = min(s["end"], zoom_end)
        ax.barh(
            y, x_end - x_start, left=x_start, height=0.6,
            color=colors[s["speaker"]], alpha=0.85,
            edgecolor="black", linewidth=0.5,
        )

    # Подсветка overlap зон
    for (o_start, o_end) in overlaps:
        if o_end < zoom_start or o_start > zoom_end:
            continue
        ax.axvspan(
            max(o_start, zoom_start), min(o_end, zoom_end),
            color="red", alpha=0.2, zorder=0,
        )

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlim(zoom_start, zoom_end)
    ax.set_xlabel("Время, сек")
    ax.set_title(
        f"Overlap zoom: {zoom_start:.1f}–{zoom_end:.1f} сек "
        f"(красным — одновременная речь)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)

    # Легенда
    handles = [Patch(facecolor=colors[spk], label=spk) for spk in speakers]
    handles.append(Patch(facecolor="red", alpha=0.2, label="Overlap"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)


def plot_context_timeline(
    segments: list[dict],
    overlaps: list[tuple[float, float]],
    ax: plt.Axes,
    zoom_start: float,
    zoom_end: float,
) -> None:
    """Общий timeline (18.5 мин) с выделенной зоной zoom."""
    speakers = sorted({s["speaker"] for s in segments})
    speaker_y = {spk: i for i, spk in enumerate(speakers)}
    colors = {"SPEAKER_00": "#1f77b4", "SPEAKER_01": "#ff7f0e"}
    for i, spk in enumerate(speakers):
        colors.setdefault(spk, plt.cm.tab10(i))

    for s in segments:
        y = speaker_y[s["speaker"]]
        ax.barh(
            y, s["end"] - s["start"], left=s["start"], height=0.6,
            color=colors[s["speaker"]], alpha=0.7,
            edgecolor="none",
        )

    # Подсветка зоны zoom
    ax.axvspan(zoom_start, zoom_end, color="yellow", alpha=0.3, label="Zoom-зона")

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Время, сек (вся запись)")
    ax.set_title(f"Контекст: вся запись ({segments[-1]['end']:.0f} сек)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"ERROR: not found: {args.json_path}")
        return 1

    segments, meta = load_diarization(args.json_path)
    overlaps = find_overlaps(segments)

    if not overlaps:
        print("Overlap не найден — нечего визуализировать.")
        return 1

    # Берём первый (или самый длинный) overlap для zoom
    zoom_start, zoom_end = max(overlaps, key=lambda o: o[1] - o[0])
    # Расширяем окно ±5 сек для контекста
    zoom_start = max(0, zoom_start - 5)
    zoom_end = zoom_end + 5

    args.out.mkdir(parents=True, exist_ok=True)

    fig, (ax_ctx, ax_zoom) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[1, 2],
    )
    plot_context_timeline(segments, overlaps, ax_ctx, zoom_start, zoom_end)
    plot_overlap_zoom(segments, overlaps, ax_zoom, zoom_start, zoom_end)

    total_overlap_sec = sum(e - s for s, e in overlaps)
    fig.suptitle(
        f"Overlap: {len(overlaps)} зон, {total_overlap_sec:.1f} сек суммарно "
        f"({total_overlap_sec / segments[-1]['end'] * 100:.1f}% записи)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    out_path = args.out / "diarization_overlap.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out_path}")
    print(f"Overlap зон: {len(overlaps)}, суммарно {total_overlap_sec:.1f} сек")
    for i, (s, e) in enumerate(overlaps[:5], 1):
        print(f"  {i}. {s:.2f}–{e:.2f} сек ({e-s:.1f} сек)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())