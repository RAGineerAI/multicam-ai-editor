"""
Визуальная проверка синхронизации.

Показывает три аудиосигнала и их сдвиги.
"""

import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from app.services.sync import extract_audio, load_audio, find_shift


def main():
    print("=" * 60)
    print("Визуальная проверка синхронизации")
    print("=" * 60)
    
    # 1. Извлечь аудио
    Path("output").mkdir(exist_ok=True)
    print("\n📥 Извлечение аудио...")
    for i in range(3):
        extract_audio(f"uploads/cam{i+1}.mp4", f"output/audio_{i}.wav")
    
    # 2. Загрузить
    print("📥 Загрузка аудио...")
    audios = [load_audio(f"output/audio_{i}.wav") for i in range(3)]
    
    # 3. Найти сдвиги
    print("🔄 Поиск сдвигов...")
    shifts = [0]
    for audio in audios[1:]:
        shifts.append(find_shift(audios[0], audio))
    
    print(f"\n✅ Сдвиги (сэмплы): {shifts}")
    print(f"✅ Сдвиги (сек): {[s/16000 for s in shifts]}")
    
    # 4. Визуализация
    print("\n📊 Построение графиков...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    for i, audio in enumerate(audios):
        # Первые 2 секунды (32000 сэмплов при 16000 Гц)
        audio_short = audio[:32000]
        time_axis = np.arange(len(audio_short)) / 16000
        
        axes[i].plot(time_axis, audio_short, color=colors[i], linewidth=0.5)
        axes[i].set_title(
            f"Cam {i+1} — сдвиг: {shifts[i]/16000:+.2f} сек",
            fontsize=12,
        )
        axes[i].set_ylabel("Амплитуда")
        axes[i].set_xlim(0, 2)
        axes[i].grid(True, alpha=0.3)
        
        # Вертикальная линия на старте сигнала
        axes[i].axvline(x=0, color="red", linestyle="--", alpha=0.5)
    
    axes[-1].set_xlabel("Время (сек)")
    
    plt.suptitle("Синхронизация: 3 камеры", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = "output/sync_visual.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"✅ График сохранён: {output_path}")


if __name__ == "__main__":
    main()