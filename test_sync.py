"""
Тест синхронизации видео.

1. Создаёт 3 тестовых видео с разным сдвигом аудио
2. Запускает sync_videos
3. Проверяет сдвиги
"""

import subprocess
from pathlib import Path


def create_test_video(output_path: str, duration: int = 10, audio_delay: float = 0.0) -> None:
    """Создаёт тестовое видео с задержкой аудио."""
    # Мы делаем видео длиной duration + audio_delay
    # Чтобы сигнал не обрезался
    total_duration = duration + audio_delay
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={total_duration}:size=320x240:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-af", f"adelay={int(audio_delay*1000)}|{int(audio_delay*1000)}",
        "-c:v", "libx264", "-c:a", "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def main():
    print("=" * 60)
    print("Тест синхронизации видео")
    print("=" * 60)
    
    # 1. Создать тестовые видео
    Path("uploads").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    print("\n📹 Создание тестовых видео...")
    create_test_video("uploads/cam1.mp4", duration=10, audio_delay=0.0)
    create_test_video("uploads/cam2.mp4", duration=10, audio_delay=0.5)
    create_test_video("uploads/cam3.mp4", duration=10, audio_delay=1.0)
    print("✅ Тестовые видео созданы")
    
    # 2. Синхронизация
    print("\n🔄 Запуск синхронизации...")
    from app.services.sync import sync_videos
    
    shifts = sync_videos([
        "uploads/cam1.mp4",
        "uploads/cam2.mp4",
        "uploads/cam3.mp4",
    ])
    
       # 3. Результат
    print("\n" + "=" * 60)
    print(f"✅ Сдвиги (сэмплы): {shifts['shifts_samples']}")
    print(f"✅ Сдвиги (сек): {shifts['shifts_seconds']}")
    print(f"✅ Файл сохранён: output/sync_result.json")
    print("=" * 60)

        # 4. Выравнивание
    print("\n🔄 Выравнивание видео...")
    from app.services.sync import align_videos
    
    aligned = align_videos(
        ["uploads/cam1.mp4", "uploads/cam2.mp4", "uploads/cam3.mp4"],
        shifts["shifts_samples"]
    )
    
    print(f"✅ Выровнено видео: {len(aligned)}")
    for path in aligned:
        print(f"  - {path}")


    
    # 4. Проверка
    print("\n📊 Проверка:")
    print(f"  Ожидалось: [0, ~8000, ~16000] (0.5 сек = 8000 сэмплов)")
    print(f"  Получено:  {shifts}")


if __name__ == "__main__":
    main()
