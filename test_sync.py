"""
Тест синхронизации видео.

1. Создаёт 3 тестовых видео с разным сдвигом аудио
2. Запускает sync_videos
3. Проверяет сдвиги
"""

import subprocess
from pathlib import Path


def create_test_video(output_path, duration=10, audio_delay=0.0):
    """Создаёт видео с задержкой аудио."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration + audio_delay}:size=320x240:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-af", f"adelay={int(audio_delay*1000)}|{int(audio_delay*1000)}",
        "-c:v", "libx264", "-c:a", "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)ы

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
    print(f"✅ Сдвиги (в сэмплах): {shifts}")
    print(f"✅ Сдвиги (в секундах): {[s / 16000 for s in shifts]}")
    print("=" * 60)
    
    # 4. Проверка
    print("\n📊 Проверка:")
    print(f"  Ожидалось: [0, ~8000, ~16000] (0.5 сек = 8000 сэмплов)")
    print(f"  Получено:  {shifts}")


if __name__ == "__main__":
    main()
