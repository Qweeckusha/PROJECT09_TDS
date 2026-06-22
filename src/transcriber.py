from faster_whisper import WhisperModel
from pathlib import Path

import torch
print(torch.__version__)
print(torch.cuda.is_available())

class Transcriber:
    """
    Класс транскрибатора для получения временных меток и сырого текста из аудио файла

    faster-whisper
    """


    def __init__(self,
                 model_size: str ="medium",
                 device: str ="cuda",
                 compute_type: str ="float16"):
        """
                Инициализация модели транскрибации
                Args:
                    model_size: размер модели (tiny, base, small, medium, large-v3)
                    device: default="cuda", auto, cpu
                    compute_type: режим вычисления или же квантизация (int8 / float16)


                """

        self.model = WhisperModel(
            model_size,
            device,
            compute_type=compute_type
        )

    def transcribe_audio(self,
                         audio_path: str | Path,
                         language: str = "ru"
                         ) -> list[dict]:
        """
        Транскрибирует аудиофайл и возвращает текст.

        Args:
            audio_path: путь к аудиофайлу

        Returns:
           list[dict]: Список сегментов вида:
                [{"start": 0.0, "end": 5.42, "text": "Привет!"}, ...]
        """

        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5
        )

        data = []
        for segment in segments:
            data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        return data

if __name__ == "__main__":

    audio_file1 = "A:/MLProjs/PR09-tds/input/3.ogg"
    audio_file2 = "A:/MLProjs/PR09-tds/input/1.ogg"

    t = Transcriber()

    # Строка вызова
    print(t.transcribe_audio(audio_path=audio_file1))
