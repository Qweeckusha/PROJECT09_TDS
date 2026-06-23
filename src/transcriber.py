from faster_whisper import WhisperModel
from pathlib import Path
import numpy as np
import subprocess

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

    def _load_and_preprocess(self, path: str) -> np.ndarray:
        """
        Обрабатывает файл по контракту faster-whisper перед транскрибацией и загружает в память

        :param path: путь к аудио фвйлу
        :return: numpy array формы (N,) [-1.0, 1.0]
        """

        cmd = [
            "ffmpeg",
            "-i", str(path),
            "-ar", "16000",
            "-ac", "1",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "pipe:1"
        ]

        proccess = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )

        return np.frombuffer(proccess.stdout, dtype=np.float32)

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
        audio = self._load_and_preprocess(audio_path)

        segments, info = self.model.transcribe(
            audio,
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

        return data

# if __name__ == "__main__":
#
#     audio_file1 = "A:/MLProjs/PR09-tds/input/3.ogg"
#     audio_file2 = "A:/MLProjs/PR09-tds/input/1.ogg"
#
#     t = Transcriber()
#
#     # Строка вызова
#     print(t.transcribe_audio(audio_path=audio_file1))
