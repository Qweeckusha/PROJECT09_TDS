from config import WHISPER_MODEL, DEVICE, COMPUTE_TYPE
from faster_whisper import WhisperModel
from pathlib import Path

class Transcriber:
    """
    Класс транскрибирования для получения временных меток и сырого текста из аудио файла

    faster-whisper
    """

    def __init__(self,
                 model_size: str = WHISPER_MODEL,
                 device: str = DEVICE,
                 compute_type: str = COMPUTE_TYPE):
        """
        Инициализация модели транскрибирования

        Args:
            model_size: размер модели
            device: при помощи чего обрабатывать CPU или GPU
            compute_type: режим вычисления или же квантизация
        """

        self.model = WhisperModel(
            model_size,
            device=device,
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

        segments, _ = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True
        )

        return [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            } for segment in segments
        ]