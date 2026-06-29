import subprocess
from pathlib import Path

import config
import torch
from pyannote.audio import Pipeline


class CustomDiarizer:
    """
        Класс для диаризации аудио (кто/когда говорил).

        Принцип работы:
        1. Пайплайн загружается один раз при создании экземпляра (базовое правило при работе с тяжёлыми моделями)
        2. Метод build_diarize_to_list() можно вызывать для множества файлов
    """

    def __init__(self, token: str = config.token, model: str = config.DIARIZATION_model ):
        """
        :param model: модель для диаризации
        :param token: уникальный токен Huggingface_HUB, обязателен только для загрузки модели
        """

        self.pipeline = Pipeline.from_pretrained(model, token=token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(device)

    def _preprocess_audio(self, path: str) -> str:
        """Конвертирует аудио в 16 кГц моно WAV для совместимости с pyannote."""
        path_obj = Path(path)

        if path_obj.suffix.lower() == ".wav":
            return path

        wav_path = path_obj.with_suffix(".wav")
        if not wav_path.exists():
            subprocess.run([
                "ffmpeg", "-i", str(path_obj),
                "-ar", "16000", "-ac", "1",
                "-y",
                str(wav_path)
            ], check=True, capture_output=True)

        return str(wav_path)

    def Diarization(self, path : str) -> list[dict]:
        """
        Запаковка данных о записи в лист словарей.

        :param path: Путь к файлу

        :returns: список словарей [{"start": 0.0, "end": 5.5, "speaker": "SPEAKER_00"}, ...]
        """
        preprocessed_path = self._preprocess_audio(path)
        try:
            result = self.pipeline(preprocessed_path).speaker_diarization.itertracks(yield_label=True)

            return [
                {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                } for turn, _, speaker in result
            ]
        finally:
            if preprocessed_path != path and Path(preprocessed_path).exists():
                Path(preprocessed_path).unlink()
