import torch
import torchaudio
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
from pyannote.core import Annotation


class CustomDiarizer:
    """
        Класс для диаризации аудио (кто/когда говорил).

        Принцип работы:
        1. Пайплайн загружается один раз при создании экземпляра (базовое правило при работе с тяжёлыми моделями)
        2. Метод build_diarize_to_list() можно вызывать для множества файлов
    """

    def __init__(self, token: str, model: str = "pyannote/speaker-diarization-community-1"):
        """
        :param model: модель для диаризации
        :param token: уникальный токен Huggingface_HUB
        """

        self.pipeline = Pipeline.from_pretrained(
            model, token=token
        )
        self.pipeline.to(torch.device("cuda"))

    def _load_and_preprocess(self, path: str) -> dict:
        """
        Приватный метод для загрузки и предобработки

        :param path: путь к аудиофайлу

        :return: словарь с тензором аудиофайла и его частота
        """
        waveform, sample_rate = torchaudio.load(path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        return {"waveform": waveform, "sample_rate": sample_rate}

    def MethodDeirize(self, path : str) -> Annotation:
        """
        Основной конвейер диаризации, который возвращает для каждого файла по сегментам времени - участников диалога

        :param path: путь к аудиофайлу

        :return: Annotation = (turn (start, end), id, specker)
        """
        return self.pipeline(self._load_and_preprocess(path)).speaker_diarization

    def build_diarize_to_list(self, path : str) -> list[dict]:
        """
        Запаковка данных о записи в лист словарей.

        :param path: путь к файлу

        :returns: список словарей [{"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"}, ...]
        """

        annotation = self.MethodDeirize(path)
        result = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            result.append({"start": turn.start, "end": turn.end, "specker": speaker})

        return result


load_dotenv()
token = os.getenv("HF_TOKEN")
audio_path = "A:/MLProjs/PR09-tds/input/3.ogg"
print("\n=== РЕЗУЛЬТАТ ДИАРИЗАЦИИ ===")
print(CustomDiarizer(token=token).build_diarize_to_list(path=audio_path))

