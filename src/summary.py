from datetime import datetime
from pathlib import Path

from config import FULL_TEXT_DIR, save_full_text_file, save_tds_file, TDS_DIR, prompt, ollama_llm
from .diarizer import CustomDiarizer
from .transcriber import Transcriber

import ollama as ol


class Summarizer:
    def __init__(self):
        self.transcriber = Transcriber()
        self.diarizer = CustomDiarizer()

    def _align_segments(self, whisper_seg: list[dict], diarizer_seg: list[dict]) -> list[dict]:
        """
        Ключевая функция агрегации для соединения двух списков словарей от разных модулей

        :param whisper_seg: сегменты от модуля транскрибирования
        :param diarizer_seg: сегменты от модуля диаризации
        :return: список словарей по каждому отрезку транскрибирования, где добавляется соответствующий спикер
        """

        # Сценарий A: модуль транскрибирования не смог определить текст в аудио или вернул ошибку.
        if not whisper_seg:
            return []

        # Сценарий B: модуль диаризации не смог определить спикеров или вернул ошибку
        if not diarizer_seg:
            return [
                {
                    **ws,
                    "speaker": "UNKNOWN",
                    "warning": "Diarization failed: module couldn't detect speakers"
                } for ws in whisper_seg
            ]

        # Сценарий C: модуль диаризации распознал только 1 спикера. Оптимизация расчётов
        speakers = {seg["speaker"] for seg in diarizer_seg}
        if len(speakers) == 1:
            speaker = next(iter(speakers))
            return [
                {
                    **ws,
                    "speaker": speaker
                } for ws in whisper_seg
            ]

        # Сценарий D: диалог/полилог распознано более 2 спикеров и есть текст. Расчёты по схождению меток времени.
        result = []
        for ws in whisper_seg:
            best_speaker = "UNKNOWN"
            max_overlap = 0.0

            for ds in diarizer_seg:
                overlap = max(0.0, min(ws["end"], ds["end"]) - max(ws["start"], ds["start"]))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = ds["speaker"]
            result.append(
                {
                    "start": ws["start"],
                    "end": ws["end"],
                    "text": ws["text"],
                    "speaker": best_speaker,
                }
            )
        return result

    def _format_full_text(self, align_segs: list[dict]) -> str:
        """
        Форматирует текст до читаемого вида и проверяет на монолог или диалог+

        При монологе возвращает текст без спикеров, только временные метки

        :param align_segs: align_segments - список объединённых сегментов
        :return: полный текст прошедший стандартный pipeline
        """
        if not align_segs:
            raise ValueError("Нет сегментов для форматирования (возможно, в аудио нет речи)")

        speakers = {seg.get("speaker", "UNKNOWN") for seg in align_segs}
        is_monologue = len(speakers) == 1

        lines = []

        for seg in align_segs:
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg["start"]
            end = seg["end"]

            if is_monologue:
                lines.append(f"[{start} - {end}] {seg['text']}")
            else:
                lines.append(f"[{start} - {end}] {speaker}: {seg['text']}")

        return "\n".join(lines)


    def get_full_text_from_audio(self, path: str) -> str:
        """
        Стандартный Pipeline (StdPl) - задействует все модули и прогоняет файл через них

        :param path: путь к аудиофайлу
        :return: возвращает полный текст транскрибированный и диаризованный
        """
        whisper_seg = self.transcriber.transcribe_audio(path)
        diarizer_seg = self.diarizer.Diarization(path)


        align = self._align_segments(whisper_seg, diarizer_seg)
        text = self._format_full_text(align)
        if save_full_text_file:
            self._save_text(path, text)
        return text

    def _save_text(self, path: str, text: str, is_LLM: bool = False) -> None:
        """
        Сохраняет текст в кэш в виде .txt файла

        :param path: путь к аудиофайлу, но нужен для названия output файлов
        :param text: текст для записи в файл
        :param is_LLM: bool переменная для разделения по директориям
        :return: None возвращать нечего, по сути функция просто выполняет свою задачу и не обязана отчитываться
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = Path(path).stem
        if is_LLM:
            TDS_DIR.mkdir(exist_ok=True)
            file_path = TDS_DIR / f"TDS-audio_{source_name}_{timestamp}.txt"
        else:
            FULL_TEXT_DIR.mkdir(exist_ok=True)
            file_path = FULL_TEXT_DIR / f"transcribed-audio_{source_name}_{timestamp}.txt"

        file_path.write_text(text, encoding="utf-8")


    def get_text_from_llm(self, path: str) -> str:
        """
        Полный Pipeline (FPl) - выполняет конечную роль - суммаризация текста, определение выводов и важных тезисов при помощи LLM

        :param model_name: переменная с точным названием модели, должна быть предварительно скачана
        :param path: Путь к аудиофайлу - первая точка вхождения
        :return: результат вызова LLM с промптом и текстом из StdPl
        """

        audio_text = self.get_full_text_from_audio(path)



        response: ol.ChatResponse = ol.chat(model=ollama_llm,
                                            messages=[
                                                    {
                                                        "role": "system",
                                                        "content": prompt
                                                    },
                                                    {
                                                        "role": "user",
                                                        "content": audio_text
                                                    }
                                                ])
        llm_output = response.message.content

        if save_tds_file:
            self._save_text(path, llm_output, is_LLM=True)

        return llm_output


# text = Summarizer().get_text_from_llm(path="A:/MLProjs/PR09-tds/input/3.ogg")
# print(text)