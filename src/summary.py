from datetime import datetime

from transcriber import Transcriber
from diarizer import CustomDiarizer

import os

from config import FULL_TEXT_DIR, save_full_text_file

class Summarizer:
    def __init__(self):
        self.transcriber = Transcriber()
        self.diarizer = CustomDiarizer(token=os.getenv("HF_TOKEN"))

    def _align_segments(self, whisper_seg: list[dict], diarizer_seg: list[dict]) -> list[dict]:


        # Сценарий A: модуль транскрибации не смог определить текст в аудио или вернул ошибку.
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

        if not align_segs:
            return "ERROR: _align_segments return did nothing"

        lines = []

        speakers = {seg.get("speaker", "UNKNOWN") for seg in align_segs}
        is_monologue = len(speakers) == 1

        for seg in align_segs:
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg["start"]
            end = seg["end"]

            if is_monologue:
                lines.append(f"[{start} - {end}] {seg['text']}")
            else:
                lines.append(f"[{start} - {end}] {speaker}: {seg['text']}")

        return "\n".join(lines)


    def get_full_text_from_audio(self, path: str):

        whisper_seg = self.transcriber.transcribe_audio(path)
        diarizer_seg = self.diarizer.build_diarize_to_list(path)

        align = self._align_segments(whisper_seg, diarizer_seg)
        text = self._format_full_text(align)
        if save_full_text_file:
            self._save_full_text(text)
        return text

    def _save_full_text(self, text: str):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = FULL_TEXT_DIR / f"saved_{timestamp}.txt"

        file_path.write_text(text, encoding="utf-8")
        return file_path

    def get_text_from_llm(self, path: str):
        pass


text = Summarizer().get_full_text_from_audio("A:/MLProjs/PR09-tds/input/3.ogg")
print(text)