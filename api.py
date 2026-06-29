import uuid
from pathlib import Path

import httpx
from pydantic import BaseModel, HttpUrl
from fastapi import FastAPI, HTTPException
from src.summary import Summarizer

app = FastAPI(title="TDS API")
summarizer = Summarizer()

CACHE_DIR = Path(__file__).parent / "cache" / "temp_audio"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class URLRequest(BaseModel):
    audio_url: HttpUrl


class PathRequest(BaseModel):
    audio_path: str


class SummarizeResponse(BaseModel):
    status: str
    summary: str | None = None
    error: str | None = None


async def _download_file(url: str) -> Path:
    """
    Загрузить файл по ссылке
    :param url: ссылка на файл
    :return: путь к временному файлу
    """
    file_path = CACHE_DIR / uuid.uuid4().hex

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

        return file_path

    except httpx.HTTPError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Ошибка скачивания: {e}")


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_audio(request: URLRequest):
    """Суммаризация аудио по URL."""

    file_path = await _download_file(str(request.audio_url))

    try:
        summary = summarizer.get_text_from_llm(str(file_path))
        return SummarizeResponse(status="success", summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {e}")

    finally:
        # Гарантированная очистка
        file_path.unlink(missing_ok=True)


@app.post("/summarize_path", response_model=SummarizeResponse)
async def summarize_from_path(request: PathRequest):
    """Суммаризация аудио по локальному пути (для тестирования)."""

    file_path = Path(request.audio_path)

    # Проверка существования файла
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {file_path}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Путь не является файлом: {file_path}")

    try:
        summary = summarizer.get_text_from_llm(str(file_path))
        return SummarizeResponse(status="success", summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {e}")

@app.post("/full_txt_path", response_model=SummarizeResponse)
async def full_txt_from_path(request: PathRequest):
    """получение текста из аудио по локальному пути (для тестирования)."""

    file_path = Path(request.audio_path)

    # Проверка существования файла
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {file_path}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Путь не является файлом: {file_path}")

    try:
        summary = summarizer.get_full_text_from_audio(str(file_path))
        return SummarizeResponse(status="success", summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}