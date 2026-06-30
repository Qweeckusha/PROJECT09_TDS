from pathlib import Path
import os

# Директории для кэша (по умолчанию в папке, где лежит config)
CACHE_DIR = Path(__file__).parent / "cache"

# Можно использовать прямой путь
# CACHE_DIR = 'C://Users/User/Desktop'

CACHE_DIR.mkdir(exist_ok=True)

FULL_TEXT_DIR = CACHE_DIR / "full_text"
TDS_DIR = CACHE_DIR / "tds"


#                   === Settings ===

save_full_text_file = False
save_tds_file = False

#                      HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"

#               == Транскрибирование ==

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-large-v3")
DEVICE = "cuda"
COMPUTE_TYPE = "float16"


#                   == Диаризация ==

TOKEN = os.getenv("HF_TOKEN")
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


#                  == Суммаризация ==

ollama_llm = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

PROMPT_FILE = Path(__file__).parent / "prompt.txt"
if PROMPT_FILE.exists():
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
else:
    prompt = "Ты AI-ассистент. Суммаризируй текст."





