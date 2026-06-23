from pathlib import Path
import os

# Директории для кэша
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

FULL_TEXT_DIR = CACHE_DIR / "full_text"
FULL_TEXT_DIR.mkdir(exist_ok=True)

# === Settings ===

save_full_text_file = True

# HuggingFace
os.environ["HF_HUB_OFFLINE"] = "0"