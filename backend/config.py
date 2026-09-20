from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ─── Application ───────────────────────────────────────────────
APP_NAME = os.getenv("APP_NAME", "Micron Tools AI Consultant")
APP_VERSION = os.getenv("APP_VERSION", "2.1.0")

# ─── Data ──────────────────────────────────────────────────────
DATA_FILE = Path(
    os.getenv(
        "DATA_FILE",
        str(BASE_DIR / "data" / "products_full.json"),
    )
)

# ─── Ollama (primary) ──────────────────────────────────────────
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", "http://127.0.0.1:11434/api/tags")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "150"))

# ─── Optional Cloud LLMs (fallback) ────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()  # e.g. gpt-4o-mini

# ─── CORS ──────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

# ─── Contact Info ──────────────────────────────────────────────
CONTACT_PHONE = os.getenv("CONTACT_PHONE", "021-66702812")
CONTACT_WHATSAPP = os.getenv("CONTACT_WHATSAPP", "09126334744")
CONTACT_ADDRESS = os.getenv(
    "CONTACT_ADDRESS",
    " تهران، حسن آباد، خیابان امام خمینی، کوچه سلطانی، پلاک ۱۸",
)
CONTACT_HOURS = os.getenv(
    "CONTACT_HOURS",
    "شنبه تا چهارشنبه ۸–۱۸ | پنجشنبه ۸–۱۴",
)
WEBSITE = os.getenv("WEBSITE", "https://www.microntoolss.ir")

