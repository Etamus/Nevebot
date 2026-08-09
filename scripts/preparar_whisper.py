from __future__ import annotations

import json
from pathlib import Path

from faster_whisper.utils import download_model


BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = BASE_DIR / "models" / "whisper"
VOICE_CONFIG = BASE_DIR / "data" / "voz_config.json"
DEFAULT_MODEL = "large-v3-turbo"
ALIASES = {
    "large turbo": DEFAULT_MODEL,
    "large-turbo": DEFAULT_MODEL,
    "large_v3_turbo": DEFAULT_MODEL,
    "turbo": DEFAULT_MODEL,
}


def selected_model() -> str:
    try:
        data = json.loads(VOICE_CONFIG.read_text(encoding="utf-8"))
        value = str(data.get("whisper_modelo", DEFAULT_MODEL)).strip()
    except (OSError, ValueError, TypeError):
        value = DEFAULT_MODEL
    return ALIASES.get(value.lower(), value) or DEFAULT_MODEL


def main() -> None:
    model = selected_model()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Whisper] Preparando {model}...")
    location = download_model(model, cache_dir=str(DOWNLOAD_DIR))
    print(f"[Whisper] Pronto em {location}")


if __name__ == "__main__":
    main()
