from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


BASE_DIR = Path(__file__).resolve().parent.parent
CHATTERBOX_DIR = BASE_DIR / "models" / "chatterbox"
BASE_MODEL_DIR = CHATTERBOX_DIR / "base"
PTBR_MODEL_DIR = CHATTERBOX_DIR / "pt-br"


def main() -> None:
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PTBR_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[Chatterbox] Baixando assets base...")
    snapshot_download(
        repo_id="ResembleAI/chatterbox",
        repo_type="model",
        revision="main",
        allow_patterns=["ve.pt"],
        local_dir=str(BASE_MODEL_DIR),
    )

    print("[Chatterbox] Baixando pacote PT-BR V3...")
    snapshot_download(
        repo_id="ResembleAI/Chatterbox-Multilingual-pt-br",
        repo_type="model",
        revision="main",
        allow_patterns=[
            "t3_pt_br.safetensors",
            "s3gen_v3.pt",
            "grapheme_mtl_merged_expanded_v1.json",
        ],
        local_dir=str(PTBR_MODEL_DIR),
    )

    cangjie_cache = hf_hub_download(
        repo_id="ResembleAI/chatterbox",
        repo_type="model",
        filename="Cangjie5_TC.json",
        cache_dir=str(PTBR_MODEL_DIR),
    )
    shutil.copy2(cangjie_cache, PTBR_MODEL_DIR / "Cangjie5_TC.json")

    print("[Chatterbox] Pronto em models/chatterbox.")


if __name__ == "__main__":
    main()
