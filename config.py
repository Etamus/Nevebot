"""
Configuracoes centrais do Nevebot.
Lê variáveis de ambiente do arquivo .env (ou do ambiente do sistema).
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv

# Carrega o .env da raiz do projeto
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ── Discord ──────────────────────────────────────────────────────────────────
DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
if not DISCORD_TOKEN:
    raise ValueError(
        "Token do Discord não encontrado. "
        "Copie .env.example para .env e preencha DISCORD_TOKEN."
    )

# ── Modelo LLM ───────────────────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "models"

def encontrar_modelo() -> str:
    """
    Retorna o caminho do primeiro arquivo .gguf encontrado em models/.
    Levanta FileNotFoundError se nenhum modelo for encontrado.
    """
    arquivos = sorted(MODELS_DIR.glob("*.gguf"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum modelo .gguf encontrado em '{MODELS_DIR}'.\n"
            "Coloque um arquivo .gguf dentro da pasta models/ e reinicie o bot."
        )
    return str(arquivos[0])

# Parâmetros do LLM
LLM_MODEL_PATH: str = encontrar_modelo()
LLM_N_CTX: int        = int(os.getenv("LLM_N_CTX", 2048))
LLM_MAX_TOKENS: int   = int(os.getenv("LLM_MAX_TOKENS", 512))
LLM_N_GPU_LAYERS: int = int(os.getenv("LLM_N_GPU_LAYERS", -1))  # -1 = toda a GPU
LLM_N_BATCH: int      = int(os.getenv("LLM_N_BATCH", 1024))     # tokens por batch no prefill

# Limite de tokens para respostas de voz (respostas curtas = resposta rápida)
LLM_VOZ_MAX_TOKENS: int = int(os.getenv("LLM_VOZ_MAX_TOKENS", 150))

# Parâmetros de qualidade / controle de repetição
LLM_TEMPERATURE: float        = float(os.getenv("LLM_TEMPERATURE",        0.7))
LLM_REPEAT_PENALTY: float     = float(os.getenv("LLM_REPEAT_PENALTY",     1.2))
LLM_FREQUENCY_PENALTY: float  = float(os.getenv("LLM_FREQUENCY_PENALTY",  0.4))
LLM_PRESENCE_PENALTY: float   = float(os.getenv("LLM_PRESENCE_PENALTY",   0.4))
