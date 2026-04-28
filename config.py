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

# Para 16GB VRAM/32GB RAM com modelo ~24B quantizado, 8192 costuma ser o ponto
# mais equilibrado. Contextos muito altos (ex.: 16k) aumentam KV cache, VRAM e
# tempo de prefill sem ganho real para chat curto/voz.
LLM_N_CTX: int        = int(os.getenv("LLM_N_CTX", 8192))

_llm_max_tokens_env = int(os.getenv("LLM_MAX_TOKENS", 320))
LLM_MAX_TOKENS: int   = _llm_max_tokens_env if _llm_max_tokens_env > 0 else 320
LLM_N_GPU_LAYERS: int = int(os.getenv("LLM_N_GPU_LAYERS", -1))  # -1 = toda a GPU
LLM_N_BATCH: int      = int(os.getenv("LLM_N_BATCH", 2048))     # tokens por batch no prefill
LLM_N_UBATCH: int     = int(os.getenv("LLM_N_UBATCH", 512))     # micro-batch interno
LLM_N_THREADS: int    = int(os.getenv("LLM_N_THREADS", max(4, (os.cpu_count() or 8) // 2)))
LLM_N_THREADS_BATCH: int = int(os.getenv("LLM_N_THREADS_BATCH", os.cpu_count() or 8))

# KV cache quantization. Q8_0 reduz uso de VRAM do KV cache sem perda perceptível
# para chat e é aplicado ao carregar o modelo, ainda no startup do bot.
LLM_KV_TYPE: str = os.getenv("LLM_KV_TYPE", "q8_0").strip().lower()

# Limite de tokens para respostas de voz (respostas curtas = resposta rápida)
LLM_VOZ_MAX_TOKENS: int = int(os.getenv("LLM_VOZ_MAX_TOKENS", 96))

# Parâmetros de qualidade / controle de repetição
LLM_TEMPERATURE: float        = float(os.getenv("LLM_TEMPERATURE",        0.7))
LLM_REPEAT_PENALTY: float     = float(os.getenv("LLM_REPEAT_PENALTY",     1.2))
LLM_FREQUENCY_PENALTY: float  = float(os.getenv("LLM_FREQUENCY_PENALTY",  0.4))
LLM_PRESENCE_PENALTY: float   = float(os.getenv("LLM_PRESENCE_PENALTY",   0.4))
