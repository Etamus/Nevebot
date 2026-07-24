"""
Configuracoes centrais do Nevebot.
Lê variáveis de ambiente do arquivo .env (ou do ambiente do sistema).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega o .env da raiz do projeto
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


def _path_env(nome_env: str, padrao: Path) -> Path:
    valor = os.getenv(nome_env, "").strip().strip('"')
    caminho = Path(valor) if valor else padrao
    return caminho if caminho.is_absolute() else BASE_DIR / caminho

# ── Discord ──────────────────────────────────────────────────────────────────
DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
if not DISCORD_TOKEN:
    raise ValueError(
        "Token do Discord não encontrado. "
        "Copie .env.example para .env e preencha DISCORD_TOKEN."
    )

# ── Modelo LLM ───────────────────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "models"
MODELS_TEXTO_DIR = MODELS_DIR / "texto"
CHATTERBOX_DIR = MODELS_DIR / "chatterbox"

def encontrar_modelo(pasta: Path | None = None, *, obrigatorio: bool = True) -> str:
    """
    Retorna o caminho do primeiro arquivo .gguf encontrado na pasta informada.
    Se pasta não for informada, procura em models/.
    """
    pasta = pasta or MODELS_DIR
    arquivos = sorted(pasta.glob("*.gguf"))
    if not arquivos:
        if not obrigatorio:
            return ""
        raise FileNotFoundError(
            f"Nenhum modelo .gguf encontrado em '{pasta}'.\n"
            "Coloque um arquivo .gguf na pasta correta e reinicie o bot."
        )
    return str(arquivos[0])

def _modelo_env_ou_pasta(nome_env: str, pasta: Path, fallback: str = "") -> str:
    valor = os.getenv(nome_env, "").strip().strip('"')
    encontrado = encontrar_modelo(pasta, obrigatorio=False)
    if valor:
        caminho = Path(valor)
        resolvido = caminho if caminho.is_absolute() else BASE_DIR / caminho
        if resolvido.exists():
            return str(resolvido)
        return encontrado or fallback
    return encontrado or fallback

# Parâmetros do LLM
_modelo_raiz = encontrar_modelo(MODELS_DIR, obrigatorio=False)
LLM_MODEL_PATH: str = _modelo_env_ou_pasta(
    "LLM_MODEL_PATH",
    MODELS_TEXTO_DIR,
    fallback=_modelo_raiz,
)
if not LLM_MODEL_PATH:
    raise FileNotFoundError(
        f"Nenhum modelo .gguf encontrado em '{MODELS_TEXTO_DIR}' ou '{MODELS_DIR}'.\n"
        "Coloque um modelo .gguf em models/texto/ e reinicie o bot."
    )

# Para chat curto/voz, 4096 reduz KV cache, VRAM e tempo de prefill sem perder
# utilidade pratica na conversa em tempo real.
LLM_N_CTX: int        = int(os.getenv("LLM_N_CTX", 4096))

_llm_max_tokens_env = int(os.getenv("LLM_MAX_TOKENS", 220))
LLM_MAX_TOKENS: int   = _llm_max_tokens_env if _llm_max_tokens_env > 0 else 220
LLM_N_GPU_LAYERS: int = int(os.getenv("LLM_N_GPU_LAYERS", -1))  # -1 = toda a GPU
LLM_N_BATCH: int      = int(os.getenv("LLM_N_BATCH", 1024))     # tokens por batch no prefill
LLM_N_UBATCH: int     = int(os.getenv("LLM_N_UBATCH", 256))     # micro-batch interno
LLM_N_THREADS: int    = int(os.getenv("LLM_N_THREADS", max(4, (os.cpu_count() or 8) // 2)))
LLM_N_THREADS_BATCH: int = int(os.getenv("LLM_N_THREADS_BATCH", os.cpu_count() or 8))

# KV cache quantization. Q8_0 reduz uso de VRAM do KV cache sem perda perceptível
# para chat e é aplicado ao carregar o modelo, ainda no startup do bot.
LLM_KV_TYPE: str = os.getenv("LLM_KV_TYPE", "q8_0").strip().lower()
LLM_CHAT_TEMPLATE: str = os.getenv("LLM_CHAT_TEMPLATE", "chatml").strip()

# llama.cpp oficial baixado pelo instalar.bat. O Python conversa com
# llama-server.exe via HTTP local, sem depender de llama-cpp-python/CUDA Toolkit.
LLAMA_CPP_DIR: Path = _path_env("LLAMA_CPP_DIR", BASE_DIR / "llama.cpp")
LLAMA_CPP_SERVER_EXE: Path = _path_env("LLAMA_CPP_SERVER_EXE", LLAMA_CPP_DIR / "llama-server.exe")
LLAMA_SERVER_HOST: str = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1").strip()
LLAMA_SERVER_PORT: int = int(os.getenv("LLAMA_SERVER_PORT", 8080))
LLAMA_SERVER_URL: str = os.getenv(
    "LLAMA_SERVER_URL",
    f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}",
).rstrip("/")
LLAMA_SERVER_STARTUP_TIMEOUT: int = int(os.getenv("LLAMA_SERVER_STARTUP_TIMEOUT", 600))
LLAMA_REQUEST_TIMEOUT: int = int(os.getenv("LLAMA_REQUEST_TIMEOUT", 600))

# Limite de tokens para respostas de voz. O prompt mantem a resposta curta; a
# folga evita corte no meio da frase quando o modelo precisa fechar a ideia.
LLM_VOZ_MAX_TOKENS: int = int(os.getenv("LLM_VOZ_MAX_TOKENS", 64))

# Chatterbox Multilingual V3 PT-BR local.
CHATTERBOX_BASE_DIR: Path = _path_env("CHATTERBOX_BASE_DIR", CHATTERBOX_DIR / "base")
CHATTERBOX_PTBR_DIR: Path = _path_env("CHATTERBOX_PTBR_DIR", CHATTERBOX_DIR / "pt-br")
CHATTERBOX_DEVICE: str = os.getenv("CHATTERBOX_DEVICE", "cuda").strip().lower()
CHATTERBOX_LANGUAGE_ID: str = "pt"
CHATTERBOX_MAX_CHARS: int = int(os.getenv("CHATTERBOX_MAX_CHARS", 280))

# Parâmetros de qualidade / controle de repetição
LLM_TEMPERATURE: float        = float(os.getenv("LLM_TEMPERATURE",        0.8))
LLM_MIN_P: float              = float(os.getenv("LLM_MIN_P",              0.05))
LLM_TOP_P: float              = float(os.getenv("LLM_TOP_P",              1.0))
LLM_TOP_K: int                = int(os.getenv("LLM_TOP_K",                0))
LLM_DRY_MULTIPLIER: float     = float(os.getenv("LLM_DRY_MULTIPLIER",     0.8))
LLM_DRY_ALLOWED_LENGTH: int   = int(os.getenv("LLM_DRY_ALLOWED_LENGTH",   3))
LLM_REPEAT_PENALTY: float     = float(os.getenv("LLM_REPEAT_PENALTY",     1.0))
LLM_FREQUENCY_PENALTY: float  = float(os.getenv("LLM_FREQUENCY_PENALTY",  0.0))
LLM_PRESENCE_PENALTY: float   = float(os.getenv("LLM_PRESENCE_PENALTY",   0.0))
