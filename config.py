"""
Configuracoes centrais do Nevebot.
Lê variáveis de ambiente do arquivo .env (ou do ambiente do sistema).
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega o .env da raiz do projeto
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


def _carregar_llm_ui() -> dict:
    try:
        data = json.loads((BASE_DIR / "data" / "config_ui.json").read_text(encoding="utf-8"))
        llm = data.get("llm", {})
        return llm if isinstance(llm, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


_LLM_UI = _carregar_llm_ui()


def _llm_valor(chave: str, env: str, padrao: object) -> object:
    valor = _LLM_UI.get(chave)
    if valor is None or valor == "":
        valor = os.getenv(env, padrao)
    return valor


def _llm_int(chave: str, env: str, padrao: int) -> int:
    return int(_llm_valor(chave, env, padrao))


def _llm_float(chave: str, env: str, padrao: float) -> float:
    return float(_llm_valor(chave, env, padrao))


def _llm_str(chave: str, env: str, padrao: str) -> str:
    return str(_llm_valor(chave, env, padrao)).strip()


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
    encontrado = encontrar_modelo(pasta, obrigatorio=False)
    candidatos = (_LLM_UI.get("model_path"), os.getenv(nome_env, ""))
    for candidato in candidatos:
        valor = str(candidato or "").strip().strip('"')
        if not valor:
            continue
        caminho = Path(valor)
        resolvido = caminho if caminho.is_absolute() else BASE_DIR / caminho
        if resolvido.exists():
            return str(resolvido)
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
LLM_N_CTX: int        = _llm_int("n_ctx", "LLM_N_CTX", 4096)

_llm_max_tokens_env = _llm_int("max_tokens", "LLM_MAX_TOKENS", 220)
LLM_MAX_TOKENS: int   = _llm_max_tokens_env if _llm_max_tokens_env > 0 else 220
LLM_N_GPU_LAYERS: int = _llm_int("n_gpu_layers", "LLM_N_GPU_LAYERS", -1)
LLM_N_BATCH: int      = _llm_int("n_batch", "LLM_N_BATCH", 1024)
LLM_N_UBATCH: int     = _llm_int("n_ubatch", "LLM_N_UBATCH", 256)
LLM_N_THREADS: int    = _llm_int("n_threads", "LLM_N_THREADS", max(4, (os.cpu_count() or 8) // 2))
LLM_N_THREADS_BATCH: int = _llm_int("n_threads_batch", "LLM_N_THREADS_BATCH", os.cpu_count() or 8)

# KV cache quantization. Q8_0 reduz uso de VRAM do KV cache sem perda perceptível
# para chat e é aplicado ao carregar o modelo, ainda no startup do bot.
LLM_KV_TYPE: str = _llm_str("kv_type", "LLM_KV_TYPE", "q8_0").lower()
LLM_CHAT_TEMPLATE: str = _llm_str("chat_template", "LLM_CHAT_TEMPLATE", "")

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
LLM_VOZ_MAX_TOKENS: int = _llm_int("voz_max_tokens", "LLM_VOZ_MAX_TOKENS", 96)
LLM_VOZ_TEMPERATURE: float = _llm_float("voz_temperature", "LLM_VOZ_TEMPERATURE", 0.5)

# Chatterbox Multilingual V3 PT-BR local.
CHATTERBOX_BASE_DIR: Path = _path_env("CHATTERBOX_BASE_DIR", CHATTERBOX_DIR / "base")
CHATTERBOX_PTBR_DIR: Path = _path_env("CHATTERBOX_PTBR_DIR", CHATTERBOX_DIR / "pt-br")
CHATTERBOX_DEVICE: str = os.getenv("CHATTERBOX_DEVICE", "cuda").strip().lower()
CHATTERBOX_LANGUAGE_ID: str = "pt"
CHATTERBOX_MAX_CHARS: int = int(os.getenv("CHATTERBOX_MAX_CHARS", 280))
CHATTERBOX_WATERMARK: bool = os.getenv("CHATTERBOX_WATERMARK", "0").strip().lower() in {"1", "true", "yes", "sim"}
CHATTERBOX_FULL_WARMUP: bool = os.getenv("CHATTERBOX_FULL_WARMUP", "1").strip().lower() in {"1", "true", "yes", "sim"}
CHATTERBOX_CFM_TIMESTEPS: int = int(os.getenv("CHATTERBOX_CFM_TIMESTEPS", 4))
CHATTERBOX_MIN_SPEECH_TOKENS: int = int(os.getenv("CHATTERBOX_MIN_SPEECH_TOKENS", 40))
CHATTERBOX_MAX_SPEECH_TOKENS: int = int(os.getenv("CHATTERBOX_MAX_SPEECH_TOKENS", 560))
CHATTERBOX_SPEECH_TOKENS_PER_CHAR: float = float(os.getenv("CHATTERBOX_SPEECH_TOKENS_PER_CHAR", 2.0))
CHATTERBOX_SPEECH_TOKEN_BIAS: int = int(os.getenv("CHATTERBOX_SPEECH_TOKEN_BIAS", 24))

# Parâmetros de qualidade / controle de repetição
LLM_TEMPERATURE: float        = _llm_float("temperature", "LLM_TEMPERATURE", 0.8)
LLM_MIN_P: float              = _llm_float("min_p", "LLM_MIN_P", 0.05)
LLM_TOP_P: float              = _llm_float("top_p", "LLM_TOP_P", 1.0)
LLM_TOP_K: int                = _llm_int("top_k", "LLM_TOP_K", 0)
LLM_DRY_MULTIPLIER: float     = _llm_float("dry_multiplier", "LLM_DRY_MULTIPLIER", 0.8)
LLM_DRY_ALLOWED_LENGTH: int   = _llm_int("dry_allowed_length", "LLM_DRY_ALLOWED_LENGTH", 3)
LLM_REPEAT_PENALTY: float     = _llm_float("repeat_penalty", "LLM_REPEAT_PENALTY", 1.0)
LLM_FREQUENCY_PENALTY: float  = _llm_float("frequency_penalty", "LLM_FREQUENCY_PENALTY", 0.0)
LLM_PRESENCE_PENALTY: float   = _llm_float("presence_penalty", "LLM_PRESENCE_PENALTY", 0.0)


_LLM_CONFIG_ATTRS = {
    "model_path": "LLM_MODEL_PATH",
    "n_ctx": "LLM_N_CTX",
    "max_tokens": "LLM_MAX_TOKENS",
    "n_gpu_layers": "LLM_N_GPU_LAYERS",
    "n_batch": "LLM_N_BATCH",
    "n_ubatch": "LLM_N_UBATCH",
    "n_threads": "LLM_N_THREADS",
    "n_threads_batch": "LLM_N_THREADS_BATCH",
    "kv_type": "LLM_KV_TYPE",
    "chat_template": "LLM_CHAT_TEMPLATE",
    "voz_max_tokens": "LLM_VOZ_MAX_TOKENS",
    "voz_temperature": "LLM_VOZ_TEMPERATURE",
    "temperature": "LLM_TEMPERATURE",
    "min_p": "LLM_MIN_P",
    "top_p": "LLM_TOP_P",
    "top_k": "LLM_TOP_K",
    "dry_multiplier": "LLM_DRY_MULTIPLIER",
    "dry_allowed_length": "LLM_DRY_ALLOWED_LENGTH",
    "repeat_penalty": "LLM_REPEAT_PENALTY",
    "frequency_penalty": "LLM_FREQUENCY_PENALTY",
    "presence_penalty": "LLM_PRESENCE_PENALTY",
}


def exportar_config_llm() -> dict:
    return {chave: globals()[attr] for chave, attr in _LLM_CONFIG_ATTRS.items()}


def validar_config_llm(valores: dict) -> dict:
    if not isinstance(valores, dict):
        raise ValueError("Configuração da LLM inválida.")

    atual = exportar_config_llm()
    resultado = dict(atual)
    inteiros = {
        "n_ctx": (512, 131072),
        "max_tokens": (1, 8192),
        "n_gpu_layers": (-1, 999),
        "n_batch": (32, 8192),
        "n_ubatch": (32, 8192),
        "n_threads": (1, 256),
        "n_threads_batch": (1, 256),
        "voz_max_tokens": (1, 2048),
        "top_k": (0, 10000),
        "dry_allowed_length": (0, 1000),
    }
    decimais = {
        "voz_temperature": (0.0, 2.0),
        "temperature": (0.0, 2.0),
        "min_p": (0.0, 1.0),
        "top_p": (0.0, 1.0),
        "dry_multiplier": (0.0, 5.0),
        "repeat_penalty": (0.0, 5.0),
        "frequency_penalty": (-2.0, 2.0),
        "presence_penalty": (-2.0, 2.0),
    }
    for chave, (minimo, maximo) in inteiros.items():
        try:
            valor = int(valores.get(chave, atual[chave]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Valor inválido para {chave}.") from exc
        if not minimo <= valor <= maximo:
            raise ValueError(f"{chave} deve ficar entre {minimo} e {maximo}.")
        resultado[chave] = valor
    for chave, (minimo, maximo) in decimais.items():
        try:
            valor = float(valores.get(chave, atual[chave]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Valor inválido para {chave}.") from exc
        if not minimo <= valor <= maximo:
            raise ValueError(f"{chave} deve ficar entre {minimo} e {maximo}.")
        resultado[chave] = valor

    if resultado["n_ubatch"] > resultado["n_batch"]:
        raise ValueError("n_ubatch não pode ser maior que n_batch.")

    modelo = Path(str(valores.get("model_path", atual["model_path"]))).expanduser()
    if not modelo.is_absolute():
        modelo = BASE_DIR / modelo
    modelo = modelo.resolve()
    if modelo.suffix.lower() != ".gguf" or not modelo.is_file():
        raise ValueError(f"Modelo GGUF não encontrado: {modelo}")
    resultado["model_path"] = str(modelo)

    kv_type = str(valores.get("kv_type", atual["kv_type"])).strip().lower()
    kv_validos = {"", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
    if kv_type not in kv_validos:
        raise ValueError("Tipo de cache KV inválido.")
    resultado["kv_type"] = kv_type
    resultado["chat_template"] = str(valores.get("chat_template", atual["chat_template"])).strip()
    return resultado


def aplicar_config_llm(valores: dict) -> dict:
    normalizada = validar_config_llm(valores)
    for chave, attr in _LLM_CONFIG_ATTRS.items():
        globals()[attr] = normalizada[chave]
    return normalizada
