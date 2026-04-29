"""
Nevebot — Entry Point
Carrega configuracoes, registra os cogs e inicia o bot.
"""

import os
import sys
import socket
import asyncio
import logging
import webbrowser
from pathlib import Path

# ── Garante que é o venv DESTE projeto (porta única, impede instâncias duplicadas)
_LOCK_PORT = 47654
_lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    _lock_socket.bind(("127.0.0.1", _LOCK_PORT))
except OSError:
    print(f"[ERRO] Já existe uma instância do bot em execução (porta {_LOCK_PORT} ocupada).")
    print("       Feche o processo anterior antes de iniciar um novo.")
    os._exit(1)

# ── Registra DLLs do CUDA (necessário quando o CUDA Toolkit não está instalado)
# Os pacotes nvidia-cuda-runtime-cu12 / nvidia-cublas-cu12 instalam as DLLs em
# site-packages/nvidia/*/bin/ e site-packages/llama_cpp/lib/.
# Os cookies devem ser mantidos em variável global para não serem coletados pelo GC.
_site_packages = Path(__file__).parent / "venv" / "Lib" / "site-packages"
_dll_cookies = []

for _dll_dir in [
    *_site_packages.glob("nvidia/*/bin"),        # cudart, cublas, etc.
    _site_packages / "llama_cpp" / "lib",        # llama.dll e ggml*.dll
    _site_packages / "ctranslate2",              # cudnn64_9.dll (faster-whisper)
]:
    if _dll_dir.is_dir():
        _dll_cookies.append(os.add_dll_directory(str(_dll_dir)))
        # Também adiciona ao PATH — llama_cpp usa winmode=0 no LoadLibraryExW,
        # que ignora AddDllDirectory e só busca no PATH.
        os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")

import discord
from discord.ext import commands

# Importa config (valida token e localiza o modelo ao iniciar)
import config
from config_loader import cfg as _bot_cfg
import web_server

# ── Logging ───────────────────────────────────────────────────────────────────
_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                             datefmt="%H:%M:%S")

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_log_fmt)

_file_handler = logging.FileHandler("logs/nevebot_error.log", encoding="utf-8")
_file_handler.setLevel(logging.WARNING)
_file_handler.setFormatter(_log_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
log = logging.getLogger("nevebot")

# ── Bot ───────────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True  # necessário para ler o conteúdo das mensagens

bot = commands.Bot(command_prefix=_bot_cfg.prefix(), intents=intents)

COGS = [
    "cogs.llm_cog",
    "cogs.voice_cog",
]


@bot.event
async def on_ready() -> None:
    log.info("Bot online como %s (ID: %s)", bot.user.name, bot.user.id)
    log.info("Modelo LLM: %s", config.LLM_MODEL_PATH)
    log.info("OmniVoice: %s", config.OMNIVOICE_MODEL_PATH)
    web_server.start(bot, loop=asyncio.get_event_loop())
    log.info("Interface web iniciada em http://127.0.0.1:5000")
    webbrowser.open("http://127.0.0.1:5000")


async def main() -> None:
    async with bot:
        for cog in COGS:
            await bot.load_extension(cog)
            log.info("Cog carregado: %s", cog)

        await bot.start(config.DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
