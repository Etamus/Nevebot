"""
NêveBot — Entry Point
Carrega configurações, registra os cogs e inicia o bot.
"""

import os
import sys
import socket
import asyncio
import logging
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

import discord
from discord.ext import commands

# Importa config (valida token e localiza o modelo ao iniciar)
import config
from config_loader import cfg as _bot_cfg
import web_server

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
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
    log.info("Modelo carregado: %s", config.LLM_MODEL_PATH)
    web_server.start(bot, loop=asyncio.get_event_loop())
    log.info("Interface web iniciada em http://127.0.0.1:5000")


async def main() -> None:
    async with bot:
        for cog in COGS:
            await bot.load_extension(cog)
            log.info("Cog carregado: %s", cog)

        await bot.start(config.DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
