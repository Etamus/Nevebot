"""
Nevebot — Entry Point
Carrega configuracoes, registra os cogs e inicia o bot.
"""

import os
import sys
import socket
import asyncio
import logging
import threading
import time
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

# ── Usa apenas os runtimes CUDA locais das dependências
# O CUDA Toolkit global não é necessário e pode expor DLLs incompatíveis.
# Os cookies devem ser mantidos em variável global para não serem coletados pelo GC.
_base_dir = Path(__file__).parent
_site_packages = _base_dir / "venv" / "Lib" / "site-packages"
_dll_cookies = []

os.environ.pop("CUDA_PATH", None)
os.environ["PATH"] = os.pathsep.join(
    parte
    for parte in os.environ.get("PATH", "").split(os.pathsep)
    if "nvidia gpu computing toolkit" not in parte.lower()
)

for _dll_dir in [
    *_site_packages.glob("nvidia/*/bin"),
    _site_packages / "ctranslate2",
]:
    if _dll_dir.is_dir():
        _dll_cookies.append(os.add_dll_directory(str(_dll_dir)))
        os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")

import discord
from discord.ext import commands

# Importa config (valida token e localiza o modelo ao iniciar)
import config
from config_loader import cfg as _bot_cfg
import desktop_ui
import web_server

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
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
intents.voice_states = True

bot = commands.Bot(command_prefix=_bot_cfg.prefix(), intents=intents)

COGS = [
    "cogs.llm_cog",
    "cogs.voice_cog",
]

_INTERFACE_URL = "http://127.0.0.1:5000"
_interface_pronta = threading.Event()
_bot_encerrado = threading.Event()
_web_iniciado = False
_erro_bot: list[BaseException] = []


@bot.event
async def on_ready() -> None:
    global _web_iniciado
    log.info("Bot online como %s (ID: %s)", bot.user.name, bot.user.id)
    log.info("Modelo LLM configurado (desligado): %s", config.LLM_MODEL_PATH)
    log.info("Chatterbox PT-BR: %s", config.CHATTERBOX_PTBR_DIR)
    if not _web_iniciado:
        web_server.start(bot, loop=asyncio.get_running_loop())
        _web_iniciado = True
        _interface_pronta.set()
        log.info("Interface web iniciada em %s", _INTERFACE_URL)


@bot.event
async def on_guild_join(guild: discord.Guild) -> None:
    web_server.registrar_guild_adicionada(guild.id)
    log.info("Bot adicionado ao servidor %s (ID: %s).", guild.name, guild.id)


async def main() -> None:
    async with bot:
        for cog in COGS:
            await bot.load_extension(cog)
            log.info("Cog carregado: %s", cog)

        await bot.start(config.DISCORD_TOKEN)


def _executar_bot() -> None:
    try:
        asyncio.run(main())
    except BaseException as exc:
        _erro_bot.append(exc)
        log.exception("O bot foi encerrado por um erro.")
    finally:
        _bot_encerrado.set()


def _aguardar_interface(timeout: float = 600.0) -> None:
    limite = time.monotonic() + timeout
    while not _interface_pronta.wait(timeout=0.25):
        if _bot_encerrado.is_set():
            if _erro_bot:
                raise RuntimeError("O bot encerrou antes de iniciar a interface.") from _erro_bot[0]
            raise RuntimeError("O bot encerrou antes de iniciar a interface.")
        if time.monotonic() >= limite:
            raise TimeoutError("A interface nao ficou pronta dentro de 10 minutos.")


def iniciar_aplicacao() -> None:
    thread_bot = threading.Thread(target=_executar_bot, name="discord-bot", daemon=True)
    thread_bot.start()
    _aguardar_interface()

    abriu_nativo = desktop_ui.iniciar_interface(_INTERFACE_URL, _bot_encerrado)
    if abriu_nativo and not _bot_encerrado.is_set():
        web_server.solicitar_desligamento(atraso=0.1)

    while not _bot_encerrado.wait(timeout=0.5):
        pass


if __name__ == "__main__":
    iniciar_aplicacao()
