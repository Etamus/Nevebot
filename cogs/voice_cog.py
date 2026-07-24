"""cogs/voice_cog.py - Gerenciamento de voz: conexão, reprodução PCM (sem FFmpeg)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path

import discord
from discord.ext import commands

log = logging.getLogger("voice_cog")

# Caminhos
_BASE_DIR = Path(__file__).parent.parent
_CAMINHO_CONFIG_VOZ = _BASE_DIR / "data" / "voz_config.json"

# Config padrão de voz
_VOZ_DEFAULT: dict = {
    "falar_discord": True,
    "tts_model": "chatterbox-ptbr-v3",
    "voz_language": "pt-BR",
    "voz_referencia": "data/voz_referencia.wav",
    "voz_exaggeration": 0.5,
    "voz_cfg_weight": 0.5,
    "voz_temperature": 0.8,
    "velocidade": 1.0,
    "volume": 1.0,
    "whisper_modelo": "small",
    "ativa": False,
    "silencio_s": 1.5,
    "voz_seed": 42,
    "pitch": 0.0,
}

_VOZ_OLD_KEYS = {"voz_age", "voz_pitch_style", "voz_instruct"}

# Estado global de voz
voz_estado: dict = {}
_playback_locks: dict[int, asyncio.Lock] = {}


def _carregar_config_voz() -> None:
    global voz_estado
    if _CAMINHO_CONFIG_VOZ.exists():
        try:
            loaded = json.loads(_CAMINHO_CONFIG_VOZ.read_text("utf-8"))
            voz_estado = {**_VOZ_DEFAULT, **loaded}
            for key in _VOZ_OLD_KEYS:
                voz_estado.pop(key, None)
            voz_estado["tts_model"] = "chatterbox-ptbr-v3"
            voz_estado["voz_language"] = "pt-BR"
            return
        except Exception as exc:
            log.warning("Falha ao ler config de voz: %s", exc)
    voz_estado = dict(_VOZ_DEFAULT)


def salvar_config_voz() -> None:
    _CAMINHO_CONFIG_VOZ.parent.mkdir(parents=True, exist_ok=True)
    _CAMINHO_CONFIG_VOZ.write_text(
        json.dumps(voz_estado, ensure_ascii=False, indent=2), "utf-8"
    )


async def reproduzir_pcm(bot: commands.Bot, guild_id: int, pcm_bytes: bytes) -> None:
    """Reproduz áudio PCM bruto (48 kHz, estéreo, 16-bit) no canal de voz.

    Usa discord.PCMAudio — não depende de FFmpeg.
    Espera até a reprodução terminar.
    """
    log.info("[VOZ] reproduzir_pcm chamado — guild_id=%s pcm_bytes=%d", guild_id, len(pcm_bytes))

    lock = _playback_locks.get(guild_id)
    if lock is None:
        lock = asyncio.Lock()
        _playback_locks[guild_id] = lock

    async with lock:
        guild = bot.get_guild(guild_id)
        if guild is None:
            log.error("[VOZ] Guild %s não encontrada!", guild_id)
            raise ValueError("Guild não encontrada")

        vc = guild.voice_client
        if vc is None or not vc.is_connected():
            log.error("[VOZ] Bot não está conectado a voz na guild %s (vc=%s)", guild_id, vc)
            raise ValueError("Bot não está conectado a um canal de voz nesta guild")

        log.info("[VOZ] Conectado ao canal: %s (guild: %s)", vc.channel.name, guild.name)

        # Aguarda se já está tocando algo
        if vc.is_playing():
            log.info("[VOZ] Aguardando reprodução anterior terminar...")
        while vc.is_playing():
            await asyncio.sleep(0.1)

        loop = asyncio.get_event_loop()
        done = loop.create_future()

        def _after(error):
            if error:
                log.error("[VOZ] Erro durante reprodução: %s", error)
                loop.call_soon_threadsafe(done.set_exception, error)
            else:
                log.info("[VOZ] Reprodução concluída com sucesso.")
                loop.call_soon_threadsafe(done.set_result, None)

        log.info("[VOZ] Iniciando vc.play() com %d bytes de PCM...", len(pcm_bytes))
        source = discord.PCMAudio(io.BytesIO(pcm_bytes))
        vc.play(source, after=_after)
        await done


class VoiceCog(commands.Cog, name="Voice"):
    """Gerencia a presença do bot em canais de voz do Discord."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        _carregar_config_voz()

    async def cog_load(self) -> None:
        log.info("[VOZ] VoiceCog carregado.")
        asyncio.create_task(self._preaquecer_pipeline_voz(), name="voice-pipeline-warmup")

    async def _preaquecer_pipeline_voz(self) -> None:
        await self._preaquecer_tts()
        await self._preaquecer_stt()

    async def _preaquecer_tts(self) -> None:
        """Carrega o Chatterbox e faz uma geração curta para aquecer CUDA/cache."""
        try:
            from services import tts_chatterbox
            await asyncio.to_thread(tts_chatterbox.precarregar_e_aquecer, voz_estado)
        except Exception as exc:
            log.warning("[VOZ] Falha ao pré-aquecer Chatterbox: %s", exc)

    async def _preaquecer_stt(self) -> None:
        """Carrega e aquece o faster-whisper com o modelo configurado."""
        try:
            from services import stt_whisper
            modelo = voz_estado.get("whisper_modelo", "small")
            await asyncio.to_thread(stt_whisper.precarregar_e_aquecer, modelo)
        except Exception as exc:
            log.warning("[VOZ] Falha ao pré-aquecer Whisper: %s", exc)

    async def cog_unload(self) -> None:
        for guild in self.bot.guilds:
            vc = guild.voice_client
            if vc and vc.is_connected():
                await vc.disconnect(force=True)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(VoiceCog(bot))
