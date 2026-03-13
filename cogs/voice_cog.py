"""
cogs/voice_cog.py — Pipeline de voz via interface web

Fluxo:
  1. Usuário fala no microfone do navegador (web/index.html)
  2. Áudio (webm/opus) é enviado via POST /api/voz/transcrever
  3. Faster-Whisper GPU transcreve para texto
  4. Texto enviado via POST /api/voz/conversar → LLM gera resposta
  5. Kokoro ONNX GPU sintetiza áudio WAV (retornado ao navegador)
  6. Se "falar_discord=true" e bot estiver em canal de voz: reproduz TTS no Discord
"""

from __future__ import annotations

import asyncio
import collections
import io
import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands

log = logging.getLogger("voice_cog")

# ── Caminhos ──────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent.parent
_CAMINHO_CONFIG_VOZ = _BASE_DIR / "data" / "voz_config.json"
_GRAVACOES_DIR = _BASE_DIR / "gravacoes"

# ── Estado de gravação de call (guild_id → (outer_sink, inner_sink)) ────────
_gravacoes_ativas: dict[int, tuple] = {}

# ── Configuração padrão de voz ────────────────────────────────────────────────
_DEFAULT_VOZ: dict = {
    "falar_discord": True,      # reproduzir TTS no canal de voz do Discord
    "voz": "af_heart",          # voz Kokoro feminina (padrão: a mais natural)
    "velocidade": 1.0,
    "volume": 1.0,
    "pitch": 0,                 # pitch em semitons (-12 a +12, 0 = sem alteração)
    "whisper_modelo": "medium",
}

# Estado global — lido/escrito pelo web_server e pela UI
voz_estado: dict = {**_DEFAULT_VOZ}

# Vozes disponíveis (as mais naturais com lang="pt-br")
# pf_dora: portuguesa nativa — sem sotaque (recomendada para PT-BR)
# af_heart / af_jessica: vozes americanas — sotaque leve em português
VOZES_PTBR: list[dict] = [
    {"id": "pf_dora",    "label": "Dora (sem sotaque - nativa PT)"},
    {"id": "af_heart",   "label": "Heart (voz suave)"},
    {"id": "af_jessica", "label": "Jessica (voz clara)"},
]


def _carregar_config_voz() -> None:
    global voz_estado
    if _CAMINHO_CONFIG_VOZ.exists():
        try:
            dados = json.loads(_CAMINHO_CONFIG_VOZ.read_text("utf-8"))
            voz_estado = {**_DEFAULT_VOZ, **dados}
            return
        except Exception as exc:
            log.warning("Falha ao ler config de voz: %s", exc)
    voz_estado = {**_DEFAULT_VOZ}


def salvar_config_voz() -> None:
    _CAMINHO_CONFIG_VOZ.parent.mkdir(parents=True, exist_ok=True)
    _CAMINHO_CONFIG_VOZ.write_text(
        json.dumps(voz_estado, ensure_ascii=False, indent=2), "utf-8"
    )


# ── Download de modelos Kokoro (GitHub Releases, sem autenticação) ───────────
_KOKORO_DIR = _BASE_DIR / "models" / "kokoro"
_KOKORO_BASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
)


def _baixar_kokoro_se_necessario(filename: str) -> Path:
    """
    Garante que o arquivo de modelo Kokoro existe em models/kokoro/.
    Se não existir, baixa do GitHub Releases (sem autenticação).
    """
    import urllib.request

    _KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    destino = _KOKORO_DIR / filename
    if destino.exists():
        log.info("[VOZ] Modelo Kokoro encontrado: %s", destino)
        return destino

    url = _KOKORO_BASE_URL + filename
    log.info("[VOZ] Baixando %s ...", url)

    tmp = destino.with_suffix(".tmp")
    try:
        def _progresso(bloco, tamanho_bloco, tamanho_total):
            if tamanho_total > 0:
                pct = bloco * tamanho_bloco * 100 // tamanho_total
                if pct % 10 == 0:
                    log.info("[VOZ] %s — %d%%", filename, min(pct, 100))

        urllib.request.urlretrieve(url, tmp, reporthook=_progresso)
        tmp.rename(destino)
        log.info("[VOZ] Download concluído: %s", destino)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    return destino


# ── Lazy-loading de modelos (thread-safe) ─────────────────────────────────────
_whisper_model = None
_kokoro_model = None
_models_lock = threading.Lock()


def _carregar_modelos_sync() -> None:
    """Carrega Faster-Whisper e Kokoro ONNX na GPU. Bloqueante, thread-safe."""
    global _whisper_model, _kokoro_model

    with _models_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            modelo = voz_estado.get("whisper_modelo", "medium")
            log.info("[VOZ] Carregando Faster-Whisper '%s' na GPU...", modelo)
            _whisper_model = WhisperModel(
                modelo,
                device="cuda",
                compute_type="float16",
            )
            log.info("[VOZ] Faster-Whisper pronto.")

        if _kokoro_model is None:
            # Configura espeakng_loader ANTES do phonemizer para fonemas PT-BR corretos
            import misaki.espeak  # noqa: F401 — efeito colateral: seta EspeakWrapper
            from kokoro_onnx import Kokoro

            model_path  = str(_baixar_kokoro_se_necessario("kokoro-v1.0.fp16.onnx"))
            voices_path = str(_baixar_kokoro_se_necessario("voices-v1.0.bin"))

            # Seleciona o melhor provider ONNX disponível no sistema
            import onnxruntime as ort
            _providers_disponiveis = ort.get_available_providers()
            log.info("[VOZ] ONNX providers disponíveis: %s", _providers_disponiveis)

            _provider_escolhido = "CPUExecutionProvider"
            if "CUDAExecutionProvider" in _providers_disponiveis:
                _provider_escolhido = "CUDAExecutionProvider"
            elif "DmlExecutionProvider" in _providers_disponiveis:
                _provider_escolhido = "DmlExecutionProvider"
            log.info("[VOZ] Provider escolhido para Kokoro: %s", _provider_escolhido)
            os.environ["ONNX_PROVIDER"] = _provider_escolhido

            try:
                log.info("[VOZ] Iniciando Kokoro ONNX (%s)...", _provider_escolhido)
                _kokoro_model = Kokoro(model_path, voices_path)
                log.info("[VOZ] Kokoro ONNX pronto. Vozes: %s", _kokoro_model.get_voices()[:12])
            except Exception as _exc_gpu:
                log.warning(
                    "[VOZ] Falha ao carregar Kokoro com %s (%s) — recaindo para CPU.",
                    _provider_escolhido, _exc_gpu,
                )
                os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
                _kokoro_model = Kokoro(model_path, voices_path)
                log.info("[VOZ] Kokoro ONNX (CPU) pronto. Vozes: %s", _kokoro_model.get_voices()[:12])


# ── Sink de gravação (voice_recv) ──────────────────────────────────────────────
try:
    from discord.ext import voice_recv as _voice_recv
    _VOICE_RECV_OK = True
except ImportError:
    _VOICE_RECV_OK = False
    _voice_recv = None  # type: ignore


if _VOICE_RECV_OK:
    # ── Patch: PacketDecoder._decode_packet resiliente a pacotes corrompidos ──
    # O router do voice_recv roda em thread própria; qualquer exceção em
    # _decode_packet mata o loop inteiro.
    #
    # IMPORTANTE: NÃO fazemos patch em Decoder.decode porque quando o codec
    # Opus C falha, seu estado interno fica corrompido — engolir a exceção e
    # retornar silêncio faz com que TODOS os frames seguintes saiam como lixo
    # (glitch distorcido).  Em vez disso, patchamos _decode_packet e, ao detectar
    # erro, RECRIAMOS o Decoder (opus_decoder_create), garantindo estado limpo.
    from discord.ext.voice_recv.opus import PacketDecoder as _PacketDecoder

    _SILENCE_FRAME = b"\x00" * 3840  # 960 samples × 2 ch × 2 bytes (s16le)
    _original_decode_packet = _PacketDecoder._decode_packet

    def _safe_decode_packet(self, packet):  # type: ignore[override]
        try:
            return _original_decode_packet(self, packet)
        except Exception as _exc:
            log.debug(
                "[VOZ REC] Opus decode error ssrc=%s: %s — recriando decoder",
                self.ssrc, _exc,
            )
            # Recria o decoder com estado totalmente limpo
            self._decoder = discord.opus.Decoder()
            return packet, _SILENCE_FRAME

    _PacketDecoder._decode_packet = _safe_decode_packet
    log.info("[VOZ] Patch de segurança aplicado em PacketDecoder._decode_packet")

    # Tamanho esperado de um frame PCM (48 kHz, stereo, 16-bit, 20 ms)
    _PCM_FRAME_SIZE = 3840

    class _GravacaoSink(_voice_recv.AudioSink):  # type: ignore[misc]
        """
        Recebe PCM decodificado (wants_opus=False) via jitter buffer da
        biblioteca.  A decodificação, ordenação de pacotes, FEC e inserção
        de silêncio são todas tratadas pelo voice_recv + SilenceGeneratorSink.
        Nós apenas acumulamos os bytes PCM por usuário.
        """

        def __init__(self) -> None:
            super().__init__()
            self._buffers: dict[str, io.BytesIO] = {}
            self._lock = threading.Lock()
            self._frames = 0
            self._bad_size = 0
            self._errors = 0

        def wants_opus(self) -> bool:
            return False  # recebe PCM já decodificado pela biblioteca

        def write(self, user, data) -> None:  # type: ignore[override]
            try:
                uid = str(user.id) if user is not None else "unknown"
                pcm = data.pcm
                if not pcm:
                    return
                # Validação: descartar frames com tamanho errado (indicam corrupção)
                if len(pcm) != _PCM_FRAME_SIZE:
                    self._bad_size += 1
                    if self._bad_size <= 5:
                        log.warning(
                            "[VOZ REC] Frame com tamanho inválido: %d bytes (esperado %d) uid=%s",
                            len(pcm), _PCM_FRAME_SIZE, uid,
                        )
                    return
                with self._lock:
                    if uid not in self._buffers:
                        self._buffers[uid] = io.BytesIO()
                        log.info("[VOZ REC] Novo stream para uid=%s", uid)
                    self._buffers[uid].write(pcm)
                    self._frames += 1
            except Exception as exc:
                self._errors += 1
                if self._errors <= 3:
                    log.warning("[VOZ REC] Erro em write(): %s", exc)

        def cleanup(self) -> None:
            total_bytes = sum(len(b.getvalue()) for b in self._buffers.values())
            dur = total_bytes / (48000 * 2 * 2) if total_bytes else 0.0
            log.info(
                "[VOZ REC] Sink cleanup — frames: %d, bad_size: %d, errors: %d, "
                "users: %d, total_bytes: %d, duração_estimada: %.1fs",
                self._frames, self._bad_size, self._errors,
                len(self._buffers), total_bytes, dur,
            )
else:
    class _GravacaoSink:  # type: ignore[no-redef]
        _buffers: dict = {}
        _lock = threading.Lock()
        _frames = 0


# ── API pública — chamada pelo web_server em threads HTTP ─────────────────────

def transcrever_audio_sync(audio_bytes: bytes, content_type: str = "audio/webm") -> str:
    """
    Transcreve áudio bruto para texto usando Faster-Whisper na GPU.
    Aceita webm, ogg, mp4 ou wav. Usa ffmpeg para converter para WAV 16 kHz.
    Bloqueante — ThreadingHTTPServer já roda em thread separada.
    """
    _carregar_modelos_sync()

    ct = content_type.lower()
    if "webm" in ct:
        ext_in = ".webm"
    elif "ogg" in ct:
        ext_in = ".ogg"
    elif "mp4" in ct or "mpeg" in ct:
        ext_in = ".mp4"
    else:
        ext_in = ".wav"

    tmp_in = tempfile.mktemp(suffix=ext_in)
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        with open(tmp_in, "wb") as f:
            f.write(audio_bytes)

        # Converte para WAV 16 kHz mono via ffmpeg
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_in,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                tmp_wav,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning("[VOZ STT] ffmpeg retornou %d: %s",
                        result.returncode,
                        result.stderr.decode(errors="replace")[:400])
            return ""

        segs, _ = _whisper_model.transcribe(
            tmp_wav,
            language="pt",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        texto = " ".join(s.text for s in segs).strip()
        log.info("[VOZ STT] → %r", texto[:120])
        return texto

    finally:
        for p in (tmp_in, tmp_wav):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


def sintetizar_bytes_sync(texto: str) -> bytes:
    """
    Gera áudio WAV a partir de texto usando Kokoro ONNX na GPU.
    Retorna bytes WAV prontos para enviar ao navegador ou reproduzir no Discord.
    Bloqueante — rode em thread.
    """
    import numpy as np
    import soundfile as sf

    _carregar_modelos_sync()

    voz = voz_estado.get("voz", "pf_dora")
    velocidade = float(voz_estado.get("velocidade", 1.0))
    volume = float(voz_estado.get("volume", 1.0))

    samples, sample_rate = _kokoro_model.create(
        texto,
        voice=voz,
        speed=velocidade,
        lang="pt-br",
    )

    if volume != 1.0:
        samples = (samples * volume).clip(-1.0, 1.0)

    # Gera bytes WAV base
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    wav_bytes = buf.read()

    # Aplica pitch shift com FFmpeg se necessário (preserva duração via atempo)
    pitch = float(voz_estado.get("pitch", 0))
    if pitch != 0.0:
        factor  = 2 ** (pitch / 12)
        atempo  = max(0.5, min(2.0, 1.0 / factor))
        new_rate = int(sample_rate * factor)
        tmp_in  = tempfile.mktemp(suffix=".wav")
        tmp_out = tempfile.mktemp(suffix=".wav")
        try:
            with open(tmp_in, "wb") as f:
                f.write(wav_bytes)
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_in,
                    "-af", f"asetrate={new_rate},aresample={sample_rate},atempo={atempo:.6f}",
                    tmp_out,
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                with open(tmp_out, "rb") as f:
                    return f.read()
            log.warning("[VOZ] ffmpeg pitch returncode %d", result.returncode)
        finally:
            for p in (tmp_in, tmp_out):
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    return wav_bytes


# ── Cog de Voz ────────────────────────────────────────────────────────────────

class VoiceCog(commands.Cog, name="Voice"):
    """
    Gerencia TTS no canal de voz do Discord.
    A entrada de voz (STT) é feita pelo microfone da interface web.
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        _carregar_config_voz()
        # guild_id → asyncio.Lock — evita TTS simultâneo por servidor
        self._tts_locks: dict[int, asyncio.Lock] = {}

    async def cog_load(self) -> None:
        log.info("[VOZ] VoiceCog carregado.")

    async def cog_unload(self) -> None:
        for guild in self.bot.guilds:
            vc = guild.voice_client
            if vc and vc.is_playing():
                vc.stop()

    # ── TTS no Discord ────────────────────────────────────────────────────────

    async def falar_no_canal(self, guild_id: int, texto: str) -> None:
        """Reproduz TTS no canal de voz onde o bot está conectado."""
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return
        vc = guild.voice_client
        if not vc or not vc.is_connected():
            return

        lock = self._tts_locks.setdefault(guild_id, asyncio.Lock())
        async with lock:
            if vc.is_playing():
                vc.stop()
                await asyncio.sleep(0.15)
            try:
                await asyncio.to_thread(self._sintetizar_e_tocar, vc, texto)
            except Exception:
                log.exception("[VOZ] Falha no TTS/playback Discord.")

    def _sintetizar_e_tocar(self, vc: discord.VoiceClient, texto: str) -> None:
        """Gera WAV e reproduz no vc de forma síncrona. Roda em thread."""
        tmp_path: Optional[str] = None
        try:
            wav_bytes = sintetizar_bytes_sync(texto)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            done = threading.Event()

            def _after(exc: Optional[Exception]) -> None:
                if exc:
                    log.warning("[VOZ] Erro na reprodução: %s", exc)
                done.set()

            vc.play(discord.FFmpegPCMAudio(tmp_path), after=_after)
            done.wait(timeout=120)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ── Gravação de call ──────────────────────────────────────────────────────

    async def iniciar_gravacao_call(self, guild_id: int) -> None:
        """Inicia gravação de todos os participantes do canal de voz."""
        if not _VOICE_RECV_OK:
            raise ValueError(
                "discord-ext-voice_recv não está instalado. "
                "Execute: pip install discord-ext-voice_recv"
            )
        guild = self.bot.get_guild(guild_id)
        if not guild:
            raise ValueError(f"Guild {guild_id} não encontrada")
        vc = guild.voice_client
        if not vc or not vc.is_connected():
            raise ValueError("Bot não está em canal de voz")
        if guild_id in _gravacoes_ativas:
            raise ValueError("Já existe uma gravação em andamento")
        if not hasattr(vc, "listen"):
            raise ValueError(
                "Desconecte o bot e reconecte pela interface web para ativar a recepção de áudio."
            )

        inner_sink = _GravacaoSink()
        from discord.ext.voice_recv import SilenceGeneratorSink
        outer_sink = SilenceGeneratorSink(inner_sink)
        _gravacoes_ativas[guild_id] = (outer_sink, inner_sink)
        vc.listen(outer_sink)
        log.info("[VOZ] Gravação iniciada na guild %d (com SilenceGeneratorSink)", guild_id)

    async def parar_gravacao_call(self, guild_id: int) -> str:
        """Para a gravação, processa o áudio e retorna o caminho do arquivo."""
        guild = self.bot.get_guild(guild_id)
        if not guild:
            raise ValueError(f"Guild {guild_id} não encontrada")

        entry = _gravacoes_ativas.pop(guild_id, None)
        if entry is None:
            raise ValueError("Nenhuma gravação ativa para esta guild")

        _outer_sink, inner_sink = entry

        vc = guild.voice_client
        if vc and hasattr(vc, "stop_listening"):
            try:
                vc.stop_listening()
            except Exception as exc:
                log.warning("[VOZ] stop_listening: %s", exc)

        log.info("[VOZ] Gravação parada na guild %d — mixando...", guild_id)
        return await asyncio.to_thread(self._mixar_e_salvar, inner_sink, guild)

    def _mixar_e_salvar(self, sink: "_GravacaoSink", guild: discord.Guild) -> str:
        """Converte PCM de cada usuário e mixa em um único WAV."""
        import datetime
        import time
        import wave

        # Aguarda threads do SilenceGenerator encerrarem completamente
        time.sleep(0.5)

        _GRAVACOES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp     = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nome_seguro   = "".join(c if c.isalnum() or c in "-_" else "_" for c in guild.name)
        output_path   = _GRAVACOES_DIR / f"call_{nome_seguro}_{timestamp}.wav"

        # Discord entrega: 48000 Hz, 2 canais, 16-bit little-endian
        RATE, CHANNELS, SAMPWIDTH = 48000, 2, 2

        with sink._lock:
            log.info(
                "[VOZ REC] Estado do sink: frames=%d, bad_size=%d, errors=%d, "
                "users=%s, buf_sizes=%s",
                sink._frames, sink._bad_size, sink._errors,
                list(sink._buffers.keys()),
                {uid: len(buf.getvalue()) for uid, buf in sink._buffers.items()},
            )
            buffers = {
                uid: buf.getvalue()
                for uid, buf in sink._buffers.items()
                if len(buf.getvalue()) > 0
            }

        if not buffers:
            log.warning("[VOZ] Nenhum áudio capturado na gravação")
            return ""

        # ── Salva WAV individual por usuário (via wave.open — sem ffmpeg) ─────
        wav_files: list[str] = []
        try:
            for uid, pcm_data in buffers.items():
                wav_path = _GRAVACOES_DIR / f"_track_{uid}_{timestamp}.wav"
                try:
                    with wave.open(str(wav_path), "wb") as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(SAMPWIDTH)
                        wf.setframerate(RATE)
                        wf.writeframes(pcm_data)
                    dur = len(pcm_data) / (RATE * CHANNELS * SAMPWIDTH)
                    log.info(
                        "[VOZ REC] Track WAV salva: %s (uid=%s, %d bytes, %.1fs)",
                        wav_path.name, uid, len(pcm_data), dur,
                    )
                    wav_files.append(str(wav_path))
                except Exception as exc:
                    log.error("[VOZ REC] Erro ao salvar track uid=%s: %s", uid, exc)

            if not wav_files:
                return ""

            # ── Mix final ─────────────────────────────────────────────────────
            if len(wav_files) == 1:
                # Um só usuário — copia direto
                cmd = ["ffmpeg", "-y", "-i", wav_files[0], str(output_path)]
            else:
                # Múltiplos — usa amix para mixar as tracks WAV
                inputs: list[str] = []
                for wf in wav_files:
                    inputs += ["-i", wf]
                cmd = [
                    "ffmpeg", "-y", *inputs,
                    "-filter_complex",
                    f"amix=inputs={len(wav_files)}:duration=longest:dropout_transition=0:normalize=0",
                    str(output_path),
                ]

            log.info("[VOZ REC] Executando: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                log.error(
                    "[VOZ] ffmpeg falhou ao mixar: %s",
                    result.stderr.decode(errors="replace")[:500],
                )
                # Fallback: usa primeira track diretamente
                import shutil
                shutil.copy2(wav_files[0], str(output_path))

            log.info("[VOZ] Gravação salva: %s (%d pista(s))", output_path, len(wav_files))
            return str(output_path)
        finally:
            # Limpa tracks individuais
            for wf in wav_files:
                try:
                    if os.path.exists(wf):
                        os.unlink(wf)
                except Exception:
                    pass


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(VoiceCog(bot))
