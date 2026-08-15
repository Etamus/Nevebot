"""Recepcao e reproducao local do audio de canais de voz do Discord."""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import sounddevice as sd
from discord.ext import voice_recv
from discord.opus import Decoder

log = logging.getLogger("discord_audio_monitor")

_SAMPLE_RATE = 48_000
_CHANNELS = 2
_FRAME_SAMPLES = 960
_FRAME_VALUES = _FRAME_SAMPLES * _CHANNELS
_FRAME_BYTES = _FRAME_VALUES * 2
_MAX_QUEUE_FRAMES = 8


def listar_dispositivos_saida() -> tuple[list[dict[str, Any]], int | None]:
    """Lista endpoints de saida reais, sem duplicatas das APIs do Windows."""
    dispositivos = sd.query_devices()
    hostapis = sd.query_hostapis()
    try:
        padrao_global = int(sd.default.device[1])
    except (TypeError, ValueError, IndexError):
        padrao_global = -1

    api_preferida: int | None = None
    if sys.platform.startswith("win"):
        api_preferida = next(
            (
                indice
                for indice, hostapi in enumerate(hostapis)
                if "wasapi" in str(hostapi.get("name", "")).casefold()
            ),
            None,
        )

    def coletar(indices: list[int], padrao: int) -> list[dict[str, Any]]:
        saidas: list[dict[str, Any]] = []
        nomes: set[str] = set()
        for indice in indices:
            info = dispositivos[indice]
            if int(info.get("max_output_channels", 0)) < _CHANNELS:
                continue
            try:
                sd.check_output_settings(
                    device=indice,
                    channels=_CHANNELS,
                    dtype="int16",
                    samplerate=_SAMPLE_RATE,
                )
            except Exception:
                continue
            nome = str(info.get("name", f"Dispositivo {indice}")).strip()
            chave = " ".join(nome.casefold().split())
            if not nome or chave in nomes:
                continue
            nomes.add(chave)
            saidas.append({"id": indice, "nome": nome, "padrao": indice == padrao})
        return saidas

    if api_preferida is not None:
        padrao_api = int(hostapis[api_preferida].get("default_output_device", -1))
        indices_api = [
            indice
            for indice, info in enumerate(dispositivos)
            if int(info.get("hostapi", -1)) == api_preferida
        ]
        saidas = coletar(indices_api, padrao_api)
        if saidas:
            padrao = padrao_api
            if not any(item["id"] == padrao for item in saidas):
                padrao = saidas[0]["id"]
            return saidas, padrao

    indices = list(range(len(dispositivos)))
    if padrao_global in indices:
        indices.remove(padrao_global)
        indices.insert(0, padrao_global)
    saidas = coletar(indices, padrao_global)
    padrao = padrao_global
    if padrao < 0 or not any(item["id"] == padrao for item in saidas):
        padrao = saidas[0]["id"] if saidas else None
    return saidas, padrao


def _normalizar_frame(pcm: bytes) -> np.ndarray:
    valores = np.frombuffer(pcm, dtype="<i2")
    if valores.size == _FRAME_VALUES:
        return valores
    frame = np.zeros(_FRAME_VALUES, dtype=np.int16)
    limite = min(valores.size, _FRAME_VALUES)
    if limite:
        frame[:limite] = valores[:limite]
    return frame


def mixar_frames(frames: list[bytes], volume: float) -> bytes:
    """Combina frames PCM simultaneos sem deixar o sinal saturar facilmente."""
    if not frames:
        return bytes(_FRAME_BYTES)
    matrizes = [_normalizar_frame(frame).astype(np.float32) for frame in frames]
    mistura = np.sum(matrizes, axis=0)
    if len(matrizes) > 1:
        mistura /= math.sqrt(len(matrizes))
    mistura *= max(0.0, min(float(volume), 1.5))
    return np.clip(mistura, -32768, 32767).astype("<i2").tobytes()


class DiscordMonitorSink(voice_recv.AudioSink):
    """Sink Opus que aplica DAVE, decodifica e entrega PCM ao mixer local."""

    def __init__(self, monitor: "DiscordAudioMonitor", bot_user_id: int):
        super().__init__()
        self.monitor = monitor
        self.bot_user_id = int(bot_user_id)
        self._decoders: dict[int, Decoder] = {}
        self._ultimo_log_erro = 0.0

    def wants_opus(self) -> bool:
        # A extensao nao decodifica DAVE. Receber Opus permite descriptografar
        # antes de entregar o pacote ao decoder, como exige o protocolo.
        return True

    def write(self, user, data) -> None:
        if user is None or int(user.id) == self.bot_user_id or bool(getattr(user, "bot", False)):
            return

        user_id = int(user.id)
        try:
            opus = data.opus
            if opus:
                estado = self.voice_client._connection
                sessao_dave = getattr(estado, "dave_session", None)
                if sessao_dave is not None and bool(getattr(estado, "can_encrypt", False)):
                    import davey

                    opus = sessao_dave.decrypt(user_id, davey.MediaType.audio, bytes(opus))
                    if not opus:
                        return
                decoder = self._decoders.setdefault(user_id, Decoder())
                pcm = decoder.decode(bytes(opus), fec=False)
            else:
                decoder = self._decoders.get(user_id)
                if decoder is None:
                    return
                pcm = decoder.decode(None, fec=False)

            self.monitor.adicionar_frame(user_id, str(getattr(user, "display_name", user)), pcm)
        except Exception as exc:
            agora = time.monotonic()
            self.monitor.registrar_erro(f"Falha ao decodificar o audio recebido: {exc}")
            if agora - self._ultimo_log_erro >= 5.0:
                log.warning("Falha ao decodificar audio de %s: %s", user_id, exc, exc_info=True)
                self._ultimo_log_erro = agora

    @voice_recv.AudioSink.listener()
    def on_voice_member_disconnect(self, member, _ssrc) -> None:
        if member is not None:
            self._decoders.pop(int(member.id), None)

    def cleanup(self) -> None:
        self._decoders.clear()
        self.monitor.sink_encerrado(self)


class DiscordAudioMonitor:
    """Mantem uma unica sessao de escuta e uma fila curta por participante."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.RLock())
        self._filas: dict[int, deque[bytes]] = {}
        self._nomes: dict[int, str] = {}
        self._atividade: dict[int, float] = {}
        self._ativo = False
        self._volume = 1.0
        self._dispositivo: int | None = None
        self._nome_dispositivo: str | None = None
        self._voice_client = None
        self._sink: DiscordMonitorSink | None = None
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._ultimo_erro: str | None = None

    def _abrir_stream(self, dispositivo: int | None):
        return sd.RawOutputStream(
            samplerate=_SAMPLE_RATE,
            blocksize=_FRAME_SAMPLES,
            device=dispositivo,
            channels=_CHANNELS,
            dtype="int16",
            latency="low",
        )

    def iniciar(self, voice_client, *, dispositivo: int | None, volume: float) -> dict[str, Any]:
        volume = max(0.0, min(float(volume), 1.5))
        dispositivos, padrao = listar_dispositivos_saida()
        ids = {item["id"] for item in dispositivos}
        if dispositivo is None:
            dispositivo = padrao
        if dispositivo not in ids:
            raise ValueError("Dispositivo de saída indisponível.")

        with self._cond:
            mesma_sessao = (
                self._ativo
                and self._voice_client is voice_client
                and self._dispositivo == dispositivo
            )
            if mesma_sessao:
                self._volume = volume
                return self.estado()

        self.parar()
        if voice_client.is_listening():
            raise RuntimeError("A conexão de voz já possui outro receptor de áudio.")

        nome_dispositivo = next(item["nome"] for item in dispositivos if item["id"] == dispositivo)
        stream = self._abrir_stream(dispositivo)
        sink = DiscordMonitorSink(self, int(voice_client.user.id))
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._reproduzir,
            args=(stream, stop_event),
            name="discord-audio-monitor",
            daemon=True,
        )

        try:
            stream.start()
            with self._cond:
                self._ativo = True
                self._volume = volume
                self._dispositivo = dispositivo
                self._nome_dispositivo = nome_dispositivo
                self._voice_client = voice_client
                self._sink = sink
                self._stream = stream
                self._thread = thread
                self._stop_event = stop_event
                self._ultimo_erro = None
                self._filas.clear()
                self._nomes.clear()
                self._atividade.clear()
            thread.start()
            voice_client.listen(sink, after=self._depois_de_escutar)
        except Exception:
            self._encerrar_saida(stream, thread, stop_event)
            with self._cond:
                self._limpar_estado_locked()
            raise

        canal = getattr(voice_client, "channel", None)
        log.info(
            "Monitor de voz iniciado: guild=%s canal=%s dispositivo=%s",
            getattr(getattr(voice_client, "guild", None), "name", "?"),
            getattr(canal, "name", "?"),
            nome_dispositivo,
        )
        return self.estado()

    def parar(self) -> dict[str, Any]:
        with self._cond:
            voice_client = self._voice_client
            sink = self._sink
            stream = self._stream
            thread = self._thread
            stop_event = self._stop_event
            estava_ativo = self._ativo
            self._limpar_estado_locked()
            if stop_event is not None:
                stop_event.set()
            self._cond.notify_all()

        if voice_client is not None:
            try:
                if voice_client.is_listening() and getattr(voice_client, "sink", None) is sink:
                    voice_client.stop_listening()
            except Exception as exc:
                log.warning("Falha ao interromper recepcao de voz: %s", exc)
        self._encerrar_saida(stream, thread, stop_event)
        if estava_ativo:
            log.info("Monitor de voz encerrado.")
        return self.estado()

    def parar_se_guild(self, guild_id: int) -> None:
        with self._cond:
            guild = getattr(self._voice_client, "guild", None)
            corresponde = guild is not None and int(guild.id) == int(guild_id)
        if corresponde:
            self.parar()

    def ajustar_volume(self, volume: float) -> None:
        with self._cond:
            self._volume = max(0.0, min(float(volume), 1.5))

    def adicionar_frame(self, user_id: int, nome: str, pcm: bytes) -> None:
        if not pcm:
            return
        with self._cond:
            if not self._ativo:
                return
            fila = self._filas.setdefault(user_id, deque(maxlen=_MAX_QUEUE_FRAMES))
            fila.append(bytes(pcm))
            self._nomes[user_id] = nome
            self._atividade[user_id] = time.monotonic()
            self._ultimo_erro = None
            self._cond.notify()

    def registrar_erro(self, mensagem: str) -> None:
        with self._cond:
            self._ultimo_erro = mensagem

    def sink_encerrado(self, sink: DiscordMonitorSink) -> None:
        with self._cond:
            if sink is not self._sink:
                return
            stream = self._stream
            thread = self._thread
            stop_event = self._stop_event
            self._limpar_estado_locked()
            if stop_event is not None:
                stop_event.set()
            self._cond.notify_all()
        self._encerrar_saida(stream, thread, stop_event)

    def _depois_de_escutar(self, erro: Exception | None) -> None:
        if erro is not None:
            self.registrar_erro(str(erro))
            log.error("Receptor de voz encerrado com erro: %s", erro)

    def _reproduzir(self, stream, stop_event: threading.Event) -> None:
        silencio = bytes(_FRAME_BYTES)
        while not stop_event.is_set():
            with self._cond:
                frames: list[bytes] = []
                agora = time.monotonic()
                for user_id, fila in list(self._filas.items()):
                    if fila:
                        frames.append(fila.popleft())
                    elif agora - self._atividade.get(user_id, 0.0) > 2.0:
                        self._filas.pop(user_id, None)
                        self._nomes.pop(user_id, None)
                        self._atividade.pop(user_id, None)
                volume = self._volume
            frame = mixar_frames(frames, volume) if frames else silencio
            try:
                stream.write(frame)
            except Exception as exc:
                if not stop_event.is_set():
                    self.registrar_erro(f"Falha no dispositivo de saída: {exc}")
                    log.exception("Falha ao reproduzir audio recebido do Discord.")
                break

    def _encerrar_saida(self, stream, thread, stop_event) -> None:
        if stop_event is not None:
            stop_event.set()
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=1.5)
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def _limpar_estado_locked(self) -> None:
        self._ativo = False
        self._voice_client = None
        self._sink = None
        self._stream = None
        self._thread = None
        self._stop_event = None
        self._filas.clear()
        self._nomes.clear()
        self._atividade.clear()

    def estado(self) -> dict[str, Any]:
        with self._cond:
            voice_client = self._voice_client
            guild = getattr(voice_client, "guild", None)
            canal = getattr(voice_client, "channel", None)
            agora = time.monotonic()
            falantes = [
                self._nomes[user_id]
                for user_id, instante in self._atividade.items()
                if agora - instante <= 0.45 and user_id in self._nomes
            ]
            return {
                "ativo": self._ativo,
                "volume": round(self._volume, 2),
                "dispositivo": self._dispositivo,
                "dispositivo_nome": self._nome_dispositivo,
                "guild_id": str(guild.id) if guild is not None else None,
                "guild_nome": getattr(guild, "name", None),
                "canal_id": str(canal.id) if canal is not None else None,
                "canal_nome": getattr(canal, "name", None),
                "falantes": falantes,
                "erro": self._ultimo_erro,
            }


_monitor = DiscordAudioMonitor()


def obter_monitor() -> DiscordAudioMonitor:
    return _monitor
