"""Transcricao isolada de um canal de voz do Discord para SRT."""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from discord.ext import voice_recv
from discord.opus import Decoder

from services.discord_voice_receive import descartar_pacotes_pendentes


log = logging.getLogger("discord_transcription")

_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_DIR = _BASE_DIR / "transcricoes"
_DISCORD_SAMPLE_RATE = 48_000
_DISCORD_CHANNELS = 2
_ANALYSIS_FRAME_MS = 20
_PRE_ROLL_FRAMES = 12
_SILENCE_SECONDS = 0.72
_STALE_SECONDS = 0.85
_MAX_UTTERANCE_SECONDS = 24.0
_MIN_VOICED_SECONDS = 0.16
_MAX_QUEUE = 64


@dataclass
class _SpeakerState:
    key: str
    name: str
    source: str
    sample_rate: int
    noise_floor: float = 0.0025
    pre_roll: deque[tuple[float, np.ndarray]] = field(
        default_factory=lambda: deque(maxlen=_PRE_ROLL_FRAMES)
    )
    utterance: list[np.ndarray] = field(default_factory=list)
    utterance_start: float | None = None
    utterance_end: float = 0.0
    voiced_seconds: float = 0.0
    silence_seconds: float = 0.0
    last_packet_monotonic: float = 0.0


@dataclass(frozen=True)
class _TranscriptionJob:
    session_id: str
    sequence: int
    speaker: str
    source: str
    start: float
    end: float
    sample_rate: int
    samples: np.ndarray


@dataclass(frozen=True)
class _Cue:
    sequence: int
    start: float
    end: float
    speaker: str
    text: str


def _safe_name(value: str, fallback: str) -> str:
    text = " ".join(str(value or "").replace("[", "").replace("]", "").split())
    return text[:80] or fallback


def _filename_part(value: str, fallback: str) -> str:
    text = re.sub(r"[^\w.-]+", "-", str(value or "").strip(), flags=re.UNICODE)
    return text.strip("-._")[:60] or fallback


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class DiscordTranscriptionSink(voice_recv.AudioSink):
    """Recebe Opus do Discord, aplica DAVE e mantem um decoder por usuario."""

    def __init__(self, service: "DiscordTranscriptionService", bot_user_id: int):
        super().__init__()
        self.service = service
        self.bot_user_id = int(bot_user_id)
        self._decoders: dict[int, Decoder] = {}
        self._dave_failures: dict[int, tuple[float, int]] = {}
        self._last_error_log = 0.0

    def wants_opus(self) -> bool:
        return True

    def write(self, user, data) -> None:
        if user is None or int(user.id) == self.bot_user_id or bool(getattr(user, "bot", False)):
            return
        if not self.service.accepting_audio:
            return

        user_id = int(user.id)
        try:
            opus = data.opus
            if opus:
                connection = self.voice_client._connection
                dave_session = getattr(connection, "dave_session", None)
                if dave_session is not None and bool(getattr(connection, "can_encrypt", False)):
                    import davey

                    try:
                        opus = dave_session.decrypt(user_id, davey.MediaType.audio, bytes(opus))
                    except Exception as exc:
                        mensagem = str(exc).casefold()
                        if "decrypt" not in mensagem and "decryption" not in mensagem:
                            raise

                        now = time.monotonic()
                        first, count = self._dave_failures.get(user_id, (now, 0))
                        count += 1
                        self._dave_failures[user_id] = (first, count)
                        if count == 1:
                            log.debug(
                                "Descartando pacote DAVE durante sincronizacao inicial de %s: %s",
                                user_id,
                                exc,
                            )
                        if now - first >= 3.0 and now - self._last_error_log >= 5.0:
                            erro = f"Falha persistente ao descriptografar audio de {user_id}: {exc}"
                            self.service.register_error(erro)
                            log.warning(erro)
                            self._last_error_log = now
                        return
                    self._dave_failures.pop(user_id, None)
                    if not opus:
                        return
                decoder = self._decoders.setdefault(user_id, Decoder())
                pcm = decoder.decode(bytes(opus), fec=False)
            else:
                decoder = self._decoders.get(user_id)
                if decoder is None:
                    return
                pcm = decoder.decode(None, fec=False)

            self.service.add_discord_pcm(
                user_id,
                str(getattr(user, "display_name", user)),
                pcm,
            )
        except Exception as exc:
            self.service.register_error(f"Falha ao decodificar audio de {user_id}: {exc}")
            now = time.monotonic()
            if now - self._last_error_log >= 5.0:
                log.warning("Falha ao decodificar audio de %s: %s", user_id, exc, exc_info=True)
                self._last_error_log = now

    @voice_recv.AudioSink.listener()
    def on_voice_member_disconnect(self, member, _ssrc) -> None:
        if member is not None:
            user_id = int(member.id)
            self._decoders.pop(user_id, None)
            self._dave_failures.pop(user_id, None)

    def cleanup(self) -> None:
        self._decoders.clear()
        self._dave_failures.clear()
        self.service.sink_closed(self)


class DiscordTranscriptionService:
    """Coordena captura, VAD, uma fila Whisper e persistencia atomica do SRT."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active = False
        self._accept_audio = False
        self._finalizing = False
        self._session_id: str | None = None
        self._started_monotonic = 0.0
        self._duration_seconds = 0.0
        self._model = "large-v3-turbo"
        self._guild_id: str | None = None
        self._guild_name: str | None = None
        self._channel_id: str | None = None
        self._channel_name: str | None = None
        self._output_path: Path | None = None
        self._voice_client = None
        self._sink: DiscordTranscriptionSink | None = None
        self._speakers: dict[str, _SpeakerState] = {}
        self._participants: dict[str, str] = {}
        self._cues: list[_Cue] = []
        self._sequence = 0
        self._queue: queue.Queue[_TranscriptionJob | None] | None = None
        self._stop_event: threading.Event | None = None
        self._finished_event: threading.Event | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._active or self._finalizing

    @property
    def accepting_audio(self) -> bool:
        with self._lock:
            return self._active and self._accept_audio

    def start(
        self,
        voice_client,
        *,
        model: str,
        output_dir: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        if voice_client is None or not voice_client.is_connected():
            raise ValueError("Conecte a Neve a um canal de voz primeiro.")
        if voice_client.is_listening():
            raise RuntimeError("A conexao de voz ja possui outro receptor de audio.")

        descartar_pacotes_pendentes(voice_client)

        folder = Path(output_dir or _DEFAULT_OUTPUT_DIR).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        if not folder.is_dir():
            raise ValueError("A pasta de transcricao e invalida.")

        guild = getattr(voice_client, "guild", None)
        channel = getattr(voice_client, "channel", None)
        if guild is None or channel is None:
            raise RuntimeError("A conexao de voz nao possui servidor ou canal valido.")

        with self._lock:
            if self._active or self._finalizing:
                raise RuntimeError("Ja existe uma transcricao em andamento ou sendo finalizada.")

            session_id = uuid.uuid4().hex
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base_name = (
                f"{_filename_part(guild.name, 'servidor')}-"
                f"{_filename_part(channel.name, 'canal')}-{timestamp}"
            )
            output_path = folder / f"{base_name}.srt"
            suffix = 2
            while output_path.exists():
                output_path = folder / f"{base_name}-{suffix}.srt"
                suffix += 1

            job_queue: queue.Queue[_TranscriptionJob | None] = queue.Queue(maxsize=_MAX_QUEUE)
            stop_event = threading.Event()
            finished_event = threading.Event()
            sink = DiscordTranscriptionSink(self, int(voice_client.user.id))

            self._active = True
            self._accept_audio = False
            self._finalizing = False
            self._session_id = session_id
            self._started_monotonic = time.monotonic()
            self._duration_seconds = 0.0
            self._model = str(model or "large-v3-turbo")
            self._guild_id = str(guild.id)
            self._guild_name = str(guild.name)
            self._channel_id = str(channel.id)
            self._channel_name = str(channel.name)
            self._output_path = output_path
            self._voice_client = voice_client
            self._sink = sink
            self._speakers.clear()
            self._participants.clear()
            self._cues.clear()
            self._sequence = 0
            self._queue = job_queue
            self._stop_event = stop_event
            self._finished_event = finished_event
            self._last_error = None

            worker = threading.Thread(
                target=self._worker_loop,
                args=(session_id, job_queue, finished_event),
                name="discord-srt-whisper",
                daemon=True,
            )
            watchdog = threading.Thread(
                target=self._watchdog_loop,
                args=(session_id, stop_event),
                name="discord-srt-vad",
                daemon=True,
            )
            self._worker_thread = worker
            self._watchdog_thread = watchdog

        try:
            self._write_srt()
            worker.start()
            watchdog.start()
            voice_client.listen(sink, after=self._after_listening)
            with self._lock:
                if self._active and self._sink is sink and self._session_id == session_id:
                    self._started_monotonic = time.monotonic()
                    self._accept_audio = True
        except Exception:
            self.stop(wait=True, timeout=5, reason="falha ao iniciar receptor")
            raise

        log.info(
            "Transcricao SRT iniciada: guild=%s canal=%s arquivo=%s modelo=%s",
            guild.name,
            channel.name,
            output_path,
            self._model,
        )
        return self.state()

    def stop(self, *, wait: bool = True, timeout: float = 35.0, reason: str = "") -> dict[str, Any]:
        job_queue: queue.Queue[_TranscriptionJob | None] | None = None
        with self._lock:
            if not self._active:
                event = self._finished_event if self._finalizing else None
            else:
                event = self._finished_event
                self._active = False
                self._accept_audio = False
                self._finalizing = True
                self._duration_seconds = max(0.0, time.monotonic() - self._started_monotonic)
                voice_client = self._voice_client
                sink = self._sink
                stop_event = self._stop_event
                job_queue = self._queue
                jobs = self._flush_all_locked()
                self._speakers.clear()
                self._participants.clear()
                self._voice_client = None
                self._sink = None
                if stop_event is not None:
                    stop_event.set()

                for job in jobs:
                    self._enqueue_locked(job)

                if voice_client is not None:
                    try:
                        if voice_client.is_listening() and getattr(voice_client, "sink", None) is sink:
                            voice_client.stop_listening()
                    except Exception as exc:
                        log.warning("Falha ao interromper receptor da transcricao: %s", exc)
                if reason:
                    log.info("Finalizando transcricao SRT (%s).", reason)

        if job_queue is not None:
            if wait:
                job_queue.put(None)
            else:
                try:
                    job_queue.put_nowait(None)
                except queue.Full:
                    threading.Thread(
                        target=job_queue.put,
                        args=(None,),
                        name="discord-srt-sentinel",
                        daemon=True,
                    ).start()
        if wait and event is not None:
            event.wait(timeout=max(0.0, timeout))
        return self.state()

    def stop_for_guild(self, guild_id: int, *, wait: bool = False) -> None:
        with self._lock:
            matches = self._guild_id == str(guild_id) and (self._active or self._finalizing)
        if matches:
            self.stop(wait=wait, reason="conexao de voz encerrada")

    def add_discord_pcm(self, user_id: int, name: str, pcm: bytes) -> None:
        self._add_pcm(
            key=f"discord:{int(user_id)}",
            name=name,
            source="discord",
            pcm=pcm,
            sample_rate=_DISCORD_SAMPLE_RATE,
            channels=_DISCORD_CHANNELS,
            start_seconds=None,
        )

    def register_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message)

    def sink_closed(self, sink: DiscordTranscriptionSink) -> None:
        with self._lock:
            unexpected = self._active and sink is self._sink
        if unexpected:
            threading.Thread(
                target=self.stop,
                kwargs={"wait": False, "reason": "receptor encerrado"},
                name="discord-srt-stop",
                daemon=True,
            ).start()

    def state(self) -> dict[str, Any]:
        with self._lock:
            elapsed = (
                time.monotonic() - self._started_monotonic
                if self._active
                else self._duration_seconds
            )
            pending = self._queue.qsize() if self._queue is not None else 0
            if self._finalizing and pending:
                pending = max(0, pending - 1)
            return {
                "ok": True,
                "ativo": self._active,
                "finalizando": self._finalizing,
                "session_id": self._session_id,
                "elapsed_s": round(max(0.0, elapsed), 1),
                "elapsed_ms": int(round(max(0.0, elapsed) * 1000)),
                "modelo": self._model,
                "guild_id": self._guild_id,
                "guild_nome": self._guild_name,
                "canal_id": self._channel_id,
                "canal_nome": self._channel_name,
                "arquivo": str(self._output_path) if self._output_path else None,
                "pasta": str(self._output_path.parent) if self._output_path else str(_DEFAULT_OUTPUT_DIR),
                "legendas": len(self._cues),
                "pendentes": pending,
                "participantes": (
                    sorted(set(self._participants.values())) if self._active else []
                ),
                "erro": self._last_error,
            }

    def open_output_folder(self) -> None:
        with self._lock:
            folder = self._output_path.parent if self._output_path else _DEFAULT_OUTPUT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(folder))  # type: ignore[attr-defined]
            return
        import subprocess

        subprocess.Popen(["xdg-open", str(folder)])

    def _add_pcm(
        self,
        *,
        key: str,
        name: str,
        source: str,
        pcm: bytes,
        sample_rate: int,
        channels: int,
        start_seconds: float | None,
    ) -> None:
        if not pcm:
            return
        values = np.frombuffer(pcm, dtype="<i2")
        if channels > 1:
            usable = values.size - (values.size % channels)
            if usable <= 0:
                return
            channel_values = values[:usable].reshape(-1, channels).astype(np.float32)
            rms_channels = np.sqrt(np.mean(channel_values * channel_values, axis=0))
            strongest = int(np.argmax(rms_channels))
            weakest = float(np.min(rms_channels))
            strongest_rms = float(np.max(rms_channels))
            if strongest_rms > 0 and strongest_rms / max(weakest, 1.0) > 1.35:
                mono = channel_values[:, strongest] / 32768.0
            else:
                mono = channel_values.mean(axis=1) / 32768.0
        else:
            mono = values.astype(np.float32) / 32768.0

        frame_size = max(1, int(sample_rate * _ANALYSIS_FRAME_MS / 1000))
        total_seconds = len(mono) / sample_rate
        with self._lock:
            if not self._active or not self._accept_audio:
                return
            now_elapsed = max(0.0, time.monotonic() - self._started_monotonic)
            block_start = (
                max(0.0, now_elapsed - total_seconds)
                if start_seconds is None
                else max(0.0, float(start_seconds))
            )
            state = self._speakers.get(key)
            if state is None:
                state = _SpeakerState(
                    key=key,
                    name=_safe_name(name, "Participante"),
                    source=source,
                    sample_rate=sample_rate,
                )
                self._speakers[key] = state
            else:
                state.name = _safe_name(name, state.name)
            self._participants[key] = state.name

            jobs: list[_TranscriptionJob] = []
            for offset in range(0, len(mono), frame_size):
                frame = mono[offset:offset + frame_size]
                if frame.size < max(1, frame_size // 2):
                    continue
                frame_start = block_start + (offset / sample_rate)
                job = self._process_frame_locked(state, frame_start, frame)
                if job is not None:
                    jobs.append(job)
            for job in jobs:
                self._enqueue_locked(job)

    def _process_frame_locked(
        self,
        state: _SpeakerState,
        frame_start: float,
        frame: np.ndarray,
    ) -> _TranscriptionJob | None:
        duration = len(frame) / state.sample_rate
        frame_end = frame_start + duration
        rms = float(np.sqrt(np.mean(frame * frame)))
        peak = float(np.max(np.abs(frame)))
        threshold = max(0.004, min(0.025, state.noise_floor * 2.8))
        voiced = rms >= threshold and peak >= 0.014

        if not state.utterance:
            if voiced:
                if state.pre_roll:
                    state.utterance_start = state.pre_roll[0][0]
                    state.utterance.extend(item[1] for item in state.pre_roll)
                else:
                    state.utterance_start = frame_start
                state.pre_roll.clear()
                state.utterance.append(frame.copy())
                state.voiced_seconds = duration
                state.silence_seconds = 0.0
            else:
                state.noise_floor = (state.noise_floor * 0.96) + (min(rms, 0.02) * 0.04)
                state.pre_roll.append((frame_start, frame.copy()))
        else:
            state.utterance.append(frame.copy())
            if voiced:
                state.voiced_seconds += duration
                state.silence_seconds = 0.0
            else:
                state.silence_seconds += duration

        state.utterance_end = frame_end
        state.last_packet_monotonic = time.monotonic()
        utterance_duration = (
            state.utterance_end - state.utterance_start
            if state.utterance and state.utterance_start is not None
            else 0.0
        )
        if state.utterance and (
            state.silence_seconds >= _SILENCE_SECONDS
            or utterance_duration >= _MAX_UTTERANCE_SECONDS
        ):
            return self._flush_speaker_locked(state)
        return None

    def _flush_speaker_locked(self, state: _SpeakerState) -> _TranscriptionJob | None:
        if not state.utterance or state.utterance_start is None:
            state.pre_roll.clear()
            return None
        samples = np.concatenate(state.utterance).astype(np.float32, copy=False)
        start = state.utterance_start
        end = max(start + (len(samples) / state.sample_rate), state.utterance_end)
        voiced_seconds = state.voiced_seconds
        state.utterance.clear()
        state.utterance_start = None
        state.utterance_end = 0.0
        state.voiced_seconds = 0.0
        state.silence_seconds = 0.0
        state.pre_roll.clear()
        if voiced_seconds < _MIN_VOICED_SECONDS or len(samples) < int(state.sample_rate * 0.20):
            return None
        self._sequence += 1
        return _TranscriptionJob(
            session_id=str(self._session_id),
            sequence=self._sequence,
            speaker=state.name,
            source=state.source,
            start=max(0.0, start),
            end=max(start + 0.20, end),
            sample_rate=state.sample_rate,
            samples=samples,
        )

    def _flush_all_locked(self) -> list[_TranscriptionJob]:
        jobs: list[_TranscriptionJob] = []
        for state in self._speakers.values():
            job = self._flush_speaker_locked(state)
            if job is not None:
                jobs.append(job)
        return jobs

    def _enqueue_locked(self, job: _TranscriptionJob) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._last_error = "A fila do Whisper encheu; um trecho de audio foi descartado."
            log.error("Fila da transcricao SRT cheia; descartando trecho de %s.", job.speaker)

    def _watchdog_loop(self, session_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(0.10):
            with self._lock:
                if not self._active or session_id != self._session_id:
                    return
                now = time.monotonic()
                jobs: list[_TranscriptionJob] = []
                for state in self._speakers.values():
                    if not state.utterance or now - state.last_packet_monotonic < _STALE_SECONDS:
                        continue
                    job = self._flush_speaker_locked(state)
                    if job is not None:
                        jobs.append(job)
                for job in jobs:
                    self._enqueue_locked(job)

    def _worker_loop(
        self,
        session_id: str,
        job_queue: queue.Queue[_TranscriptionJob | None],
        finished_event: threading.Event,
    ) -> None:
        unexpected_voice_client = None
        unexpected_sink = None
        try:
            from services import stt_whisper

            while True:
                job = job_queue.get()
                try:
                    if job is None:
                        break
                    segments = stt_whisper.transcrever_segmentos_pcm(
                        job.samples,
                        job.sample_rate,
                        self._model,
                    )
                    new_cues: list[_Cue] = []
                    for index, segment in enumerate(segments):
                        relative_start = float(segment["start"])
                        relative_end = float(segment["end"])
                        start = max(job.start, job.start + relative_start)
                        end = min(job.end, max(start + 0.08, job.start + relative_end))
                        new_cues.append(
                            _Cue(
                                sequence=(job.sequence * 1000) + index,
                                start=start,
                                end=end,
                                speaker=job.speaker,
                                text=str(segment["text"]),
                            )
                        )
                    if new_cues:
                        with self._lock:
                            if session_id == self._session_id:
                                self._cues.extend(new_cues)
                        self._write_srt()
                except Exception as exc:
                    self.register_error(f"Falha ao transcrever trecho de {getattr(job, 'speaker', '?')}: {exc}")
                    log.exception("Falha no worker da transcricao SRT.")
                finally:
                    job_queue.task_done()
        except Exception as exc:
            self.register_error(f"Worker do Whisper encerrado: {exc}")
            log.exception("Worker principal da transcricao SRT foi encerrado.")
        finally:
            self._write_srt()
            with self._lock:
                if session_id == self._session_id:
                    if self._active:
                        self._duration_seconds = max(
                            0.0,
                            time.monotonic() - self._started_monotonic,
                        )
                        unexpected_voice_client = self._voice_client
                        unexpected_sink = self._sink
                    self._active = False
                    self._accept_audio = False
                    self._finalizing = False
                    self._voice_client = None
                    self._sink = None
                    self._speakers.clear()
                    self._participants.clear()
                    self._queue = None
                    self._stop_event = None
                    self._watchdog_thread = None
                    self._worker_thread = None
            if unexpected_voice_client is not None:
                try:
                    if (
                        unexpected_voice_client.is_listening()
                        and getattr(unexpected_voice_client, "sink", None) is unexpected_sink
                    ):
                        unexpected_voice_client.stop_listening()
                except Exception:
                    log.exception("Falha ao encerrar receptor apos erro do worker SRT.")
            finished_event.set()
            log.info("Transcricao SRT finalizada: %s", self._output_path)

    def _write_srt(self) -> None:
        with self._lock:
            path = self._output_path
            cues = sorted(self._cues, key=lambda cue: (cue.start, cue.end, cue.sequence))
        if path is None:
            return
        blocks = []
        for index, cue in enumerate(cues, 1):
            blocks.append(
                f"{index}\n{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}\n"
                f"[{cue.speaker}] {cue.text}\n"
            )
        content = "\n".join(blocks)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(content, encoding="utf-8-sig")
            os.replace(temporary, path)
        except OSError as exc:
            self.register_error(f"Falha ao salvar SRT: {exc}")
            log.exception("Falha ao gravar %s.", path)

    def _after_listening(self, error: Exception | None) -> None:
        if error is not None:
            self.register_error(f"Recepcao de voz encerrada: {error}")
            log.error("Receptor da transcricao encerrado com erro: %s", error)


_service = DiscordTranscriptionService()


def get_transcription_service() -> DiscordTranscriptionService:
    return _service
