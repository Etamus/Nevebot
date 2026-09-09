"""Transcricao isolada do audio de saida do Windows para SRT."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from services.discord_transcription import (
    DiscordTranscriptionService,
    _DEFAULT_OUTPUT_DIR,
    _TranscriptionJob,
)


log = logging.getLogger("local_transcription")
_SAMPLE_RATE = 48_000
_BLOCK_FRAMES = 4_800
_MAX_QUEUE = 64


class LocalTranscriptionService(DiscordTranscriptionService):
    """Captura somente o loopback do dispositivo de saida padrao."""

    def __init__(self) -> None:
        super().__init__()
        self._capture_thread: threading.Thread | None = None
        self._speaker_id: str | None = None
        self._speaker_name: str | None = None

    def start(self, *, model: str, output_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        try:
            import soundcard as sc
        except ImportError as exc:
            raise RuntimeError("A captura local requer SoundCard. Execute instalar.bat novamente.") from exc

        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("Nenhum dispositivo de saida padrao foi encontrado no Windows.")
        loopback = sc.get_microphone(id=speaker.id, include_loopback=True)
        if loopback is None or not bool(getattr(loopback, "isloopback", False)):
            raise RuntimeError("O dispositivo de saida padrao nao oferece captura loopback WASAPI.")

        folder = Path(output_dir or _DEFAULT_OUTPUT_DIR).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        if not folder.is_dir():
            raise ValueError("A pasta de transcricao e invalida.")

        with self._lock:
            if self._active or self._finalizing:
                raise RuntimeError("Ja existe uma transcricao local em andamento.")
            session_id = uuid.uuid4().hex
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = folder / f"Local-{timestamp}.srt"
            suffix = 2
            while output_path.exists():
                output_path = folder / f"Local-{timestamp}-{suffix}.srt"
                suffix += 1

            job_queue: queue.Queue[_TranscriptionJob | None] = queue.Queue(maxsize=_MAX_QUEUE)
            stop_event = threading.Event()
            finished_event = threading.Event()
            self._active = True
            self._accept_audio = True
            self._finalizing = False
            self._session_id = session_id
            self._started_monotonic = time.monotonic()
            self._duration_seconds = 0.0
            self._model = str(model or "large-v3-turbo")
            self._guild_id = None
            self._guild_name = None
            self._channel_id = None
            self._channel_name = "Audio local"
            self._output_path = output_path
            self._voice_client = None
            self._sink = None
            self._speakers.clear()
            self._participants.clear()
            self._cues.clear()
            self._sequence = 0
            self._queue = job_queue
            self._stop_event = stop_event
            self._finished_event = finished_event
            self._last_error = None
            self._speaker_id = str(speaker.id)
            self._speaker_name = str(speaker.name)

            worker = threading.Thread(target=self._worker_loop, args=(session_id, job_queue, finished_event), name="local-srt-whisper", daemon=True)
            watchdog = threading.Thread(target=self._watchdog_loop, args=(session_id, stop_event), name="local-srt-vad", daemon=True)
            capture = threading.Thread(target=self._capture_loop, args=(session_id, stop_event, self._speaker_id), name="windows-loopback-capture", daemon=True)
            self._worker_thread = worker
            self._watchdog_thread = watchdog
            self._capture_thread = capture

        try:
            self._write_srt()
            worker.start()
            watchdog.start()
            capture.start()
        except Exception:
            self.stop(wait=True, timeout=5, reason="falha ao iniciar captura local")
            raise
        log.info("Transcricao local iniciada: dispositivo=%s arquivo=%s modelo=%s", self._speaker_name, output_path, self._model)
        return self.state()

    def state(self) -> dict[str, Any]:
        state = super().state()
        state.update({
            "modo": "local",
            "canal_nome": self._speaker_name,
            "participantes": ["Audio do Windows"] if state["ativo"] else [],
        })
        return state

    def _capture_loop(self, session_id: str, stop_event: threading.Event, speaker_id: str) -> None:
        try:
            import soundcard as sc

            loopback = sc.get_microphone(id=speaker_id, include_loopback=True)
            if loopback is None or not bool(getattr(loopback, "isloopback", False)):
                raise RuntimeError("A captura loopback do dispositivo padrao ficou indisponivel.")
            with loopback.recorder(samplerate=_SAMPLE_RATE, channels=None, blocksize=_BLOCK_FRAMES) as recorder:
                while not stop_event.is_set():
                    started = time.monotonic()
                    samples = np.asarray(recorder.record(numframes=_BLOCK_FRAMES), dtype=np.float32)
                    if stop_event.is_set():
                        break
                    if samples.ndim == 1:
                        samples = samples[:, None]
                    if samples.ndim != 2 or not samples.shape[0]:
                        stop_event.wait(0.05)
                        continue
                    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2", copy=False).tobytes()
                    self._add_pcm(
                        key="local:windows",
                        name="Audio do Windows",
                        source="local",
                        pcm=pcm,
                        sample_rate=_SAMPLE_RATE,
                        channels=int(samples.shape[1]),
                        start_seconds=None,
                    )
                    expected = samples.shape[0] / _SAMPLE_RATE
                    remaining = expected - (time.monotonic() - started)
                    if remaining > 0:
                        stop_event.wait(min(remaining, expected))
        except Exception as exc:
            with self._lock:
                current = self._active and self._session_id == session_id
            if current:
                self.register_error(f"Falha na captura do audio do Windows: {exc}")
                log.exception("Captura loopback local encerrada.")
                self.stop(wait=False, reason="captura loopback encerrada")
        finally:
            with self._lock:
                if self._session_id == session_id:
                    self._capture_thread = None


_service = LocalTranscriptionService()


def get_local_transcription_service() -> LocalTranscriptionService:
    return _service
