"""
services/stt_whisper.py — Speech-to-Text com faster-whisper (CTranslate2).

Transcreve áudio WAV recebido do browser para texto em PT-BR.
Usa faster-whisper (~4x mais rápido que openai-whisper) com VAD integrado.
Lê WAV com o módulo `wave` da stdlib e resampla com torchaudio — sem FFmpeg.
"""

import io
import logging
import threading
import wave

import numpy as np

log = logging.getLogger("stt_whisper")

_model = None
_lock = threading.Lock()


def carregar(modelo: str = "medium") -> None:
    """Carrega o modelo faster-whisper (lazy, thread-safe)."""
    global _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            log.info("[STT] Carregando faster-whisper '%s'...", modelo)
            _model = WhisperModel(modelo, device="cuda", compute_type="int8_float16")
            log.info("[STT] faster-whisper carregado com sucesso.")
        except Exception as exc:
            log.error("[STT] ERRO ao carregar faster-whisper: %s", exc, exc_info=True)
            raise


def transcrever(wav_bytes: bytes) -> str:
    """Transcreve áudio WAV (qualquer sample rate) para texto PT-BR.

    O WAV é lido com a stdlib (sem FFmpeg). Se a taxa de amostragem
    não for 16 kHz, o resample é feito com torchaudio.
    """
    log.info("[STT] Recebido WAV: %d bytes", len(wav_bytes))
    try:
        carregar()
        import torch
        import torchaudio.functional as F

        # Lê WAV — stdlib, sem dependência externa
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        log.info("[STT] WAV: sr=%d ch=%d sw=%d frames=%d duração=%.2fs",
                 sr, n_ch, sw, n_frames, n_frames / sr if sr else 0)

        # Converte bytes → float32
        if sw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        # Mono
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)

        # Resample para 16 kHz (exigido pelo Whisper)
        if sr != 16000:
            log.info("[STT] Resampleando %d → 16000 Hz", sr)
            tensor = torch.from_numpy(samples).float().unsqueeze(0)
            samples = F.resample(tensor, sr, 16000).squeeze(0).numpy()

        # Normaliza volume do áudio para melhorar transcrição
        peak = np.abs(samples).max()
        if peak > 0 and peak < 0.1:
            samples = samples / peak * 0.95
            log.info("[STT] Áudio normalizado (peak original=%.4f)", peak)

        log.info("[STT] Transcrevendo %d amostras (%.2fs)...", len(samples), len(samples) / 16000)
        segments, info = _model.transcribe(
            samples,
            language="pt",
            beam_size=1,
            initial_prompt="Conversa em português brasileiro.",
            condition_on_previous_text=False,
            no_speech_threshold=0.4,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        texto = " ".join(seg.text.strip() for seg in segments).strip()
        log.info("[STT] Transcrição: %r (lang=%s prob=%.2f)", texto,
                 info.language, info.language_probability)
        return texto
    except Exception as exc:
        log.error("[STT] ERRO ao transcrever: %s", exc, exc_info=True)
        raise
