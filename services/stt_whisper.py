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


def _normalizar_rms(samples: np.ndarray, alvo_rms: float = 0.08) -> np.ndarray:
    """Normaliza fala baixa por RMS sem clipar."""
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    peak = float(np.max(np.abs(samples)))
    if rms <= 1e-6 or peak <= 1e-6:
        return samples
    ganho = min(alvo_rms / rms, 0.95 / peak, 8.0)
    if ganho > 1.05:
        log.info("[STT] Normalização RMS aplicada: rms=%.4f ganho=%.2fx", rms, ganho)
        return np.clip(samples * ganho, -1.0, 1.0).astype(np.float32)
    return samples.astype(np.float32)


def _trim_silencio_vad(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Remove silêncio inicial/final com VAD simples por energia em frames."""
    if samples.size < int(sr * 0.15):
        return samples
    frame = int(sr * 0.03)  # 30ms
    hop = int(sr * 0.01)    # 10ms
    if len(samples) < frame:
        return samples
    rms_vals = []
    for start in range(0, len(samples) - frame + 1, hop):
        trecho = samples[start:start + frame]
        rms_vals.append(float(np.sqrt(np.mean(trecho * trecho))))
    if not rms_vals:
        return samples
    rms_arr = np.asarray(rms_vals, dtype=np.float32)
    noise = float(np.percentile(rms_arr, 20))
    threshold = max(0.012, noise * 3.0)
    voiced = np.where(rms_arr > threshold)[0]
    if voiced.size == 0:
        log.info("[STT] VAD local descartou áudio sem fala clara.")
        return np.asarray([], dtype=np.float32)
    pre = int(sr * 0.18)
    post = int(sr * 0.25)
    ini = max(0, int(voiced[0] * hop) - pre)
    fim = min(len(samples), int(voiced[-1] * hop + frame) + post)
    if ini > 0 or fim < len(samples):
        log.info("[STT] Silêncio cortado: %.2fs → %.2fs", len(samples) / sr, (fim - ini) / sr)
    return samples[ini:fim].astype(np.float32)


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

        # VAD/trim local antes do Whisper: reduz silêncio e melhora latência.
        samples = _trim_silencio_vad(samples, 16000)
        if samples.size < int(16000 * 0.20):
            log.info("[STT] Áudio útil muito curto após VAD local; ignorando.")
            return ""

        # Normaliza volume do áudio para melhorar transcrição sem distorcer.
        samples = _normalizar_rms(samples)

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
