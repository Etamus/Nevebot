"""
services/stt_whisper.py — Speech-to-Text com faster-whisper (CTranslate2).

Transcreve áudio WAV recebido do browser para texto em PT-BR.
Usa faster-whisper (~4x mais rápido que openai-whisper) com VAD integrado.
Lê WAV com o módulo `wave` da stdlib e resampla com torchaudio — sem FFmpeg.
"""

import io
import logging
import os
import re
import threading
import unicodedata
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("stt_whisper")

_model = None
_model_name = None
_lock = threading.Lock()
_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = "large-v3-turbo"
_DOWNLOAD_DIR = _BASE_DIR / "models" / "whisper"
_MODEL_ALIASES = {
    "large turbo": "large-v3-turbo",
    "large-turbo": "large-v3-turbo",
    "turbo": "large-v3-turbo",
    "large_v3_turbo": "large-v3-turbo",
}


def _env_int(nome: str, padrao: int) -> int:
    try:
        return int(os.getenv(nome, str(padrao)))
    except ValueError:
        return padrao


_BEAM_SIZE = max(1, _env_int("WHISPER_BEAM_SIZE", 3))
_RETRY_BEAM_SIZE = max(_BEAM_SIZE, _env_int("WHISPER_RETRY_BEAM_SIZE", 5))
_RETRY_MAX_SECONDS = float(os.getenv("WHISPER_RETRY_MAX_SECONDS", "8.0"))
_HOTWORDS = os.getenv("WHISPER_HOTWORDS", "").strip() or None
_PROMPT_PTBR = (
    "Transcrição literal de fala casual em português brasileiro. "
    "Não traduza, não complete frases e preserve nomes próprios, gírias e pausas naturais."
)
_HALLUCINATIONS = (
    "legendas pela comunidade amara.org",
    "transcrição e legendas pedro negri",
    "transcricao e legendas pedro negri",
    "transcrição e legenda pedro negri",
    "transcricao e legenda pedro negri",
    "inscreva-se no canal",
    "obrigado por assistir",
    "ative o sininho",
    "se gostou deixe o like",
)


def _normalizar_nome_modelo(modelo: str | None) -> str:
    nome = (modelo or _DEFAULT_MODEL).strip()
    return _MODEL_ALIASES.get(nome.lower(), nome)


def _device_e_compute() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "int8_float16"
    except Exception:
        pass
    return "cpu", "int8"


def _sem_acentos(texto: str) -> str:
    normalizado = unicodedata.normalize("NFD", texto or "")
    return "".join(ch for ch in normalizado if unicodedata.category(ch) != "Mn")


def _normalizar_texto_lixo(texto: str) -> str:
    texto = _sem_acentos(texto).lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return " ".join(texto.split())


def _eh_credito_legenda(texto: str) -> bool:
    normalizado = _normalizar_texto_lixo(texto)
    if not normalizado:
        return False
    if "pedro negri" in normalizado and "legenda" in normalizado and "transcri" in normalizado:
        return True
    if "transcricao e legenda" in normalizado and len(normalizado.split()) <= 7:
        return True
    return False


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


def _remover_dc_offset(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples
    media = float(np.mean(samples))
    if abs(media) < 1e-4:
        return samples.astype(np.float32, copy=False)
    log.info("[STT] Removendo DC offset: %.5f", media)
    return np.clip(samples - media, -1.0, 1.0).astype(np.float32)


def _trim_silencio_vad(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Remove silêncio inicial/final sem cortar fonemas fracos."""
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
    global_rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    peak = float(np.max(np.abs(samples)))
    threshold = max(0.003, noise * 1.45, global_rms * 0.18)
    voiced = np.where(rms_arr > threshold)[0]
    if voiced.size == 0:
        if peak > 0.018 or global_rms > 0.004:
            log.info("[STT] VAD incerto; mantendo áudio completo para o Whisper.")
            return samples.astype(np.float32)
        log.info("[STT] VAD local descartou áudio sem fala clara.")
        return np.asarray([], dtype=np.float32)
    pre = int(sr * 0.45)
    post = int(sr * 0.60)
    ini = max(0, int(voiced[0] * hop) - pre)
    fim = min(len(samples), int(voiced[-1] * hop + frame) + post)
    if fim - ini < int(sr * 0.35) and len(samples) > int(sr * 0.70):
        log.info("[STT] Corte VAD curto demais; mantendo áudio completo.")
        return samples.astype(np.float32)
    if ini > 0 or fim < len(samples):
        log.info("[STT] Silêncio cortado: %.2fs → %.2fs", len(samples) / sr, (fim - ini) / sr)
    return samples[ini:fim].astype(np.float32)


def _texto_suspeito(texto: str) -> bool:
    normalizado = " ".join((texto or "").lower().split())
    if not normalizado:
        return True
    if _eh_credito_legenda(texto):
        return True
    if any(frase in normalizado for frase in _HALLUCINATIONS):
        return True
    palavras = re.findall(r"\w+", normalizado, flags=re.UNICODE)
    if len(palavras) >= 6:
        repetidas = sum(1 for a, b in zip(palavras, palavras[1:]) if a == b)
        if repetidas >= 3:
            return True
    return False


def _metricas_segmentos(segmentos: list) -> dict[str, float]:
    if not segmentos:
        return {"avg_logprob": -99.0, "compression_ratio": 99.0, "no_speech_prob": 1.0}
    avg_logprob = float(np.mean([float(getattr(s, "avg_logprob", -99.0)) for s in segmentos]))
    compression_ratio = float(max(float(getattr(s, "compression_ratio", 0.0)) for s in segmentos))
    no_speech_prob = float(max(float(getattr(s, "no_speech_prob", 0.0)) for s in segmentos))
    return {
        "avg_logprob": avg_logprob,
        "compression_ratio": compression_ratio,
        "no_speech_prob": no_speech_prob,
    }


def _transcricao_suspeita(texto: str, metricas: dict[str, float], info, duracao_s: float) -> bool:
    if _texto_suspeito(texto):
        return True
    lang_prob = float(getattr(info, "language_probability", 1.0) or 0.0)
    if lang_prob < 0.55:
        return True
    if duracao_s > 0.7 and len(texto.strip()) < 3:
        return True
    if metricas["avg_logprob"] < -0.85:
        return True
    if metricas["compression_ratio"] > 2.8:
        return True
    if metricas["no_speech_prob"] > 0.85 and metricas["avg_logprob"] < -0.45:
        return True
    return False


def _pontuar_transcricao(texto: str, metricas: dict[str, float], info) -> float:
    lang_prob = float(getattr(info, "language_probability", 1.0) or 0.0)
    penalidade_vazio = 2.0 if not texto.strip() else 0.0
    return (
        metricas["avg_logprob"]
        + (lang_prob * 0.35)
        - max(0.0, metricas["compression_ratio"] - 2.2) * 0.25
        - max(0.0, metricas["no_speech_prob"] - 0.65) * 0.4
        - penalidade_vazio
    )


def _executar_transcricao(samples: np.ndarray, *, beam_size: int, vad_filter: bool):
    segments, info = _model.transcribe(
        samples,
        language="pt",
        task="transcribe",
        beam_size=beam_size,
        best_of=1,
        patience=1.0,
        temperature=0.0,
        initial_prompt=_PROMPT_PTBR,
        condition_on_previous_text=False,
        no_speech_threshold=0.72,
        compression_ratio_threshold=2.8,
        log_prob_threshold=-1.15,
        suppress_blank=True,
        without_timestamps=True,
        word_timestamps=False,
        vad_filter=vad_filter,
        vad_parameters={
            "threshold": 0.35,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 360,
        } if vad_filter else None,
        hotwords=_HOTWORDS,
    )
    segmentos = list(segments)
    texto = " ".join(seg.text.strip() for seg in segmentos).strip()
    return texto, info, _metricas_segmentos(segmentos)


def carregar(modelo: str = _DEFAULT_MODEL) -> None:
    """Carrega o modelo faster-whisper (lazy, thread-safe)."""
    global _model, _model_name
    modelo = _normalizar_nome_modelo(modelo)
    if _model is not None and _model_name == modelo:
        return
    with _lock:
        if _model is not None and _model_name == modelo:
            return
        try:
            if _model is not None and _model_name != modelo:
                log.info("[STT] Trocando faster-whisper '%s' -> '%s'...", _model_name, modelo)
                _model = None

            from faster_whisper import WhisperModel

            device, compute_type = _device_e_compute()
            cpu_threads = max(4, (os.cpu_count() or 8) // 2)
            _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
            log.info(
                "[STT] Carregando faster-whisper '%s' em %s/%s...",
                modelo,
                device,
                compute_type,
            )
            _model = WhisperModel(
                modelo,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=1,
                download_root=str(_DOWNLOAD_DIR),
            )
            _model_name = modelo
            log.info("[STT] faster-whisper carregado com sucesso.")
        except Exception as exc:
            log.error("[STT] ERRO ao carregar faster-whisper: %s", exc, exc_info=True)
            raise


def precarregar_e_aquecer(modelo: str = _DEFAULT_MODEL) -> None:
    """Carrega o faster-whisper e aquece com um WAV real quando disponivel."""
    modelo = _normalizar_nome_modelo(modelo)
    carregar(modelo)

    wav_teste = Path(__file__).parent.parent / "data" / "debug_tts" / "chatterbox_sem_duplicar_normal.wav"
    if not wav_teste.exists():
        log.info("[STT] faster-whisper '%s' pre-carregado; WAV de warmup nao encontrado.", modelo)
        return
    try:
        texto = transcrever(wav_teste.read_bytes(), modelo)
        log.info("[STT] Warmup faster-whisper '%s' concluido: %r", modelo, texto)
    except Exception as exc:
        log.warning("[STT] Warmup faster-whisper falhou: %s", exc)


def transcrever(wav_bytes: bytes, modelo: str = _DEFAULT_MODEL) -> str:
    """Transcreve áudio WAV (qualquer sample rate) para texto PT-BR.

    O WAV é lido com a stdlib (sem FFmpeg). Se a taxa de amostragem
    não for 16 kHz, o resample é feito com torchaudio.
    """
    log.info("[STT] Recebido WAV: %d bytes", len(wav_bytes))
    try:
        modelo = _normalizar_nome_modelo(modelo)
        carregar(modelo)
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

        # Mono. Se houver canais com níveis muito diferentes, usa o canal mais claro.
        if n_ch > 1:
            canais = samples.reshape(-1, n_ch)
            rms_canais = np.sqrt(np.mean(canais * canais, axis=0))
            melhor = int(np.argmax(rms_canais))
            menor = float(np.min(rms_canais))
            maior = float(np.max(rms_canais))
            if maior > 0 and (menor <= 1e-5 or maior / max(menor, 1e-5) > 1.35):
                log.info("[STT] Usando canal %d/%d com melhor RMS para evitar cancelamento.", melhor + 1, n_ch)
                samples = canais[:, melhor]
            else:
                samples = canais.mean(axis=1)

        # Resample para 16 kHz (exigido pelo Whisper)
        if sr != 16000:
            log.info("[STT] Resampleando %d → 16000 Hz", sr)
            tensor = torch.from_numpy(samples).float().unsqueeze(0)
            samples = F.resample(tensor, sr, 16000).squeeze(0).numpy()

        samples = _remover_dc_offset(samples)

        # VAD/trim local antes do Whisper: reduz silêncio sem cortar início/fim de fala.
        samples = _trim_silencio_vad(samples, 16000)
        if samples.size < int(16000 * 0.20):
            log.info("[STT] Áudio útil muito curto após VAD local; ignorando.")
            return ""

        # Normaliza volume do áudio para melhorar transcrição sem distorcer.
        samples = _normalizar_rms(samples)

        duracao_s = len(samples) / 16000
        log.info("[STT] Transcrevendo %d amostras (%.2fs, beam=%d)...", len(samples), duracao_s, _BEAM_SIZE)
        texto, info, metricas = _executar_transcricao(samples, beam_size=_BEAM_SIZE, vad_filter=False)
        suspeita = _transcricao_suspeita(texto, metricas, info, duracao_s)
        log.info(
            "[STT] Transcrição: %r (lang=%s prob=%.2f avg=%.2f comp=%.2f nospeech=%.2f suspeita=%s)",
            texto,
            info.language,
            info.language_probability,
            metricas["avg_logprob"],
            metricas["compression_ratio"],
            metricas["no_speech_prob"],
            suspeita,
        )
        if suspeita and duracao_s <= _RETRY_MAX_SECONDS:
            log.info("[STT] Transcrição suspeita; tentando retry com beam=%d + VAD interno.", _RETRY_BEAM_SIZE)
            texto_retry, info_retry, metricas_retry = _executar_transcricao(
                samples,
                beam_size=_RETRY_BEAM_SIZE,
                vad_filter=True,
            )
            suspeita_retry = _transcricao_suspeita(texto_retry, metricas_retry, info_retry, duracao_s)
            score = _pontuar_transcricao(texto, metricas, info)
            score_retry = _pontuar_transcricao(texto_retry, metricas_retry, info_retry)
            log.info(
                "[STT] Retry: %r (prob=%.2f avg=%.2f comp=%.2f nospeech=%.2f suspeita=%s score=%.2f/%0.2f)",
                texto_retry,
                info_retry.language_probability,
                metricas_retry["avg_logprob"],
                metricas_retry["compression_ratio"],
                metricas_retry["no_speech_prob"],
                suspeita_retry,
                score_retry,
                score,
            )
            if texto_retry and (not suspeita_retry or score_retry >= score + 0.08 or not texto):
                texto = texto_retry
                info = info_retry

        if _eh_credito_legenda(texto):
            log.warning("[STT] Crédito de legenda detectado como alucinação do Whisper; ignorando: %r", texto)
            return ""

        return texto
    except Exception as exc:
        log.error("[STT] ERRO ao transcrever: %s", exc, exc_info=True)
        raise
