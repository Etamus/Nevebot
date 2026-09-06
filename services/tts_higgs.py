"""Higgs Audio v3 TTS 4B local via servidor nativo audio.cpp."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

import config


log = logging.getLogger("tts_higgs")
_MODEL_ID = "higgs-tts-3-4b"
_MODEL_MIN_BYTES = 5_095_354_048
_lock = threading.RLock()
_process: subprocess.Popen | None = None
_log_handle = None
_session = requests.Session()


def _referencia(voz_cfg: dict) -> Path:
    raw = str(voz_cfg.get("voz_referencia") or "data/voz_referencia.wav")
    path = Path(raw)
    return path if path.is_absolute() else config.BASE_DIR / path


def _verificar_runtime() -> None:
    if not config.HIGGS_SERVER_EXE.is_file():
        raise FileNotFoundError(
            "Runtime do Higgs ausente. Execute instalar.bat para preparar o audio.cpp."
        )
    if not config.HIGGS_MODEL_PATH.is_file() or config.HIGGS_MODEL_PATH.stat().st_size < _MODEL_MIN_BYTES:
        raise FileNotFoundError(
            "Modelo Higgs Audio v3 Q8_0 ausente ou incompleto. Execute instalar.bat."
        )


def _server_config() -> Path:
    path = config.BASE_DIR / "logs" / "higgs-server.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": config.HIGGS_SERVER_HOST,
        "port": config.HIGGS_SERVER_PORT,
        "backend": "cuda",
        "device": 0,
        "threads": max(2, min(8, (os.cpu_count() or 8) // 2)),
        "lazy_load": False,
        "max_request_body_bytes": 67_108_864,
        "models": [
            {
                "id": _MODEL_ID,
                "family": "higgs_audio_tts",
                "path": str(config.HIGGS_MODEL_PATH.resolve()),
                "task": "tts",
                "mode": "offline",
                "lazy": False,
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _saudavel() -> bool:
    try:
        return _session.get(f"{config.HIGGS_SERVER_URL}/health", timeout=1.5).status_code == 200
    except requests.RequestException:
        return False


def carregar() -> None:
    """Inicia uma unica instancia persistente e aguarda o modelo ficar pronto."""
    global _process, _log_handle
    if _saudavel():
        return
    with _lock:
        if _saudavel():
            return
        _verificar_runtime()
        if _process is not None and _process.poll() is None:
            _process.terminate()
            try:
                _process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("[TTS:Higgs] Servidor anterior nao encerrou; finalizando processo preso.")
                _process.kill()
                _process.wait(timeout=5)
        if _log_handle is not None:
            _log_handle.close()
            _log_handle = None

        log_path = config.BASE_DIR / "logs" / "higgs-server.log"
        _log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        env = os.environ.copy()
        env["PATH"] = str(config.HIGGS_RUNTIME_DIR.resolve()) + os.pathsep + env.get("PATH", "")
        env.pop("CUDA_PATH", None)
        cmd = [
            str(config.HIGGS_SERVER_EXE.resolve()),
            "--config", str(_server_config()),
            "--no-ui",
            "--max-loaded-models", "1",
            "--busy-timeout-ms", str(config.HIGGS_REQUEST_TIMEOUT * 1000),
            "--log",
            "--log-file", str(log_path.resolve()),
        ]
        log.info("[TTS:Higgs] Iniciando audio.cpp persistente em %s...", config.HIGGS_SERVER_URL)
        _process = subprocess.Popen(
            cmd,
            cwd=str(config.HIGGS_RUNTIME_DIR),
            stdout=_log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + config.HIGGS_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if _process.poll() is not None:
                code = _process.returncode
                descarregar()
                raise RuntimeError(
                    f"Servidor Higgs encerrou com codigo {code}. Confira logs/higgs-server.log."
                )
            if _saudavel():
                log.info("[TTS:Higgs] Modelo Higgs Audio v3 pronto.")
                return
            time.sleep(0.25)
        descarregar()
        raise TimeoutError("O Higgs Audio v3 nao ficou pronto dentro do tempo limite.")


def descarregar() -> None:
    global _process, _log_handle
    with _lock:
        process, _process = _process, None
        if process is not None and process.poll() is None:
            log.info("[TTS:Higgs] Encerrando servidor nativo.")
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        if _log_handle is not None:
            _log_handle.close()
            _log_handle = None


def gerar(
    texto: str,
    voz_cfg: dict,
) -> tuple[np.ndarray, int]:
    texto = " ".join((texto or "").split())
    if not texto:
        return np.zeros(0, dtype=np.float32), 24000
    referencia = _referencia(voz_cfg)
    if not referencia.is_file():
        raise FileNotFoundError(f"Referencia de voz nao encontrada: {referencia}")
    carregar()
    texto_sem_tags = re.sub(r"<\|[^|<>]{1,80}\|>", "", texto)
    total_palavras = max(1, len(re.findall(r"\w+", texto_sem_tags, flags=re.UNICODE)))
    # Precisa ser identico no warm-up e nas requisicoes reais. O servidor nativo
    # reutiliza o estado preparado e evita recompilar/reorganizar a geracao por frase.
    max_tokens = max(512, min(1536, int(config.HIGGS_MAX_TOKENS)))
    payload = {
        "model": _MODEL_ID,
        "input": texto,
        "voice_ref": str(referencia.resolve()),
        "reference_text": str(voz_cfg.get("voz_referencia_texto") or "").strip(),
        "language": "pt-BR",
        "seed": int(voz_cfg.get("voz_seed", 42) or 42),
        "temperature": max(0.05, min(2.0, float(voz_cfg.get("voz_temperature", 0.8) or 0.8))),
        "top_k": max(1, min(100, int(voz_cfg.get("voz_top_k", 30) or 30))),
        "top_p": max(0.05, min(1.0, float(voz_cfg.get("voz_top_p", 0.8) or 0.8))),
        "max_tokens": max_tokens,
        "response_format": "wav",
    }
    inicio = time.perf_counter()
    response = _session.post(
        f"{config.HIGGS_SERVER_URL}/v1/audio/speech",
        json=payload,
        timeout=config.HIGGS_REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Higgs HTTP {response.status_code}: {response.text[:500]}")
    audio, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    speed = max(0.85, min(1.2, float(voz_cfg.get("velocidade", 1.0) or 1.0)))
    if not math.isclose(speed, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        import librosa

        audio = librosa.effects.time_stretch(audio, rate=speed).astype(np.float32, copy=False)
    duracao_maxima = max(4.0, min(18.0, 2.0 + total_palavras * 0.8))
    max_amostras = int(sample_rate * duracao_maxima)
    if len(audio) > max_amostras:
        log.warning(
            "[TTS:Higgs] Audio anormalmente longo (%.2fs para %d palavras); "
            "limitando a %.2fs.",
            len(audio) / sample_rate,
            total_palavras,
            duracao_maxima,
        )
        audio = audio[:max_amostras].copy()
        fade = min(len(audio), max(1, int(sample_rate * 0.08)))
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    log.info("[TTS:Higgs] Geracao concluida em %.2fs (%d amostras).", time.perf_counter() - inicio, len(audio))
    return audio, int(sample_rate)


def para_pcm_discord(
    audio: np.ndarray,
    sample_rate: int,
    *,
    volume: float = 1.0,
    pitch_semitones: float = 0.0,
    start_pad_s: float = 0.18,
    end_pad_s: float = 1.2,
    tail_frames: int = 60,
) -> bytes:
    if audio is None or len(audio) == 0:
        return b""
    import torch
    import torchaudio.functional as F

    tensor = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
    if pitch_semitones:
        ratio = 2.0 ** (float(pitch_semitones) / 12.0)
        intermediate = max(8000, int(round(sample_rate * ratio)))
        tensor = F.resample(tensor, sample_rate, intermediate)
        output = F.resample(tensor, intermediate, 48000).squeeze(0).numpy()
    else:
        output = F.resample(tensor, sample_rate, 48000).squeeze(0).numpy()
    output = np.concatenate([
        np.zeros(int(48000 * max(0.0, start_pad_s)), dtype=np.float32),
        output.astype(np.float32, copy=False),
        np.zeros(int(48000 * max(0.0, end_pad_s)), dtype=np.float32),
    ])
    peak = float(np.abs(output).max()) if output.size else 0.0
    if peak > 0:
        output = output / peak * 0.95 * min(max(float(volume), 0.0), 2.0)
    pcm16 = (output * 32767).clip(-32768, 32767).astype(np.int16)
    pcm = np.column_stack([pcm16, pcm16]).tobytes()
    frame_size = 3840
    if len(pcm) % frame_size:
        pcm += b"\x00" * (frame_size - len(pcm) % frame_size)
    return pcm + b"\x00" * (frame_size * max(0, int(tail_frames)))


def precarregar_e_aquecer(voz_cfg: dict, *, full_warmup: bool = True) -> None:
    carregar()
    if not full_warmup:
        return
    audio, sr = gerar("Oi.", voz_cfg)
    _ = para_pcm_discord(audio, sr, volume=float(voz_cfg.get("volume", 1.0)))
    log.info("[TTS:Higgs] Warmup concluido.")
