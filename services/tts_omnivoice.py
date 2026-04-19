"""
services/tts_omnivoice.py — Text-to-Speech com OmniVoice (k2-fsa).

Gera áudio a partir de texto usando voice cloning com um áudio de referência
para manter a voz consistente entre gerações.
Converte para formato PCM compatível com Discord (48 kHz estéreo 16-bit).
Não depende de FFmpeg.
"""

import io
import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("tts_omnivoice")

_model = None
_voice_prompt = None   # VoiceClonePrompt reutilizável
_lock = threading.Lock()

_BASE_DIR = Path(__file__).parent.parent
_REF_AUDIO_PATH = _BASE_DIR / "data" / "voz_referencia.wav"
_REF_TEXT = "Oi, tudo bem? Eu sou a Lou, é muito legal te conhecer."


def carregar(device: str = "cuda:0") -> None:
    """Carrega o modelo OmniVoice (lazy, thread-safe)."""
    global _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        try:
            import torch
            from omnivoice import OmniVoice

            log.info("[TTS] Carregando OmniVoice (device=%s)...", device)
            log.info("[TTS] CUDA disponível: %s", torch.cuda.is_available())
            if torch.cuda.is_available():
                log.info("[TTS] GPU: %s", torch.cuda.get_device_name(0))
            _model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map=device,
                dtype=torch.float16,
            )
            log.info("[TTS] OmniVoice carregado com sucesso.")
        except Exception as exc:
            log.error("[TTS] ERRO ao carregar OmniVoice: %s", exc, exc_info=True)
            raise


def _garantir_referencia(instruct: str, language: str, seed: int) -> None:
    """Gera o áudio de referência se ainda não existir e cria o VoiceClonePrompt."""
    global _voice_prompt
    if _voice_prompt is not None:
        return

    import torch
    import soundfile as sf

    carregar()

    if not _REF_AUDIO_PATH.exists():
        log.info("[TTS] Gerando áudio de referência com voice design...")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        chunks = _model.generate(
            text=_REF_TEXT, instruct=instruct, speed=1.0, language=language,
        )
        ref_audio = chunks[0]
        sf.write(str(_REF_AUDIO_PATH), ref_audio, 24000)
        log.info("[TTS] Áudio de referência salvo em %s (%.2fs)",
                 _REF_AUDIO_PATH, len(ref_audio) / 24000)

    log.info("[TTS] Criando VoiceClonePrompt a partir de %s...", _REF_AUDIO_PATH)
    _voice_prompt = _model.create_voice_clone_prompt(
        ref_audio=str(_REF_AUDIO_PATH),
        ref_text=_REF_TEXT,
    )
    log.info("[TTS] VoiceClonePrompt criado com sucesso.")


def gerar(texto: str, instruct: str = "female, teenager, low pitch",
          speed: float = 1.0, seed: int = 42,
          language: str = "Portuguese") -> np.ndarray:
    """Gera áudio (np.ndarray, 24 kHz mono float32) a partir de texto."""
    log.info("[TTS] Gerando áudio — texto=%r speed=%.1f lang=%s", texto[:80], speed, language)
    try:
        import torch
        carregar()
        _garantir_referencia(instruct, language, seed)
        chunks = _model.generate(
            text=texto, voice_clone_prompt=_voice_prompt,
            speed=speed, language=language,
        )
        audio = chunks[0]
        log.info("[TTS] Áudio gerado: shape=%s dtype=%s duração=%.2fs",
                 audio.shape, audio.dtype, len(audio) / 24000)
        return audio
    except Exception as exc:
        log.error("[TTS] ERRO ao gerar áudio: %s", exc, exc_info=True)
        raise


def regenerar_referencia(instruct: str = "female, teenager, low pitch",
                         language: str = "Portuguese", seed: int = 42) -> None:
    """Apaga o áudio de referência e recria (chamado quando o usuário muda a voz)."""
    global _voice_prompt
    _voice_prompt = None
    if _REF_AUDIO_PATH.exists():
        _REF_AUDIO_PATH.unlink()
        log.info("[TTS] Áudio de referência anterior removido.")
    _garantir_referencia(instruct, language, seed)


def para_pcm_discord(audio_24k: np.ndarray, volume: float = 1.0,
                     pitch_semitones: float = 0.0) -> bytes:
    """Converte áudio 24 kHz mono → PCM 48 kHz estéreo 16-bit (formato Discord).

    Usa torchaudio para resample — sem FFmpeg.
    pitch_semitones: altera o pitch em semitons (positivo = mais agudo, negativo = mais grave).
    """
    log.info("[TTS] Convertendo para PCM Discord — input shape=%s volume=%.1f pitch=%.1f st",
             audio_24k.shape, volume, pitch_semitones)
    try:
        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(audio_24k).float().unsqueeze(0)  # (1, T)

        # Pitch shift via resample trick: resample para taxa alterada, depois para 48 kHz
        if pitch_semitones != 0.0:
            ratio = 2.0 ** (pitch_semitones / 12.0)
            intermediate_rate = int(round(24000 * ratio))
            # Resample 24k → intermediate (muda o pitch)
            tensor = F.resample(tensor, 24000, intermediate_rate)
            # Resample intermediate → 48k (corrige a duração)
            audio_48k = F.resample(tensor, intermediate_rate, 48000).squeeze(0).numpy()
        else:
            audio_48k = F.resample(tensor, 24000, 48000).squeeze(0).numpy()

        # Normaliza e aplica volume
        peak = np.abs(audio_48k).max()
        if peak > 0:
            audio_48k = audio_48k / peak * 0.95 * min(max(volume, 0.0), 2.0)

        pcm16 = (audio_48k * 32767).clip(-32768, 32767).astype(np.int16)
        stereo = np.column_stack([pcm16, pcm16])
        pcm_bytes = stereo.tobytes()

        # Discord lê em frames de 3840 bytes (20ms a 48kHz estéreo 16-bit).
        # Frames incompletos são descartados e o encoder Opus tem look-ahead
        # interno (~26ms) — quando o stream acaba, amostras no buffer do Opus
        # são perdidas. Alinhamos ao frame e adicionamos 5 frames de silêncio
        # para o Opus processar todo o áudio real antes do EOF.
        frame_size = 3840  # 960 samples × 2 canais × 2 bytes
        resto = len(pcm_bytes) % frame_size
        if resto:
            pcm_bytes += b'\x00' * (frame_size - resto)
        pcm_bytes += b'\x00' * (frame_size * 5)  # 100ms de silêncio para flush do Opus

        log.info("[TTS] PCM gerado: %d bytes (%.2fs a 48kHz estéreo)",
                 len(pcm_bytes), len(pcm_bytes) / (48000 * 4))
        return pcm_bytes
    except Exception as exc:
        log.error("[TTS] ERRO ao converter PCM: %s", exc, exc_info=True)
        raise
