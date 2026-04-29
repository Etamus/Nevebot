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
import config

log = logging.getLogger("tts_omnivoice")

_model = None
_voice_prompt = None   # VoiceClonePrompt reutilizável
_voice_prompt_key = None
_lock = threading.Lock()

_BASE_DIR = Path(__file__).parent.parent
_REF_AUDIO_PATH = _BASE_DIR / "data" / "voz_referencia.wav"
_REF_TEXT = "Oi, tudo bem? Eu sou a Lou, é muito legal te conhecer."
REF_TEXT_PADRAO = _REF_TEXT

def montar_instruct(age: str = "teenager", pitch_style: str = "low pitch", style: str = "calma") -> str:
    """Monta um instruct aceito pelo OmniVoice.

    O OmniVoice rejeita descritores livres como "calm", "soft voice", "shy".
    Por isso mantemos apenas os itens oficiais suportados.
    """
    age_map = {
        "teenager": "teenager",
        "young adult": "young adult",
        "adult": "middle-aged",
        "middle-aged": "middle-aged",
    }
    pitch_map = {
        "low pitch": "low pitch",
        "medium pitch": "moderate pitch",
        "moderate pitch": "moderate pitch",
        "high pitch": "high pitch",
    }
    age_token = age_map.get((age or "").strip().lower(), "teenager")
    pitch_token = pitch_map.get((pitch_style or "").strip().lower(), "low pitch")
    return f"female, {age_token}, {pitch_token}"


def _duracao_segura(texto: str, speed: float = 1.0) -> float:
    """Estima duração com folga para evitar que o OmniVoice corte o fim da frase."""
    texto = (texto or "").strip()
    speed = max(0.5, min(float(speed or 1.0), 2.0))
    letras = sum(1 for ch in texto if not ch.isspace())
    pontuacoes = sum(1 for ch in texto if ch in ".,;:?!")
    # Português tende a precisar de mais folga no OmniVoice; a margem evita finais como "ago...".
    dur = (letras / 11.5 + pontuacoes * 0.18 + 1.35) / speed
    return max(1.6, min(dur, 18.0))


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

            log.info("[TTS] Carregando OmniVoice (device=%s, model=%s)...", device, config.OMNIVOICE_MODEL_PATH)
            log.info("[TTS] CUDA disponível: %s", torch.cuda.is_available())
            if torch.cuda.is_available():
                log.info("[TTS] GPU: %s", torch.cuda.get_device_name(0))
            _model = OmniVoice.from_pretrained(
                config.OMNIVOICE_MODEL_PATH,
                device_map=device,
                dtype=torch.float16,
            )
            log.info("[TTS] OmniVoice carregado com sucesso.")
        except Exception as exc:
            log.error("[TTS] ERRO ao carregar OmniVoice: %s", exc, exc_info=True)
            raise


def _referencia_atual() -> tuple[Path, str, bool]:
    """Retorna (caminho, texto_ref, deve_gerar_auto)."""
    return _REF_AUDIO_PATH, _REF_TEXT, True


def limpar_prompt_voz() -> None:
    """Força recriação do VoiceClonePrompt na próxima geração."""
    global _voice_prompt, _voice_prompt_key
    _voice_prompt = None
    _voice_prompt_key = None


def _garantir_referencia(instruct: str, language: str, seed: int) -> None:
    """Gera o áudio de referência se ainda não existir e cria o VoiceClonePrompt."""
    global _voice_prompt, _voice_prompt_key
    ref_path, ref_text, gerar_auto = _referencia_atual()
    prompt_key = (str(ref_path.resolve()), ref_text, instruct, language, int(seed))
    if _voice_prompt is not None and _voice_prompt_key == prompt_key:
        return

    import torch
    import soundfile as sf

    carregar()

    if gerar_auto and not ref_path.exists():
        log.info("[TTS] Gerando áudio de referência com voice design...")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        chunks = _model.generate(
            text=ref_text,
            instruct=instruct,
            duration=_duracao_segura(ref_text, 1.0),
            language=language,
            postprocess_output=False,
        )
        ref_audio = chunks[0]
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(ref_path), ref_audio, 24000)
        log.info("[TTS] Áudio de referência salvo em %s (%.2fs)",
                 ref_path, len(ref_audio) / 24000)

    log.info("[TTS] Criando VoiceClonePrompt a partir de %s...", ref_path)
    _voice_prompt = _model.create_voice_clone_prompt(
        ref_audio=str(ref_path),
        ref_text=ref_text,
    )
    _voice_prompt_key = prompt_key
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
        duration = _duracao_segura(texto, speed)
        chunks = _model.generate(
            text=texto, voice_clone_prompt=_voice_prompt,
            duration=duration, language=language,
            postprocess_output=False,
        )
        audio = chunks[0]
        log.info("[TTS] Áudio gerado: shape=%s dtype=%s duração=%.2fs alvo=%.2fs",
                 audio.shape, audio.dtype, len(audio) / 24000, duration)
        return audio
    except Exception as exc:
        log.error("[TTS] ERRO ao gerar áudio: %s", exc, exc_info=True)
        raise


def regenerar_referencia(instruct: str = "female, teenager, low pitch",
                         language: str = "Portuguese", seed: int = 42) -> None:
    """Apaga o áudio de referência e recria (chamado quando o usuário muda a voz)."""
    limpar_prompt_voz()
    if _REF_AUDIO_PATH.exists():
        _REF_AUDIO_PATH.unlink()
        log.info("[TTS] Áudio de referência anterior removido.")
    _garantir_referencia(instruct, language, seed)


def precarregar_e_aquecer(voz_cfg: dict | None = None) -> None:
    """Carrega OmniVoice, cria prompt de voz e gera uma frase curta para aquecer GPU."""
    voz_cfg = voz_cfg or {}
    instruct = voz_cfg.get("voz_instruct") or montar_instruct(
        voz_cfg.get("voz_age", "teenager"),
        voz_cfg.get("voz_pitch_style", "low pitch"),
    )
    language = voz_cfg.get("voz_language", "Portuguese")
    seed = int(voz_cfg.get("voz_seed", 42))
    carregar()
    _garantir_referencia(instruct, language, seed)
    _ = _model.generate(
        text="Oi.",
        voice_clone_prompt=_voice_prompt,
        duration=_duracao_segura("Oi.", 1.0),
        language=language,
        postprocess_output=False,
    )
    log.info("[TTS] Warmup OmniVoice concluído.")


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

        # Margens/fades para não perder sílabas no início/fim do Discord/Opus.
        start_pad = np.zeros(int(48000 * 0.30), dtype=np.float32)
        end_pad = np.zeros(int(48000 * 2.25), dtype=np.float32)
        audio_48k = np.concatenate([start_pad, audio_48k.astype(np.float32), end_pad])

        fade_in = min(int(48000 * 0.03), len(audio_48k))
        fade_out = min(int(48000 * 0.05), len(audio_48k))
        if fade_in > 1:
            audio_48k[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
        if fade_out > 1:
            audio_48k[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

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
        # são perdidas. Alinhamos ao frame e adicionamos silêncio extra
        # para o Opus processar todo o áudio real antes do EOF.
        frame_size = 3840  # 960 samples × 2 canais × 2 bytes
        resto = len(pcm_bytes) % frame_size
        if resto:
            pcm_bytes += b'\x00' * (frame_size - resto)
        pcm_bytes += b'\x00' * (frame_size * 75)  # 1.5s de silêncio para flush do Opus

        log.info("[TTS] PCM gerado: %d bytes (%.2fs a 48kHz estéreo)",
                 len(pcm_bytes), len(pcm_bytes) / (48000 * 4))
        return pcm_bytes
    except Exception as exc:
        log.error("[TTS] ERRO ao converter PCM: %s", exc, exc_info=True)
        raise
