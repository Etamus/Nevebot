"""
services/tts_chatterbox.py - Text-to-Speech local com Chatterbox Multilingual V3 PT-BR.

Usa o pacote dedicado ResembleAI/Chatterbox-Multilingual-pt-br e clona a voz a
partir de data/voz_referencia.wav. A referencia e preparada uma vez e so e
recriada quando o arquivo muda.
"""

from __future__ import annotations

import logging
import math
import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("tts_chatterbox")

_BASE_DIR = Path(__file__).parent.parent
_REF_AUDIO_PATH = _BASE_DIR / "data" / "voz_referencia.wav"

_BASE_REQUIRED = ("ve.pt",)
_PTBR_REQUIRED = (
    "t3_pt_br.safetensors",
    "s3gen_v3.pt",
    "grapheme_mtl_merged_expanded_v1.json",
)

_model: "_ChatterboxPTBR | None" = None
_ref_key: tuple[str, int, int, float] | None = None
_lock = threading.RLock()


@dataclass
class _Conditionals:
    t3: object
    gen: dict

    def to(self, device: str) -> "_Conditionals":
        import torch

        self.t3 = self.t3.to(device=device)
        for key, value in self.gen.items():
            if torch.is_tensor(value):
                self.gen[key] = value.to(device=device)
        return self


class _ChatterboxPTBR:
    """Loader enxuto para o Single Language Pack PT-BR do Chatterbox V3."""

    def __init__(self, t3, s3gen, ve, tokenizer, device: str) -> None:
        from chatterbox.models.s3gen import S3GEN_SR
        import perth

        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds: _Conditionals | None = None
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, base_dir: Path, ptbr_dir: Path, device: str) -> "_ChatterboxPTBR":
        import torch
        from safetensors.torch import load_file as load_safetensors
        from chatterbox.models.t3 import T3
        from chatterbox.models.t3.modules.t3_config import T3Config
        from chatterbox.models.s3gen import S3Gen
        from chatterbox.models.tokenizers import MTLTokenizer
        from chatterbox.models.voice_encoder import VoiceEncoder

        _patch_alignment_analyzer()
        base_dir = Path(base_dir)
        ptbr_dir = Path(ptbr_dir)
        map_location = torch.device("cpu")

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(base_dir / "ve.pt", map_location=map_location, weights_only=True))
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ptbr_dir / "t3_pt_br.safetensors")
        if "model" in t3_state:
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ptbr_dir / "s3gen_v3.pt", map_location=map_location, weights_only=True),
            strict=False,
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(str(ptbr_dir / "grapheme_mtl_merged_expanded_v1.json"))
        return cls(t3, s3gen, ve, tokenizer, device)

    def prepare_conditionals(self, wav_path: Path, exaggeration: float) -> None:
        import librosa
        import torch
        from chatterbox.models.s3gen import S3GEN_SR
        from chatterbox.models.s3tokenizer import S3_SR
        from chatterbox.models.t3.modules.cond_enc import T3Cond

        ref_wav, _ = librosa.load(str(wav_path), sr=S3GEN_SR, mono=True)
        ref_wav = _preparar_audio_referencia(ref_wav, S3GEN_SR)
        if ref_wav.size < S3GEN_SR:
            raise ValueError("data/voz_referencia.wav precisa ter pelo menos 1 segundo de audio.")

        ref_16k = librosa.resample(ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = ref_wav[: 10 * S3GEN_SR]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            prompt_tokens, _ = self.s3gen.tokenizer.forward([ref_16k[: 6 * S3_SR]], max_len=plen)
            prompt_tokens = torch.atleast_2d(prompt_tokens).to(self.device)

        speaker_emb = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR))
        speaker_emb = speaker_emb.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=prompt_tokens,
            emotion_adv=float(exaggeration) * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = _Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text: str,
        *,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        repetition_penalty: float,
        min_p: float,
        top_p: float,
    ):
        import torch
        import torch.nn.functional as F
        from chatterbox.models.s3gen import S3GEN_SR
        from chatterbox.models.s3tokenizer import S3_TOKEN_RATE, drop_invalid_tokens

        if self.conds is None:
            raise RuntimeError("Referencia de voz nao foi preparada.")

        current_exag = float(self.conds.t3.emotion_adv[0, 0, 0].item())
        if not math.isclose(float(exaggeration), current_exag, rel_tol=1e-4, abs_tol=1e-4):
            from chatterbox.models.t3.modules.cond_enc import T3Cond

            cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=cond.speaker_emb,
                cond_prompt_speech_tokens=cond.cond_prompt_speech_tokens,
                emotion_adv=float(exaggeration) * torch.ones(1, 1, 1),
            ).to(device=self.device)

        text = _normalizar_pontuacao(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id="pt").to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        text_tokens = F.pad(text_tokens, (1, 0), value=self.t3.hp.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.t3.hp.stop_text_token)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=_estimar_max_speech_tokens(text),
                temperature=float(temperature),
                cfg_weight=float(cfg_weight),
                repetition_penalty=float(repetition_penalty),
                min_p=float(min_p),
                top_p=float(top_p),
            )[0]
            speech_tokens = drop_invalid_tokens(speech_tokens).to(self.device)
            wav, _ = self.s3gen.inference(speech_tokens=speech_tokens, ref_dict=self.conds.gen)
            wav = wav.squeeze(0).detach().cpu().numpy()

        n_tokens = int(speech_tokens.shape[-1])
        st_len = max(1, n_tokens - 1)
        wav = wav[: st_len * (S3GEN_SR // S3_TOKEN_RATE)]
        watermarked = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return watermarked.astype(np.float32, copy=False)


def _patch_alignment_analyzer() -> None:
    """Corrige parada precoce por repeticao no pacote chatterbox-tts 0.1.7."""
    from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer

    if getattr(AlignmentStreamAnalyzer, "_nevebot_patch", False):
        return

    def step(self, logits, next_token=None):
        import torch

        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            a_chunk = aligned_attn[j:, i:j].clone().cpu()
        else:
            a_chunk = aligned_attn[:, i:j].clone().cpu()

        a_chunk[:, self.curr_frame_pos + 1:] = 0
        self.alignment = torch.cat((self.alignment, a_chunk), dim=0)

        a_mat = self.alignment
        _, text_len = a_mat.shape
        cur_text_pos = a_chunk[-1].argmax()
        discontinuity = not (-4 < cur_text_pos - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_pos

        false_start = (not self.started) and (a_mat[-2:, -2:].max() > 0.1 or a_mat[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = a_mat.shape[0]

        self.complete = self.complete or self.text_position >= text_len - 3
        if self.complete and self.completed_at is None:
            self.completed_at = a_mat.shape[0]

        long_tail = self.complete and (a_mat[self.completed_at:, -3:].sum(dim=0).max() >= 5)
        alignment_repetition = self.complete and (a_mat[self.completed_at:, :-5].max(dim=1).values.sum() > 5)

        if next_token is not None:
            token_id = next_token.item() if isinstance(next_token, torch.Tensor) else int(next_token)
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 12:
                self.generated_tokens = self.generated_tokens[-12:]

        token_repetition = False

        if cur_text_pos < text_len - 3 and text_len > 5:
            logits[..., self.eos_idx] = -2**15

        if long_tail or alignment_repetition or token_repetition:
            log.warning(
                "[TTS] Forcando EOS: long_tail=%s alignment_repetition=%s token_repetition=%s",
                bool(long_tail),
                bool(alignment_repetition),
                bool(token_repetition),
            )
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits

    AlignmentStreamAnalyzer.step = step
    AlignmentStreamAnalyzer._nevebot_patch = True


def _estimar_max_speech_tokens(text: str) -> int:
    chars = max(1, len(text))
    return max(90, min(1000, int(chars * 2.8) + 45))


def _dbfs(valor: float) -> float:
    return 20.0 * math.log10(max(float(valor), 1e-12))


def _preparar_audio_referencia(audio: np.ndarray, sr: int) -> np.ndarray:
    """Deixa a referencia mais previsivel para o prompt de clonagem."""
    x = np.asarray(audio, dtype=np.float32)
    x = np.nan_to_num(x, copy=False)
    if x.size == 0:
        return x

    dur_original = x.size / float(sr)
    x = x - float(np.mean(x))

    frame = max(256, int(sr * 0.04))
    hop = max(128, int(sr * 0.01))
    if x.size >= frame:
        rms_frames = []
        for start in range(0, x.size - frame + 1, hop):
            trecho = x[start : start + frame]
            rms_frames.append(float(np.sqrt(np.mean(trecho * trecho))))
        rms = np.asarray(rms_frames, dtype=np.float32)
        if rms.size:
            ruido = float(np.percentile(rms, 20))
            fala = float(np.percentile(rms, 95))
            limite = max(10 ** (-42 / 20), ruido * 2.5, fala * 0.08)
            ativo = rms > limite
            if ativo.any():
                pad = max(1, int(0.16 / (hop / sr)))
                kernel = np.ones(pad * 2 + 1, dtype=np.int16)
                ativo = np.convolve(ativo.astype(np.int16), kernel, mode="same") > 0
                indices = np.flatnonzero(ativo)
                inicio = max(0, int(indices[0] * hop - sr * 0.08))
                fim = min(x.size, int(indices[-1] * hop + frame + sr * 0.20))
                x = x[inicio:fim]

    max_len = int(sr * 10.0)
    if x.size > max_len:
        x = x[:max_len]

    if x.size:
        rms_final = float(np.sqrt(np.mean(x * x)))
        if rms_final > 0:
            alvo = 10 ** (-20 / 20)
            ganho = max(0.5, min(4.0, alvo / rms_final))
            x = x * ganho
        peak = float(np.max(np.abs(x)))
        if peak > 0.92:
            x = x * (0.92 / peak)

        fade = min(int(sr * 0.02), x.size // 2)
        if fade > 1:
            curva = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            x[:fade] *= curva
            x[-fade:] *= curva[::-1]

    peak_final = float(np.max(np.abs(x))) if x.size else 0.0
    rms_final = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    log.info(
        "[TTS] Referencia normalizada: %.2fs -> %.2fs, peak=%.1f dBFS, rms=%.1f dBFS",
        dur_original,
        x.size / float(sr),
        _dbfs(peak_final),
        _dbfs(rms_final),
    )
    return x.astype(np.float32, copy=False)


def baixar_modelos() -> None:
    """Baixa os pesos necessarios para models/chatterbox."""
    from huggingface_hub import hf_hub_download, snapshot_download

    config.CHATTERBOX_BASE_DIR.mkdir(parents=True, exist_ok=True)
    config.CHATTERBOX_PTBR_DIR.mkdir(parents=True, exist_ok=True)

    log.info("[TTS] Baixando assets base do Chatterbox...")
    snapshot_download(
        repo_id="ResembleAI/chatterbox",
        repo_type="model",
        revision="main",
        allow_patterns=list(_BASE_REQUIRED),
        local_dir=str(config.CHATTERBOX_BASE_DIR),
    )

    log.info("[TTS] Baixando Chatterbox Multilingual V3 PT-BR...")
    snapshot_download(
        repo_id="ResembleAI/Chatterbox-Multilingual-pt-br",
        repo_type="model",
        revision="main",
        allow_patterns=list(_PTBR_REQUIRED),
        local_dir=str(config.CHATTERBOX_PTBR_DIR),
    )
    hf_hub_download(
        repo_id="ResembleAI/chatterbox",
        repo_type="model",
        filename="Cangjie5_TC.json",
        cache_dir=str(config.CHATTERBOX_PTBR_DIR),
    )


def _verificar_modelos() -> None:
    missing = []
    for name in _BASE_REQUIRED:
        if not (config.CHATTERBOX_BASE_DIR / name).exists():
            missing.append(str(config.CHATTERBOX_BASE_DIR / name))
    for name in _PTBR_REQUIRED:
        if not (config.CHATTERBOX_PTBR_DIR / name).exists():
            missing.append(str(config.CHATTERBOX_PTBR_DIR / name))
    if missing:
        raise FileNotFoundError(
            "Pesos do Chatterbox PT-BR ausentes. Rode scripts/preparar_chatterbox_ptbr.py. "
            + "Faltando: "
            + ", ".join(missing)
        )


def carregar(device: str | None = None) -> None:
    """Carrega o Chatterbox PT-BR de forma lazy e thread-safe."""
    global _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import torch

        _verificar_modelos()
        chosen_device = (device or config.CHATTERBOX_DEVICE or "cuda").strip().lower()
        if chosen_device.startswith("cuda") and not torch.cuda.is_available():
            chosen_device = "cpu"
        log.info("[TTS] Carregando Chatterbox PT-BR V3 em %s...", chosen_device)
        if torch.cuda.is_available():
            log.info("[TTS] CUDA: %s", torch.cuda.get_device_name(0))
        _model = _ChatterboxPTBR.from_local(
            config.CHATTERBOX_BASE_DIR,
            config.CHATTERBOX_PTBR_DIR,
            chosen_device,
        )
        log.info("[TTS] Chatterbox PT-BR carregado.")


def limpar_cache_referencia() -> None:
    global _ref_key
    with _lock:
        _ref_key = None
        if _model is not None:
            _model.conds = None


def _referencia_key(exaggeration: float) -> tuple[str, int, int, float]:
    if not _REF_AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Referencia de voz nao encontrada: {_REF_AUDIO_PATH}. "
            "Coloque um WAV em data/voz_referencia.wav."
        )
    stat = _REF_AUDIO_PATH.stat()
    return (str(_REF_AUDIO_PATH.resolve()), stat.st_mtime_ns, stat.st_size, round(float(exaggeration), 4))


def _garantir_referencia(exaggeration: float) -> None:
    global _ref_key
    carregar()
    key = _referencia_key(exaggeration)
    if _ref_key == key and _model is not None and _model.conds is not None:
        return
    with _lock:
        if _ref_key == key and _model is not None and _model.conds is not None:
            return
        log.info("[TTS] Preparando clone por referencia: %s", _REF_AUDIO_PATH)
        _model.prepare_conditionals(_REF_AUDIO_PATH, exaggeration=float(exaggeration))
        _ref_key = key


def _set_seed(seed: int) -> None:
    import torch

    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _normalizar_pontuacao(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "Oi."
    replacements = {
        "...": ", ",
        "…": ", ",
        ":": ",",
        ";": ",",
        " - ": ", ",
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?,-":
        text += "."
    return text


def _quebrar_texto(texto: str, max_chars: int) -> list[str]:
    texto = " ".join((texto or "").split())
    if len(texto) <= max_chars:
        return [texto]
    partes: list[str] = []
    sentencas = re.split(r"(?<=[.!?])\s+", texto)
    atual = ""
    for sentenca in sentencas:
        if not sentenca:
            continue
        if len(sentenca) > max_chars:
            palavras = sentenca.split()
            for palavra in palavras:
                candidato = f"{atual} {palavra}".strip()
                if len(candidato) > max_chars and atual:
                    partes.append(atual)
                    atual = palavra
                else:
                    atual = candidato
            continue
        candidato = f"{atual} {sentenca}".strip()
        if len(candidato) > max_chars and atual:
            partes.append(atual)
            atual = sentenca
        else:
            atual = candidato
    if atual:
        partes.append(atual)
    return partes or [texto[:max_chars]]


def gerar(
    texto: str,
    *,
    speed: float = 1.0,
    seed: int = 42,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
    min_p: float = 0.05,
    top_p: float = 0.95,
) -> np.ndarray:
    """Gera audio mono float32 no sample-rate nativo do Chatterbox."""
    texto = " ".join((texto or "").split())
    if not texto:
        return np.zeros(0, dtype=np.float32)

    speed = max(0.85, min(float(speed or 1.0), 1.2))
    exaggeration = max(0.25, min(float(exaggeration or 0.5), 0.95))
    cfg_weight = max(0.2, min(float(cfg_weight or 0.5), 1.0))
    temperature = max(0.05, min(float(temperature or 0.8), 2.0))
    repetition_penalty = max(1.0, min(float(repetition_penalty or 1.2), 2.0))

    _set_seed(int(seed or 0))
    _garantir_referencia(exaggeration)

    chunks = _quebrar_texto(texto, int(config.CHATTERBOX_MAX_CHARS))
    audios: list[np.ndarray] = []
    silence = np.zeros(int(_model.sr * 0.18), dtype=np.float32)
    log.info(
        "[TTS] Gerando Chatterbox PT-BR: chunks=%d speed=%.2f exag=%.2f cfg=%.2f temp=%.2f",
        len(chunks),
        speed,
        exaggeration,
        cfg_weight,
        temperature,
    )
    for i, chunk in enumerate(chunks, start=1):
        log.info("[TTS] Chunk %d/%d: %r", i, len(chunks), chunk[:90])
        wav = _model.generate(
            chunk,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        audios.append(wav)
        if i != len(chunks):
            audios.append(silence)

    audio = np.concatenate(audios).astype(np.float32, copy=False)
    if not math.isclose(speed, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        import librosa

        audio = librosa.effects.time_stretch(audio, rate=speed).astype(np.float32, copy=False)
    return audio


def precarregar_e_aquecer(voz_cfg: dict | None = None) -> None:
    voz_cfg = voz_cfg or {}
    carregar()
    _garantir_referencia(float(voz_cfg.get("voz_exaggeration", 0.5)))
    audio = gerar(
        "Oi, tudo bem.",
        speed=float(voz_cfg.get("velocidade", 1.0)),
        seed=int(voz_cfg.get("voz_seed", 42)),
        exaggeration=float(voz_cfg.get("voz_exaggeration", 0.5)),
        cfg_weight=float(voz_cfg.get("voz_cfg_weight", 0.5)),
        temperature=float(voz_cfg.get("voz_temperature", 0.8)),
    )
    _ = para_pcm_discord(
        audio,
        volume=float(voz_cfg.get("volume", 1.0)),
        pitch_semitones=float(voz_cfg.get("pitch", 0.0)),
    )
    log.info("[TTS] Warmup Chatterbox PT-BR concluido.")


def para_pcm_discord(
    audio: np.ndarray,
    volume: float = 1.0,
    pitch_semitones: float = 0.0,
    *,
    start_pad_s: float = 0.18,
    end_pad_s: float = 1.2,
    tail_frames: int = 60,
) -> bytes:
    """Converte audio mono float32 para PCM 48 kHz stereo 16-bit do Discord."""
    if audio is None or len(audio) == 0:
        return b""
    log.info(
        "[TTS] Convertendo para PCM Discord: samples=%d sr=%d volume=%.2f pitch=%.1f",
        len(audio),
        _model.sr if _model is not None else 24000,
        volume,
        pitch_semitones,
    )
    try:
        import torch
        import torchaudio.functional as F

        src_sr = int(_model.sr if _model is not None else 24000)
        tensor = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)

        if pitch_semitones != 0.0:
            ratio = 2.0 ** (float(pitch_semitones) / 12.0)
            intermediate_rate = max(8000, int(round(src_sr * ratio)))
            tensor = F.resample(tensor, src_sr, intermediate_rate)
            audio_48k = F.resample(tensor, intermediate_rate, 48000).squeeze(0).numpy()
        else:
            audio_48k = F.resample(tensor, src_sr, 48000).squeeze(0).numpy()

        start_pad = np.zeros(int(48000 * max(0.0, float(start_pad_s))), dtype=np.float32)
        end_pad = np.zeros(int(48000 * max(0.0, float(end_pad_s))), dtype=np.float32)
        audio_48k = np.concatenate([start_pad, audio_48k.astype(np.float32), end_pad])

        fade_in = min(int(48000 * 0.02), len(audio_48k))
        fade_out = min(int(48000 * 0.05), len(audio_48k))
        if fade_in > 1:
            audio_48k[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
        if fade_out > 1:
            audio_48k[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

        peak = float(np.abs(audio_48k).max()) if audio_48k.size else 0.0
        if peak > 0:
            audio_48k = audio_48k / peak * 0.95 * min(max(float(volume), 0.0), 2.0)

        pcm16 = (audio_48k * 32767).clip(-32768, 32767).astype(np.int16)
        stereo = np.column_stack([pcm16, pcm16])
        pcm_bytes = stereo.tobytes()

        frame_size = 3840
        resto = len(pcm_bytes) % frame_size
        if resto:
            pcm_bytes += b"\x00" * (frame_size - resto)
        pcm_bytes += b"\x00" * (frame_size * max(0, int(tail_frames)))
        return pcm_bytes
    except Exception:
        log.exception("[TTS] ERRO ao converter PCM")
        raise
