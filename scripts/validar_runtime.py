from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import sys
import warnings
from importlib import metadata
from pathlib import Path


REQUIRED_MODULES = (
    "discord",
    "discord.ext.voice_recv",
    "davey",
    "dotenv",
    "requests",
    "pynput",
    "webview",
    "PIL",
    "numpy",
    "scipy",
    "soundfile",
    "sounddevice",
    "librosa",
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "torch",
    "torchaudio",
    "chatterbox",
    "safetensors",
    "huggingface_hub",
    "perth",
    "s3tokenizer",
    "diffusers",
    "conformer",
    "spacy_pkuseg",
    "pykakasi",
    "pyloudnorm",
    "omegaconf",
    "resampy",
    "silero_vad",
    "einops",
    "antlr4",
    "jaconv",
    "onnx",
    "ml_dtypes",
    "transformers",
    "deprecated",
    "srsly",
    "catalogue",
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
CHATTERBOX_VERSION = "0.1.7"
CHATTERBOX_IMPORTS = (
    "chatterbox.models.t3",
    "chatterbox.models.s3gen",
    "chatterbox.models.tokenizers",
    "chatterbox.models.voice_encoder",
)


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _required_distributions() -> dict[str, str]:
    requirements_path = BASE_DIR / "requirements.txt"
    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    result: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)$")
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        match = pattern.fullmatch(line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


def _quick_check(*, chatterbox_only: bool) -> list[str]:
    modules = ("chatterbox",) if chatterbox_only else REQUIRED_MODULES
    errors: list[str] = []
    for module in modules:
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, AttributeError, ValueError) as exc:
            errors.append(f"{module}: {type(exc).__name__}: {exc}")
            continue
        if not available:
            errors.append(f"modulo ausente: {module}")

    chatterbox_version = _distribution_version("chatterbox-tts")
    if chatterbox_version != CHATTERBOX_VERSION:
        errors.append(
            "chatterbox-tts ausente ou em versao incorreta "
            f"(esperado {CHATTERBOX_VERSION}, encontrado {chatterbox_version or 'nenhum'})"
        )

    if not chatterbox_only:
        for distribution, expected in _required_distributions().items():
            installed = _distribution_version(distribution)
            if installed is None:
                errors.append(f"distribuicao ausente: {distribution}")
            elif installed.split("+", 1)[0] != expected.split("+", 1)[0]:
                errors.append(
                    f"{distribution} em versao incorreta "
                    f"(esperado {expected}, encontrado {installed})"
                )
        torch_version = _distribution_version("torch")
        torchaudio_version = _distribution_version("torchaudio")
        if not torch_version:
            errors.append("distribuicao ausente: torch")
        if not torchaudio_version:
            errors.append("distribuicao ausente: torchaudio")
        if torch_version and torchaudio_version:
            torch_base = torch_version.split("+", 1)[0]
            torchaudio_base = torchaudio_version.split("+", 1)[0]
            if torch_base != torchaudio_base:
                errors.append(
                    f"torch {torch_version} e torchaudio {torchaudio_version} nao estao alinhados"
                )
    return errors


def _deep_check(*, chatterbox_only: bool) -> list[str]:
    errors = _quick_check(chatterbox_only=chatterbox_only)
    modules = CHATTERBOX_IMPORTS if chatterbox_only else (
        "torch",
        "torchaudio",
        "faster_whisper",
        *CHATTERBOX_IMPORTS,
    )
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"{module}: {type(exc).__name__}: {exc}")
    if not chatterbox_only:
        try:
            from discord.ext import voice_recv
            from discord.ext.voice_recv import video
            from services.discord_voice_receive import aplicar_compatibilidade_voice_recv

            aplicar_compatibilidade_voice_recv()
            if not getattr(video.VideoStreamInfo.__init__, "_nevebot_compat", False):
                errors.append("compatibilidade de VideoStreamInfo nao foi aplicada")
            if not getattr(voice_recv.VoiceRecvClient._remove_ssrc, "_nevebot_compat", False):
                errors.append("compatibilidade de VoiceRecvClient nao foi aplicada")
        except Exception as exc:
            errors.append(f"compatibilidade voice-recv: {type(exc).__name__}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida o runtime Python do Nevebot.")
    parser.add_argument("--deep", action="store_true", help="Importa os runtimes pesados.")
    parser.add_argument(
        "--chatterbox-only",
        action="store_true",
        help="Valida somente o pacote do Chatterbox.",
    )
    args = parser.parse_args()

    errors = (
        _deep_check(chatterbox_only=args.chatterbox_only)
        if args.deep
        else _quick_check(chatterbox_only=args.chatterbox_only)
    )
    if errors:
        for error in dict.fromkeys(errors):
            print(f"[ERRO] {error}")
        return 1
    print("[OK] Runtime Python pronto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
