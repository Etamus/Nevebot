from __future__ import annotations

import argparse
import importlib
import json
import platform
import struct
import subprocess
import sys
import warnings
import wave
from pathlib import Path

from dotenv import dotenv_values


warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CHATTERBOX_BASE_FILES = ("ve.pt",)
CHATTERBOX_PTBR_FILES = (
    "t3_pt_br.safetensors",
    "s3gen_v3.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "Cangjie5_TC.json",
)
WHISPER_ALIASES = {
    "large turbo": "large-v3-turbo",
    "large-turbo": "large-v3-turbo",
    "large_v3_turbo": "large-v3-turbo",
    "turbo": "large-v3-turbo",
}
REQUIRED_MODULES = {
    "discord": "discord.py",
    "discord.ext.voice_recv": "discord-ext-voice-recv",
    "dotenv": "python-dotenv",
    "requests": "requests",
    "pynput": "pynput",
    "webview": "pywebview",
    "PIL": "Pillow",
    "numpy": "numpy",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "sounddevice": "sounddevice",
    "librosa": "librosa",
    "faster_whisper": "faster-whisper",
    "ctranslate2": "ctranslate2",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "chatterbox": "chatterbox-tts",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface-hub",
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.pending: list[str] = []

    @staticmethod
    def ok(message: str) -> None:
        print(f"[OK] {message}")

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"[ERRO] {message}")

    def warning(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[AVISO] {message}")

    def user_action(self, message: str) -> None:
        self.pending.append(message)
        print(f"[PENDENTE] {message}")


def resolve_path(value: object, default: Path) -> Path:
    raw = str(value or "").strip().strip('"')
    path = Path(raw) if raw else default
    return path if path.is_absolute() else BASE_DIR / path


def version_base(value: object) -> str:
    return str(value or "").split("+", 1)[0]


def validate_json(report: Report, path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("a raiz precisa ser um objeto JSON")
        report.ok(f"Configuração válida: {path.relative_to(BASE_DIR)}")
        return value
    except (OSError, ValueError) as exc:
        report.error(f"Configuração inválida em {path}: {exc}")
        return {}


def validate_python(report: Report) -> None:
    version = sys.version_info
    if version[:2] == (3, 11):
        report.ok(f"Python {platform.python_version()} compatível")
    else:
        report.error("Use Python 3.11 de 64 bits.")

    if struct.calcsize("P") * 8 == 64:
        report.ok("Python de 64 bits")
    else:
        report.error("Python de 32 bits não é compatível com os runtimes do projeto.")


def validate_imports(report: Report) -> tuple[object | None, object | None]:
    loaded: dict[str, object] = {}
    failed: list[str] = []
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except Exception as exc:
            failed.append(f"{package_name} ({type(exc).__name__}: {exc})")

    if failed:
        report.error("Dependências que não importaram: " + "; ".join(failed))
    else:
        report.ok("Dependências Python principais importadas")

    torch = loaded.get("torch")
    torchaudio = loaded.get("torchaudio")
    if torch is not None and torchaudio is not None:
        torch_version = version_base(getattr(torch, "__version__", ""))
        audio_version = version_base(getattr(torchaudio, "__version__", ""))
        if torch_version != audio_version:
            report.error(
                f"torch {torch_version} e torchaudio {audio_version} precisam ter a mesma versão."
            )
        else:
            report.ok(f"PyTorch e torchaudio alinhados em {torch_version}")
        cuda_available = bool(torch.cuda.is_available())
        backend = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        report.ok(f"Backend PyTorch disponível: {backend}")
    return torch, torchaudio


def validate_llama(report: Report, env: dict) -> None:
    llama_dir = resolve_path(env.get("LLAMA_CPP_DIR"), BASE_DIR / "llama.cpp")
    server = resolve_path(
        env.get("LLAMA_CPP_SERVER_EXE"), llama_dir / "llama-server.exe"
    )
    if not server.is_file():
        report.error(f"llama-server.exe não encontrado em {server}")
        return
    try:
        result = subprocess.run(
            [str(server), "--version"],
            cwd=str(server.parent),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        report.error(f"llama-server.exe não pôde ser executado: {exc}")
        return
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[:240]
        report.error(f"llama-server.exe retornou código {result.returncode}: {detail}")
        return
    first_line = (result.stdout or result.stderr).strip().splitlines()
    report.ok(f"llama-server executável: {first_line[0] if first_line else server.name}")


def validate_models(report: Report, env: dict, voice_config: dict) -> None:
    chatterbox_base = resolve_path(
        env.get("CHATTERBOX_BASE_DIR"), BASE_DIR / "models" / "chatterbox" / "base"
    )
    chatterbox_ptbr = resolve_path(
        env.get("CHATTERBOX_PTBR_DIR"), BASE_DIR / "models" / "chatterbox" / "pt-br"
    )
    missing = [str(chatterbox_base / name) for name in CHATTERBOX_BASE_FILES if not (chatterbox_base / name).is_file()]
    missing.extend(
        str(chatterbox_ptbr / name)
        for name in CHATTERBOX_PTBR_FILES
        if not (chatterbox_ptbr / name).is_file()
    )
    if missing:
        report.error("Pesos do Chatterbox ausentes: " + ", ".join(missing))
    else:
        report.ok("Pesos do Chatterbox Multilingual V3 PT-BR completos")

    whisper_name = str(voice_config.get("whisper_modelo", "large-v3-turbo") or "large-v3-turbo").strip()
    whisper_name = WHISPER_ALIASES.get(whisper_name.lower(), whisper_name)
    try:
        from faster_whisper.utils import download_model

        location = download_model(
            whisper_name,
            cache_dir=str(BASE_DIR / "models" / "whisper"),
            local_files_only=True,
        )
        report.ok(f"Whisper {whisper_name} disponível em cache ({location})")
    except Exception as exc:
        report.error(f"Whisper {whisper_name} não está preparado: {exc}")


def validate_user_files(report: Report, env: dict, ui_config: dict, voice_config: dict) -> None:
    token = str(env.get("DISCORD_TOKEN", "") or "").strip()
    if not token or token == "SEU_TOKEN_AQUI" or len(token) < 30:
        report.user_action("Preencha DISCORD_TOKEN no arquivo .env.")
    else:
        report.ok("Token do Discord configurado")

    model_candidates: list[Path] = []
    ui_model = ui_config.get("llm", {}).get("model_path") if isinstance(ui_config.get("llm"), dict) else None
    for value in (ui_model, env.get("LLM_MODEL_PATH")):
        if value:
            model_candidates.append(resolve_path(value, BASE_DIR / "models" / "texto"))
    model_candidates.extend((BASE_DIR / "models" / "texto").glob("*.gguf"))
    model_candidates.extend((BASE_DIR / "models").glob("*.gguf"))
    valid_models = [path for path in model_candidates if path.is_file() and path.suffix.lower() == ".gguf"]
    if valid_models:
        report.ok(f"Modelo GGUF disponível: {valid_models[0].name}")
    else:
        report.user_action("Adicione um modelo GGUF em models/texto/.")

    reference = resolve_path(
        voice_config.get("voz_referencia"), BASE_DIR / "data" / "voz_referencia.wav"
    )
    if not reference.is_file():
        report.user_action("Adicione sua referência em data/voz_referencia.wav.")
        return
    try:
        with wave.open(str(reference), "rb") as wav_file:
            rate = wav_file.getframerate()
            duration = wav_file.getnframes() / rate if rate else 0.0
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
        if duration < 1.0:
            report.user_action("voz_referencia.wav precisa ter pelo menos 1 segundo de áudio.")
        elif channels not in {1, 2} or sample_width not in {1, 2, 3, 4}:
            report.user_action("voz_referencia.wav possui um formato PCM não suportado.")
        else:
            report.ok(f"Referência de voz válida ({duration:.1f}s, {channels} canal/canais)")
    except (OSError, wave.Error) as exc:
        report.user_action(f"voz_referencia.wav não é um WAV PCM válido: {exc}")


def validate_project_files(report: Report) -> None:
    required = (
        "nevebot.py",
        "iniciar.bat",
        "web/index.html",
        "web/app.css",
        "web/favicon.png",
        "services/discord_audio_monitor.py",
        "services/discord_transcription.py",
        "services/discord_voice_receive.py",
        "data/config_ui.json",
        "data/voz_config.json",
        "personality_prompt.json",
    )
    missing = [name for name in required if not (BASE_DIR / name).is_file()]
    if missing:
        report.error("Arquivos do projeto ausentes: " + ", ".join(missing))
    else:
        report.ok("Arquivos essenciais do projeto presentes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida a instalação local do Nevebot.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Retorna código 2 quando ainda existem ações obrigatórias do usuário.",
    )
    args = parser.parse_args()

    print("================================================")
    print(" Nevebot - diagnóstico da instalação")
    print("================================================")
    report = Report()
    env_path = BASE_DIR / ".env"
    env = dict(dotenv_values(env_path)) if env_path.is_file() else {}
    if env_path.is_file():
        report.ok("Arquivo .env presente")
    else:
        report.error("Arquivo .env ausente; execute instalar.bat novamente.")

    validate_python(report)
    validate_project_files(report)
    ui_config = validate_json(report, BASE_DIR / "data" / "config_ui.json")
    voice_config = validate_json(report, BASE_DIR / "data" / "voz_config.json")
    validate_imports(report)
    validate_llama(report, env)
    validate_models(report, env, voice_config)
    validate_user_files(report, env, ui_config, voice_config)

    print()
    if report.errors:
        print(f"Instalação incompleta: {len(report.errors)} erro(s) técnico(s).")
        return 1
    if report.pending:
        print("Instalação técnica concluída, mas o Nevebot ainda precisa dos itens marcados como PENDENTE.")
        return 2 if args.strict else 0
    print("Instalação pronta para uso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
