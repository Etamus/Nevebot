"""
config_loader.py — Carrega e persiste configurações editáveis via UI web.
Thread-safe: pode ser lido/escrito do bot (asyncio) e do servidor web (thread).
"""

import json
import threading
from pathlib import Path

_CONFIG_PATH = Path("data/config_ui.json")
_lock = threading.Lock()

# ── Config padrão ─────────────────────────────────────────────────────────────
_DEFAULT: dict = {
    "prefix": "!",
    "prompts": {
        "lou": "",
        "lou_voz": "",
    },
    "llm": {
        "model_path": None,
        "n_ctx": None,
        "max_tokens": None,
        "n_gpu_layers": None,
        "n_batch": None,
        "n_ubatch": None,
        "n_threads": None,
        "n_threads_batch": None,
        "kv_type": None,
        "chat_template": None,
        "voz_max_tokens": None,
        "voz_temperature": None,
        "temperature": None,
        "min_p": None,
        "top_p": None,
        "top_k": None,
        "dry_multiplier": None,
        "dry_allowed_length": None,
        "repeat_penalty": None,
        "frequency_penalty": None,
        "presence_penalty": None,
    },
    "commands": {
        "lou": {
            "name": "ligar",
            "descricao": "Ativa a Neve no canal.",
            "messages": {
                "ja_ativo": "Já estou ligada neste canal.",
                "ativado": "Neve ligada neste canal."
            }
        },
        "limpar": {
            "name": "limpar",
            "descricao": "Apaga o histórico de conversa do canal.",
            "messages": {
                "apagado": "Histórico apagado."
            }
        },
        "desligar": {
            "name": "desligar",
            "descricao": "Desativa a Neve no canal.",
            "messages": {
                "ja_desligado": "Já estou desligada neste canal.",
                "desligado": "Desligada neste canal."
            }
        },
        "limitar": {
            "name": "bloquear",
            "descricao": "[Apenas dono] Bloqueia um usuário de receber respostas.",
            "messages": {
                "sem_mencao": "Mencione um usuário. Ex: `!bloquear @alguem`",
                "auto_bloqueio": "Você não pode se bloquear.",
                "bloquear_bot": "Não posso me bloquear.",
                "bloqueado": "**{nome}** não receberá mais respostas minhas."
            }
        },
        "desbloquear": {
            "name": "desbloquear",
            "descricao": "[Apenas dono] Remove o bloqueio de um usuário.",
            "messages": {
                "sem_mencao": "Mencione um usuário. Ex: `!desbloquear @alguem`",
                "desbloqueado": "**{nome}** voltou a receber respostas.",
                "nao_bloqueado": "**{nome}** não estava bloqueado."
            }
        }
    }
}


class BotConfig:
    """Configuração mutável do bot, persistida em JSON."""

    def __init__(self) -> None:
        self._data: dict = {}
        self.reload()

    # ── Persistência ──────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Carrega (ou cria) o arquivo de config."""
        with _lock:
            if _CONFIG_PATH.exists():
                try:
                    with open(_CONFIG_PATH, encoding="utf-8") as f:
                        loaded = json.load(f)
                    # Garante que chaves novas do padrão existam
                    self._data = _merge_defaults(loaded, _DEFAULT)
                except (json.JSONDecodeError, KeyError):
                    self._data = _deep_copy(_DEFAULT)
            else:
                self._data = _deep_copy(_DEFAULT)
            self._save_locked()

    def save(self, new_data: dict | None = None) -> None:
        """Salva uma nova config (ou re-salva a atual)."""
        with _lock:
            if new_data is not None:
                self._data = new_data
            self._save_locked()

    def _save_locked(self) -> None:
        """Salva sem adquirir o lock (já deve estar adquirido)."""
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── Leitura ───────────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        with _lock:
            return _deep_copy(self._data)

    def prefix(self) -> str:
        with _lock:
            return self._data.get("prefix", "!")

    def cmd_name(self, cmd_key: str) -> str:
        """Retorna o nome atual de um comando pelo seu identificador interno."""
        with _lock:
            return self._data["commands"][cmd_key]["name"]

    def msg(self, cmd_key: str, msg_key: str) -> str:
        """Retorna o texto de uma mensagem configurável."""
        with _lock:
            return self._data["commands"][cmd_key]["messages"][msg_key]

    def prompt(self, prompt_key: str, fallback: str) -> str:
        """Retorna um system prompt configurado via UI, ou fallback se estiver vazio."""
        with _lock:
            valor = self._data.get("prompts", {}).get(prompt_key, "")
        if isinstance(valor, str) and valor.strip():
            return valor
        return fallback

    # ── Utilitários ───────────────────────────────────────────────────────────

    def original_names(self) -> dict[str, str]:
        """Retorna mapa {chave_interna: nome_atual} de todos os comandos."""
        with _lock:
            return {k: v["name"] for k, v in self._data["commands"].items()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deep_copy(d: dict) -> dict:
    return json.loads(json.dumps(d))


def _merge_defaults(loaded: dict, default: dict) -> dict:
    """Adiciona chaves presentes no padrão que estejam faltando no carregado."""
    result = _deep_copy(default)
    result.update({k: v for k, v in loaded.items() if k not in ("commands", "prompts", "llm")})
    if "prompts" in loaded:
        for prompt_key in result["prompts"]:
            if prompt_key in loaded.get("prompts", {}):
                result["prompts"][prompt_key] = loaded["prompts"][prompt_key]
    if "llm" in loaded:
        for llm_key in result["llm"]:
            if llm_key in loaded.get("llm", {}):
                result["llm"][llm_key] = loaded["llm"][llm_key]
    if "commands" in loaded:
        for cmd_key, cmd_val in loaded["commands"].items():
            if cmd_key in result["commands"]:
                command_name = cmd_val.get("name", result["commands"][cmd_key]["name"])
                if cmd_key == "lou" and command_name == "casual":
                    command_name = "ligar"
                result["commands"][cmd_key]["name"] = command_name
                msgs = result["commands"][cmd_key]["messages"]
                msgs.update(cmd_val.get("messages", {}))
    return result


# Instância global — importada pelo cog e pelo servidor web
cfg = BotConfig()
