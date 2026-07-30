"""
web_server.py — Servidor HTTP embarcado para configuração e chat de voz via UI web.

Rota                    Método  Descrição
/                       GET     Serve web/index.html
/api/config             GET     Retorna a config atual como JSON
/api/config             POST    Salva nova config; aplica mudanças ao bot em tempo real
/api/guilds             GET     Lista guilds do bot
/api/guilds/remover     POST    Remove o bot de uma guild
/api/voz/canais         GET     Lista canais de voz de uma guild
/api/voz/conectar       POST    Conecta bot a um canal de voz
/api/voz/desconectar    POST    Desconecta bot do canal de voz
/api/voz/chat           POST    Recebe WAV, transcreve, gera resposta LLM, fala no Discord
/api/voz/falar          POST    Recebe texto, gera TTS e fala no Discord
/api/voz/config         GET     Retorna config de voz
/api/voz/config         POST    Salva config de voz
/api/voz/limpar         POST    Limpa histórico do chat de voz

Inicie com:  start(bot, host="127.0.0.1", port=5000)
"""

import asyncio
import json
import logging
import re
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config

log = logging.getLogger("web_server")

_WEB_DIR = Path(__file__).parent / "web"
_SHUTDOWN_FLAG = Path(__file__).parent / "logs" / "ui_shutdown.flag"
_bot_ref = None        # discord.ext.commands.Bot
_loop_ref = None       # asyncio event loop do bot
_voz_connect_locks: dict[int, asyncio.Lock] = {}
_voz_connect_tasks: dict[int, asyncio.Task] = {}
_guilds_removidas_ui: set[int] = set()
_tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voz-tts")
_TTS_STARTUP_ESTIMATE_MS = 1800
_tts_state_lock = threading.Lock()
_tts_session_id = 0
_tts_futures: set[Future] = set()

# ── Histórico do chat de voz (via web) ────────────────────────────────────────
_voz_historico: deque = deque(maxlen=4)
_voz_lock = threading.Lock()

# ── Push-to-Talk global (funciona fora do navegador) ─────────────────────────
_ptt_global_pressionado = False


def _voz_connect_lock(guild_id: int) -> asyncio.Lock:
    lock = _voz_connect_locks.get(guild_id)
    if lock is None:
        lock = asyncio.Lock()
        _voz_connect_locks[guild_id] = lock
    return lock


async def _cancelar_tentativa_conexao_voz(guild_id: int, motivo: str = "") -> None:
    task = _voz_connect_tasks.get(guild_id)
    if task is None or task.done() or task is asyncio.current_task():
        return
    log.warning(
        "[WEB] Cancelando tentativa de conexão de voz%s na guild %s.",
        f" ({motivo})" if motivo else "",
        guild_id,
    )
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=3)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _limpar_conexao_voz(guild_id: int, motivo: str = "") -> dict:
    """Força a remoção do VoiceClient, inclusive quando handshake fica preso."""
    if _bot_ref is None:
        raise RuntimeError("Bot indisponivel")
    guild = _bot_ref.get_guild(guild_id)
    if not guild:
        raise ValueError("Guild não encontrada")

    vc = guild.voice_client
    if vc is None:
        return {"ok": True, "status": "sem_conexao"}

    canal = getattr(getattr(vc, "channel", None), "name", None)
    log.warning(
        "[WEB] Limpando conexao de voz%s: guild=%s canal=%s conectado=%s",
        f" ({motivo})" if motivo else "",
        guild.name,
        canal,
        vc.is_connected(),
    )
    try:
        if vc.is_playing():
            vc.stop()
    except Exception:
        pass
    try:
        await asyncio.wait_for(vc.disconnect(force=True), timeout=8)
    except Exception as exc:
        log.warning("[WEB] Falha ao limpar VoiceClient da guild %s: %s", guild.name, exc)
    try:
        if guild.voice_client is vc:
            vc.cleanup()
    except Exception as exc:
        log.warning("[WEB] Falha ao executar cleanup() do VoiceClient da guild %s: %s", guild.name, exc)
    try:
        if guild.voice_client is not None:
            _bot_ref._connection._remove_voice_client(guild_id)
    except Exception as exc:
        log.warning("[WEB] Falha ao remover VoiceClient do cache da guild %s: %s", guild.name, exc)
    await asyncio.sleep(0.35)
    return {"ok": True, "status": "removido", "canal": canal}


def _agendar_reproducao_pcm(
    guild_id: int,
    pcm: bytes,
    origem: str = "",
    *,
    interromper: bool = False,
) -> bool:
    """Agenda reproducao no Discord sem bloquear a resposta HTTP."""
    if _loop_ref is None or _bot_ref is None:
        log.warning("[WEB] Nao foi possivel agendar audio%s: bot/loop indisponivel.", origem)
        return False
    try:
        from cogs.voice_cog import reproduzir_pcm

        future = asyncio.run_coroutine_threadsafe(
            reproduzir_pcm(_bot_ref, guild_id, pcm, interromper=interromper),
            _loop_ref,
        )

        def _registrar_resultado(fut) -> None:
            try:
                fut.result()
            except Exception as exc:
                log.error("[WEB] ERRO ao reproduzir TTS%s: %s", origem, exc, exc_info=True)
            else:
                log.info("[WEB] Reproducao no Discord concluida%s.", origem)

        future.add_done_callback(_registrar_resultado)
        return True
    except Exception as exc:
        log.error("[WEB] ERRO ao agendar TTS%s: %s", origem, exc, exc_info=True)
        return False


def _iniciar_sessao_tts(origem: str) -> int:
    global _tts_session_id
    cancelados = 0
    with _tts_state_lock:
        _tts_session_id += 1
        sessao = _tts_session_id
        for futuro in list(_tts_futures):
            if futuro.done():
                _tts_futures.discard(futuro)
            elif futuro.cancel():
                cancelados += 1
                _tts_futures.discard(futuro)
    log.info("[WEB] Nova sessao TTS %d%s; chunks pendentes cancelados=%d", sessao, f" {origem}" if origem else "", cancelados)
    return sessao


def _sessao_tts_atual(sessao: int) -> bool:
    with _tts_state_lock:
        return sessao == _tts_session_id


def _duracao_pcm_ms(pcm: bytes) -> int:
    # PCM bruto do Discord: 48 kHz, stereo, 16-bit = 192000 bytes/s.
    return max(0, int((len(pcm) / 192000) * 1000))


_ABREVIACOES_TTS = {
    "sr.",
    "sra.",
    "dr.",
    "dra.",
    "etc.",
    "ex.",
    "obs.",
}
_PALAVRAS_LIGACAO_TTS = {"que", "de", "do", "da", "para", "pra", "com", "por", "se", "e", "ou"}
_INICIO_FRASE_SEM_PONTO = re.compile(
    r"\s+(?=(?:Eu|Voc(?:e|ê)s?|Tudo|Isso|Mas|Ent(?:ao|ão)|S(?:o|ó)|A(?:i|í)|Agora|Talvez|N(?:ao|ão)|Sim|T(?:a|á)|Claro|Calma)\b)",
)


def _corrigir_mojibake(texto: str) -> str:
    if not texto or not any(marcador in texto for marcador in ("Ã", "Â", "â€", "â€œ", "â€™")):
        return texto
    try:
        return texto.encode("latin1").decode("utf-8")
    except UnicodeError:
        reparos = {
            "Ã¡": "á", "Ã ": "à", "Ã¢": "â", "Ã£": "ã", "Ã©": "é", "Ãª": "ê",
            "Ã­": "í", "Ã³": "ó", "Ã´": "ô", "Ãµ": "õ", "Ãº": "ú", "Ã§": "ç",
            "Ã": "Á", "Ã€": "À", "Ã‚": "Â", "Ãƒ": "Ã", "Ã‰": "É", "ÃŠ": "Ê",
            "Ã": "Í", "Ã“": "Ó", "Ã”": "Ô", "Ã•": "Õ", "Ãš": "Ú", "Ã‡": "Ç",
            "Â¿": "¿", "Â¡": "¡", "Âº": "º", "Âª": "ª",
            "â€“": "-", "â€”": "-", "â€¦": "...", "â€œ": '"', "â€": '"',
            "â€˜": "'", "â€™": "'",
        }
        for errado, certo in reparos.items():
            texto = texto.replace(errado, certo)
        return texto


def _normalizar_frase_tts(texto: str) -> str:
    texto = _corrigir_mojibake(texto)
    texto = " ".join((texto or "").split())
    texto = re.sub(r"<\|[^|]+\|>", "", texto)
    texto = re.sub(r"</?[a-zA-Z][^>]*/?>", "", texto)
    texto = re.sub(r"^\[[^\]]{1,50}\]\s*:?\s*", "", texto).strip()
    texto = re.sub(r"\*[^*]+\*", "", texto)
    texto = re.sub(r"(?<![\w])_([^_]+)_(?![\w])", "", texto)
    texto = texto.strip().strip("\"'")
    if not texto:
        return ""
    if texto[-1] in ",;:-":
        texto = texto[:-1].rstrip()
    if texto and texto[-1] not in ".!?":
        texto += "."
    return texto


def _fim_frase_pronta(texto: str) -> int | None:
    for i, char in enumerate(texto):
        if char not in ".!?":
            continue
        if char == "." and i > 0 and i + 1 < len(texto) and texto[i - 1].isdigit() and texto[i + 1].isdigit():
            continue
        fim = i + 1
        while fim < len(texto) and texto[fim] in ".!?":
            fim += 1
        while fim < len(texto) and texto[fim] in "\"')]}":
            fim += 1
        ultima = texto[:fim].strip().lower().split()
        if ultima and ultima[-1] in _ABREVIACOES_TTS:
            continue
        if fim == len(texto) or texto[fim].isspace():
            return fim
    return None


def _corte_frase_longa(texto: str, limite: int = 135, minimo: int = 60) -> int | None:
    candidatos = [m.start() for m in _INICIO_FRASE_SEM_PONTO.finditer(texto)]
    for pos in candidatos:
        prefixo = texto[:pos].rstrip().lower().split()
        ultima = prefixo[-1] if prefixo else ""
        if max(10, minimo - 20) <= pos <= limite and ultima not in _PALAVRAS_LIGACAO_TTS:
            return pos
    if len(texto) < limite:
        return None
    for marcador in (",", ";", ":"):
        pos = texto.rfind(marcador, minimo, limite)
        if pos >= minimo:
            return pos + 1
    trecho = texto[:limite]
    for match in reversed(list(re.finditer(r"\s+", trecho))):
        pos = match.start()
        if pos < minimo:
            break
        prefixo = texto[:pos].rstrip().lower().split()
        ultima = prefixo[-1] if prefixo else ""
        if ultima not in _PALAVRAS_LIGACAO_TTS:
            return pos
    return None


def _extrair_frases_tts(
    buffer: str,
    *,
    force: bool = False,
    limite_corte: int = 135,
    minimo_corte: int = 60,
) -> tuple[list[str], str]:
    texto = _corrigir_mojibake(buffer or "").replace("\r", " ").replace("\n", " ").lstrip()
    frases: list[str] = []
    while texto:
        fim_pontuacao = _fim_frase_pronta(texto)
        fim_corte = _corte_frase_longa(texto, limite=limite_corte, minimo=minimo_corte)
        if fim_corte is not None and (fim_pontuacao is None or fim_corte < fim_pontuacao):
            fim = fim_corte
        else:
            fim = fim_pontuacao
        if fim is None:
            break
        frase = _normalizar_frase_tts(texto[:fim])
        if frase:
            frases.append(frase)
        texto = texto[fim:].lstrip()
    if force and texto.strip():
        frase = _normalizar_frase_tts(texto)
        if frase:
            frases.append(frase)
        texto = ""
    return frases, texto


def _normalizar_resposta_voz_texto(texto: str) -> str:
    frases, resto = _extrair_frases_tts(_corrigir_mojibake(texto), force=True)
    if frases:
        return " ".join(frases)
    return _normalizar_frase_tts(resto or texto)


# ── Aplicação de mudanças ao bot em tempo real ────────────────────────────────

async def _aplicar_mudancas(nova_config: dict, config_antiga: dict) -> None:
    """Aplica prefix e renomes de comandos ao bot sem reiniciar."""
    from config_loader import cfg as _cfg

    bot = _bot_ref
    if bot is None:
        return

    # 1. Prefixo
    novo_prefix = nova_config.get("prefix", "!")
    if bot.command_prefix != novo_prefix:
        bot.command_prefix = novo_prefix
        log.info("Prefixo atualizado: '%s'", novo_prefix)

    # 2. Nomes de comandos
    # Garante que cfg._data está em sincronia (já foi salvo antes desta chamada)
    cmds_novos = nova_config.get("commands", {})
    cmds_antigos = config_antiga.get("commands", {})
    for cmd_key, cmd_data in cmds_novos.items():
        novo_nome = cmd_data.get("name", cmd_key)
        nome_atual = cmds_antigos.get(cmd_key, {}).get("name", cmd_key)
        if novo_nome != nome_atual:
            # Tenta pelo nome atual salvo; fallback: pelo nome original do decorador (cmd_key)
            cmd = bot.get_command(nome_atual) or bot.get_command(cmd_key)
            if cmd is not None:
                bot.remove_command(cmd.name)
                cmd.name = novo_nome
                bot.add_command(cmd)
                log.info("Comando renomeado ao vivo: '%s' → '%s'", nome_atual, novo_nome)
            else:
                log.warning("Comando não encontrado para renomear: key='%s' nome_atual='%s'",
                            cmd_key, nome_atual)


# ── Handler HTTP ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # silencia log padrão do http.server
        log.debug("HTTP %s %s", self.command, self.path)

    def do_OPTIONS(self):  # suporte a CORS preflight do navegador
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file(_WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/logo.png":
            self._serve_file(_WEB_DIR / "logo.png", "image/png")
        elif self.path == "/api/config":
            from config_loader import cfg
            data = cfg.as_dict()
            try:
                from cogs.llm_cog import prompt_defaults
                data["_prompt_defaults"] = prompt_defaults()
            except Exception as exc:
                log.warning("Falha ao anexar prompts padrao na config: %s", exc)
                data["_prompt_defaults"] = {}
            payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self._respond(200, "application/json", payload)
        elif self.path.startswith("/api/voz/canais"):
            self._handle_get_voz_canais()
        elif self.path.startswith("/api/texto/canais"):
            self._handle_get_texto_canais()
        elif self.path == "/api/voz/config":
            self._handle_get_voz_config()
        elif self.path == "/api/voz/ptt-estado":
            payload = json.dumps({"pressionado": _ptt_global_pressionado}).encode("utf-8")
            self._respond(200, "application/json", payload)
        elif self.path == "/api/guilds":
            # Retorna lista de servidores onde o bot está instalado.
            guilds = []
            if _bot_ref:
                for g in _bot_ref.guilds:
                    if g.id in _guilds_removidas_ui:
                        continue
                    vc = g.voice_client
                    em_voz = vc is not None and vc.is_connected()
                    conectando = vc is not None and not vc.is_connected()
                    canal = getattr(vc, "channel", None) if vc else None
                    guilds.append({
                        "id": str(g.id),
                        "name": g.name,
                        "em_voz": em_voz,
                        "conectando": conectando,
                        "canal_voz": canal.name if canal and em_voz else None,
                        "canal_id": str(canal.id) if canal and em_voz else None,
                    })
            self._respond(200, "application/json",
                          json.dumps(guilds, ensure_ascii=False).encode("utf-8"))
        else:
            self._respond(404, "text/plain", b"Not found")

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        log.info("[WEB] POST %s", self.path)
        if self.path == "/api/shutdown":
            self._respond(200, "application/json", b'{"ok": true}')
            log.info("Desligamento solicitado via UI web.")
            try:
                _SHUTDOWN_FLAG.parent.mkdir(parents=True, exist_ok=True)
                _SHUTDOWN_FLAG.write_text("shutdown via web ui\n", encoding="utf-8")
            except Exception:
                log.exception("Falha ao registrar flag de desligamento da UI.")
            import os, threading
            threading.Timer(0.3, lambda: os._exit(0)).start()
            return

        if self.path == "/api/voz/conectar":
            self._handle_voz_conectar()
            return

        if self.path == "/api/voz/desconectar":
            self._handle_voz_desconectar()
            return

        if self.path == "/api/guilds/remover":
            self._handle_guild_remover()
            return

        if self.path == "/api/voz/chat":
            self._handle_voz_chat()
            return

        if self.path == "/api/voz/falar":
            self._handle_voz_falar()
            return

        if self.path == "/api/voz/testar":
            self._handle_voz_testar()
            return

        if self.path == "/api/voz/chat-texto":
            self._handle_voz_chat_texto()
            return

        if self.path == "/api/voz/config":
            self._handle_post_voz_config()
            return

        if self.path == "/api/voz/limpar":
            with _voz_lock:
                _voz_historico.clear()
            self._respond(200, "application/json", b'{"ok": true}')
            return

        if self.path == "/api/texto/enviar":
            self._handle_texto_enviar()
            return

        if self.path != "/api/config":
            self._respond(404, "text/plain", b"Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            nova_config = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        from config_loader import cfg
        nova_config.pop("_prompt_defaults", None)
        config_antiga = cfg.as_dict()

        # Salva a nova config
        cfg.save(nova_config)
        log.info("Config salva via UI web.")

        # Aplica ao bot de forma thread-safe (bot roda em event loop asyncio)
        if _loop_ref is not None and _bot_ref is not None:
            future = asyncio.run_coroutine_threadsafe(
                _aplicar_mudancas(nova_config, config_antiga), _loop_ref
            )
            try:
                future.result(timeout=5)
            except Exception as exc:
                log.warning("Falha ao aplicar mudanças ao bot: %s", exc)

        self._respond(200, "application/json", b'{"ok": true}')

    # ── Handlers Voz: Canais / Conectar / Desconectar ─────────────────────────

    def _handle_get_voz_canais(self) -> None:
        """GET /api/voz/canais?guild_id=... — lista canais de voz de uma guild."""
        from urllib.parse import urlparse, parse_qs
        import discord as _discord
        qs = parse_qs(urlparse(self.path).query)
        guild_id_str = (qs.get("guild_id") or [None])[0]
        if not guild_id_str or _bot_ref is None:
            self._respond(400, "application/json", b'{"erro": "guild_id obrigatorio"}')
            return
        guild_id = int(guild_id_str)
        if guild_id in _guilds_removidas_ui:
            self._respond(404, "application/json", b'{"erro": "guild removida"}')
            return
        guild = _bot_ref.get_guild(guild_id)
        if not guild:
            self._respond(404, "application/json", b'{"erro": "guild nao encontrada"}')
            return
        canais = sorted(
            [{"id": str(c.id), "name": c.name}
             for c in guild.channels if isinstance(c, _discord.VoiceChannel)],
            key=lambda c: c["name"],
        )
        self._respond(200, "application/json",
                      json.dumps(canais, ensure_ascii=False).encode("utf-8"))

    def _handle_get_texto_canais(self) -> None:
        """GET /api/texto/canais?guild_id=... — lista canais de texto onde o bot pode falar."""
        from urllib.parse import urlparse, parse_qs
        import discord as _discord
        qs = parse_qs(urlparse(self.path).query)
        guild_id_str = (qs.get("guild_id") or [None])[0]
        if not guild_id_str or _bot_ref is None:
            self._respond(400, "application/json", b'{"erro": "guild_id obrigatorio"}')
            return
        guild_id = int(guild_id_str)
        if guild_id in _guilds_removidas_ui:
            self._respond(404, "application/json", b'{"erro": "guild removida"}')
            return
        guild = _bot_ref.get_guild(guild_id)
        if not guild:
            self._respond(404, "application/json", b'{"erro": "guild nao encontrada"}')
            return

        canais = []
        member = guild.me
        for c in guild.channels:
            if not isinstance(c, _discord.TextChannel):
                continue
            perms = c.permissions_for(member) if member else None
            if perms and not (perms.view_channel and perms.send_messages):
                continue
            canais.append({
                "id": str(c.id),
                "name": c.name,
                "category": c.category.name if c.category else "Sem categoria",
            })

        canais.sort(key=lambda c: (c["category"].lower(), c["name"].lower()))
        self._respond(200, "application/json",
                      json.dumps(canais, ensure_ascii=False).encode("utf-8"))

    def _handle_texto_enviar(self) -> None:
        """POST /api/texto/enviar — body: {guild_id, channel_id, texto}."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        guild_id = int(data.get("guild_id", 0))
        channel_id = int(data.get("channel_id", 0))
        texto = str(data.get("texto", "")).strip()
        if not guild_id or not channel_id or not texto or _bot_ref is None or _loop_ref is None:
            self._respond(400, "application/json",
                          b'{"erro": "guild_id, channel_id e texto obrigatorios"}')
            return
        if guild_id in _guilds_removidas_ui:
            self._respond(404, "application/json", b'{"erro": "guild removida"}')
            return
        if len(texto) > 2000:
            self._respond(400, "application/json", b'{"erro": "texto excede 2000 caracteres"}')
            return

        async def _enviar():
            import discord as _discord
            guild = _bot_ref.get_guild(guild_id)
            if not guild:
                raise ValueError("Guild não encontrada")
            channel = guild.get_channel(channel_id)
            if not channel or not isinstance(channel, _discord.TextChannel):
                raise ValueError("Canal de texto não encontrado")
            member = guild.me
            perms = channel.permissions_for(member) if member else None
            if perms and not (perms.view_channel and perms.send_messages):
                raise PermissionError("Bot sem permissão para enviar mensagens neste canal")
            msg = await channel.send(texto)
            return msg.id

        try:
            msg_id = asyncio.run_coroutine_threadsafe(_enviar(), _loop_ref).result(timeout=10)
            payload = json.dumps({"ok": True, "message_id": str(msg_id)}).encode("utf-8")
            self._respond(200, "application/json", payload)
        except Exception as exc:
            log.exception("Erro em /api/texto/enviar")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}, ensure_ascii=False).encode("utf-8"))

    def _handle_voz_conectar(self) -> None:
        """POST /api/voz/conectar — body: {guild_id, channel_id}."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        guild_id   = int(data.get("guild_id", 0))
        channel_id = int(data.get("channel_id", 0))
        if not guild_id or not channel_id or _bot_ref is None or _loop_ref is None:
            self._respond(400, "application/json",
                          b'{"erro": "guild_id e channel_id obrigatorios"}')
            return
        if guild_id in _guilds_removidas_ui:
            self._respond(404, "application/json", b'{"erro": "guild removida"}')
            return

        async def _conectar():
            import discord as _discord

            async with _voz_connect_lock(guild_id):
                if guild_id in _guilds_removidas_ui:
                    raise ValueError("Guild removida")
                task_atual = asyncio.current_task()
                if task_atual is not None:
                    _voz_connect_tasks[guild_id] = task_atual
                guild = _bot_ref.get_guild(guild_id)
                try:
                    if not guild:
                        raise ValueError("Guild não encontrada")
                    channel = guild.get_channel(channel_id)
                    if not channel or not isinstance(channel, _discord.VoiceChannel):
                        raise ValueError("Canal de voz não encontrado")

                    member = guild.me
                    perms = channel.permissions_for(member) if member else None
                    if perms and not (perms.view_channel and perms.connect):
                        raise PermissionError("Bot sem permissão para conectar neste canal")

                    vc = guild.voice_client
                    if vc is not None:
                        vc_channel = getattr(vc, "channel", None)
                        if vc.is_connected():
                            if vc_channel and vc_channel.id == channel_id:
                                return {"ok": True, "status": "ja_conectado", "canal": channel.name}
                            log.info("[WEB] Movendo voz: %s -> %s", getattr(vc_channel, "name", "?"), channel.name)
                            try:
                                await asyncio.wait_for(vc.move_to(channel), timeout=15)
                                return {"ok": True, "status": "movido", "canal": channel.name}
                            except Exception as exc:
                                log.warning("[WEB] Falha ao mover voz; limpando e reconectando: %s", exc)
                                await _limpar_conexao_voz(guild_id, "falha ao mover")
                        else:
                            await _limpar_conexao_voz(guild_id, "voice client preso antes de conectar")

                    try:
                        vc = await asyncio.wait_for(
                            channel.connect(timeout=20, reconnect=False),
                            timeout=25,
                        )
                    except asyncio.CancelledError:
                        await _limpar_conexao_voz(guild_id, "tentativa cancelada")
                        raise
                    except Exception as exc:
                        atual = guild.voice_client
                        atual_canal = getattr(atual, "channel", None) if atual else None
                        if atual and atual.is_connected() and atual_canal and atual_canal.id == channel_id:
                            return {"ok": True, "status": "conectado", "canal": channel.name}
                        await _limpar_conexao_voz(guild_id, "falha no handshake")
                        raise RuntimeError(
                            "Falha ao conectar ao canal de voz. "
                            "A conexão foi limpa; tente novamente em alguns segundos."
                        ) from exc

                    if not vc.is_connected():
                        await _limpar_conexao_voz(guild_id, "connect retornou sem is_connected")
                        raise RuntimeError("Discord retornou conexão de voz incompleta.")
                    return {"ok": True, "status": "conectado", "canal": channel.name}
                finally:
                    if _voz_connect_tasks.get(guild_id) is task_atual:
                        _voz_connect_tasks.pop(guild_id, None)

        try:
            future = asyncio.run_coroutine_threadsafe(_conectar(), _loop_ref)
            resultado = future.result(timeout=35)
            self._respond(200, "application/json",
                          json.dumps(resultado, ensure_ascii=False).encode("utf-8"))
        except FutureTimeoutError:
            future.cancel()
            log.warning("Timeout em /api/voz/conectar; cancelando tentativa e limpando VoiceClient.")
            try:
                asyncio.run_coroutine_threadsafe(
                    _limpar_conexao_voz(guild_id, "timeout HTTP"),
                    _loop_ref,
                ).result(timeout=12)
            except Exception:
                log.exception("Falha ao limpar conexao de voz apos timeout.")
            self._respond(
                504,
                "application/json",
                json.dumps(
                    {
                        "erro": (
                            "Tempo limite ao conectar no Discord. "
                            "A conexão foi removida; tente novamente em alguns segundos."
                        )
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
        except Exception as exc:
            log.exception("Erro em /api/voz/conectar")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc) or type(exc).__name__},
                                     ensure_ascii=False).encode("utf-8"))

    def _handle_voz_desconectar(self) -> None:
        """POST /api/voz/desconectar — body: {guild_id}."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        guild_id = int(data.get("guild_id", 0))
        if not guild_id or _bot_ref is None or _loop_ref is None:
            self._respond(400, "application/json", b'{"erro": "guild_id obrigatorio"}')
            return

        async def _desconectar():
            await _cancelar_tentativa_conexao_voz(guild_id, "desconectar")
            async with _voz_connect_lock(guild_id):
                return await _limpar_conexao_voz(guild_id, "desconectar")

        try:
            resultado = asyncio.run_coroutine_threadsafe(_desconectar(), _loop_ref).result(timeout=12)
            self._respond(200, "application/json",
                          json.dumps(resultado, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc) or type(exc).__name__},
                                     ensure_ascii=False).encode("utf-8"))

    def _handle_guild_remover(self) -> None:
        """POST /api/guilds/remover — body: {guild_id}; faz o bot sair da guild."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        guild_id = int(data.get("guild_id", 0) or 0)
        if not guild_id:
            self._respond(400, "application/json", b'{"erro": "guild_id obrigatorio"}')
            return
        if _bot_ref is None or _loop_ref is None:
            self._respond(400, "application/json", b'{"erro": "bot indisponivel"}')
            return

        async def _remover_guild():
            guild = _bot_ref.get_guild(guild_id)
            if guild is None:
                raise ValueError("Servidor não encontrado")

            guild_nome = guild.name
            _guilds_removidas_ui.add(guild_id)
            await _cancelar_tentativa_conexao_voz(guild_id, "remover servidor")
            async with _voz_connect_lock(guild_id):
                if guild.voice_client is not None:
                    await _limpar_conexao_voz(guild_id, "remover servidor")
                log.warning("[WEB] Removendo bot do servidor '%s' (%s) via UI.", guild_nome, guild_id)
                await guild.leave()

            _voz_connect_tasks.pop(guild_id, None)
            _voz_connect_locks.pop(guild_id, None)
            return {"ok": True, "guild_id": str(guild_id), "name": guild_nome}

        try:
            resultado = asyncio.run_coroutine_threadsafe(_remover_guild(), _loop_ref).result(timeout=30)
            self._respond(200, "application/json",
                          json.dumps(resultado, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:
            _guilds_removidas_ui.discard(guild_id)
            log.exception("Erro em /api/guilds/remover")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc) or type(exc).__name__},
                                     ensure_ascii=False).encode("utf-8"))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._respond(404, "text/plain", b"File not found")
            return
        data = path.read_bytes()
        self._respond(200, content_type, data)

    def _respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # ── Handlers: Chat de Voz (STT → LLM → TTS → Discord) ───────────────────

    def _handle_voz_chat(self) -> None:
        """POST /api/voz/chat — recebe áudio WAV, transcreve, gera resposta, fala."""
        inicio_total = time.perf_counter()
        log.info("[WEB] POST /api/voz/chat recebido")
        length = int(self.headers.get("Content-Length", 0))
        wav_bytes = self.rfile.read(length)
        log.info("[WEB] WAV recebido: %d bytes", len(wav_bytes))

        if not wav_bytes:
            log.warning("[WEB] Áudio vazio recebido!")
            self._respond(400, "application/json", b'{"erro": "audio vazio"}')
            return

        try:
            from services import stt_whisper
            from cogs.voice_cog import voz_estado

            # Carrega whisper com o modelo configurado
            whisper_modelo = voz_estado.get("whisper_modelo", "large-v3-turbo")
            log.info("[WEB] Carregando Whisper '%s'...", whisper_modelo)
            stt_whisper.carregar(whisper_modelo)

            # 1. Transcrever áudio
            log.info("[WEB] Etapa 1: Transcrevendo áudio...")
            inicio_stt = time.perf_counter()
            texto_usuario = stt_whisper.transcrever(wav_bytes, whisper_modelo)
            tempo_stt = time.perf_counter() - inicio_stt
            log.info("[WEB] Transcrição: %r", texto_usuario)
            if not texto_usuario:
                log.warning("[WEB] Transcrição vazia!")
                self._respond(200, "application/json",
                              json.dumps({"transcript": "", "resposta": ""}).encode("utf-8"))
                return

            # 2. Enviar ao LLM
            log.info("[WEB] Etapa 2: Gerando resposta LLM...")
            guild_com_voz = _encontrar_guild_com_voz()
            inicio_llm = time.perf_counter()
            resposta_llm, falou, audio_ms = _gerar_resposta_voz_streaming(
                texto_usuario,
                voz_estado,
                guild_com_voz,
                "/api/voz/chat",
            )
            tempo_llm = time.perf_counter() - inicio_llm
            log.info("[WEB] Resposta LLM: %r", resposta_llm[:100] if resposta_llm else "(vazio)")

            # 3. Gerar TTS e reproduzir no Discord
            guild_com_voz = _encontrar_guild_com_voz()
            log.info("[WEB] Etapa 3: TTS → Discord — guild_com_voz=%s falar_discord=%s",
                     guild_com_voz, voz_estado.get("falar_discord"))
            if not guild_com_voz:
                log.warning("[WEB] Bot NÃO está em nenhum canal de voz!")
            elif not resposta_llm:
                log.warning("[WEB] Resposta LLM vazia — sem TTS")

            payload = {
                "transcript": texto_usuario,
                "resposta": resposta_llm,
                "falou_discord": falou,
                "audio_ms": audio_ms if falou else 0,
                "timings": {
                    "stt_s": round(tempo_stt, 3),
                    "llm_stream_s": round(tempo_llm, 3),
                    "http_total_s": round(time.perf_counter() - inicio_total, 3),
                },
            }
            log.info(
                "[WEB] Pipeline voz concluído: stt=%.2fs llm_stream=%.2fs http_total=%.2fs",
                tempo_stt,
                tempo_llm,
                time.perf_counter() - inicio_total,
            )
            self._respond(200, "application/json",
                          json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        except Exception as exc:
            log.exception("[WEB] ERRO GERAL em /api/voz/chat")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

    def _handle_voz_falar(self) -> None:
        """POST /api/voz/falar — recebe texto, gera TTS e fala no Discord."""
        log.info("[WEB] POST /api/voz/falar recebido")
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        texto = data.get("texto", "").strip()
        if not texto:
            self._respond(400, "application/json", b'{"erro": "texto vazio"}')
            return

        try:
            from cogs.voice_cog import voz_estado

            guild_id = _encontrar_guild_com_voz()
            if not guild_id:
                self._respond(400, "application/json",
                              b'{"erro": "Bot nao esta conectado a um canal de voz"}')
                return

            log.info("[WEB] /api/voz/falar texto='%s'", texto[:60])
            _iniciar_sessao_tts("/api/voz/falar")
            pcm = _gerar_pcm_tts(texto, voz_estado)
            audio_ms = _duracao_pcm_ms(pcm)
            log.info("[WEB] /api/voz/falar PCM gerado, %d bytes; agendando no Discord", len(pcm))
            if not _agendar_reproducao_pcm(guild_id, pcm, " em /api/voz/falar", interromper=True):
                self._respond(500, "application/json", b'{"erro": "falha ao agendar audio"}')
                return
            log.info("[WEB] /api/voz/falar audio agendado com sucesso")

            payload = json.dumps({"ok": True, "audio_ms": audio_ms}).encode("utf-8")
            self._respond(200, "application/json", payload)
        except Exception as exc:
            log.exception("[WEB] ERRO em /api/voz/falar")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

    def _handle_voz_testar(self) -> None:
        """POST /api/voz/testar — gera uma frase curta com a configuração atual."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except Exception:
            data = {}
        texto = str(data.get("texto") or "Oi, eu sou a Neve. Assim ficou minha voz agora.").strip()
        try:
            from cogs.voice_cog import voz_estado

            guild_id = _encontrar_guild_com_voz()
            if not guild_id:
                self._respond(400, "application/json",
                              b'{"erro": "Bot nao esta conectado a um canal de voz"}')
                return

            _iniciar_sessao_tts("/api/voz/testar")
            pcm = _gerar_pcm_tts(texto, voz_estado)
            audio_ms = _duracao_pcm_ms(pcm)
            if not _agendar_reproducao_pcm(guild_id, pcm, " em /api/voz/testar", interromper=True):
                self._respond(500, "application/json", b'{"erro": "falha ao agendar audio"}')
                return
            payload = json.dumps({"ok": True, "audio_ms": audio_ms}).encode("utf-8")
            self._respond(200, "application/json", payload)
        except Exception as exc:
            log.exception("[WEB] ERRO em /api/voz/testar")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}, ensure_ascii=False).encode("utf-8"))

    def _handle_voz_chat_texto(self) -> None:
        """POST /api/voz/chat-texto — recebe texto, envia ao LLM, gera TTS, fala."""
        log.info("[WEB] POST /api/voz/chat-texto recebido")
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            log.error("[WEB] Erro ao ler JSON: %s", exc)
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        texto = data.get("texto", "").strip()
        log.info("[WEB] Texto recebido: %r", texto[:100] if texto else "(vazio)")
        if not texto:
            self._respond(400, "application/json", b'{"erro": "texto vazio"}')
            return

        try:
            from cogs.voice_cog import voz_estado

            # 1. LLM
            log.info("[WEB] Etapa 1: Gerando resposta LLM...")
            guild_com_voz = _encontrar_guild_com_voz()
            resposta_llm, falou, audio_ms = _gerar_resposta_voz_streaming(
                texto,
                voz_estado,
                guild_com_voz,
                "/api/voz/chat-texto",
            )
            log.info("[WEB] Resposta LLM: %r", resposta_llm[:100] if resposta_llm else "(vazio)")

            # 2. TTS + Discord
            guild_com_voz = _encontrar_guild_com_voz()
            log.info("[WEB] Etapa 2: TTS → Discord — guild=%s falar_discord=%s",
                     guild_com_voz, voz_estado.get("falar_discord"))
            if not guild_com_voz:
                log.warning("[WEB] Bot NÃO está em nenhum canal de voz!")
            elif not resposta_llm:
                log.warning("[WEB] Resposta LLM vazia — sem TTS")

            payload = {
                "resposta": resposta_llm,
                "falou_discord": falou,
                "audio_ms": audio_ms if falou else 0,
            }
            self._respond(200, "application/json",
                          json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:
            log.exception("[WEB] ERRO GERAL em /api/voz/chat-texto")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

    def _handle_get_voz_config(self) -> None:
        """GET /api/voz/config — retorna config de voz."""
        from cogs.voice_cog import voz_estado
        self._respond(200, "application/json",
                      json.dumps(voz_estado, ensure_ascii=False).encode("utf-8"))

    def _handle_post_voz_config(self) -> None:
        """POST /api/voz/config — salva config de voz."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        from cogs.voice_cog import voz_estado, salvar_config_voz
        from services import tts_chatterbox

        old_exaggeration = voz_estado.get("voz_exaggeration")
        for old_key in ("voz_age", "voz_pitch_style", "voz_instruct"):
            data.pop(old_key, None)
        data["tts_model"] = "chatterbox-ptbr-v3"
        data["voz_language"] = "pt-BR"
        data["voz_referencia"] = "data/voz_referencia.wav"

        voz_estado.update(data)
        salvar_config_voz()
        log.info("Config de voz salva via UI web.")

        if voz_estado.get("voz_exaggeration") != old_exaggeration:
            tts_chatterbox.limpar_cache_referencia()

        self._respond(200, "application/json", b'{"ok": true}')


# ── Helpers de chat de voz ─────────────────────────────────────────────────────

def _gerar_pcm_tts(texto: str, voz_cfg: dict, *, stream_chunk: bool = False) -> bytes:
    """Gera TTS Chatterbox PT-BR e converte para PCM do Discord."""
    from services import tts_chatterbox

    inicio = time.perf_counter()
    speed = float(voz_cfg.get("velocidade", 1.0))
    volume = float(voz_cfg.get("volume", 1.0))
    seed = int(voz_cfg.get("voz_seed", 42))
    pitch = float(voz_cfg.get("pitch", 0.0))
    exaggeration = float(voz_cfg.get("voz_exaggeration", 0.5))
    cfg_weight = float(voz_cfg.get("voz_cfg_weight", 0.5))
    temperature = float(voz_cfg.get("voz_temperature", 0.8))
    log.info(
        "[WEB] TTS Chatterbox PT-BR: speed=%.2f vol=%.2f seed=%d pitch=%.1f exag=%.2f cfg=%.2f temp=%.2f",
        speed,
        volume,
        seed,
        pitch,
        exaggeration,
        cfg_weight,
        temperature,
    )
    audio = tts_chatterbox.gerar(
        texto,
        speed=speed,
        seed=seed,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    if stream_chunk:
        pcm = tts_chatterbox.para_pcm_discord(
            audio,
            volume=volume,
            pitch_semitones=pitch,
            start_pad_s=0.0,
            end_pad_s=0.08,
            tail_frames=2,
        )
        log.info("[WEB] TTS chunk completo em %.2fs (%d bytes)", time.perf_counter() - inicio, len(pcm))
        return pcm
    pcm = tts_chatterbox.para_pcm_discord(audio, volume=volume, pitch_semitones=pitch)
    log.info("[WEB] TTS completo em %.2fs (%d bytes)", time.perf_counter() - inicio, len(pcm))
    return pcm


def _agendar_frase_tts(
    frase: str,
    voz_cfg: dict,
    guild_id: int,
    origem: str,
    indice: int,
) -> tuple[bool, int]:
    pcm = _gerar_pcm_tts(frase, voz_cfg, stream_chunk=True)
    audio_ms = _duracao_pcm_ms(pcm)
    log.info("[WEB] Frase TTS %d%s: %r", indice, origem, frase[:120])
    falou = _agendar_reproducao_pcm(guild_id, pcm, f" {origem} frase {indice}", interromper=(indice == 1))
    return falou, audio_ms if falou else 0


def _estimar_audio_tts_ms(texto: str, voz_cfg: dict) -> int:
    palavras = max(1, len((texto or "").split()))
    chars = len(texto or "")
    speed = max(0.85, min(float(voz_cfg.get("velocidade", 1.0) or 1.0), 1.2))
    return max(800, int(((palavras * 390) + (chars * 18)) / speed))


def _submeter_frase_tts_async(
    frase: str,
    voz_cfg: dict,
    guild_id: int,
    origem: str,
    indice: int,
    sessao: int,
) -> Future:
    voz_cfg_snapshot = dict(voz_cfg)
    inicio = time.perf_counter()
    futuro = _tts_executor.submit(_gerar_pcm_tts, frase, voz_cfg_snapshot, stream_chunk=True)
    with _tts_state_lock:
        _tts_futures.add(futuro)

    def _quando_pronto(fut: Future) -> None:
        try:
            pcm = fut.result()
            audio_ms = _duracao_pcm_ms(pcm)
        except CancelledError:
            log.info("[WEB] TTS frase %d%s cancelado antes de gerar.", indice, origem)
            return
        except Exception as exc:
            log.error("[WEB] ERRO ao gerar TTS assíncrono frase %d: %s", indice, exc, exc_info=True)
            return
        finally:
            with _tts_state_lock:
                _tts_futures.discard(fut)

        if not _sessao_tts_atual(sessao):
            log.info("[WEB] Descartando TTS frase %d%s: sessao antiga %d.", indice, origem, sessao)
            return
        log.info(
            "[WEB] Frase TTS %d%s pronta em background em %.2fs: %r",
            indice,
            origem,
            time.perf_counter() - inicio,
            frase[:120],
        )
        if not _agendar_reproducao_pcm(
            guild_id,
            pcm,
            f" {origem} frase {indice}",
            interromper=(indice == 1),
        ):
            log.error("[WEB] Falha ao agendar TTS assíncrono frase %d.", indice)
            return
        log.info("[WEB] Frase TTS %d%s agendada (%d ms).", indice, origem, audio_ms)

    futuro.add_done_callback(_quando_pronto)
    return futuro


def _gerar_resposta_voz_streaming(
    texto_usuario: str,
    voz_cfg: dict,
    guild_id: int | None,
    origem: str,
) -> tuple[str, bool, int]:
    """Gera resposta por streaming e agenda TTS assim que cada frase fica pronta."""
    log.info("[LLM-VOZ] Gerando resposta streaming para: %r", texto_usuario[:80])
    cog = _bot_ref.get_cog("LLM") if _bot_ref else None
    if cog is None:
        log.error("[LLM-VOZ] Cog LLM nao encontrado para streaming.")
        return "LLM nao carregado.", False, 0

    system_prompt = cog._construir_prompt_lou_voz(0)
    with _voz_lock:
        _voz_historico.append({"role": "user", "content": texto_usuario})
        historico = list(_voz_historico)

    tts_ativo = bool(guild_id and voz_cfg.get("falar_discord", True))
    sessao_tts = _iniciar_sessao_tts(origem) if tts_ativo else 0
    partes: list[str] = []
    buffer_tts = ""
    falou = False
    audio_ms_total = 0
    frase_idx = 0
    inicio_llm = time.perf_counter()

    try:
        for delta in cog._stream_resposta(
            system_prompt,
            historico,
            max_tokens=config.LLM_VOZ_MAX_TOKENS,
            temperature=config.LLM_VOZ_TEMPERATURE,
        ):
            partes.append(delta)
            if not tts_ativo:
                continue
            buffer_tts += delta
            frases, buffer_tts = _extrair_frases_tts(
                buffer_tts,
                limite_corte=36 if frase_idx == 0 else 58,
                minimo_corte=16 if frase_idx == 0 else 26,
            )
            for frase in frases:
                frase_idx += 1
                _submeter_frase_tts_async(frase, voz_cfg, int(guild_id), origem, frase_idx, sessao_tts)
                falou = True
                audio_ms_total += _estimar_audio_tts_ms(frase, voz_cfg)
                if frase_idx == 1:
                    audio_ms_total += _TTS_STARTUP_ESTIMATE_MS
    except Exception as exc:
        log.warning("[LLM-VOZ] Streaming falhou: %s", exc, exc_info=True)
        if not partes:
            resposta_fallback = cog._gerar_resposta(
                system_prompt,
                historico,
                max_tokens=config.LLM_VOZ_MAX_TOKENS,
                continuar_se_cortar=False,
                temperature=config.LLM_VOZ_TEMPERATURE,
            )
            resposta_fallback = _normalizar_resposta_voz_texto(resposta_fallback)
            if tts_ativo and resposta_fallback:
                try:
                    pcm = _gerar_pcm_tts(resposta_fallback, voz_cfg)
                    falou = _agendar_reproducao_pcm(int(guild_id), pcm, f" {origem} fallback", interromper=True)
                    audio_ms_total = _duracao_pcm_ms(pcm) if falou else 0
                except Exception as tts_exc:
                    log.error("[WEB] ERRO ao gerar TTS fallback: %s", tts_exc, exc_info=True)
            if resposta_fallback:
                with _voz_lock:
                    _voz_historico.append({"role": "assistant", "content": resposta_fallback})
            return resposta_fallback, falou, audio_ms_total

    resposta_bruta = "".join(partes)
    resposta = _normalizar_resposta_voz_texto(cog._limpar_resposta(resposta_bruta))
    if not resposta and resposta_bruta.strip():
        resposta = _normalizar_resposta_voz_texto(resposta_bruta)

    if tts_ativo:
        frases_finais, _ = _extrair_frases_tts(buffer_tts, force=True)
        for frase in frases_finais:
            frase_idx += 1
            _submeter_frase_tts_async(frase, voz_cfg, int(guild_id), origem, frase_idx, sessao_tts)
            falou = True
            audio_ms_total += _estimar_audio_tts_ms(frase, voz_cfg)
            if frase_idx == 1:
                audio_ms_total += _TTS_STARTUP_ESTIMATE_MS

    log.info(
        "[LLM-VOZ] Streaming concluido em %.2fs; frases_tts=%d resposta=%r",
        time.perf_counter() - inicio_llm,
        frase_idx,
        resposta[:100] if resposta else "(vazio)",
    )

    if resposta:
        with _voz_lock:
            _voz_historico.append({"role": "assistant", "content": resposta})
    return resposta or "", falou, audio_ms_total


def _gerar_resposta_voz(texto_usuario: str) -> str:
    """Envia texto ao LLM (via cog) e retorna a resposta. Mantém histórico próprio."""
    log.info("[LLM-VOZ] Gerando resposta para: %r", texto_usuario[:80])
    cog = _bot_ref.get_cog("LLM") if _bot_ref else None
    if cog is None:
        log.error("[LLM-VOZ] Cog LLM não encontrado! bot_ref=%s cogs=%s",
                  _bot_ref is not None,
                  [c.qualified_name for c in _bot_ref.cogs.values()] if _bot_ref else [])
        return "LLM não carregado."

    system_prompt = cog._construir_prompt_lou_voz(0)
    log.info("[LLM-VOZ] Prompt sistema: %d chars, histórico: %d msgs",
             len(system_prompt), len(_voz_historico))

    with _voz_lock:
        _voz_historico.append({"role": "user", "content": texto_usuario})
        historico = list(_voz_historico)

    try:
        inicio_llm = time.perf_counter()
        resposta = cog._gerar_resposta(
            system_prompt,
            historico,
            max_tokens=config.LLM_VOZ_MAX_TOKENS,
            continuar_se_cortar=False,
            temperature=config.LLM_VOZ_TEMPERATURE,
        )
        log.info("[LLM-VOZ] Tempo LLM: %.2fs", time.perf_counter() - inicio_llm)
        log.info("[LLM-VOZ] Resposta gerada: %r", resposta[:100] if resposta else "(vazio)")
    except Exception as exc:
        log.error("[LLM-VOZ] ERRO ao gerar resposta: %s", exc, exc_info=True)
        return ""

    if resposta:
        with _voz_lock:
            _voz_historico.append({"role": "assistant", "content": resposta})
    return resposta or ""


def _encontrar_guild_com_voz() -> int | None:
    """Retorna o ID da primeira guild onde o bot está em um canal de voz."""
    if _bot_ref is None:
        log.warning("[WEB] _bot_ref é None!")
        return None
    for g in _bot_ref.guilds:
        vc = g.voice_client
        if vc and vc.is_connected():
            log.info("[WEB] Guild com voz encontrada: %s (canal: %s)", g.name, vc.channel.name)
            return g.id
    log.warning("[WEB] Nenhuma guild com voz ativa! Guilds: %s",
                [(g.name, g.voice_client is not None) for g in _bot_ref.guilds])
    return None


# ── Inicialização pública ─────────────────────────────────────────────────────

def start(bot, host: str = "127.0.0.1", port: int = 5000,
          loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Inicia o servidor web em uma thread daemon (não bloqueia o bot)."""
    global _bot_ref, _loop_ref
    _bot_ref = bot
    _loop_ref = loop or asyncio.get_event_loop()

    # Inicia listener global de PTT (Shift direito)
    _iniciar_ptt_global()

    server = ThreadingHTTPServer((host, port), _Handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True, name="web-ui")
    thread.start()
    log.info("Interface web disponível em http://%s:%d", host, port)


def _iniciar_ptt_global() -> None:
    """Inicia listener global de teclado para Push-to-Talk (Shift direito)."""
    global _ptt_global_pressionado
    try:
        from pynput import keyboard

        def on_press(key):
            global _ptt_global_pressionado
            if key == keyboard.Key.shift_r and not _ptt_global_pressionado:
                _ptt_global_pressionado = True

        def on_release(key):
            global _ptt_global_pressionado
            if key == keyboard.Key.shift_r:
                _ptt_global_pressionado = False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release, daemon=True)
        listener.start()
        log.info("[PTT] Listener global de teclado iniciado (Shift direito).")
    except Exception as exc:
        log.warning("[PTT] Falha ao iniciar listener global: %s", exc)
