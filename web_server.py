"""
web_server.py — Servidor HTTP embarcado para configuração via UI web.

Rota          Método  Descrição
/             GET     Serve web/index.html
/api/config   GET     Retorna a config atual como JSON
/api/config   POST    Salva nova config; aplica mudanças ao bot em tempo real

Inicie com:  start(bot, host="127.0.0.1", port=5000)
"""

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("web_server")

_WEB_DIR = Path(__file__).parent / "web"
_bot_ref = None        # discord.ext.commands.Bot
_loop_ref = None       # asyncio event loop do bot


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
        elif self.path == "/api/config":
            from config_loader import cfg
            payload = json.dumps(cfg.as_dict(), ensure_ascii=False, indent=2).encode("utf-8")
            self._respond(200, "application/json", payload)
        elif self.path == "/api/voz":
            from cogs.voice_cog import voz_estado, VOZES_PTBR
            payload = json.dumps(
                {**voz_estado, "vozes_disponiveis": VOZES_PTBR},
                ensure_ascii=False,
            ).encode("utf-8")
            self._respond(200, "application/json", payload)
        elif self.path.startswith("/api/voz/canais"):
            self._handle_get_voz_canais()
        elif self.path == "/api/guilds":
            # Retorna lista de servidores onde o bot está conectado a canais de voz
            guilds = []
            if _bot_ref:
                for g in _bot_ref.guilds:
                    vc = g.voice_client
                    guilds.append({
                        "id": str(g.id),
                        "name": g.name,
                        "em_voz": vc is not None and vc.is_connected(),
                        "canal_voz": vc.channel.name if vc and vc.is_connected() else None,
                        "canal_id": str(vc.channel.id) if vc and vc.is_connected() else None,
                    })
            self._respond(200, "application/json",
                          json.dumps(guilds, ensure_ascii=False).encode("utf-8"))
        else:
            self._respond(404, "text/plain", b"Not found")

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        if self.path == "/api/shutdown":
            self._respond(200, "application/json", b'{"ok": true}')
            log.info("Desligamento solicitado via UI web.")
            import os, threading
            threading.Timer(0.3, lambda: os._exit(0)).start()
            return

        if self.path == "/api/voz":
            self._handle_post_voz()
            return

        if self.path == "/api/voz/transcrever":
            self._handle_transcrever()
            return

        if self.path == "/api/voz/conversar":
            self._handle_conversar()
            return

        if self.path == "/api/voz/conectar":
            self._handle_voz_conectar()
            return

        if self.path == "/api/voz/desconectar":
            self._handle_voz_desconectar()
            return

        if self.path == "/api/voz/gravar/iniciar":
            self._handle_gravar_iniciar()
            return

        if self.path == "/api/voz/gravar/parar":
            self._handle_gravar_parar()
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

    # ── Handler POST /api/voz ─────────────────────────────────────────────────

    def _handle_post_voz(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            nova = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        from cogs.voice_cog import voz_estado, salvar_config_voz
        campos_validos = {"falar_discord", "voz", "velocidade", "volume", "pitch", "whisper_modelo"}
        for campo in campos_validos:
            if campo in nova:
                voz_estado[campo] = nova[campo]

        salvar_config_voz()
        log.info("Config de voz salva via UI web.")
        self._respond(200, "application/json", b'{"ok": true}')

    # ── Handler POST /api/voz/transcrever ────────────────────────────────────

    def _handle_transcrever(self) -> None:
        """Recebe blob de áudio e retorna texto transcrito pelo Whisper."""
        content_type = self.headers.get("Content-Type", "audio/webm")
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._respond(400, "application/json", b'{"erro": "sem audio"}')
            return
        audio_bytes = self.rfile.read(length)
        try:
            from cogs.voice_cog import transcrever_audio_sync
            texto = transcrever_audio_sync(audio_bytes, content_type)
            payload = json.dumps({"texto": texto}, ensure_ascii=False).encode("utf-8")
            self._respond(200, "application/json", payload)
        except Exception as exc:
            log.exception("Erro ao transcrever áudio")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

    # ── Handler POST /api/voz/conversar ──────────────────────────────────────

    def _handle_conversar(self) -> None:
        """Recebe texto do usuário, gera resposta LLM + TTS no Discord."""
        import collections

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        texto_usuario = data.get("texto", "").strip()
        guild_id_str = data.get("guild_id")  # pode ser None
        canal_id = int(data.get("canal_id", -1))

        if not texto_usuario:
            self._respond(400, "application/json", b'{"erro": "texto vazio"}')
            return

        # Obtém LLMCog
        llm_cog = _bot_ref.cogs.get("LLM") if _bot_ref else None
        if llm_cog is None:
            self._respond(503, "application/json", b'{"erro": "LLMCog indisponivel"}')
            return

        try:
            # Para a interface de voz, o padrão é sempre o modo casual (Lou).
            # Só troca se o canal do Discord tiver modo explicitamente configurado.
            modo = llm_cog.canais_modo.get(canal_id)  # None quando canal_id == -1
            if modo == "terapeuta":
                from cogs.llm_cog import _PROMPT_TERAPEUTA
                sys_prompt = _PROMPT_TERAPEUTA
            elif modo == "assistente":
                sys_prompt = llm_cog._construir_prompt_assistente(canal_id)
            else:
                # Padrão de voz: personalidade casual (Lou)
                sys_prompt = llm_cog._construir_prompt_lou(canal_id)

            # A interface web é local (localhost), acessível apenas pelo dono.
            # Injeta o token compacto para que o LLM reconheça o pai corretamente.
            from cogs.llm_cog import _USERNAME_PAI
            texto_para_llm = f"[{_USERNAME_PAI}✓]: {texto_usuario}"

            historico = list(llm_cog._historico.get(canal_id, []))
            historico.append({"role": "user", "content": texto_para_llm})

            resposta = llm_cog._gerar_resposta(sys_prompt, historico)

            # Atualiza histórico
            if canal_id not in llm_cog._historico:
                llm_cog._historico[canal_id] = collections.deque(maxlen=20)
            llm_cog._historico[canal_id].append({"role": "user", "content": texto_para_llm})
            llm_cog._historico[canal_id].append({"role": "assistant", "content": resposta})

            # Reproduz no Discord se configurado e bot estiver em canal de voz
            from cogs.voice_cog import voz_estado
            if guild_id_str and voz_estado.get("falar_discord", True):
                if _loop_ref is not None and _bot_ref is not None:
                    voice_cog = _bot_ref.cogs.get("Voice")
                    if voice_cog is not None:
                        asyncio.run_coroutine_threadsafe(
                            voice_cog.falar_no_canal(int(guild_id_str), resposta),
                            _loop_ref,
                        )

            payload = json.dumps(
                {"resposta": resposta},
                ensure_ascii=False,
            ).encode("utf-8")
            self._respond(200, "application/json", payload)

        except Exception as exc:
            log.exception("Erro em /api/voz/conversar")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

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
        guild = _bot_ref.get_guild(int(guild_id_str))
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
        if not guild_id or not channel_id or _bot_ref is None:
            self._respond(400, "application/json",
                          b'{"erro": "guild_id e channel_id obrigatorios"}')
            return

        async def _conectar():
            import discord as _discord
            guild = _bot_ref.get_guild(guild_id)
            if not guild:
                raise ValueError("Guild não encontrada")
            channel = guild.get_channel(channel_id)
            if not channel or not isinstance(channel, _discord.VoiceChannel):
                raise ValueError("Canal de voz não encontrado")
            if guild.voice_client is not None:
                if guild.voice_client.channel.id == channel_id:
                    return  # já conectado neste canal
                await guild.voice_client.disconnect()
            # Conecta com VoiceRecvClient para suportar gravação
            try:
                from discord.ext import voice_recv as _vr
                await channel.connect(cls=_vr.VoiceRecvClient)
            except ImportError:
                await channel.connect()

        try:
            asyncio.run_coroutine_threadsafe(_conectar(), _loop_ref).result(timeout=10)
            self._respond(200, "application/json", b'{"ok": true}')
        except Exception as exc:
            log.exception("Erro em /api/voz/conectar")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

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
        if not guild_id or _bot_ref is None:
            self._respond(400, "application/json", b'{"erro": "guild_id obrigatorio"}')
            return

        async def _desconectar():
            guild = _bot_ref.get_guild(guild_id)
            if guild and guild.voice_client:
                await guild.voice_client.disconnect()

        try:
            asyncio.run_coroutine_threadsafe(_desconectar(), _loop_ref).result(timeout=5)
            self._respond(200, "application/json", b'{"ok": true}')
        except Exception as exc:
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

    # ── Handlers Gravação de Call ───────────────────────────────────────────────────

    def _handle_gravar_iniciar(self) -> None:
        """POST /api/voz/gravar/iniciar — body: {guild_id}."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        guild_id = int(data.get("guild_id", 0))
        if not guild_id or _bot_ref is None:
            self._respond(400, "application/json", b'{"erro": "guild_id obrigatorio"}')
            return

        voice_cog = _bot_ref.cogs.get("Voice")
        if voice_cog is None:
            self._respond(503, "application/json", b'{"erro": "VoiceCog indisponivel"}')
            return

        try:
            asyncio.run_coroutine_threadsafe(
                voice_cog.iniciar_gravacao_call(guild_id), _loop_ref
            ).result(timeout=10)
            self._respond(200, "application/json", b'{"ok": true}')
        except Exception as exc:
            log.exception("Erro em /api/voz/gravar/iniciar")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

    def _handle_gravar_parar(self) -> None:
        """POST /api/voz/gravar/parar — body: {guild_id}. Aguarda mixagem e retorna caminho."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._respond(400, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))
            return

        guild_id = int(data.get("guild_id", 0))
        if not guild_id or _bot_ref is None:
            self._respond(400, "application/json", b'{"erro": "guild_id obrigatorio"}')
            return

        voice_cog = _bot_ref.cogs.get("Voice")
        if voice_cog is None:
            self._respond(503, "application/json", b'{"erro": "VoiceCog indisponivel"}')
            return

        try:
            # Timeout generoso: inclui ffmpeg amix para calls longas
            arquivo = asyncio.run_coroutine_threadsafe(
                voice_cog.parar_gravacao_call(guild_id), _loop_ref
            ).result(timeout=180)
            nome_arquivo = Path(arquivo).name if arquivo else ""
            self._respond(200, "application/json",
                          json.dumps({"ok": True, "arquivo": nome_arquivo, "caminho": arquivo},
                                     ensure_ascii=False).encode("utf-8"))
        except Exception as exc:
            log.exception("Erro em /api/voz/gravar/parar")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

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


# ── Inicialização pública ─────────────────────────────────────────────────────

def start(bot, host: str = "127.0.0.1", port: int = 5000,
          loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Inicia o servidor web em uma thread daemon (não bloqueia o bot)."""
    global _bot_ref, _loop_ref
    _bot_ref = bot
    _loop_ref = loop or asyncio.get_event_loop()

    server = ThreadingHTTPServer((host, port), _Handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True, name="web-ui")
    thread.start()
    log.info("Interface web disponível em http://%s:%d", host, port)
