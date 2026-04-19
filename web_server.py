"""
web_server.py — Servidor HTTP embarcado para configuração e chat de voz via UI web.

Rota                    Método  Descrição
/                       GET     Serve web/index.html
/api/config             GET     Retorna a config atual como JSON
/api/config             POST    Salva nova config; aplica mudanças ao bot em tempo real
/api/guilds             GET     Lista guilds do bot
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
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config

log = logging.getLogger("web_server")

_WEB_DIR = Path(__file__).parent / "web"
_bot_ref = None        # discord.ext.commands.Bot
_loop_ref = None       # asyncio event loop do bot

# ── Histórico do chat de voz (via web) ────────────────────────────────────────
_voz_historico: deque = deque(maxlen=20)
_voz_lock = threading.Lock()

# ── Push-to-Talk global (funciona fora do navegador) ─────────────────────────
_ptt_global_pressionado = False


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
        elif self.path.startswith("/api/voz/canais"):
            self._handle_get_voz_canais()
        elif self.path == "/api/voz/config":
            self._handle_get_voz_config()
        elif self.path == "/api/voz/ptt-estado":
            payload = json.dumps({"pressionado": _ptt_global_pressionado}).encode("utf-8")
            self._respond(200, "application/json", payload)
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
        log.info("[WEB] POST %s", self.path)
        if self.path == "/api/shutdown":
            self._respond(200, "application/json", b'{"ok": true}')
            log.info("Desligamento solicitado via UI web.")
            import os, threading
            threading.Timer(0.3, lambda: os._exit(0)).start()
            return

        if self.path == "/api/voz/conectar":
            self._handle_voz_conectar()
            return

        if self.path == "/api/voz/desconectar":
            self._handle_voz_desconectar()
            return

        if self.path == "/api/voz/chat":
            self._handle_voz_chat()
            return

        if self.path == "/api/voz/falar":
            self._handle_voz_falar()
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
        log.info("[WEB] POST /api/voz/chat recebido")
        length = int(self.headers.get("Content-Length", 0))
        wav_bytes = self.rfile.read(length)
        log.info("[WEB] WAV recebido: %d bytes", len(wav_bytes))

        if not wav_bytes:
            log.warning("[WEB] Áudio vazio recebido!")
            self._respond(400, "application/json", b'{"erro": "audio vazio"}')
            return

        try:
            from services import stt_whisper, tts_omnivoice
            from cogs.voice_cog import voz_estado, reproduzir_pcm

            # Carrega whisper com o modelo configurado
            whisper_modelo = voz_estado.get("whisper_modelo", "medium")
            log.info("[WEB] Carregando Whisper '%s'...", whisper_modelo)
            stt_whisper.carregar(whisper_modelo)

            # 1. Transcrever áudio
            log.info("[WEB] Etapa 1: Transcrevendo áudio...")
            texto_usuario = stt_whisper.transcrever(wav_bytes)
            log.info("[WEB] Transcrição: %r", texto_usuario)
            if not texto_usuario:
                log.warning("[WEB] Transcrição vazia!")
                self._respond(200, "application/json",
                              json.dumps({"transcript": "", "resposta": ""}).encode("utf-8"))
                return

            # 2. Enviar ao LLM
            log.info("[WEB] Etapa 2: Gerando resposta LLM...")
            resposta_llm = _gerar_resposta_voz(texto_usuario)
            log.info("[WEB] Resposta LLM: %r", resposta_llm[:100] if resposta_llm else "(vazio)")

            # 3. Gerar TTS e reproduzir no Discord
            guild_com_voz = _encontrar_guild_com_voz()
            log.info("[WEB] Etapa 3: TTS → Discord — guild_com_voz=%s falar_discord=%s",
                     guild_com_voz, voz_estado.get("falar_discord"))
            falou = False
            if guild_com_voz and resposta_llm and voz_estado.get("falar_discord", True):
                try:
                    instruct = voz_estado.get("voz_instruct", "female, teenager, low pitch")
                    speed = float(voz_estado.get("velocidade", 1.0))
                    volume = float(voz_estado.get("volume", 1.0))
                    seed = int(voz_estado.get("voz_seed", 42))
                    pitch = float(voz_estado.get("pitch", 0.0))
                    language = voz_estado.get("voz_language", "Portuguese")
                    log.info("[WEB] TTS params: instruct=%r speed=%.1f volume=%.1f seed=%d pitch=%.1f lang=%s",
                             instruct, speed, volume, seed, pitch, language)

                    audio = tts_omnivoice.gerar(resposta_llm, instruct=instruct, speed=speed, seed=seed, language=language)
                    pcm = tts_omnivoice.para_pcm_discord(audio, volume=volume, pitch_semitones=pitch)
                    log.info("[WEB] Enviando PCM ao Discord (%d bytes)...", len(pcm))
                    asyncio.run_coroutine_threadsafe(
                        reproduzir_pcm(_bot_ref, guild_com_voz, pcm), _loop_ref
                    ).result(timeout=120)
                    falou = True
                    log.info("[WEB] Reprodução no Discord concluída!")
                except Exception as exc:
                    log.error("[WEB] ERRO ao reproduzir TTS no Discord: %s", exc, exc_info=True)
            elif not guild_com_voz:
                log.warning("[WEB] Bot NÃO está em nenhum canal de voz!")
            elif not resposta_llm:
                log.warning("[WEB] Resposta LLM vazia — sem TTS")

            payload = {
                "transcript": texto_usuario,
                "resposta": resposta_llm,
                "falou_discord": falou,
            }
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
            from services import tts_omnivoice
            from cogs.voice_cog import voz_estado, reproduzir_pcm

            guild_id = _encontrar_guild_com_voz()
            if not guild_id:
                self._respond(400, "application/json",
                              b'{"erro": "Bot nao esta conectado a um canal de voz"}')
                return

            instruct = voz_estado.get("voz_instruct", "female, teenager, low pitch")
            speed = float(voz_estado.get("velocidade", 1.0))
            volume = float(voz_estado.get("volume", 1.0))
            seed = int(voz_estado.get("voz_seed", 42))
            pitch = float(voz_estado.get("pitch", 0.0))
            language = voz_estado.get("voz_language", "Portuguese")

            log.info("[WEB] /api/voz/falar texto='%s' instruct='%s' speed=%.1f vol=%.1f seed=%d pitch=%.1f lang=%s", texto[:60], instruct, speed, volume, seed, pitch, language)
            audio = tts_omnivoice.gerar(texto, instruct=instruct, speed=speed, seed=seed, language=language)
            log.info("[WEB] /api/voz/falar TTS gerado, %d amostras", len(audio) if audio is not None else 0)
            pcm = tts_omnivoice.para_pcm_discord(audio, volume=volume, pitch_semitones=pitch)
            log.info("[WEB] /api/voz/falar PCM gerado, %d bytes — enviando ao Discord", len(pcm))
            asyncio.run_coroutine_threadsafe(
                reproduzir_pcm(_bot_ref, guild_id, pcm), _loop_ref
            ).result(timeout=120)
            log.info("[WEB] /api/voz/falar concluido com sucesso")

            self._respond(200, "application/json", b'{"ok": true}')
        except Exception as exc:
            log.exception("[WEB] ERRO em /api/voz/falar")
            self._respond(500, "application/json",
                          json.dumps({"erro": str(exc)}).encode("utf-8"))

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
            from services import tts_omnivoice
            from cogs.voice_cog import voz_estado, reproduzir_pcm

            # 1. LLM
            log.info("[WEB] Etapa 1: Gerando resposta LLM...")
            resposta_llm = _gerar_resposta_voz(texto)
            log.info("[WEB] Resposta LLM: %r", resposta_llm[:100] if resposta_llm else "(vazio)")

            # 2. TTS + Discord
            guild_com_voz = _encontrar_guild_com_voz()
            log.info("[WEB] Etapa 2: TTS → Discord — guild=%s falar_discord=%s",
                     guild_com_voz, voz_estado.get("falar_discord"))
            falou = False
            if guild_com_voz and resposta_llm and voz_estado.get("falar_discord", True):
                try:
                    instruct = voz_estado.get("voz_instruct", "female, teenager, low pitch")
                    speed = float(voz_estado.get("velocidade", 1.0))
                    volume = float(voz_estado.get("volume", 1.0))
                    seed = int(voz_estado.get("voz_seed", 42))
                    pitch = float(voz_estado.get("pitch", 0.0))
                    language = voz_estado.get("voz_language", "Portuguese")
                    log.info("[WEB] TTS params: instruct=%r speed=%.1f volume=%.1f seed=%d pitch=%.1f lang=%s",
                             instruct, speed, volume, seed, pitch, language)

                    audio = tts_omnivoice.gerar(resposta_llm, instruct=instruct, speed=speed, seed=seed, language=language)
                    pcm = tts_omnivoice.para_pcm_discord(audio, volume=volume, pitch_semitones=pitch)
                    log.info("[WEB] Enviando PCM ao Discord (%d bytes)...", len(pcm))
                    asyncio.run_coroutine_threadsafe(
                        reproduzir_pcm(_bot_ref, guild_com_voz, pcm), _loop_ref
                    ).result(timeout=120)
                    falou = True
                    log.info("[WEB] Reprodução no Discord concluída!")
                except Exception as exc:
                    log.error("[WEB] ERRO ao reproduzir TTS: %s", exc, exc_info=True)
            elif not guild_com_voz:
                log.warning("[WEB] Bot NÃO está em nenhum canal de voz!")
            elif not resposta_llm:
                log.warning("[WEB] Resposta LLM vazia — sem TTS")

            payload = {
                "resposta": resposta_llm,
                "falou_discord": falou,
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

        # Detecta se parâmetros de voz mudaram (requer nova referência)
        old_instruct = voz_estado.get("voz_instruct")
        old_language = voz_estado.get("voz_language")
        old_seed = voz_estado.get("voz_seed")

        voz_estado.update(data)
        salvar_config_voz()
        log.info("Config de voz salva via UI web.")

        new_instruct = voz_estado.get("voz_instruct")
        new_language = voz_estado.get("voz_language")
        new_seed = voz_estado.get("voz_seed")
        if (new_instruct != old_instruct or new_language != old_language
                or new_seed != old_seed):
            try:
                from services import tts_omnivoice
                tts_omnivoice.regenerar_referencia(
                    instruct=new_instruct or "female, teenager, low pitch",
                    language=new_language or "Portuguese",
                    seed=int(new_seed or 42),
                )
                log.info("Referência de voz regenerada após mudança de config.")
            except Exception as exc:
                log.warning("Falha ao regenerar referência: %s", exc)

        self._respond(200, "application/json", b'{"ok": true}')


# ── Helpers de chat de voz ─────────────────────────────────────────────────────

def _gerar_resposta_voz(texto_usuario: str) -> str:
    """Envia texto ao LLM (via cog) e retorna a resposta. Mantém histórico próprio."""
    log.info("[LLM-VOZ] Gerando resposta para: %r", texto_usuario[:80])
    cog = _bot_ref.get_cog("LLM") if _bot_ref else None
    if cog is None:
        log.error("[LLM-VOZ] Cog LLM não encontrado! bot_ref=%s cogs=%s",
                  _bot_ref is not None,
                  [c.qualified_name for c in _bot_ref.cogs.values()] if _bot_ref else [])
        return "LLM não carregado."

    # Usa o prompt Lou (personalidade) para o chat de voz
    system_prompt = cog._construir_prompt_lou(0)
    # Instrução extra para voz: respostas curtas, naturais, no tom da Lou
    system_prompt += (
        "\n\n[SISTEMA] Você está conversando por voz em tempo real, como se fosse "
        "um papo casual pelo Discord. Fale como a Lou falaria — natural, descontraída, "
        "sem formalidade. 1 a 2 frases curtas no máximo. Nada de listas, parágrafos "
        "ou oferecer ajuda como se fosse assistente. Você é a Lou, não uma IA."
    )
    log.info("[LLM-VOZ] Prompt sistema: %d chars, histórico: %d msgs",
             len(system_prompt), len(_voz_historico))

    with _voz_lock:
        _voz_historico.append({"role": "user", "content": texto_usuario})
        historico = list(_voz_historico)

    try:
        resposta = cog._gerar_resposta(system_prompt, historico,
                                       max_tokens=config.LLM_VOZ_MAX_TOKENS)
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
