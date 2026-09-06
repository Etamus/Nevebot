"""Utilitarios compartilhados para iniciar a recepcao de voz do Discord."""

from __future__ import annotations

import asyncio
import logging
import select
import threading
from typing import Any


log = logging.getLogger("discord_voice_receive")
_compat_lock = threading.Lock()
_compat_aplicada = False


def aplicar_compatibilidade_voice_recv() -> None:
    """Corrige falhas conhecidas do voice-recv sem alterar o pacote instalado."""
    global _compat_aplicada
    if _compat_aplicada:
        return

    with _compat_lock:
        if _compat_aplicada:
            return

        from discord.ext import voice_recv
        from discord.ext.voice_recv import video

        original_video_init = video.VideoStreamInfo.__init__

        def _video_init_compativel(self: Any, *, data: dict[str, Any]) -> None:
            seguro = dict(data or {})
            seguro.setdefault("active", False)
            seguro.setdefault("max_bitrate", 0)
            seguro.setdefault("max_framerate", 0)
            seguro.setdefault("max_resolution", {"width": 0, "height": 0, "type": "unknown"})
            seguro.setdefault("quality", 0)
            seguro.setdefault("rid", "")
            seguro.setdefault("rtx_ssrc", 0)
            seguro.setdefault("ssrc", 0)
            original_video_init(self, data=seguro)

        original_remove_ssrc = voice_recv.VoiceRecvClient._remove_ssrc

        def _remove_ssrc_compativel(self: Any, *, user_id: int) -> None:
            ssrc = self._id_to_ssrc.pop(user_id, None)
            if ssrc is None:
                return
            reader = getattr(self, "_reader", None)
            timer = getattr(reader, "speaking_timer", None)
            if timer is not None:
                timer.drop_ssrc(ssrc)
            self._ssrc_to_id.pop(ssrc, None)

        # Os marcadores evitam empilhar wrappers em reloads durante desenvolvimento.
        if not getattr(original_video_init, "_nevebot_compat", False):
            _video_init_compativel._nevebot_compat = True
            video.VideoStreamInfo.__init__ = _video_init_compativel
        if not getattr(original_remove_ssrc, "_nevebot_compat", False):
            _remove_ssrc_compativel._nevebot_compat = True
            voice_recv.VoiceRecvClient._remove_ssrc = _remove_ssrc_compativel

        _compat_aplicada = True
        log.info("Compatibilidade do receptor de voz do Discord aplicada.")


def conexao_voz_saudavel(voice_client: Any) -> bool:
    """Detecta VoiceClient que aparenta estar conectado, mas perdeu o poller."""
    if voice_client is None or not voice_client.is_connected():
        return False
    connection = getattr(voice_client, "_connection", None)
    runner = getattr(connection, "_runner", None)
    if runner is None:
        return False
    return not runner.done()


def erro_conexao_voz(voice_client: Any) -> BaseException | None:
    """Retorna a falha terminal do poller, quando disponivel."""
    connection = getattr(voice_client, "_connection", None)
    runner = getattr(connection, "_runner", None)
    if runner is None or not runner.done() or runner.cancelled():
        return None
    try:
        return runner.exception()
    except (RuntimeError, asyncio.CancelledError):
        return None


def preparar_recebimento_dave(voice_client: Any, *, segundos: int = 15) -> None:
    """Permite a transicao inicial entre Opus puro e frames DAVE."""
    connection = getattr(voice_client, "_connection", None)
    session = getattr(connection, "dave_session", None)
    if session is None:
        return
    try:
        session.set_passthrough_mode(True, max(1, int(segundos)))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        log.debug("Nao foi possivel preparar o passthrough DAVE: %s", exc)


def descriptografar_opus_dave(voice_client: Any, user_id: int, opus: bytes) -> bytes:
    """Descriptografa DAVE sem rejeitar frames Opus em passthrough."""
    payload = bytes(opus)
    if payload == b"\xf8\xff\xfe":
        return payload

    connection = getattr(voice_client, "_connection", None)
    session = getattr(connection, "dave_session", None)
    if session is None:
        return payload
    if not bool(getattr(session, "ready", False)):
        if int(getattr(connection, "dave_protocol_version", 0) or 0) > 0:
            raise RuntimeError("A sessao DAVE ainda nao esta pronta para receber audio.")
        return payload

    import davey

    try:
        decrypted = session.decrypt(int(user_id), davey.MediaType.audio, payload)
    except Exception as exc:
        message = str(exc).casefold()
        if "unencrypted" in message and "passthrough" in message:
            return payload
        raise
    return bytes(decrypted)


def descartar_pacotes_pendentes(voice_client: Any, *, limite: int = 4096) -> int:
    """Esvazia datagramas acumulados enquanto o SocketReader estava pausado."""
    connection = getattr(voice_client, "_connection", None)
    socket_reader = getattr(connection, "_socket_reader", None)
    callbacks = getattr(socket_reader, "_callbacks", None)
    sock = getattr(connection, "socket", None)
    if sock is None or callbacks:
        return 0

    descartados = 0
    try:
        while descartados < max(1, int(limite)):
            readable, _, _ = select.select([sock], [], [], 0)
            if not readable:
                break
            sock.recv(65_535)
            descartados += 1
    except (OSError, ValueError, TypeError) as exc:
        log.debug("Nao foi possivel concluir a limpeza do socket de voz: %s", exc)

    if descartados:
        log.info("Descartados %s pacotes de voz anteriores a ativacao.", descartados)
    return descartados
