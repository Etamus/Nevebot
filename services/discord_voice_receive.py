"""Utilitarios compartilhados para iniciar a recepcao de voz do Discord."""

from __future__ import annotations

import logging
import select
from typing import Any


log = logging.getLogger("discord_voice_receive")


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
