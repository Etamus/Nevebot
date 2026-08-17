"""Utilitarios compartilhados para iniciar a recepcao de voz do Discord."""

from __future__ import annotations

import logging
import select
from typing import Any


log = logging.getLogger("discord_voice_receive")


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
