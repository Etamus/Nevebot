"""Roteamento exclusivo entre os backends de sintese de voz."""

from __future__ import annotations

import logging
import threading


log = logging.getLogger("tts_manager")
HIGGS = "higgs-tts-3-4b"
CHATTERBOX = "chatterbox-ptbr-v3"
MODELOS = {HIGGS, CHATTERBOX}
_switch_lock = threading.RLock()
_ativo: str | None = None


def modelo(voz_cfg: dict) -> str:
    value = str(voz_cfg.get("tts_model") or HIGGS).strip().lower()
    if value not in MODELOS:
        raise ValueError(f"Modelo de voz nao suportado: {value}")
    return value


def _liberar_chatterbox() -> None:
    from services import tts_chatterbox

    descarregar = getattr(tts_chatterbox, "descarregar", None)
    if callable(descarregar):
        descarregar()


def ativar(voz_cfg: dict) -> str:
    global _ativo
    escolhido = modelo(voz_cfg)
    with _switch_lock:
        if _ativo == escolhido:
            return escolhido
        if escolhido == HIGGS:
            _liberar_chatterbox()
        else:
            from services import tts_higgs

            tts_higgs.descarregar()
        _ativo = escolhido
        log.info("Backend TTS selecionado: %s", escolhido)
        return escolhido


def gerar_pcm(
    voz_cfg: dict,
    texto: str,
    *,
    stream_chunk: bool = False,
    expressao: dict | None = None,
    permitir_tags_manuais: bool = False,
) -> bytes:
    with _switch_lock:
        escolhido = ativar(voz_cfg)
        from services import tts_expression

        volume = float(voz_cfg.get("volume", 1.0))
        pitch = float(voz_cfg.get("pitch", 0.0))
        padding = {"start_pad_s": 0.0, "end_pad_s": 0.08, "tail_frames": 2} if stream_chunk else {}
        if escolhido == HIGGS:
            from services import tts_higgs

            if permitir_tags_manuais:
                texto_higgs, tags = tts_expression.preparar_manual(texto)
                if not texto_higgs:
                    raise ValueError("Digite uma mensagem alem das tags de expressao.")
                aprovada = tts_expression.Expression()
                tags_console = " ".join(tags) if tags else "nenhuma (fala neutra)"
                log.info(
                    "[TTS:Higgs:Manual] tags aplicadas=%s | entrada=%r",
                    tags_console,
                    texto_higgs,
                )
            else:
                texto_higgs, tags, aprovada = tts_expression.preparar(texto, expressao)
            if expressao is not None and not permitir_tags_manuais:
                proposta = (
                    f"emotion={expressao.get('emotion', 'neutral')}, "
                    f"style={expressao.get('style', 'normal')}, "
                    f"effect={expressao.get('effect', 'none')}, "
                    f"confidence={expressao.get('confidence', 'low')}"
                )
                tags_console = " ".join(tags) if tags else "nenhuma (fala neutra)"
                log.info(
                    "[TTS:Higgs:Expressividade] proposta={%s} | tags aplicadas=%s | entrada=%r",
                    proposta,
                    tags_console,
                    texto_higgs,
                )
                valores_aprovados = (aprovada.emotion, aprovada.style, aprovada.effect)
                valores_propostos = tuple(
                    str(expressao.get(chave) or padrao)
                    for chave, padrao in (
                        ("emotion", "neutral"),
                        ("style", "normal"),
                        ("effect", "none"),
                    )
                )
                if tags and valores_aprovados != valores_propostos:
                    log.info(
                        "[TTS:Higgs:Expressividade] proposta ajustada com seguranca para "
                        "emotion=%s, style=%s, effect=%s.",
                        *valores_aprovados,
                    )
                if not tags and any(
                    str(expressao.get(chave) or padrao) != padrao
                    for chave, padrao in (
                        ("emotion", "neutral"),
                        ("style", "normal"),
                        ("effect", "none"),
                    )
                ):
                    log.info(
                        "[TTS:Higgs:Expressividade] proposta neutralizada pela validacao segura "
                        "(emotion=%s, style=%s, effect=%s).",
                        aprovada.emotion,
                        aprovada.style,
                        aprovada.effect,
                    )
            audio, sample_rate = tts_higgs.gerar(texto_higgs, voz_cfg)
            return tts_higgs.para_pcm_discord(
                audio, sample_rate, volume=volume, pitch_semitones=pitch, **padding
            )

        from services import tts_chatterbox

        audio = tts_chatterbox.gerar(
            tts_expression.limpar_tags(texto),
            speed=float(voz_cfg.get("velocidade", 1.0)),
            seed=int(voz_cfg.get("voz_seed", 42)),
            exaggeration=float(voz_cfg.get("voz_exaggeration", 0.5)),
            cfg_weight=float(voz_cfg.get("voz_cfg_weight", 0.5)),
            temperature=float(voz_cfg.get("voz_temperature", 0.8)),
        )
        return tts_chatterbox.para_pcm_discord(
            audio, volume=volume, pitch_semitones=pitch, **padding
        )


def precarregar_e_aquecer(voz_cfg: dict, *, full_warmup: bool = True) -> None:
    with _switch_lock:
        escolhido = ativar(voz_cfg)
        if escolhido == HIGGS:
            from services import tts_higgs

            tts_higgs.precarregar_e_aquecer(voz_cfg, full_warmup=full_warmup)
        else:
            from services import tts_chatterbox

            tts_chatterbox.precarregar_e_aquecer(voz_cfg, full_warmup=full_warmup)


def referencia_alterada(voz_cfg: dict) -> None:
    escolhido = modelo(voz_cfg)
    if escolhido == CHATTERBOX:
        from services import tts_chatterbox

        tts_chatterbox.limpar_cache_referencia()


def desligar() -> None:
    global _ativo
    with _switch_lock:
        from services import tts_higgs

        tts_higgs.descarregar()
        _liberar_chatterbox()
        _ativo = None
