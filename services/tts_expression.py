"""Controles expressivos seguros e exclusivos do Higgs Audio."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping


_TAG_RE = re.compile(r"<\|[^|<>]{1,80}\|>")

# Controles oficiais aceitos pelo Higgs. A gramatica da LLM usa a mesma lista.
EMOTIONS = {
    "neutral",
    "elation",
    "amusement",
    "enthusiasm",
    "determination",
    "pride",
    "affection",
    "contentment",
    "relief",
    "contemplation",
    "confusion",
    "surprise",
    "awe",
    "longing",
    "arousal",
    "anger",
    "fear",
    "disgust",
    "bitterness",
    "sadness",
    "shame",
    "helplessness",
}
STYLES = {"normal", "singing", "shouting", "whispering"}
EFFECTS = {
    "none", "cough", "laughter", "crying", "screaming", "burping",
    "humming", "sigh", "sniff", "sneeze",
}
CONFIDENCES = {"low", "high"}


@dataclass(frozen=True)
class Expression:
    emotion: str = "neutral"
    style: str = "normal"
    effect: str = "none"
    confidence: str = "low"


def limpar_tags(texto: str) -> str:
    """Remove qualquer controle Higgs vindo de texto nao confiavel."""
    return " ".join(_TAG_RE.sub("", str(texto or "")).split())


def _normalizar(texto: str) -> str:
    valor = unicodedata.normalize("NFKD", texto.casefold())
    return " ".join(
        re.findall(r"[a-z0-9]+", "".join(c for c in valor if not unicodedata.combining(c)))
    )


def _tem(texto: str, termos: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(termo)}\b", texto) for termo in termos)


def _tem_afirmado(texto: str, termos: tuple[str, ...]) -> bool:
    """Ignora pistas negadas, preservando expressoes cujo sentido ja e negativo."""
    for termo in termos:
        for ocorrencia in re.finditer(rf"\b{re.escape(termo)}\b", texto):
            if termo.startswith(("nao ", "nunca ", "nem ", "sem ")):
                return True
            prefixo = texto[max(0, ocorrencia.start() - 32):ocorrencia.start()]
            if re.search(r"\b(?:nao|nunca|nem|sem)(?:\s+\w+){0,3}\s*$", prefixo):
                continue
            return True
    return False


def _emocao_solicitada(contexto: str) -> str:
    """Extrai apenas pedidos explicitos de tom/relato, nao o humor casual do usuario."""
    if not _tem(
        contexto,
        (
            "conte", "conta", "historia", "acontecimento", "momento", "lembranca",
            "fale", "fala", "diga", "tom", "voz", "interprete",
        ),
    ):
        return "neutral"
    pedidos = (
        ("elation", ("euforia", "euforica", "extase", "gritando de alegria")),
        ("amusement", ("engracado", "divertido", "com humor", "risada", "piada")),
        ("enthusiasm", ("entusiasmo", "empolgada", "animada")),
        ("determination", ("determinacao", "determinada", "decidida")),
        ("pride", ("orgulho", "orgulhosa")),
        ("affection", ("carinho", "carinhosa", "amor", "romantica")),
        ("contentment", (
            "muito feliz", "felicidade", "alegria", "alegre", "feliz",
            "contente", "satisfeita", "tranquila",
        )),
        ("relief", ("alivio", "aliviada")),
        ("contemplation", ("pensativa", "reflexiva", "contemplativa")),
        ("confusion", ("confusa", "confusao")),
        ("surprise", ("surpresa", "surpreendente")),
        ("awe", ("admiracao", "encantamento", "deslumbrada")),
        ("longing", ("saudade", "nostalgia")),
        ("anger", ("raiva", "furiosa", "brava")),
        ("fear", ("medo", "assustada", "aterrorizante")),
        ("disgust", ("nojo", "repulsa", "nojento")),
        ("bitterness", ("amargura", "ressentimento", "amarga")),
        ("sadness", ("muito triste", "tristeza", "triste", "abatida", "melancolica")),
        ("shame", ("vergonha", "envergonhada")),
        ("helplessness", ("desamparo", "impotencia", "sem saida")),
    )
    return next((nome for nome, termos in pedidos if _tem(contexto, termos)), "neutral")


def _moderar_emocao(emocao: str, texto: str, contexto: str) -> str:
    """Evita interpretacoes teatrais em uma conversa cotidiana."""
    if emocao == "elation":
        euforia_explicita = ("euforia", "euforica", "extase", "gritando de alegria")
        if _tem(contexto, euforia_explicita) or _tem(texto, euforia_explicita):
            return emocao
        return "contentment"
    return emocao


def validar(texto: str, dados: Mapping[str, object] | None) -> Expression:
    """Converte a proposta da LLM apenas quando o texto fornece evidencia segura."""
    if not dados:
        return Expression()

    emotion_proposta = str(dados.get("emotion") or "neutral").strip().lower()
    style_proposto = str(dados.get("style") or "normal").strip().lower()
    effect_proposto = str(dados.get("effect") or "none").strip().lower()
    confidence = str(dados.get("confidence") or "low").strip().lower()
    if confidence not in CONFIDENCES:
        return Expression()

    emotion_proposta = emotion_proposta if emotion_proposta in EMOTIONS else "neutral"
    style_proposto = style_proposto if style_proposto in STYLES else "normal"
    effect_proposto = effect_proposto if effect_proposto in EFFECTS else "none"

    normalizado = _normalizar(texto)
    contexto = _normalizar(str(dados.get("context") or ""))
    if normalizado in {"acho que nao entendi direito"}:
        # Resposta de recuperacao do pipeline: deve soar clara e neutra, nunca
        # pensativa, confusa, triste ou satisfeita por causa de palavras soltas.
        return Expression(confidence=confidence)
    emotion_solicitada = _emocao_solicitada(contexto)
    pistas = {
        "elation": ("radiante", "maravilhoso", "melhor dia", "felicissima"),
        "amusement": (
            "haha", "hahaha", "kkk", "achei graca", "foi engracado",
            "muito engracado", "essa foi boa", "hilario",
        ),
        "enthusiasm": ("animada", "empolgada", "mal posso esperar", "bora"),
        "affection": (
            "te amo", "amo voce", "com carinho", "querido", "querida", "beijo",
            "apaixonada", "romantico", "romantica", "abraco", "meu amor",
        ),
        "contentment": (
            "que bom", "otimo", "perfeito", "estou feliz", "estou bem",
            "muito feliz", "gostei", "adorei", "foi especial",
            "lembranca boa", "inesquecivel",
        ),
        "relief": ("alivio", "ainda bem", "ufa"),
        "contemplation": (
            "fiquei pensando", "estou pensando", "me peguei pensando",
            "estive refletindo", "estou refletindo",
        ),
        "confusion": ("estou confusa", "fiquei confusa", "que confusao"),
        "surprise": ("nossa", "serio", "caramba", "uau", "surpresa"),
        "awe": ("impressionante", "incrivel", "magnifico"),
        "longing": ("saudade", "sinto falta"),
        "anger": ("raiva", "furiosa", "irritada", "odeio"),
        "fear": ("medo", "receio", "assustada", "apavorada"),
        "disgust": ("nojo", "nojento", "repugnante"),
        "bitterness": ("amargura", "ressentida", "injusto", "nao e justo"),
        "sadness": (
            "triste", "chateada", "doeu", "sinto muito", "faleceu", "morreu",
            "perdi", "perda", "luto", "doloroso", "abatida", "vazio",
        ),
        "shame": ("vergonha", "envergonhada", "me arrependo", "arrependida"),
        "helplessness": (
            "impotente", "nao posso fazer nada", "nao consigo fazer nada", "sem saida",
            "sem saber o que fazer",
        ),
        "determination": ("vou conseguir", "vamos", "decidida", "com certeza"),
        "pride": (
            "orgulho", "orgulhosa", "consegui", "conquista", "bolsa de estudos",
            "me formei", "fui aprovada", "fui aprovado",
        ),
    }
    prioridade = (
        "amusement", "sadness", "anger", "fear", "disgust", "shame",
        "helplessness", "confusion", "relief", "surprise", "awe", "longing", "affection",
        "determination", "pride", "elation", "enthusiasm", "bitterness",
        "contentment", "contemplation",
    )
    emotion_inferida = next(
        (nome for nome in prioridade if _tem_afirmado(normalizado, pistas[nome])),
        "neutral",
    )
    positivas = {
        "elation", "amusement", "enthusiasm", "determination", "pride",
        "contentment", "affection", "relief", "awe",
    }
    if emotion_solicitada == "contentment" and emotion_inferida in positivas:
        # "Feliz" define valencia, nao uma emocao unica. Preserve a nuance mais
        # especifica da resposta, como carinho, orgulho, diversao ou alivio.
        emotion = emotion_inferida
    elif emotion_solicitada == "contentment":
        # Um assunto feliz nao torna toda frase feliz. Detalhes factuais e
        # perguntas dentro desse assunto permanecem com a prosodia natural.
        emotion = "neutral"
    elif emotion_solicitada != "neutral":
        # Quando a pessoa pede explicitamente um tom ou tipo de relato, esse e o
        # objetivo vocal correto, independentemente de um rotulo aleatorio da LLM.
        emotion = emotion_solicitada
    else:
        # A LLM e apenas uma proposta. Fora de pedidos explicitos, uma emocao so
        # chega ao TTS quando a propria fala contem evidencia semantica suficiente.
        # Isso impede um rotulo aleatorio, mesmo com confidence=high, de alterar a voz.
        emotion = emotion_inferida

    emotion = _moderar_emocao(emotion, normalizado, contexto)

    pedidos_estilo = {
        "whispering": ("sussurra", "sussurre", "sussurrando", "fala baixinho", "fale baixinho"),
        "shouting": ("grita", "grite", "gritando", "fala alto", "fale alto"),
        "singing": ("canta", "cante", "cantando"),
    }
    estilo_pedido = next(
        (nome for nome, termos in pedidos_estilo.items() if _tem(contexto, termos)),
        "normal",
    )
    pista_sussurro = _tem(
        normalizado, ("segredo", "baixinho", "sussurro", "nao conta", "entre nos")
    )
    if estilo_pedido != "normal":
        style = estilo_pedido
    elif pista_sussurro:
        style = "whispering"
    else:
        style = "normal"

    # Efeitos so sao aplicados com vocalizacao ou intencao explicita reconhecivel.
    tem_risada = bool(re.search(r"\b(?:ha){2,}|\bk{3,}", normalizado))
    tem_suspiro = _tem(normalizado, ("ufa",)) or (
        effect_proposto == "sigh" and _tem(normalizado, ("ah", "ai"))
    )
    pistas_efeito = {
        "cough": ("cof", "cof cof"),
        "crying": ("chorando", "choro", "snif snif"),
        "screaming": ("aaaa", "socorro"),
        "burping": ("arroto", "burp"),
        "humming": ("hmm", "humm"),
        "sniff": ("snif",),
        "sneeze": ("atchim", "espirro"),
    }
    efeito_inferido = next(
        (nome for nome, termos in pistas_efeito.items() if _tem(normalizado, termos)),
        "none",
    )
    pedidos_efeito = {
        "cough": ("tosse", "tussa"),
        "laughter": ("risada", "rir", "de uma risada", "quero que voce ria"),
        "crying": ("chora", "chore", "chorando"),
        "screaming": ("grito", "grita", "grite"),
        "burping": ("arroto", "arrota", "arrote"),
        "humming": ("cantarola", "cantarole"),
        "sigh": ("suspiro", "suspira", "suspire"),
        "sniff": ("funga", "fungue"),
        "sneeze": ("espirra", "espirre", "espirro"),
    }
    efeito_pedido = next(
        (nome for nome, termos in pedidos_efeito.items() if _tem(contexto, termos)),
        "none",
    )
    if tem_risada:
        effect = "laughter"
    elif tem_suspiro:
        effect = "sigh"
    elif efeito_inferido != "none":
        effect = efeito_inferido
    elif efeito_pedido != "none":
        effect = efeito_pedido
    else:
        effect = "none"

    return Expression(emotion, style, effect, confidence)


def preparar(
    texto: str,
    dados: Mapping[str, object] | None,
) -> tuple[str, tuple[str, ...], Expression]:
    """Retorna a entrada Higgs, as tags aplicadas e a expressao aprovada."""
    limpo = limpar_tags(texto)
    expressao = validar(limpo, dados)
    tags: list[str] = []
    if expressao.emotion != "neutral":
        tags.append(f"<|emotion:{expressao.emotion}|>")
    # No modo de clonagem, algumas emocoes alteram o timbre mas comprimem a
    # modulacao. expressive_high recupera a prosodia sem trocar a emocao.
    reforco_prosodico = {
        "contentment", "pride", "relief", "contemplation", "confusion",
        "surprise", "awe", "longing", "determination", "enthusiasm",
        "anger", "fear", "disgust", "bitterness", "sadness", "shame",
        "helplessness",
    }
    contexto = _normalizar(str((dados or {}).get("context") or ""))
    if (
        expressao.style == "normal"
        and expressao.emotion in reforco_prosodico
        and expressao.confidence == "high"
    ):
        tags.append("<|prosody:expressive_high|>")
    if expressao.style != "normal":
        tags.append(f"<|style:{expressao.style}|>")
    if expressao.effect != "none":
        tags.append(f"<|sfx:{expressao.effect}|>")
    return "".join(tags) + limpo, tuple(tags), expressao


def compilar(texto: str, dados: Mapping[str, object] | None) -> str:
    """Converte metadados validados em tags somente no instante de gerar Higgs."""
    return preparar(texto, dados)[0]
