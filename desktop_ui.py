"""Janela desktop da interface web, com fallback para navegadores instalados."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlparse


log = logging.getLogger("desktop_ui")
_BASE_DIR = Path(__file__).parent
_window = None


def _habilitar_microfone_local() -> None:
    """Autoriza somente o microfone solicitado pela interface HTTP local."""
    if sys.platform != "win32":
        return

    from webview.platforms import edgechromium
    from Microsoft.Web.WebView2.Core import (
        CoreWebView2PermissionKind,
        CoreWebView2PermissionState,
    )

    edge_class = edgechromium.EdgeChrome
    if getattr(edge_class, "_neve_microfone_configurado", False):
        return

    on_ready_original = edge_class.on_webview_ready

    def on_webview_ready(self, sender, init_args) -> None:
        if init_args.IsSuccess:
            url_local = urlparse(str(self.pywebview_window.real_url or ""))
            if url_local.scheme == "http" and url_local.hostname in {
                "127.0.0.1",
                "localhost",
            }:
                porta = f":{url_local.port}" if url_local.port else ""
                origem_permissao = f"http://{url_local.hostname}{porta}"
                sender.CoreWebView2.Profile.SetPermissionStateAsync(
                    CoreWebView2PermissionKind.Microphone,
                    origem_permissao,
                    CoreWebView2PermissionState.Allow,
                )

            def permission_requested(_sender, args) -> None:
                origem = urlparse(str(args.Uri))
                origem_local = origem.scheme == "http" and origem.hostname in {
                    "127.0.0.1",
                    "localhost",
                }
                if (
                    origem_local
                    and args.PermissionKind == CoreWebView2PermissionKind.Microphone
                ):
                    args.State = CoreWebView2PermissionState.Allow
                    args.SavesInProfile = True
                    args.Handled = True
                    log.info("Permissao de microfone concedida para %s.", args.Uri)

            self._neve_permission_handler = permission_requested
            sender.CoreWebView2.PermissionRequested += permission_requested
        on_ready_original(self, sender, init_args)

    edge_class.on_webview_ready = on_webview_ready
    edge_class._neve_microfone_configurado = True


def _url_http_valida(url: str) -> bool:
    parsed = urlparse(str(url))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def abrir_no_navegador(url: str) -> bool:
    """Abre uma URL no navegador padrao e tenta alternativas conhecidas."""
    if not _url_http_valida(url):
        log.warning("URL externa recusada: %r", url)
        return False

    try:
        if webbrowser.open(url, new=2, autoraise=True):
            return True
    except Exception:
        log.exception("Falha ao abrir URL no navegador padrao.")

    for nome in ("windows-default", "edge", "chrome", "firefox"):
        try:
            if webbrowser.get(nome).open(url, new=2, autoraise=True):
                return True
        except Exception:
            continue

    locais = [
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Mozilla Firefox/firefox.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Mozilla Firefox/firefox.exe",
    ]
    for executavel in locais:
        if not str(executavel.parent) or not executavel.is_file():
            continue
        try:
            subprocess.Popen([str(executavel), url])
            return True
        except OSError:
            continue

    log.error("Nenhum navegador disponivel conseguiu abrir %s.", url)
    return False


class _DesktopApi:
    def abrir_url(self, url: str) -> bool:
        return abrir_no_navegador(url)

    def fechar_janela(self) -> bool:
        window = _window
        if window is None:
            return False
        try:
            window.destroy()
            return True
        except Exception:
            log.exception("Falha ao fechar a janela pywebview.")
            return False


def iniciar_interface(url: str, bot_encerrado: threading.Event) -> bool:
    """Abre a UI nativa. Retorna True se o pywebview chegou a iniciar."""
    global _window

    renderer_compativel = threading.Event()
    try:
        import webview
    except ImportError:
        log.exception("pywebview nao esta instalado; usando navegador externo.")
        abrir_no_navegador(url)
        return False

    try:
        _habilitar_microfone_local()
        _window = webview.create_window(
            "Nevebot",
            url=url,
            js_api=_DesktopApi(),
            width=1280,
            height=820,
            min_size=(900, 620),
            background_color="#0a0a0a",
            text_select=True,
        )

        def fechar_quando_bot_encerrar() -> None:
            bot_encerrado.wait()
            if _window is not None:
                try:
                    _window.destroy()
                except Exception:
                    pass

        threading.Thread(
            target=fechar_quando_bot_encerrar,
            daemon=True,
            name="ui-bot-monitor",
        ).start()

        storage_path = _BASE_DIR / "data" / "pywebview-v2"
        storage_path.mkdir(parents=True, exist_ok=True)

        def validar_renderer() -> None:
            if webview.renderer != "edgechromium":
                log.error(
                    "Renderer pywebview incompativel (%s); usando navegador externo.",
                    webview.renderer,
                )
                try:
                    _window.destroy()
                except Exception:
                    pass
                return
            renderer_compativel.set()

        log.info("Abrindo interface nativa com Edge WebView2.")
        webview.start(
            validar_renderer,
            gui="edgechromium",
            private_mode=False,
            storage_path=str(storage_path),
        )
        if not renderer_compativel.is_set():
            abrir_no_navegador(url)
            return False
        return True
    except Exception:
        log.exception("Falha ao iniciar pywebview; tentando navegador externo.")
        abrir_no_navegador(url)
        return False
    finally:
        _window = None
