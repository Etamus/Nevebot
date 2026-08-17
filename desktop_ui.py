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

_WINDOW_WIDTH = 1280
_WINDOW_HEIGHT = 820


def _configurar_identidade_windows() -> None:
    """Define uma identidade própria para a janela na barra de tarefas."""
    if sys.platform != "win32":
        return
    try:
        from ctypes import windll

        windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Nevebot.ControlCenter"
        )
    except Exception:
        log.warning("Nao foi possivel definir a identidade da janela do Nevebot.")


def _posicao_centralizada(width: int, height: int) -> tuple[int | None, int | None]:
    """Calcula o centro da area util do monitor principal no Windows."""
    if sys.platform != "win32":
        return None, None
    try:
        from ctypes import byref, windll
        from ctypes.wintypes import RECT

        work_area = RECT()
        if not windll.user32.SystemParametersInfoW(0x0030, 0, byref(work_area), 0):
            raise OSError("SystemParametersInfoW falhou")
        available_width = work_area.right - work_area.left
        available_height = work_area.bottom - work_area.top
        x = work_area.left + max(0, (available_width - width) // 2)
        y = work_area.top + max(0, (available_height - height) // 2)
        return int(x), int(y)
    except Exception:
        log.warning("Nao foi possivel centralizar a janela automaticamente.")
        return None, None


def _preparar_icone_janela() -> str | None:
    """Gera o ICO exigido pelo WinForms usando o favicon do projeto."""
    favicon_path = _BASE_DIR / "web" / "favicon.png"
    icon_path = _BASE_DIR / "data" / "nevebot-favicon.ico"
    if not favicon_path.is_file():
        log.warning("Favicon da interface nao encontrado em %s.", favicon_path)
        return None

    try:
        if (
            icon_path.is_file()
            and icon_path.stat().st_mtime_ns >= favicon_path.stat().st_mtime_ns
        ):
            return str(icon_path)

        from PIL import Image

        icon_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(favicon_path) as source:
            favicon = source.convert("RGBA")
            favicon.save(
                icon_path,
                format="ICO",
                sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                       (128, 128), (256, 256)],
            )
        return str(icon_path)
    except Exception:
        log.exception("Falha ao preparar o icone da janela a partir de %s.", favicon_path)
        return None


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
        _configurar_identidade_windows()
        _habilitar_microfone_local()
        window_x, window_y = _posicao_centralizada(_WINDOW_WIDTH, _WINDOW_HEIGHT)
        _window = webview.create_window(
            "Nevebot",
            url=url,
            js_api=_DesktopApi(),
            width=_WINDOW_WIDTH,
            height=_WINDOW_HEIGHT,
            x=window_x,
            y=window_y,
            min_size=(900, 620),
            maximized=False,
            background_color="#000000",
            text_select=True,
        )

        def definir_modo_janela(modo: str) -> None:
            try:
                _window.evaluate_js(
                    f'document.documentElement.dataset.windowState = "{modo}";'
                )
            except Exception:
                log.debug("Nao foi possivel atualizar o estado visual da janela.")

        _window.events.maximized += lambda: definir_modo_janela("maximized")
        _window.events.restored += lambda: definir_modo_janela("normal")

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
        window_icon = _preparar_icone_janela()

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
            icon=window_icon,
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
