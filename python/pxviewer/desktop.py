"""Self-contained PyQt desktop viewer for pxviewer.

This opens the same pxviewer webapp in a `QWebEngineView` window. The HTTP server
that serves the app and generated volume files is started inside the Python
process, so the only external requirement is the PySide6 package.
"""

from __future__ import annotations

import sys
import threading

from .webapp import Webapp


def _check_qt() -> None:
    try:
        import PySide6  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The desktop viewer needs PySide6. Install it with: "
            "pip install 'pxviewer[desktop]'"
        ) from exc


class DesktopWindow:
    """A minimal PyQt window wrapping the pxviewer webapp in a webview."""

    def __init__(self, webapp: Webapp, *, title: str = "pxviewer"):
        _check_qt()

        from PySide6.QtCore import QUrl
        from PySide6.QtWebEngineCore import QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QApplication

        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)
        self._webapp = webapp

        self._view = QWebEngineView()
        self._view.setWindowTitle(title)
        self._view.setMinimumSize(1024, 768)

        settings = self._view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)

        self._view.load(QUrl(webapp.url))
        self._view.show()

    def run(self) -> int:
        """Run the Qt event loop and shut down the webapp on exit."""
        try:
            return self._app.exec()
        finally:
            self._webapp.stop()


def run_desktop(host: str = "127.0.0.1", port: int = 5173) -> int:
    """Start the webapp and open it in a PyQt webview.

    Returns the Qt application exit code.
    """
    _check_qt()

    webapp = Webapp(host=host, port=port)
    actual_port = webapp.start()
    print(f"pxviewer desktop viewer running at {webapp.url}", flush=True)

    window = DesktopWindow(webapp, title="pxviewer")
    return window.run()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(run_desktop())
