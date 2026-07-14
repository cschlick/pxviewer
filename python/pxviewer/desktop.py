"""Self-contained PyQt desktop viewer for pxviewer.

The desktop app opens two side-by-side windows:

1. **Viewport** — a `QWebEngineView` that loads the Mol* viewer and a volume demo.
2. **Controls** — a native Qt window with demo chooser and live interaction toggles.

A `LiveSession` is started in the background so the controls can toggle mouse
selection mode and receive click-built selections. The whole thing is served by
the local `Webapp` server, so no external browser is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .data import Atom
from .volume_demos import create_volume_demo, list_volume_demos
from .webapp import Webapp


def _check_qt() -> None:
    try:
        import PySide6  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The desktop viewer needs PySide6. Install it with: "
            "pip install 'pxviewer[desktop]'"
        ) from exc


class ViewportWindow:
    """A Qt window wrapping the Mol* viewer in a QWebEngineView."""

    def __init__(self, title: str = "pxviewer — viewport"):
        _check_qt()

        from PySide6.QtCore import QUrl
        from PySide6.QtWebEngineCore import QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        self._window = QWidget()
        self._window.setWindowTitle(title)
        self._window.setMinimumSize(640, 480)

        layout = QVBoxLayout(self._window)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view = QWebEngineView()
        settings = self._view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        layout.addWidget(self._view)

    def load(self, url: str) -> None:
        from PySide6.QtCore import QUrl
        self._view.load(QUrl(url))

    def show(self) -> None:
        self._window.show()

    def set_geometry(self, rect) -> None:
        self._window.setGeometry(rect)

    def widget(self):
        return self._window


class ControlsWindow:
    """A Qt window with controls for the viewport."""

    def __init__(self, desktop: "DesktopApp", title: str = "pxviewer — controls"):
        _check_qt()

        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QComboBox,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        self._desktop = desktop
        self._window = QWidget()
        self._window.setWindowTitle(title)
        self._window.setMinimumSize(320, 480)

        layout = QVBoxLayout(self._window)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel("<h2>pxviewer</h2>"))

        layout.addWidget(QLabel("Volume demo"))
        self._demo_select = QComboBox()
        for name, description in list_volume_demos():
            self._demo_select.addItem(f"{name}: {description}", name)
        layout.addWidget(self._demo_select)

        self._load_btn = QPushButton("Load demo")
        self._load_btn.clicked.connect(self._on_load_demo)
        layout.addWidget(self._load_btn)

        self._select_btn = QPushButton("Enable selection mode")
        self._select_btn.setCheckable(True)
        self._select_btn.clicked.connect(self._on_toggle_select)
        layout.addWidget(self._select_btn)

        self._clear_btn = QPushButton("Clear selection")
        self._clear_btn.clicked.connect(self._on_clear_selection)
        layout.addWidget(self._clear_btn)

        layout.addWidget(QLabel("Selected atoms:"))
        self._selection_label = QLabel("none")
        self._selection_label.setWordWrap(True)
        layout.addWidget(self._selection_label)

        layout.addStretch()

        self._status_label = QLabel("Ready")
        layout.addWidget(self._status_label)

    def show(self) -> None:
        self._window.show()

    def set_geometry(self, rect) -> None:
        self._window.setGeometry(rect)

    def widget(self):
        return self._window

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_load_demo(self) -> None:
        name = self._demo_select.currentData()
        if name:
            self._desktop.load_volume_demo(name)

    def _on_toggle_select(self, checked: bool) -> None:
        if checked:
            self._desktop.enable_mouse_selection(self._on_selection_changed)
            self._select_btn.setText("Disable selection mode")
        else:
            self._desktop.disable_mouse_selection()
            self._select_btn.setText("Enable selection mode")

    def _on_clear_selection(self) -> None:
        self._desktop.clear_selection()
        self._selection_label.setText("none")

    def _on_selection_changed(self, selection) -> None:
        indices = selection.indices
        self._selection_label.setText(
            f"{len(indices)} atom(s): {indices[:12]}{'...' if len(indices) > 12 else ''}"
        )


class DesktopApp:
    """Run the pxviewer desktop app with viewport and controls windows."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5173):
        _check_qt()

        from PySide6.QtWidgets import QApplication

        self._host = host
        self._port = port

        # Qt must be initialized before any widgets are created.
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv[:1])

        self._webapp = Webapp(host=host, port=port)

        # A single off-screen atom is enough to keep the LiveSession WebSocket
        # open and route click-mode / mouse-selection messages.
        dummy_atom = Atom(
            id=1, element="C", name="C", resname="UNL", resseq=1, chain="A",
            x=100.0, y=0.0, z=0.0,
        )
        self._session = self._make_live_session([dummy_atom])

        self._viewport = ViewportWindow()
        self._controls = ControlsWindow(self)

    def _make_live_session(self, atoms):
        try:
            from .live import LiveSession
            return LiveSession(atoms)
        except Exception as exc:  # pragma: no cover - missing websockets
            raise ImportError(
                "The desktop viewer needs websockets. Install it with: "
                "pip install 'pxviewer[live]'"
            ) from exc

    def start(self) -> int:
        self._webapp.start()
        self._session.start(host=self._host, port=0)

        print(f"pxviewer desktop viewer running at {self._webapp.url}", flush=True)

        ws_url = f"ws://{self._host}:{self._session.port}"

        self._viewport.show()
        self._controls.show()
        self._arrange_windows()

        # Load the first demo by default.
        first_demo = self._controls._demo_select.currentData()
        if first_demo:
            self.load_volume_demo(first_demo, ws_url=ws_url)

        return self._app.exec()

    def stop(self) -> None:
        self._session.stop()
        self._webapp.stop()

    def _arrange_windows(self) -> None:
        """Place the two windows side by side on the primary screen."""
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        screen = app.primaryScreen()
        if screen is None:
            # Fallback geometry if no screen is reported.
            rect = QRect(0, 0, 1920, 1080)
        else:
            rect = screen.availableGeometry()

        x, y, total_width, total_height = rect.x(), rect.y(), rect.width(), rect.height()
        half_width = total_width // 2

        self._viewport.set_geometry(QRect(x, y, half_width, total_height))
        self._controls.set_geometry(QRect(x + half_width, y, total_width - half_width, total_height))

    def load_volume_demo(self, name: str, *, ws_url: Optional[str] = None) -> None:
        """Generate a volume demo and load it in the viewport."""
        if ws_url is None:
            ws_url = f"ws://{self._host}:{self._session.port}"

        demo_dir = self._webapp.volume_dir / name
        demo_dir.mkdir(parents=True, exist_ok=True)
        create_volume_demo(
            name,
            mrc_path=demo_dir / "volume.mrc",
            mvsj_path=demo_dir / "volume.mvsj",
            shape=(32, 32, 32),
        )

        base = self._webapp.url
        mvsj_url = f"/demo/{name}/volume.mvsj"
        url = f"{base}index.html?mvsj={mvsj_url}&ws={ws_url}"
        self._viewport.load(url)

    def enable_mouse_selection(self, on_change) -> None:
        self._session.enable_mouse_selection(on_change)

    def disable_mouse_selection(self) -> None:
        self._session.disable_mouse_selection()

    def clear_selection(self) -> None:
        self._session.clear_selection()


def run_desktop(host: str = "127.0.0.1", port: int = 5173) -> int:
    """Start the desktop app with viewport and controls windows."""
    _check_qt()

    desktop = DesktopApp(host=host, port=port)
    try:
        return desktop.start()
    finally:
        desktop.stop()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(run_desktop())
