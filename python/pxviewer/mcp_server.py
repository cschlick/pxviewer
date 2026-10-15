"""An MCP server that drives a live pxviewer session.

The server owns a real ``DesktopApp`` — the same windows a user opens — so an
agent's figure work happens in the viewer the user is watching, and everything
the desktop can do (models, maps, selections, camera, saved views) is
available to tools verbatim.

Threading: Qt must live on the main thread (macOS requires it), so the MCP
stdio loop runs on a daemon thread. Tool calls that touch widgets are
marshalled onto the Qt thread through a queued signal; ``LiveSession`` calls
(screenshot, camera, background) are already thread-safe — they block on the
browser round trip while the Qt loop delivers the answer — and run directly
on the MCP thread.

Run it as ``python -m pxviewer mcp`` and point a client at the stdio
transport, e.g. in an MCP client config::

    {"command": "/path/to/env/bin/python", "args": ["-m", "pxviewer", "mcp"]}
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

_app = None          # DesktopApp, main thread
_invoker = None      # _Invoker, main thread

MCP_PORT = 8753      # fixed so MCP client configs can name a stable URL


def _make_invoker():
    from PySide6.QtCore import QObject, Signal

    class _Invoker(QObject):
        run_job = Signal(object)

        def __init__(self):
            super().__init__()
            self.run_job.connect(self._run)

        def _run(self, job):
            fn, fut = job
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(fn())
                except BaseException as exc:
                    fut.set_exception(exc)

    return _Invoker()


def _invoke(fn):
    """Run ``fn`` on the Qt thread, return a Future for its result."""
    fut = Future()
    _invoker.run_job.emit((fn, fut))
    return fut


async def _qt(fn):
    """awaitable _invoke for use inside async tool handlers."""
    return await asyncio.wrap_future(_invoke(fn))


def _session():
    session = _app._control_session()
    if session is None:
        raise RuntimeError("viewport not connected yet — try again in a moment")
    return session


def _wait_ready(timeout: float = 60.0) -> None:
    """Block until the Mol* page answers a screenshot — the signal it is live.

    Requests sent before the viewer's handler exists are dropped, so this is
    short repeated tries rather than one long wait (the wait_viewer pattern).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        session = _app._control_session()
        if session is not None:
            try:
                if session.screenshot(timeout=5):
                    return
            except Exception:
                pass
        time.sleep(0.5)
    raise RuntimeError("viewport never came up")


def _model_id(name):
    """Resolve a model by name (or the active/first model when name is None)."""
    if name is None:
        return _app._active_model_id or (_app._models[0]["id"] if _app._models else None)
    for entry in _app._models:
        if entry["id"] == name or entry.get("name") == name:
            return entry["id"]
    return None


def _volume_id(name):
    if name is None:
        return _app._volumes[0]["id"] if _app._volumes else None
    for entry in _app._volumes:
        if entry["id"] == name or entry.get("name") == name:
            return entry["id"]
    return None


def attach(app, port: int = MCP_PORT) -> "str | None":
    """Serve the MCP tool surface over HTTP for an already-running DesktopApp.

    Called on the Qt thread during ``DesktopApp.start`` so the invoker's
    signals land on the right loop. Returns the URL clients point at, or None
    if the ``mcp`` package is absent or the port is taken (a second pxviewer
    instance keeps working, just without an endpoint).
    """
    global _app, _invoker
    try:
        import mcp  # noqa: F401
    except ImportError:
        return None
    import socket

    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        print(f"pxviewer: MCP port {port} in use — this instance serves no endpoint",
              flush=True)
        return None
    finally:
        probe.close()

    _app = app
    _invoker = _make_invoker()
    server = _make_server()
    server.settings.host = "127.0.0.1"
    server.settings.port = port

    def serve():
        try:
            server.run(transport="streamable-http")
        except Exception as exc:  # noqa: BLE001 — nothing else reports it
            print("pxviewer: MCP server stopped: %r" % exc, file=sys.stderr)

    threading.Thread(target=serve, name="pxviewer-mcp", daemon=True).start()
    url = f"http://127.0.0.1:{port}/mcp"
    print(f"pxviewer: MCP tools at {url}  (PXVIEWER_MCP=0 to disable)", flush=True)
    return url


def _start_app():
    """Build the desktop app the way run_desktop starts it; return the app."""
    global _app, _invoker
    from PySide6.QtWidgets import QApplication

    from pxviewer import gpu as gpu_backend

    gpu_mode = gpu_backend.configure(None)
    qt_app = QApplication(sys.argv[:1])
    qt_app.setApplicationName("pxviewer")

    from pxviewer.desktop import DesktopApp

    can_hide = gpu_mode == "hardware"
    app = DesktopApp(port=0, can_hide=can_hide)
    app._webapp.start()
    app._main.show()
    app._controls.widget().show()
    app._main.resize(1400, 900)
    _app = app
    _invoker = _make_invoker()
    return qt_app, app


def _make_server():
    # FastMCP resolves annotations with eval_str against the module globals, so
    # Image must be visible there even though mcp is imported lazily.
    global Image
    from mcp.server.fastmcp import FastMCP, Image

    mcp = FastMCP("pxviewer")

    @mcp.tool()
    async def status() -> dict:
        """Scene state: loaded objects, selections, camera."""
        def work():
            return {
                "models": [{"id": e["id"], "name": e.get("name"),
                            "rep": e.get("rep"), "color": e.get("color")}
                           for e in _app._models],
                "volumes": [{"id": e["id"], "name": e.get("name"),
                             "iso": e.get("iso"), "opacity": e.get("opacity")}
                            for e in _app._volumes],
                "selection": {mid: len(ix) for mid, ix in
                              _app._scene_selection.items()},
            }
        state = await _qt(work)
        state["camera"] = _session().camera_state()
        state["background"] = _session().background_color()
        return state

    @mcp.tool()
    async def load(path: str) -> dict:
        """Load a structure (PDB/mmCIF) or map (MRC/CCP4) file into the scene."""
        def work():
            return _app.load_file(str(Path(path).expanduser()))
        result = await _qt(work)
        _wait_ready()
        return {"loaded": result}

    @mcp.tool()
    async def load_view(path: str) -> dict:
        """Restore a saved .pxview.json — objects, representations, camera."""
        from pxviewer import viewstate
        await _qt(lambda: viewstate.load_view(
            _app, str(Path(path).expanduser())))
        _wait_ready()
        return {"loaded_view": path}

    @mcp.tool()
    async def save_view(path: str, bundle_data: bool = True) -> dict:
        """Save the current scene as a .pxview.json (with bundled data by
        default, so the view reproduces anywhere)."""
        from pxviewer import viewstate
        doc = await _qt(lambda: viewstate.save_view(
            _app, str(Path(path).expanduser()), bundle_data=bundle_data))
        return {"saved": path, "objects": len(doc.get("objects", []))}

    @mcp.tool()
    async def select(expression: str) -> dict:
        """Apply a cctbx selection string (e.g. 'chain A and resseq 5:14')."""
        def work():
            controls = _app._controls
            controls._select_expr.setText(expression)
            controls._on_select_expression()
            return {mid: len(ix) for mid, ix in _app._scene_selection.items()}
        return {"selected": await _qt(work)}

    @mcp.tool()
    async def clear_selection() -> dict:
        """Clear the standing selection."""
        await _qt(lambda: _app.clear_selection())
        return {"selected": {}}

    @mcp.tool()
    async def set_representation(rep: str, model: str = None,
                                 color: str = None) -> dict:
        """Set a model's representation ('cartoon', 'ball-and-stick',
        'spacefill', …) and optionally its color theme."""
        def work():
            mid = _model_id(model)
            if mid is None:
                raise ValueError("no such model: %r" % model)
            _app.set_model_representation(mid, rep)
            if color is not None:
                _app.set_model_color(mid, color)
            return mid
        return {"model": await _qt(work), "rep": rep, "color": color}

    @mcp.tool()
    async def set_volume(name: str = None, iso: float = None,
                         opacity: float = None, color: str = None,
                         visible: bool = None) -> dict:
        """Adjust a map's isosurface: iso level, opacity, color, visibility."""
        def work():
            vid = _volume_id(name)
            if vid is None:
                raise ValueError("no such volume: %r" % name)
            if iso is not None:
                _app.set_volume_iso(vid, iso)
            if opacity is not None:
                _app.set_volume_opacity(vid, opacity)
            if color is not None:
                _app.set_volume_color(vid, color)
            if visible is not None:
                _app.set_volume_visible(vid, visible)
            return vid
        return {"volume": await _qt(work)}

    @mcp.tool()
    async def set_background(color: str) -> dict:
        """Set the viewport background (hex color, e.g. '#101018')."""
        _session().set_background(color)
        return {"background": color}

    @mcp.tool()
    async def reset_view() -> dict:
        """Reframe the camera on the whole scene."""
        _session().reset_view()
        return {"ok": True}

    @mcp.tool()
    async def set_camera(state: dict) -> dict:
        """Set the camera from a state dict as returned by camera_state."""
        _session().set_camera(state)
        return {"ok": True}

    @mcp.tool()
    async def screenshot() -> Image:
        """Capture the current Mol* viewport frame and return it as an image.
        Call this after making changes to *see* the scene — it is the way to
        check that a figure looks right before saving the view."""
        _wait_ready(timeout=15)
        png = _session().screenshot(timeout=30)
        if not png:
            raise RuntimeError("screenshot came back empty")
        return Image(data=png, format="png")

    return mcp


def run() -> int:
    """Start the app on the main thread, the MCP stdio loop on a daemon
    thread, and run the Qt event loop until the app exits."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("pxviewer mcp needs the 'mcp' package: pip install 'mcp>=1.10,<2'",
              file=sys.stderr)
        return 1

    qt_app, app = _start_app()
    server = _make_server()

    def serve():
        try:
            server.run()
        except Exception as exc:  # noqa: BLE001 — nothing else reports it
            print("mcp server stopped: %r" % exc, file=sys.stderr)

    threading.Thread(target=serve, name="pxviewer-mcp", daemon=True).start()

    # A client may take a moment to send its first tool call; the viewport
    # page connects as soon as the window is up, which also warms the session
    # so the first screenshot does not race the page load.
    return qt_app.exec()


if __name__ == "__main__":
    sys.exit(run())
