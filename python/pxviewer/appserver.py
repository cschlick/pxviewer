"""A tiny static file server for the pxviewer frontend.

Serving the built frontend and the `LiveSession` WebSocket from a single command
avoids the classic trap of pointing a browser straight at the WebSocket port. The
server redirects the root URL to `index.html?ws=<ws_url>`, so opening the printed
http:// address just works — the page's JavaScript connects the WebSocket itself.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import threading
from pathlib import Path
from typing import Optional, Tuple


def find_frontend_dir() -> Optional[Path]:
    """Locate the frontend directory in an editable checkout, if present."""
    candidate = Path(__file__).resolve().parents[2] / "frontend"
    if (candidate / "index.html").exists():
        return candidate
    return None


def frontend_is_built(frontend_dir: Path) -> bool:
    return (frontend_dir / "build" / "index.js").exists()


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, ws_url: str = "", **kwargs):
        self._ws_url = ws_url
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (name required by base class)
        # Send visitors of the bare root to the viewer page wired to the WS URL.
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", f"/index.html?ws={self._ws_url}")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        # Never cache: the frontend bundle is rebuilt in place during development, and
        # SimpleHTTPRequestHandler's Last-Modified otherwise lets a viewer keep running
        # a stale build/index.js after a rebuild.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, *args) -> None:  # keep the console focused on the demo
        pass


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve_frontend(
    frontend_dir: Path,
    ws_url: str,
    *,
    host: str = "127.0.0.1",
    port: int = 5173,
) -> Tuple[_Server, int]:
    """Serve ``frontend_dir`` in a background thread. Returns (server, actual_port).

    Falls back to an ephemeral port if the requested one is taken.
    """
    handler = functools.partial(_Handler, directory=str(frontend_dir), ws_url=ws_url)
    try:
        httpd = _Server((host, port), handler)
    except OSError:
        httpd = _Server((host, 0), handler)
    actual_port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, name="pxviewer-http", daemon=True)
    thread.start()
    return httpd, actual_port


def stop_all(*steps) -> None:
    """Run each teardown step, retrying so a repeated Ctrl-C can't abort cleanup.

    Steps must be idempotent (they may run more than once if interrupted).
    """
    while True:
        try:
            for step in steps:
                step()
            return
        except KeyboardInterrupt:
            continue


def stop_frontend(httpd) -> None:
    """Stop a server from `serve_frontend`/`announce_viewer`, ignoring errors."""
    if httpd is None:
        return
    try:
        httpd.shutdown()
    except Exception:
        pass
    try:
        httpd.server_close()
    except Exception:
        pass


def announce_viewer(host: str, ws_url: str, *, http_port: int = 5173, serve: bool = True):
    """Serve the frontend if possible and print how to open the viewer.

    Returns the http server (call ``.shutdown()`` to stop it) or None.
    """
    frontend_dir = find_frontend_dir() if serve else None
    if frontend_dir is not None and frontend_is_built(frontend_dir):
        httpd, actual = serve_frontend(frontend_dir, ws_url, host=host, port=http_port)
        print(f"Open the viewer in your browser:  http://{host}:{actual}/", flush=True)
        return httpd

    if serve and frontend_dir is not None:
        print("(frontend found but not built — run `cd frontend && npm run build`)", flush=True)
    print(f"Then point the frontend page at:  ?ws={ws_url}", flush=True)
    return None
