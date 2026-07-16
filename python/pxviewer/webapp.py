"""Web application for presenting pxviewer volume demos in a browser.

The webapp serves a small HTML app at the root URL. The app lets users pick a
volume demo, generates the corresponding MRC/MVSJ files on demand, and loads them
in the bundled Mol* viewer. The same app URL can be opened by a desktop PyQt
webview so the presentation layer is shared between the two formats.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import tempfile
import threading
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple

from .appserver import find_frontend_dir, frontend_is_built
from .volume_demos import create_volume_demo, list_volume_demos


class _WebappHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the webapp, viewer, and generated demo files.

    Routing:
      - /api/volume-demos        -> JSON list of available volume demos
      - /api/volume-demo/<name>  -> generate files and return {"mvsj_url": ...}
      - /demo/<name>/<file>      -> serve generated demo files
      - /app.html, /             -> serve the webapp HTML
      - /index.html, /build/*    -> serve the bundled viewer frontend
    """

    def __init__(
        self,
        *args,
        volume_dir: Path,
        frontend_dir: Path,
        **kwargs,
    ):
        self.volume_dir = volume_dir
        self.frontend_dir = frontend_dir
        super().__init__(*args, directory=str(volume_dir), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (name required by base class)
        # Route on the path alone: the viewer page is loaded as
        # /index.html?mvsj=...&ws=..., and the query must not defeat the match.
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == "/api/volume-demos":
            self._send_json([{"name": n, "description": d} for n, d in list_volume_demos()])
            return
        if path.startswith("/api/volume-demo/"):
            parts = path.split("/")
            if len(parts) >= 4:
                name = urllib.parse.unquote(parts[3])
                self._load_volume_demo(name)
                return
            self.send_error(404)
            return
        if path == "/" or path == "/app.html":
            self._serve_app_html()
            return
        if path == "/index.html":
            self._serve_frontfile("index.html")
            return
        if path == "/favicon.png":
            self._serve_frontfile("favicon.png")
            return
        if path.startswith("/build/"):
            self._serve_frontfile(path.lstrip("/"))
            return
        super().do_GET()

    def _load_volume_demo(self, name: str) -> None:
        try:
            demo_dir = self.volume_dir / name
            demo_dir.mkdir(parents=True, exist_ok=True)
            mrc_path = demo_dir / "volume.mrc"
            mvsj_path = demo_dir / "volume.mvsj"
            create_volume_demo(
                name,
                mrc_path=mrc_path,
                mvsj_path=mvsj_path,
                shape=(32, 32, 32),
            )
            self._send_json({"mvsj_url": f"/demo/{urllib.parse.quote(name, safe='')}/volume.mvsj"})
        except ValueError as exc:
            self.send_error(400, explain=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self.send_error(500, explain=str(exc))

    def _serve_app_html(self) -> None:
        app_html = self.frontend_dir / "app.html"
        if not app_html.exists():
            self.send_error(404, explain="app.html not found")
            return
        self._serve_file(app_html, "text/html")

    def _serve_frontfile(self, relative: str) -> None:
        target = self.frontend_dir / relative
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return
        ctype = self.guess_type(str(target))
        self._serve_file(target, ctype)

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # Never cache: the viewport is a long-lived QWebEngine view and the frontend
        # bundle is rebuilt in place during development — a cached build/index.js
        # silently keeps running old code after a rebuild.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict | list) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keep the console focused on the app
        pass

    def translate_path(self, path: str) -> str:
        """Resolve requests against the volume directory or the frontend directory."""
        # Strip query string and fragment, then normalise.
        path = path.split("?", 1)[0]
        path = path.split("#", 1)[0]
        path = urllib.parse.unquote(path)
        path = os.path.normpath(path)
        words = [w for w in path.split("/") if w and w not in (os.curdir, os.pardir)]

        if len(words) >= 2 and words[0] == "demo":
            demo_dir = self.volume_dir / words[1]
            candidate = demo_dir.joinpath(*words[2:])
            if candidate.exists():
                return str(candidate)
            return str(candidate)

        if words and words[0] == "build":
            candidate = self.frontend_dir.joinpath(*words)
            if candidate.exists():
                return str(candidate)
            return str(candidate)

        # Default to the volume directory; 404 will be produced for missing files.
        return str(self.volume_dir.joinpath(*words))


class _WebappServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Webapp:
    """Run the pxviewer webapp.

    The app is served from a background thread so the calling code can open a
    browser or webview and keep it alive until the user stops it.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.volume_dir = Path(tempfile.mkdtemp(prefix="pxviewer-webapp-"))
        self._server: Optional[_WebappServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> int:
        """Start the webapp server and return the bound port."""
        frontend_dir = find_frontend_dir()
        if frontend_dir is None or not frontend_is_built(frontend_dir):
            raise RuntimeError(
                "frontend not built. Run `cd frontend && npm install && npm run build`"
            )

        handler = functools.partial(
            _WebappHandler,
            volume_dir=self.volume_dir,
            frontend_dir=frontend_dir,
        )
        try:
            self._server = _WebappServer((self.host, self.port), handler)
        except OSError:
            self._server = _WebappServer((self.host, 0), handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="pxviewer-webapp", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        """Shut down the webapp server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        """The URL to open the webapp."""
        return f"http://{self.host}:{self.port}/"


def run_webapp(host: str = "127.0.0.1", port: int = 5173, open_browser: bool = True) -> None:
    """Start the webapp and block until the user stops it."""
    import webbrowser

    app = Webapp(host=host, port=port)
    actual = app.start()
    url = f"http://{host}:{actual}/"
    print(f"pxviewer webapp running at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\nstopping...", flush=True)
    finally:
        app.stop()
