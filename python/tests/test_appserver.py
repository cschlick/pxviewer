"""Tests for the bundled frontend server and the WebSocket port's HTTP guard."""

import urllib.error
import urllib.request

import pytest

from pxviewer import Atom, LiveSession
from pxviewer import appserver

websockets = pytest.importorskip("websockets")


def _get(url, timeout=5):
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 (localhost test)


def test_root_redirects_to_wired_index():
    frontend = appserver.find_frontend_dir()
    if frontend is None or not appserver.frontend_is_built(frontend):
        pytest.skip("frontend not built")

    ws_url = "ws://127.0.0.1:9999"
    httpd, port = appserver.serve_frontend(frontend, ws_url, port=0)
    try:
        # Do not follow the redirect: inspect the Location header directly.
        req = urllib.request.Request(f"http://127.0.0.1:{port}/")

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        try:
            opener.open(req)
            location = None
        except urllib.error.HTTPError as exc:
            assert exc.code == 302
            location = exc.headers.get("Location")
        assert location == f"/index.html?ws={ws_url}"

        page = _get(f"http://127.0.0.1:{port}/index.html?ws={ws_url}").read().decode()
        assert "<!DOCTYPE html>" in page
    finally:
        httpd.shutdown()


def test_stop_all_survives_repeated_interrupt():
    events = []

    class ShutdownOnce:
        """Raises KeyboardInterrupt the first time (like a Ctrl-C mid-cleanup)."""

        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            events.append("shutdown")

    appserver.stop_all(lambda: events.append("stop_session"), ShutdownOnce())

    # Cleanup still completed despite the interrupt during the second step.
    assert "shutdown" in events


def test_ws_port_answers_plain_http_without_crashing():
    session = LiveSession([Atom(id=1, element="C", x=0, y=0, z=0), Atom(id=2, element="C", x=1, y=0, z=0)])
    session.start(port=0)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"http://127.0.0.1:{session.port}/")
        assert exc.value.code == 426  # Upgrade Required, served politely
        assert b"WebSocket endpoint" in exc.value.read()
    finally:
        session.stop()
