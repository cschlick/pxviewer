"""The bundled frontend server, and the WebSocket port's HTTP guard."""

from __future__ import absolute_import, division, print_function

import sys
import urllib.error
import urllib.request

from libtbx.test_utils import Exception_expected

from pxviewer import appserver
from pxviewer.regression.tst_utils import have, skip

if not have("websockets"):
    skip("websockets not available")

from pxviewer import LiveSession                    # noqa: E402


def get(url, timeout=5):
    """Fetch a URL and return its body, always reading it (an unread body makes the
    server log a broken pipe and buries the test's own output)."""
    resp = urllib.request.urlopen(url, timeout=timeout)     # localhost only
    try:
        return resp.read()
    finally:
        resp.close()


def exercise_root_redirects_to_a_wired_index():
    frontend = appserver.find_frontend_dir()
    if frontend is None or not appserver.frontend_is_built(frontend):
        print("  skipping: frontend not built")
        return

    ws_url = "ws://127.0.0.1:9999"
    httpd, port = appserver.serve_frontend(frontend, ws_url, port=0)
    try:
        # Do not follow the redirect: inspect the Location header directly.
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        location = None
        try:
            opener.open(urllib.request.Request("http://127.0.0.1:%d/" % port))
            raise Exception_expected
        except urllib.error.HTTPError as exc:
            assert exc.code == 302
            location = exc.headers.get("Location")
        assert location == "/index.html?ws=%s" % ws_url

        page = get("http://127.0.0.1:%d/index.html?ws=%s" % (port, ws_url)).decode()
        assert "<!DOCTYPE html>" in page
    finally:
        httpd.shutdown()


def exercise_stop_all_survives_a_repeated_interrupt():
    events = []

    class ShutdownOnce(object):
        """Raises KeyboardInterrupt the first time, like a Ctrl-C mid-cleanup."""

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


def exercise_the_ws_port_answers_plain_http_without_crashing():
    session = LiveSession.from_sites([[0, 0, 0], [1, 0, 0]])
    session.start(port=0)
    try:
        # try/except rather than libtbx's raises(): that helper instantiates the exception
        # class with no arguments to test isinstance, and HTTPError needs five.
        try:
            get("http://127.0.0.1:%d/" % session.port)
            raise Exception_expected
        except urllib.error.HTTPError as exc:
            assert exc.code == 426              # Upgrade Required, served politely
            assert b"WebSocket endpoint" in exc.read()
    finally:
        session.stop()


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
