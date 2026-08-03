"""The webapp server that backs the browser app and the desktop viewport."""

from __future__ import absolute_import, division, print_function

import contextlib
import json
import sys
import urllib.error
import urllib.request

from libtbx.test_utils import Exception_expected

from pxviewer import appserver
from pxviewer.regression.tst_utils import skip
from pxviewer.webapp import Webapp


def get(url, timeout=5):
    """Fetch a URL and return (status, content_type, body).

    The body is always read. Leaving it unread makes the server log a broken pipe when the
    connection closes, which buries a test's own output in noise.
    """
    resp = urllib.request.urlopen(url, timeout=timeout)       # localhost only
    try:
        return resp.status, resp.headers.get_content_type(), resp.read()
    finally:
        resp.close()


@contextlib.contextmanager
def webapp():
    """A running server, stopped afterwards even if the body raises."""
    app = Webapp(port=0)
    app.start()
    try:
        yield app
    finally:
        app.stop()


def exercise_the_viewer_page_is_served_with_a_query_string():
    """The desktop viewport always loads /index.html with ?mvsj=...&ws=... attached.

    Routing on the raw request path (query string included) sent this to the static
    handler, which resolved it against the volume dir and 404'd -- so the viewport showed
    an error page instead of the viewer.
    """
    with webapp() as app:
        url = ("%sindex.html?mvsj=/demo/gaussian/volume.mvsj"
               "&ws=ws://127.0.0.1:9999" % app.url)
        _status, _ctype, page = get(url)
        page = page.decode()
        assert "<!DOCTYPE html>" in page
        assert "build/index.js" in page


def exercise_the_app_and_viewer_pages_are_served():
    with webapp() as app:
        app_page = get(app.url)[2].decode()
        viewer_page = get("%sindex.html" % app.url)[2].decode()
        assert "<!DOCTYPE html>" in app_page
        assert "<!DOCTYPE html>" in viewer_page
        assert get("%sbuild/index.js" % app.url)[0] == 200
        # Both pages point the browser tab at the favicon.
        assert "/favicon.png" in app_page
        assert "/favicon.png" in viewer_page


def exercise_the_favicon_is_served():
    """The favicon lives in the frontend dir; the webapp handler must serve /favicon.png
    from there. It does not match the app/index/build routes, so without an explicit case
    it fell through to the static handler (rooted at the volume dir) and 404'd -- a blank
    browser-tab icon."""
    with webapp() as app:
        status, content_type, body = get("%sfavicon.png" % app.url)
        assert status == 200
        assert content_type == "image/png"
        assert len(body) > 0


def exercise_the_volume_demo_api_generates_files():
    with webapp() as app:
        demos = json.loads(get("%sapi/volume-demos" % app.url)[2])
        assert demos and all("name" in d and "description" in d for d in demos)

        name = demos[0]["name"]
        payload = json.loads(get("%sapi/volume-demo/%s" % (app.url, name))[2])
        mvsj_url = payload["mvsj_url"]
        assert mvsj_url == "/demo/%s/volume.mvsj" % name

        # The generated scene and its density map are both reachable.
        scene = json.loads(get("%s%s" % (app.url, mvsj_url.lstrip("/")))[2])
        assert scene["root"]
        assert get("%sdemo/%s/volume.mrc" % (app.url, name))[0] == 200


def exercise_an_unknown_demo_is_rejected():
    with webapp() as app:
        # try/except rather than libtbx's raises(): that helper instantiates the exception
        # class with no arguments to test isinstance, and HTTPError needs five.
        try:
            get("%sapi/volume-demo/not-a-demo" % app.url)
            raise Exception_expected
        except urllib.error.HTTPError as e:
            assert e.code == 400


def run():
    frontend = appserver.find_frontend_dir()
    if frontend is None or not appserver.frontend_is_built(frontend):
        skip("frontend not built (run scripts/build_frontend.sh)")
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
