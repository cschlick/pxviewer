"""Tests for the webapp server that backs the browser app and the desktop viewport."""

import json
import urllib.error
import urllib.request

import pytest

from pxviewer import appserver
from pxviewer.webapp import Webapp


def _get(url, timeout=5):
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 (localhost test)


@pytest.fixture
def webapp():
    frontend = appserver.find_frontend_dir()
    if frontend is None or not appserver.frontend_is_built(frontend):
        pytest.skip("frontend not built")

    app = Webapp(port=0)
    app.start()
    try:
        yield app
    finally:
        app.stop()


def test_viewer_page_is_served_with_a_query_string(webapp):
    """The desktop viewport always loads /index.html with ?mvsj=…&ws=… attached.

    Routing on the raw request path (query string included) sent this to the
    static handler, which resolved it against the volume dir and 404'd — so the
    viewport showed an error page instead of the viewer.
    """
    url = f"{webapp.url}index.html?mvsj=/demo/gaussian/volume.mvsj&ws=ws://127.0.0.1:9999"
    page = _get(url).read().decode()
    assert "<!DOCTYPE html>" in page
    assert "build/index.js" in page


def test_app_and_viewer_pages_are_served(webapp):
    app_page = _get(f"{webapp.url}").read().decode()
    viewer_page = _get(f"{webapp.url}index.html").read().decode()
    assert "<!DOCTYPE html>" in app_page
    assert "<!DOCTYPE html>" in viewer_page
    assert _get(f"{webapp.url}build/index.js").status == 200
    # Both pages point the browser tab at the favicon.
    assert "/favicon.png" in app_page
    assert "/favicon.png" in viewer_page


def test_favicon_is_served(webapp):
    """The favicon lives in the frontend dir; the webapp handler must serve
    /favicon.png from there. It doesn't match the app/index/build routes, so without
    an explicit case it fell through to the static handler (rooted at the volume dir)
    and 404'd — a blank browser-tab icon."""
    resp = _get(f"{webapp.url}favicon.png")
    assert resp.status == 200
    assert resp.headers.get_content_type() == "image/png"
    assert len(resp.read()) > 0


def test_volume_demo_api_generates_files(webapp):
    demos = json.loads(_get(f"{webapp.url}api/volume-demos").read())
    assert demos and all("name" in d and "description" in d for d in demos)

    name = demos[0]["name"]
    payload = json.loads(_get(f"{webapp.url}api/volume-demo/{name}").read())
    mvsj_url = payload["mvsj_url"]
    assert mvsj_url == f"/demo/{name}/volume.mvsj"

    # The generated scene and its density map are both reachable.
    scene = json.loads(_get(f"{webapp.url}{mvsj_url.lstrip('/')}").read())
    assert scene["root"]
    assert _get(f"{webapp.url}demo/{name}/volume.mrc").status == 200


def test_unknown_demo_is_rejected(webapp):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{webapp.url}api/volume-demo/not-a-demo")
    assert exc.value.code == 400
