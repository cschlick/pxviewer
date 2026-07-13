"""Tests for the static volume demos."""

import functools
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from pxviewer import appserver
from pxviewer.volume_demos import (
    VOLUME_DEMOS,
    _VolumeDemoHandler,
    _VolumeDemoServer,
    create_volume_demo,
    list_volume_demos,
)


def test_list_volume_demos():
    demos = list_volume_demos()
    assert len(demos) == len(VOLUME_DEMOS)
    for name, description in demos:
        assert name in VOLUME_DEMOS
        assert description


@pytest.mark.parametrize("name", list(VOLUME_DEMOS))
def test_create_volume_demo_writes_files(name, tmp_path):
    mrc_path = tmp_path / "volume.mrc"
    mvsj_path = tmp_path / "volume.mvsj"

    mvsj = create_volume_demo(
        name,
        mrc_path=mrc_path,
        mvsj_path=mvsj_path,
        voxel_size=1.0,
        shape=(16, 16, 16),
    )

    assert mrc_path.exists()
    assert mvsj_path.exists()
    assert mvsj

    state = json.loads(mvsj)
    assert state["kind"] == "single"
    root = state["root"]
    download = root["children"][0]
    assert download["kind"] == "download"
    assert download["params"]["url"] == "volume.mrc"

    parse_node = download["children"][0]
    assert parse_node["kind"] == "parse"
    assert parse_node["params"]["format"] == "map"

    volume = parse_node["children"][0]
    assert volume["kind"] == "volume"

    repr = volume["children"][0]
    assert repr["kind"] == "volume_representation"
    assert repr["params"]["type"] == "isosurface"


def test_create_volume_demo_view_kwargs_override(tmp_path):
    mrc_path = tmp_path / "volume.mrc"
    mvsj_path = tmp_path / "volume.mvsj"

    mvsj = create_volume_demo(
        "gaussian",
        mrc_path=mrc_path,
        mvsj_path=mvsj_path,
        shape=(16, 16, 16),
        view_kwargs={"color": "green", "isosurface_value": 3.5, "isosurface_kind": "absolute"},
    )

    state = json.loads(mvsj)
    repr = state["root"]["children"][0]["children"][0]["children"][0]["children"][0]
    assert repr["params"]["absolute_isovalue"] == pytest.approx(3.5)
    color_node = repr["children"][0]
    assert color_node["kind"] == "color"
    assert color_node["params"]["color"] == "green"


def test_create_volume_demo_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown volume demo"):
        create_volume_demo(
            "not_a_demo",
            mrc_path=tmp_path / "volume.mrc",
            mvsj_path=tmp_path / "volume.mvsj",
        )


def test_volume_demo_data_finite():
    for demo in VOLUME_DEMOS.values():
        data = demo.make_data((16, 16, 16))
        assert data.shape == (16, 16, 16)
        assert np.isfinite(data).all()


def test_volume_demo_server_serves_volume_and_frontend(tmp_path):
    frontend = appserver.find_frontend_dir()
    if frontend is None or not appserver.frontend_is_built(frontend):
        pytest.skip("frontend not built")

    mrc_path = tmp_path / "volume.mrc"
    mvsj_path = tmp_path / "volume.mvsj"
    create_volume_demo("gaussian", mrc_path=mrc_path, mvsj_path=mvsj_path, shape=(16, 16, 16))

    handler = functools.partial(
        _VolumeDemoHandler,
        volume_dir=str(tmp_path),
        frontend_dir=str(frontend),
        mvsj_url="volume.mvsj",
    )
    httpd = _VolumeDemoServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        # Root redirects to the volume scene.
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
        assert location == "/index.html?mvsj=volume.mvsj"

        # The page and generated volume are both reachable.
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html?mvsj=volume.mvsj", timeout=5).read().decode()
        assert "<!DOCTYPE html>" in page

        mvsj = urllib.request.urlopen(f"http://127.0.0.1:{port}/volume.mvsj", timeout=5).read()
        assert mvsj
        assert json.loads(mvsj)["root"]["children"][0]["params"]["url"] == "volume.mrc"

        build = urllib.request.urlopen(f"http://127.0.0.1:{port}/build/index.js", timeout=5).read()
        assert build
    finally:
        httpd.shutdown()
        httpd.server_close()
