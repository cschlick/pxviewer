"""The built-in volume demos.

Each demo synthesises a density on a grid and writes it with the scene that displays it,
so this is the one place a map is produced end to end without any real data: it is what
``pxviewer.volume_demo`` and the webapp's demo endpoint both stand on.
"""

from __future__ import absolute_import, division, print_function

import functools
import json
import os
import sys
import threading
import urllib.error
import urllib.request

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import have, tmp_dir

if not have("numpy"):
    from pxviewer.regression.tst_utils import skip
    skip("numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer import appserver                       # noqa: E402
from pxviewer.volume_demos import (                  # noqa: E402
    VOLUME_DEMOS,
    _VolumeDemoHandler,
    _VolumeDemoServer,
    create_volume_demo,
    list_volume_demos,
)

SHAPE = (16, 16, 16)          # small enough that every demo can be built in a loop


def exercise_the_catalogue_is_complete():
    demos = list_volume_demos()
    assert len(demos) == len(VOLUME_DEMOS)
    for name, description in demos:
        assert name in VOLUME_DEMOS
        assert description


def exercise_every_demo_produces_finite_data():
    """A NaN or an inf here renders as an empty viewport with no error anywhere."""
    for name, demo in sorted(VOLUME_DEMOS.items()):
        data = demo.make_data(SHAPE)
        arrays = data if isinstance(data, list) else [data]
        for arr in arrays:
            assert arr.shape == SHAPE, name
            assert np.isfinite(arr).all(), name


def exercise_every_demo_writes_a_map_and_a_scene():
    """Run the whole catalogue, not a sample -- a demo that fails to write is a 404 in
    the app, and the demos differ enough (single vs multi-channel) that one standing in
    for the others would miss it."""
    for name in sorted(VOLUME_DEMOS):
        with tmp_dir() as path:
            mrc_path = os.path.join(path, "volume.mrc")
            mvsj_path = os.path.join(path, "volume.mvsj")

            mvsj = create_volume_demo(
                name, mrc_path=mrc_path, mvsj_path=mvsj_path,
                voxel_size=1.0, shape=SHAPE)

            assert os.path.exists(mrc_path), name
            assert os.path.exists(mvsj_path), name

            state = json.loads(mvsj)
            assert state["kind"] == "single", name

            download = state["root"]["children"][0]
            assert download["kind"] == "download", name
            # Relative, so the scene works wherever the pair is served from.
            assert download["params"]["url"] == "volume.mrc", name

            parse = download["children"][0]
            assert parse["kind"] == "parse", name
            assert parse["params"]["format"] == "map", name

            volume = parse["children"][0]
            assert volume["kind"] == "volume", name

            rep = volume["children"][0]
            assert rep["kind"] == "volume_representation", name
            assert rep["params"]["type"] == "isosurface", name


def exercise_view_kwargs_override_the_demo_defaults():
    with tmp_dir() as path:
        mvsj = create_volume_demo(
            "gaussian",
            mrc_path=os.path.join(path, "volume.mrc"),
            mvsj_path=os.path.join(path, "volume.mvsj"),
            shape=SHAPE,
            view_kwargs={"color": "green", "isosurface_value": 3.5,
                         "isosurface_kind": "absolute"},
        )
        state = json.loads(mvsj)
        rep = state["root"]["children"][0]["children"][0]["children"][0]["children"][0]
        assert approx_equal(rep["params"]["absolute_isovalue"], 3.5)

        color = rep["children"][0]
        assert color["kind"] == "color"
        assert color["params"]["color"] == "green"


def exercise_an_unknown_demo_is_rejected():
    with tmp_dir() as path:
        with raises(ValueError) as e:
            create_volume_demo(
                "not_a_demo",
                mrc_path=os.path.join(path, "volume.mrc"),
                mvsj_path=os.path.join(path, "volume.mvsj"))
        assert "unknown volume demo" in str(e.value)


def exercise_the_demo_server_serves_the_scene_and_the_frontend():
    """``pxviewer.volume_demo`` starts this server and points a browser at the root.

    Three things have to line up for that to work: the root redirects to the viewer page
    with the scene in the query string, the scene and its map are reachable at the URLs
    the scene names, and the frontend bundle is served alongside them.
    """
    frontend = appserver.find_frontend_dir()
    if frontend is None or not appserver.frontend_is_built(frontend):
        print("    (skipped: frontend not built -- run scripts/build_frontend.sh)")
        return

    with tmp_dir() as path:
        create_volume_demo(
            "gaussian",
            mrc_path=os.path.join(path, "volume.mrc"),
            mvsj_path=os.path.join(path, "volume.mvsj"),
            shape=SHAPE)

        handler = functools.partial(
            _VolumeDemoHandler,
            volume_dir=path, frontend_dir=str(frontend), mvsj_url="volume.mvsj")
        httpd = _VolumeDemoServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = "http://127.0.0.1:%d/" % port

            class No_redirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *args, **kwargs):
                    return None

            location = None
            try:
                # Read and close: an unread body makes the server log a broken pipe.
                urllib.request.build_opener(No_redirect).open(base).close()
            except urllib.error.HTTPError as e:
                assert e.code == 302
                location = e.headers.get("Location")
                e.close()
            assert location == "/index.html?mvsj=volume.mvsj"

            page = fetch(base + "index.html?mvsj=volume.mvsj").decode()
            assert "<!DOCTYPE html>" in page

            scene = json.loads(fetch(base + "volume.mvsj"))
            assert scene["root"]["children"][0]["params"]["url"] == "volume.mrc"

            assert fetch(base + "build/index.js")
        finally:
            httpd.shutdown()
            httpd.server_close()


def fetch(url, timeout=5):
    resp = urllib.request.urlopen(url, timeout=timeout)       # localhost only
    try:
        return resp.read()
    finally:
        resp.close()


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
