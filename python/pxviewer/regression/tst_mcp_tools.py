"""The MCP tools exercised end-to-end against a real ``DesktopApp``.

``tst_mcp_server`` pins which tools exist; this file checks they do what the
docstring claims — a selection is a selection, a saved screenshot is on disk,
a failed background job is readable in ``status``. The browser is the only
piece faked: a stand-in page connects to the app's control session over the
real websocket and answers the four round-trip requests ``live.ts`` answers
(screenshot, camera-state, ping, background). Everything the tools touch
Python-side — models, selections, markers, primitives, job state — is the
app's own.
"""

from __future__ import absolute_import, division, print_function

import asyncio
import contextlib
import json
import os
import sys
import threading
import time

from pxviewer.regression.tst_utils import (
    closing_modals, data_path, dispose, have, process_events, qt_application,
    shipped_defaults, skip, tmp_dir)

if not have("mcp", "PySide6.QtWebEngineWidgets", "websockets.sync.client",
            "iotbx.data_manager"):
    skip("mcp / PySide6 QtWebEngine / websockets / iotbx.data_manager "
         "not available")

qt_application()

from mcp.server.fastmcp.exceptions import ToolError   # noqa: E402

#: A 1x1 PNG. The fake page's answer to `screenshot` — enough for the wire
#: (`screenshot-result` carries a data URI) and for `save_screenshot` to write.
PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg==")


def _call(server, name, **arguments):
    """Call a tool by name, the way a client does — through the server."""
    result = asyncio.run(server.call_tool(name, arguments))
    if isinstance(result, dict):
        return result
    first = result[0] if result else None
    if getattr(first, "type", None) == "text":
        return json.loads(first.text)
    return result  # e.g. the ImageContent list `screenshot` returns


def _expect_error(server, name, **arguments):
    """A tool's failure must surface to the caller, not vanish."""
    try:
        _call(server, name, **arguments)
    except ToolError as exc:
        return str(exc)
    raise AssertionError("%s(%r) did not fail" % (name, arguments))


class _FakePage(threading.Thread):
    """A minimal Mol* page: connects to whichever session is the control
    session and answers the round-trip requests.

    The follow-the-control-session loop is deliberate: a model load moves the
    control session from the dummy to the model's session, exactly like the
    real page following its `ws=` URL on a viewport reload.
    """

    def __init__(self, app):
        super().__init__(name="fake-page", daemon=True)
        self._app = app
        self._stopped = threading.Event()

    def stop(self):
        self._stopped.set()

    def run(self):
        import websockets.sync.client as ws_sync
        while not self._stopped.is_set():
            session = self._app._control_session()
            port = getattr(session, "port", None)
            if session is None or not port:
                time.sleep(0.05)
                continue
            try:
                url = "ws://%s:%d" % (session.host, port)
                with ws_sync.connect(url) as ws:
                    while not self._stopped.is_set():
                        if self._app._control_session() is not session:
                            break
                        try:
                            raw = ws.recv(timeout=0.1)
                        except TimeoutError:
                            continue
                        if isinstance(raw, (bytes, bytearray)):
                            continue  # topology/frame payloads, not requests
                        try:
                            message = json.loads(raw)
                        except ValueError:
                            continue
                        reply = self._answer(message)
                        if reply is not None:
                            ws.send(json.dumps(reply))
            except Exception:
                time.sleep(0.05)

    @staticmethod
    def _answer(message):
        kind, req_id = message.get("type"), message.get("reqId")
        if kind == "screenshot":
            return {"type": "screenshot-result", "reqId": req_id,
                    "dataUri": "data:image/png;base64," + PNG_1PX,
                    "error": None}
        if kind == "camera-state":
            return {"type": "camera-state-result", "reqId": req_id,
                    "state": {"target": [0.0, 0.0, 0.0],
                              "position": [0.0, 0.0, 50.0],
                              "up": [0.0, 1.0, 0.0],
                              "radius": 40.0, "fov": 30.0}}
        if kind == "ping":
            return {"type": "pong", "reqId": req_id, "pending": 0,
                    "busy": None, "busyMs": 0, "viewer": True}
        if kind == "query-background":
            return {"type": "background-result", "reqId": req_id,
                    "color": "#000000", "error": None}
        return None


async def _inline_qt(fn):
    """`_qt` normally marshals onto the Qt thread; the test *is* the Qt
    thread, so the work runs inline — same thread, one hop fewer."""
    return fn()


@contextlib.contextmanager
def _mcp_app():
    """A ``DesktopApp`` with the fake page attached, driven through its real
    MCP surface. The real webview is held back (``_viewport.load`` is the one
    stubbed edge): the fake page replaces it on the wire."""
    from pxviewer import mcp_server
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    app._viewport.load = lambda url: None
    saved = mcp_server._app, mcp_server._qt
    mcp_server._app = app
    mcp_server._qt = _inline_qt
    page = _FakePage(app)
    page.start()
    with closing_modals():
        try:
            yield app, mcp_server._make_server()
        finally:
            page.stop()
            page.join(timeout=5)
            mcp_server._app, mcp_server._qt = saved
            dispose(app)


def exercise_tools_drive_a_real_app():
    """The core journey an agent takes: load, inspect, select, draw, measure,
    mark, screenshot, save — every step read back through a tool."""
    with shipped_defaults(), _mcp_app() as (app, server):
        # Before anything loads, the honest answers are "nobody's here".
        assert _call(server, "ping")["alive"] is False
        _expect_error(server, "screenshot")

        loaded = _call(server, "load", path=data_path("1ubq.pdb"))
        assert loaded["loaded"], loaded

        state = _call(server, "status")
        assert state["models"][0]["id"] == "model-1", state
        assert state["models"][0]["active"] is True
        assert state["camera"]["radius"] == 40.0     # the fake page's answer
        assert state["background"] == "#000000"
        assert state["busy"] == [] and state["job_errors"] == []

        info = _call(server, "structure_info")["info"]
        assert info["atoms"] == 660
        assert info["chains"] == [
            {"chain": "A", "residues": 76, "first": 1, "last": 76}]
        assert info["waters"] == 58

        ss = _call(server, "secondary_structure")
        assert ss["helices"] and all(r["chain"] == "A" for r in ss["helices"])

        # Selection is a selection: what select() reports, selection_state()
        # and the scene agree on.
        picked = _call(server, "select", expression="resseq 40:45")
        assert picked["atoms"] > 0 and picked["residues"]
        scene = _call(server, "selection_state")["selection"]
        assert scene["model-1"]["atoms"] == picked["atoms"]

        gly = _call(server, "atoms", expression="resname GLY")
        assert gly["matched"] > 0
        assert {"i", "name", "xyz", "b", "occ"} <= set(gly["atoms"][0])
        assert gly["centroid"] is not None

        table = _call(server, "atom_table")
        assert table["n"] == 660 and len(table["x"]) == 660
        assert "ATOM" in _call(server, "model_text")["text"]

        model = _call(server, "model_state", model="model-1")["model"]
        assert model["atoms"] == 660
        assert model["selected"] == picked["atoms"]

        _call(server, "set_representation", rep="cartoon", model="model-1")
        assert _call(server, "model_state",
                   model="model-1")["model"]["rep"] == "cartoon"

        # Hiding is refused on software WebGL (the app's `_can_hide` gate);
        # the tool must then report the *unchanged* state, not the request.
        hidden = _call(server, "set_object_visible", kind="model",
                       ident="model-1", visible=False)
        assert hidden["visible"] is (not app._can_hide), hidden
        assert _call(server, "model_state",
                   model="model-1")["model"]["visible"] is (not app._can_hide)
        if app._can_hide:
            _call(server, "set_object_visible", kind="model", ident="model-1",
                  visible=True)

        sphere = _call(server, "set_clip_sphere", mode="view", radius=20.0,
                       model="model-1")["sphere"]
        assert sphere["mode"] == "view"

        distance = _call(server, "measure", kind="distance", atoms=[0, 9])
        label = _call(server, "add_label", atom=0, text="N-term")
        prims = _call(server, "primitives")["primitives"]
        ids = {p["id"] for p in prims}
        assert {distance["primitive"], label["primitive"]} <= ids
        _call(server, "remove_primitive", primitive_id=label["primitive"])
        assert len(_call(server, "primitives")["primitives"]) == 1
        _call(server, "clear_primitives")
        assert _call(server, "primitives")["primitives"] == []

        marker = _call(server, "place_marker", position=[20.0, 20.0, 20.0])
        assert marker["marker"] in {m["id"] for m in
                                    _call(server, "status")["markers"]}
        _call(server, "remove", kind="marker", ident=marker["marker"])
        assert _call(server, "status")["markers"] == []

        pong = _call(server, "ping")
        assert pong["alive"] is True and pong["pending"] == 0

        # Viewport-facing commands all take the round-trip answers the fake
        # page provides.
        assert _call(server, "mouse_selection")["indices"] == []
        _call(server, "zoom_view", direction=-1)
        _call(server, "step", step=1)
        _call(server, "focus_residue", chain="A", resid="42")
        _call(server, "reset_view")
        _call(server, "set_camera",
              state={"target": [0, 0, 0], "position": [0, 0, 50],
                     "up": [0, 1, 0], "radius": 40.0, "fov": 30.0})
        _call(server, "set_background", color="#101018")
        _call(server, "set_active_model", model="model-1")

        image = _call(server, "screenshot")
        assert image[0].type == "image", image

        with tmp_dir() as directory:
            png = os.path.join(directory, "view.png")
            _call(server, "save_screenshot", path=png)
            assert os.path.getsize(png) > 0   # the file really landed
            pdb = os.path.join(directory, "model.pdb")
            _call(server, "write_object", kind="model", ident="model-1",
                  path=pdb)
            assert "ATOM" in open(pdb).read()

        _call(server, "clear_selection")
        assert _call(server, "selection_state")["selection"] == {}

        # Errors surface as errors — the contract that replaced silent lies.
        assert "no such model" in _expect_error(
            server, "model_state", model="nope")
        _expect_error(server, "atoms", expression="name CA", model="nope")
        _expect_error(server, "remove", kind="model", ident="nope")
        _expect_error(server, "remove", kind="bogus", ident="x")
        _expect_error(server, "measure", kind="bogus", atoms=[0, 1])
        _expect_error(server, "volume_state", volume="volume-1")


def exercise_a_failed_background_job_is_reported():
    """run_background feeds status.busy/job_errors — the reporting that made
    a failed fetch visible to agents."""
    with shipped_defaults(), _mcp_app() as (app, server):
        def boom():
            raise RuntimeError("no such structure factors")

        app.run_background(boom, name="mcp-test-boom", label="Boom")
        deadline = time.time() + 15
        while time.time() < deadline:
            process_events()
            state = _call(server, "status")
            errors = [e for e in state["job_errors"]
                      if e["name"] == "mcp-test-boom"]
            if errors:
                assert "no such structure factors" in errors[0]["error"]
                break
            time.sleep(0.1)
        else:
            raise AssertionError("the failed job never reached job_errors")


def exercise_validation_results_read_back():
    """run_validation + validation_results are a pair: the agent can start the
    job and read the same tables the Validation tab shows."""
    with shipped_defaults(), _mcp_app() as (app, server):
        _call(server, "load", path=data_path("1ubq.pdb"))
        assert _call(server, "validation_results")["results"] == {}

        _call(server, "run_validation")
        deadline = time.time() + 180
        results = {}
        while time.time() < deadline:
            process_events()
            results = _call(server, "validation_results")["results"]
            if results:
                break
            time.sleep(0.5)
        assert results, "run_validation produced no results in 180s"

        key = sorted(results)[0]
        row = results[key]
        assert row["title"] and row["summary"]
        assert isinstance(row["columns"], list) and isinstance(row["rows"], list)

        marked = _call(server, "set_validation_markers", key=key, visible=True)
        assert marked["results_cached"] is True
        _call(server, "set_validation_markers", key=key, visible=False)


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
