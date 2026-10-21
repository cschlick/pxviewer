"""The MCP tools exercised end-to-end against a real ``DesktopApp``.

``tst_mcp_server`` pins which tools exist; this file checks they do what the
docstring claims — ``python`` executes in the app's shared kernel namespace
(real objects, persistent bindings, ``app`` marshalled onto the Qt thread), a
saved screenshot is on disk, a failed background job is readable in ``status``.
The browser is the only piece faked: a stand-in page connects to the app's
control session over the real websocket and answers the four round-trip
requests ``live.ts`` answers (screenshot, camera-state, ping, background).

Threading, faithfully exercised: tool calls run ``asyncio.run`` on a worker
thread while the test's main thread pumps the Qt event loop — which is exactly
how ``invoke_on_gui`` gets serviced in the running app. Nothing is stubbed
except the webview page itself.
"""

from __future__ import absolute_import, division, print_function

import asyncio
import contextlib
import json
import os
import sys
import threading
import time

from pxviewer.regression.tst_utils import (closing_modals, data_path,
    dispose, _exercise_selected, have, process_events, qt_application,
    shipped_defaults, skip, tmp_dir)

if not have("mcp", "PySide6.QtWebEngineWidgets", "websockets.sync.client",
            "iotbx.data_manager", "ipykernel", "jupyter_client", "qtconsole"):
    skip("mcp / PySide6 QtWebEngine / websockets / iotbx / ipykernel "
         "not available")

qt_application()

from mcp.server.fastmcp.exceptions import ToolError   # noqa: E402

#: A 1x1 PNG. The fake page's answer to `screenshot` — enough for the wire
#: (`screenshot-result` carries a data URI) and for `save_screenshot` to write.
PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg==")


def _call(server, name, **arguments):
    """Call a tool by name, the way a client does — through the server.

    The call runs on a worker thread while THIS thread (the main thread, where
    the QApplication lives) pumps Qt events — so an exec'd ``app.*`` call that
    the kernel proxies onto the GUI thread really runs while the tool call is
    in flight, the same as in the live app.
    """
    box = {}

    def go():
        try:
            box["value"] = asyncio.run(server.call_tool(name, arguments))
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=go, name="mcp-call", daemon=True)
    thread.start()
    deadline = time.time() + 300
    while thread.is_alive():
        process_events()
        thread.join(timeout=0.05)
        if time.time() > deadline:
            raise AssertionError("%s(%r) never returned" % (name, arguments))
    if "error" in box:
        raise box["error"]
    result = box.get("value")
    if isinstance(result, dict):
        return result
    first = result[0] if result else None
    if getattr(first, "type", None) == "text":
        return json.loads(first.text)
    return result  # e.g. the ImageContent list `screenshot` returns


def _python(server, code, **kw):
    """Exec through the python tool; assert the cell itself succeeded."""
    result = _call(server, "python", code=code, **kw)
    assert result["status"] == "ok", "exec failed: %r" % result
    return result


def _expect_error(server, name, **arguments):
    """A tool's failure must surface to the caller, not vanish."""
    try:
        _call(server, name, **arguments)
    except ToolError as exc:
        return str(exc)
    except Exception as exc:  # call_tool may re-raise the raw error
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
    saved = mcp_server._app
    mcp_server._app = app
    page = _FakePage(app)
    page.start()
    with closing_modals():
        try:
            yield app, mcp_server._make_server()
        finally:
            page.stop()
            page.join(timeout=5)
            mcp_server._app = saved
            app.shutdown_kernel()
            dispose(app)


def exercise_python_exec_drives_a_real_app():
    """The core journey an agent takes — through the shared kernel namespace:
    load via app.*, verify via status(), drive selection via exec, read the
    frame via screenshot — with the kernel's objects staying live between
    calls the way the console's do."""
    with shipped_defaults(), _mcp_app() as (app, server):
        # Before anything loads, the honest answers are "nobody's here".
        assert _call(server, "ping")["alive"] is False
        _expect_error(server, "screenshot")
        assert _call(server, "status")["kernel"] is None  # lazy: not started

        # Loading is an app call — exec'd, marshalled onto the Qt thread.
        path = data_path("1ubq.pdb")
        result = _python(server, "app.load_file(%r)" % path)
        assert result["result"] in ("'model'", '"model"'), result

        state = _call(server, "status")
        assert state["models"][0]["id"] == "model-1", state
        assert state["models"][0]["active"] is True
        assert state["camera"]["radius"] == 40.0     # the fake page's answer
        assert state["background"] == "#000000"
        assert state["busy"] == [] and state["job_errors"] == []
        kernel = state["kernel"]
        assert kernel and kernel["connection_file"].endswith(".json"), state

        # Raw objects: session.model is the real cctbx model — a flex array's
        # size comes back through the repr, the object stays alive kernel-side.
        result = _python(
            server, "n_atoms = session.model.get_sites_cart().size()")
        # ...and persists: a name bound in one call is there for the next.
        assert _python(server, "n_atoms")["result"] == "660", "n_atoms lost"

        # stdout and structured errors behave like a real kernel's.
        r = _python(server, "print('working')")
        assert r["stdout"] == "working\n", r
        r = _call(server, "python", code="1/0")
        assert r["status"] == "error", r
        assert r["error"]["ename"] == "ZeroDivisionError"
        assert any("ZeroDivisionError" in l for l in r["error"]["traceback"])

        # The api guide is the agent's discovery surface.
        r = _python(server, "str(api.find('color'))")
        assert "matching 'color'" in r["result"], r
        assert "session." in r["result"], r

        # Selection is a selection: exec'd, and the scene state agrees.
        r = _python(server, "app.select_by_expression('resseq 40:45')")
        picked = int(r["result"])
        assert picked > 0, r
        state = _call(server, "status")
        assert state["selection"]["model-1"] == picked, state

        # Representation through the same path, read back via exec too.
        _python(server, "app.set_model_representation('model-1', 'cartoon')")
        r = _python(server, "app._model_entry('model-1').get('reps')")
        assert "cartoon" in r["result"], r

        # Viewport-facing tools still take the round trip the fake page answers.
        pong = _call(server, "ping")
        assert pong["alive"] is True and pong["pending"] == 0

        image = _call(server, "screenshot")
        assert image[0].type == "image", image

        with tmp_dir() as directory:
            png = os.path.join(directory, "view.png")
            _call(server, "save_screenshot", path=png)
            assert os.path.getsize(png) > 0   # the file really landed

        # And the kernel outside the tool is the same kernel — an attached
        # Jupyter client sees the names exec set.
        k = app._kernel
        assert k is not None
        client = k.client()
        try:
            mid = client.execute("n_atoms + 1")
            seen = None
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    msg = client.get_iopub_msg(timeout=0.5)
                except Exception:
                    msg = None
                if (msg and msg["parent_header"].get("msg_id") == mid
                        and msg["msg_type"] == "execute_result"):
                    seen = msg["content"]["data"]["text/plain"]
                    break
            assert seen == "661", "external client did not see the namespace"
        finally:
            client.stop_channels()


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


def exercise_exec_sees_validation_results():
    """run_validation through exec lands results where exec can read them —
    the same dict the Validation tab shows."""
    with shipped_defaults(), _mcp_app() as (app, server):
        _python(server, "app.load_file(%r)" % data_path("1ubq.pdb"))
        r = _python(server, "app.run_validation()")
        assert r["status"] == "ok", r

        deadline = time.time() + 180
        while time.time() < deadline:
            process_events()
            r = _python(
                server,
                "bool(app._model_entry('model-1').get('validation'))")
            if r["result"] == "True":
                break
            time.sleep(0.5)
        else:
            raise AssertionError("run_validation produced no results in 180s")


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise") and _exercise_selected(name):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
