"""Shared driver for the visual passes in VISUAL_TESTS.md.

A visual pass drives the real desktop app -- a real QtWebEngine viewport and a
real Mol* render, not an offscreen stub -- through a chained sequence where each
step leaves standing state the next step leans on. Assertions read the state the
step produces (selection markers, clip bookkeeping, camera state, table row
selection) and viewport screenshots go to the run directory for review.

Conventions:

- ``check(label, ok, detail)`` prints ``[PASS]``/``[FAIL]`` and accumulates a
  report written next to the screenshots; ``note`` adds a free-form line.
- ``pump``/``settle`` advance Qt like a user would -- ``pump`` waits on a
  condition, ``settle`` lets a fixed interval's worth of events land. Both use
  ``tst_utils.process_events`` so deferred deletes are delivered like a real
  event loop's.
- ``shot_viewport`` captures the browser-side WebGL render (the only faithful
  picture of the canvas -- ``QWidget.grab`` on the view gets a blank frame);
  ``shot``/``main_window``/``controls_window`` grab chrome widgets.
- ``cam_state`` and ``run_js`` read Mol* state back through the live session and
  the page respectively, for assertions screenshots cannot carry (camera
  radius, clip-object counts).
- ``img_diff`` compares two captured frames as a percentage of pixels that
  changed; within-run comparisons (clipped vs. lifted, first clip vs.
  re-applied) are portable in a way golden images are not, since they ask only
  "did the pixels move" and "did they come back", not "does this GPU render
  like mine did".

Run directory is ``$PXVIEWER_VISUAL_DIR`` or ``$TMPDIR/pxviewer-visual``; files
accumulate there for review and are never cleaned automatically. Settings are
isolated by ``tst_utils``'s import-time ``PXVIEWER_SETTINGS_DIR`` redirect, and
``make_app`` additionally runs under ``shipped_defaults`` the way every
DesktopApp-constructing test must.

A pass script looks like::

    from pxviewer.regression.visual.harness import *

    app = make_app()
    try:
        ...
        check("clip stands", entry.get("_auto_clip") is True)
    except Exception:
        check("pass ran clean", False, traceback.format_exc(limit=3))
    finish(app)   # writes the report, disposes the app, exits 0 or 1
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from pxviewer.regression import tst_utils          # noqa: F401 -- settings isolation at import
from pxviewer.regression.tst_utils import (
    dispose,
    have,
    process_events,
    qt_application,
    shipped_defaults,
    skip,
)

if not have("PySide6.QtWebEngineWidgets", "websockets"):
    skip("visual passes need PySide6 QtWebEngine and websockets")
if not have("PIL"):
    skip("visual passes need pillow for frame diffs")

from PySide6.QtWidgets import QApplication         # noqa: E402

#: Every run's artifacts -- screenshots plus report.txt -- land here. Stable across
#: runs so a review session can browse them; override with PXVIEWER_VISUAL_DIR.
OUT_DIR = Path(os.environ.get("PXVIEWER_VISUAL_DIR") or
               os.path.join(tempfile.gettempdir(), "pxviewer-visual"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT: list[str] = []
FAILED = 0

_stack = None          # ExitStack holding shipped_defaults() for the process


def note(msg: str) -> None:
    line = "      %s" % msg
    print(line, flush=True)
    REPORT.append(line)


def check(label: str, ok, detail: str = "") -> bool:
    """Assert-and-record; returns ``ok`` so callers can gate follow-up work."""
    global FAILED
    ok = bool(ok)
    if not ok:
        FAILED += 1
    line = "[%s] %s%s" % ("PASS" if ok else "FAIL", label,
                          " -- %s" % detail if detail else "")
    print(line, flush=True)
    REPORT.append(line)
    return ok


def section(title: str) -> None:
    line = "\n=== %s ===" % title
    print(line, flush=True)
    REPORT.append(line)


def pump(until=None, timeout: float = 15.0, interval: float = 0.03) -> bool:
    """Drive the event loop until ``until()`` or ``timeout``; True on success."""
    deadline = time.time() + timeout
    while True:
        process_events()
        if until is not None and until():
            return True
        if time.time() > deadline:
            return False
        time.sleep(interval)


def settle(seconds: float = 0.6) -> None:
    """Let ``seconds`` of events land -- standing in for a human pausing to look."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        process_events()
        time.sleep(0.03)


def _call_session(app, method: str, timeout: float, *args, **kw):
    """Run a LiveSession method off the GUI thread; ``None`` if it never answers.

    Session calls block on the browser round trip, so they must not run on the
    thread that has to deliver the answer.
    """
    session = app._control_session()
    if session is None:
        return None
    out: dict = {}

    def work():
        try:
            out["v"] = getattr(session, method)(*args, timeout=timeout, **kw)
        except Exception as exc:
            out["err"] = exc

    t = threading.Thread(target=work, daemon=True, name="pxviewer-visual-call")
    t.start()
    pump(until=lambda: not t.is_alive(), timeout=timeout + 5)
    return out.get("v")


def shot_viewport(app, name: str):
    """Browser-side render of the Mol* canvas -> ``OUT_DIR/<name>.png``."""
    png = _call_session(app, "screenshot", 30)
    if not png:
        check("viewport shot %s" % name, False, "no answer from the browser")
        return None
    path = OUT_DIR / ("%s.png" % name)
    path.write_bytes(png)
    return str(path)


def shot(widget, name: str) -> str:
    path = OUT_DIR / ("%s.png" % name)
    widget.grab().save(str(path))
    return str(path)


def main_window(app, name: str) -> str:
    return shot(app._main, name)


def controls_window(app, name: str) -> str:
    return shot(app._controls.widget(), name)


def cam_state(app, timeout: float = 10.0):
    """The Mol* camera's serialized state (target, radius, radiusMax, ...)."""
    return _call_session(app, "camera_state", timeout)


def run_js(app, script: str, timeout: float = 10.0):
    """Evaluate ``script`` in the viewport page; returns the JSON-able result."""
    page = app._viewport._view.page()
    out = {"set": False, "v": None}
    page.runJavaScript(script, lambda v: out.update(set=True, v=v))
    if not pump(until=lambda: out["set"], timeout=timeout):
        return None
    return out["v"]


def img_diff(path_a, path_b, threshold: int = 8):
    """Percentage of pixels whose channels differ by more than ``threshold``.

    For comparing frames within one run -- same window, same GPU -- so the number
    is a behavioral claim ("the lifted view differs", "the re-applied view
    matches"), not a rendering-fidelity claim.
    """
    import numpy as np
    from PIL import Image, ImageChops

    a = Image.open(path_a).convert("RGB")
    b = Image.open(path_b).convert("RGB")
    if a.size != b.size:
        return 100.0
    diff = np.asarray(ImageChops.difference(a, b))
    return float((diff.max(axis=-1) > threshold).mean() * 100.0)


def make_app():
    """A real DesktopApp, windows shown, started the way ``run_desktop`` starts.

    ``gpu.configure`` picks the GL backend before QApplication exists and the
    app is told whether it can hide objects in place (hardware only); the first
    call enters ``shipped_defaults`` so construction reads fresh-install
    preferences regardless of the runner's own settings.
    """
    global _stack
    # A visual pass needs the real compositor: an offscreen WebEngine renders
    # nothing to screenshot. qt_application() defaults to the offscreen platform
    # when DISPLAY is unset -- correct for the headless suite, but wrong here on
    # macOS, where cocoa is native and DISPLAY is an X11-ism. Pin cocoa so a
    # display-less *shell* on a machine that has a display still renders.
    if sys.platform == "darwin" and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "cocoa"
    # The app's real startup sequence: pick the GL backend *before* QApplication
    # exists (the flags cannot move after), and tell the app whether the chosen
    # backend survives in-place hide -- the same call run_desktop makes. A
    # software backend refuses hiding entirely, so passes that hide objects can
    # only run on hardware.
    from pxviewer import gpu as gpu_backend

    gpu_mode = gpu_backend.configure(None)
    qt_application()
    if _stack is None:
        _stack = contextlib.ExitStack()
        _stack.enter_context(shipped_defaults())
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0, can_hide=(gpu_mode == "hardware"))
    app._webapp.start()
    app._main.show()
    app._controls.widget().show()
    app._main.resize(1400, 900)
    app._app.setActiveWindow(app._main)
    pump(until=lambda: app._control_session() is not None, timeout=30)
    return app


def wait_viewer(app, timeout: float = 60.0):
    """Pump until the embedded Mol* answers a screenshot -- the signal it is live.

    A request sent before the viewer's handler exists is dropped, never answered
    -- so the tries are short and repeated rather than one long wait that only
    proves nobody was listening when it was sent.
    """
    out: dict = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        out["v"] = _call_session(app, "screenshot", 5)
        if out["v"]:
            return True, out
        settle(0.5)
    return False, out


def load_bundled(app, filename: str):
    """Load one of the shipped sample structures by name (``1ubq.pdb``, ...)."""
    from pxviewer.loader import sample_structure_path

    path = sample_structure_path(filename)
    app.load_file(str(path))
    return path


def write_report() -> None:
    (OUT_DIR / "report.txt").write_text("\n".join(REPORT))


def finish(app=None) -> None:
    """Report, free the app, exit 0 or 1 -- the ``tst_`` convention's contract."""
    write_report()
    if app is not None:
        dispose(app)
    if _stack is not None:
        _stack.close()
    if FAILED:
        print("FAILED (%d checks) -- artifacts in %s" % (FAILED, OUT_DIR))
        sys.stdout.flush()
        sys.exit(1)
    print("OK -- artifacts in %s" % OUT_DIR)
    sys.stdout.flush()
    sys.exit(0)
