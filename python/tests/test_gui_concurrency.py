"""The GUI fuzzer's concurrency layer: mutate state while a background job is in flight.

Every threaded job in the app — phasing (``make_maps``/``update_maps``), minimization,
dragging — captures the objects it works on and finishes *later*, on the GUI thread. If
the user unloads one of those objects in between, the job lands against a world that has
changed under it. This is where the session's real bugs lived (a control session pointed
at a stopped socket, a callback referencing a removed entry), and none of them showed up
in the synchronous walk, because the whole point is the gap between start and finish.

The trick that makes this *deterministic* rather than a flaky race: the jobs marshal
their result back with ``run_on_main``, a queued callback that only runs when the GUI
loop is pumped. So a test can let the worker thread finish its computation (``_drain``,
which waits for the thread WITHOUT pumping the loop), mutate the registry, and only then
pump — landing the callback against the mutated state, the same interleaving a real
unload-mid-job produces, but reproducibly. One genuinely-threaded stress test
(``minimize`` streaming while another model is removed) covers the shared-list case the
marshalling trick cannot.

After every case, the invariant bank (see gui_invariants) must still hold.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("websockets")
pytest.importorskip("PySide6.QtWebEngineWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui_invariants import assert_viewer_consistent  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "pxviewer", "data")


def _drain(name: str, timeout: float = 120.0) -> None:
    """Wait for a named worker thread to finish, WITHOUT pumping the GUI event loop.

    Not pumping is the whole point: the worker's ``run_on_main`` callback is left queued
    and pending, so the test can change the registry before letting it land — the
    unload-while-the-job-was-running window, made reproducible.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(t.name == name and t.is_alive() for t in threading.enumerate()):
            return
        time.sleep(0.02)
    raise TimeoutError(f"worker {name!r} did not finish within {timeout}s")


@pytest.fixture
def guarded_modals(monkeypatch):
    """Auto-answer any modal so a background job's status/error path cannot block."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))


@pytest.fixture
def xray_app(guarded_modals):
    """An app with a model and reflections loaded but unpaired — ready to phase."""
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    app._controls.widget().show()
    try:
        app.load_xray_demo(d_min=3.0)
        QApplication.processEvents()
        yield app
    finally:
        app.stop()


# -- phasing lands after the model/reflections were unloaded ------------------

@pytest.mark.parametrize("unload", ["none", "model", "reflections", "both"])
def test_make_maps_survives_unload_before_it_lands(xray_app, unload):
    """Phase on a thread, unload the source in the window before the maps land, then let
    them land. The maps belong to objects that no longer exist, so nothing orphaned may
    appear — no volumes, no group paired to a model that is gone."""
    app = xray_app
    rid = app._reflections[0]["id"]
    mid = app._models[0]["id"]

    app.make_maps(rid, mid)
    _drain("pxviewer-phasing")  # computed; its add-on-main callback is now queued

    if unload in ("model", "both"):
        app.remove_model(mid)
    if unload in ("reflections", "both"):
        app.remove_reflections(rid)

    QApplication.processEvents()  # land the callback against the mutated registry
    assert_viewer_consistent(app)

    if unload == "none":
        assert len(app._volumes) == 2 and len(app._groups) == 1, "maps should have landed"
    else:
        # The source was pulled out from under the job: its results are discarded, not
        # left as maps paired to nothing.
        assert not app._volumes, f"orphaned volumes after unloading {unload}"
        assert not app._groups, f"orphaned group after unloading {unload}"


def _phase_and_land(app):
    rid = app._reflections[0]["id"]
    mid = app._models[0]["id"]
    app.make_maps(rid, mid)
    _drain("pxviewer-phasing")
    QApplication.processEvents()
    assert_viewer_consistent(app)
    return rid


@pytest.mark.parametrize("unload", ["reflections", "group"])
def test_update_chain_survives_unload(xray_app, unload):
    """After a minimization the app auto-re-phases on the GUI thread. If the reflections
    or the whole group were unloaded first, that must be a quiet no-op — update_maps
    itself raises on a missing pairing, and an exception here escapes into the event
    loop."""
    app = xray_app
    rid = _phase_and_land(app)
    gid = app._reflection_entry(rid).get("group")

    if unload == "reflections":
        app.remove_reflections(rid)
    else:
        app.remove_group(gid)

    # Exactly what the post-minimization chain emits to run on the GUI thread.
    app._update_maps_if_live(rid)
    QApplication.processEvents()
    assert_viewer_consistent(app)


# -- minimize streaming while another model is removed underneath it ----------

def test_minimize_survives_concurrent_removal(guarded_modals):
    """A genuinely concurrent case (no marshalling trick): the minimizer streams frames
    on its thread while the GUI thread removes a *different* model, mutating the shared
    registry list mid-run. The run must complete and the app stay consistent."""
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("minimization needs the monomer library")

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    app._controls.widget().show()
    try:
        app.load_file(os.path.join(DATA, "1ubq.pdb"))
        app.load_file(os.path.join(DATA, "1tec.pdb"))
        QApplication.processEvents()
        victim = app._models[1]["id"]
        app.set_active_model(app._models[0]["id"])

        app.minimize_model()  # streams on pxviewer-minimize
        time.sleep(0.05)      # let it get going
        app.remove_model(victim)  # mutate the shared list while it streams
        QApplication.processEvents()
        assert_viewer_consistent(app)

        _drain("pxviewer-minimize")
        QApplication.processEvents()
        assert_viewer_consistent(app)
        assert app._model_entry(victim) is None
    finally:
        app.stop()


# -- a drag whose model is unloaded while it is held --------------------------

def test_drag_ends_when_its_model_is_unloaded(guarded_modals):
    """Begin a drag, remove the model being dragged, then deliver the rest of the drag.
    Serving each message re-resolves the model, so the removed one is a no-op — and the
    in-flight Tug (which holds the gone model) is closed out rather than left for the
    free-run loop to keep stepping."""
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("dragging needs the monomer library")

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    app._controls.widget().show()
    try:
        app.load_file(os.path.join(DATA, "1ubq.pdb"))
        QApplication.processEvents()
        mid = app._models[0]["id"]

        # Drive the drag synchronously (bypassing the worker thread) so the interleaving
        # is deterministic: begin, then the model vanishes, then move/end arrive.
        app._serve_tug(mid, "begin", 0, None)
        assert app._tug is not None, "drag should have started"
        app.remove_model(mid)

        app._serve_tug(mid, "move", 0, (1.0, 2.0, 3.0))  # must not raise
        app._serve_tug(mid, "end", 0, None)
        assert app._tug is None, "drag was not closed out after its model was unloaded"

        QApplication.processEvents()
        assert_viewer_consistent(app)
    finally:
        app.stop()
