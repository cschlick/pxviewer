"""Mutating the registry while a background job is in flight.

Every threaded job in the app -- phasing, minimization, dragging -- captures the objects
it works on and finishes *later*, on the GUI thread. If the user unloads one of those in
between, the job lands against a world that has changed under it. This is where the real
bugs lived (a control session pointed at a stopped socket, a callback referencing a
removed entry), and none of them showed up in the synchronous walk, because the whole
point is the gap between start and finish.

What makes this deterministic rather than a flaky race: the jobs marshal their result
back with ``run_on_main``, a queued callback that only runs when the GUI loop is pumped.
So an exercise can let the worker finish its computation (:func:`drain`, which waits for
the thread *without* pumping), mutate the registry, and only then pump -- landing the
callback against the mutated state. The same interleaving a real unload-mid-job produces,
but reproducibly. One genuinely threaded case (minimize streaming while another model is
removed) covers the shared-list case the marshalling trick cannot.

After every case the invariant bank in ``gui_invariants`` must still hold.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import sys
import threading
import time

from pxviewer.regression.tst_utils import (
    closing_modals, data_path, have, qt_application, shipped_defaults, skip)

if not have("PySide6.QtWebEngineWidgets", "websockets"):
    skip("PySide6 QtWebEngine / websockets not available")

QAPP = qt_application()

from PySide6.QtWidgets import QApplication        # noqa: E402

from pxviewer.regression.gui_invariants import assert_viewer_consistent   # noqa: E402

#: Phasing a 3.0 A demo and a full minimization are both slow on a loaded machine.
DRAIN_TIMEOUT_S = 120.0


def drain(name, timeout=DRAIN_TIMEOUT_S):
    """Wait for a named worker thread to finish, **without** pumping the GUI loop.

    Not pumping is the whole point: the worker's ``run_on_main`` callback is left queued
    and pending, so the caller can change the registry before letting it land -- the
    unload-while-the-job-was-running window, made reproducible.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(t.name == name and t.is_alive() for t in threading.enumerate()):
            return
        time.sleep(0.02)
    raise RuntimeError("worker %r did not finish within %ss" % (name, timeout))


@contextlib.contextmanager
def desktop():
    """A running app with its controls shown, stopped afterwards.

    The controls are shown because several invariants read widget state -- the focused
    object, and whether the Loaded tree can be rebuilt from the summary.
    """
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    app._controls.widget().show()
    with closing_modals():
        try:
            yield app
        finally:
            app.stop()


@contextlib.contextmanager
def xray_desktop():
    """An app with a model and reflections loaded but unpaired -- ready to phase."""
    with desktop() as app:
        app.load_xray_demo(d_min=3.0)
        QApplication.processEvents()
        yield app


def phase_and_land(app):
    """Phase, let the result land, and return the reflections id."""
    rid = app._reflections[0]["id"]
    mid = app._models[0]["id"]
    app.make_maps(rid, mid)
    drain("pxviewer-phasing")
    QApplication.processEvents()
    assert_viewer_consistent(app)
    return rid


# -- phasing lands after its source was unloaded ------------------------------


def exercise_make_maps_survives_unload_before_it_lands():
    """Phase on a thread, unload the source in the window before the maps land, then let
    them land.

    The maps belong to objects that no longer exist, so nothing orphaned may appear: no
    volumes, and no group paired to a model that is gone.
    """
    for unload in ["none", "model", "reflections", "both"]:
        with xray_desktop() as app:
            rid = app._reflections[0]["id"]
            mid = app._models[0]["id"]

            app.make_maps(rid, mid)
            drain("pxviewer-phasing")     # computed; the add-on-main callback is queued

            if unload in ("model", "both"):
                app.remove_model(mid)
            if unload in ("reflections", "both"):
                app.remove_reflections(rid)

            QApplication.processEvents()  # land it against the mutated registry
            assert_viewer_consistent(app)

            if unload == "none":
                assert len(app._volumes) == 2, unload
                assert len(app._groups) == 1, unload
            else:
                # The source was pulled out from under the job, so its results are
                # discarded rather than left as maps paired to nothing.
                assert not app._volumes, "orphaned volumes after unloading %s" % unload
                assert not app._groups, "orphaned group after unloading %s" % unload


def exercise_the_update_chain_survives_unload():
    """After a minimization the app re-phases on the GUI thread. If the reflections or
    the whole group were unloaded first, that has to be a quiet no-op.

    ``update_maps`` itself raises on a missing pairing, and an exception raised here
    escapes into the event loop, where nothing catches it.
    """
    for unload in ["reflections", "group"]:
        with xray_desktop() as app:
            rid = phase_and_land(app)
            gid = app._reflection_entry(rid).get("group")

            if unload == "reflections":
                app.remove_reflections(rid)
            else:
                app.remove_group(gid)

            # Exactly what the post-minimization chain emits to run on the GUI thread.
            app._update_maps_if_live(rid)
            QApplication.processEvents()
            assert_viewer_consistent(app)


# -- a job streaming while the registry changes -------------------------------


def exercise_minimize_survives_concurrent_removal():
    """A genuinely concurrent case, with no marshalling trick: the minimizer streams
    frames on its thread while the GUI thread removes a *different* model, mutating the
    shared registry list mid-run."""
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        print("    (skipped: minimization needs the monomer library)")
        return

    with desktop() as app:
        app.load_file(data_path("1ubq.pdb"))
        app.load_file(data_path("1tec.pdb"))
        QApplication.processEvents()
        victim = app._models[1]["id"]
        app.set_active_model(app._models[0]["id"])

        app.minimize_model()                  # streams on pxviewer-minimize
        time.sleep(0.05)                      # let it get going
        app.remove_model(victim)              # mutate the shared list while it streams
        QApplication.processEvents()
        assert_viewer_consistent(app)

        # The run is continuous -- it holds at its minimum until ended -- so stop it, then
        # let the thread wind down and land its result against the mutated registry.
        app.stop_minimization()
        drain("pxviewer-minimize")
        QApplication.processEvents()
        assert_viewer_consistent(app)
        assert app._model_entry(victim) is None


def exercise_a_drag_ends_when_its_model_is_unloaded():
    """Begin a drag, remove the model being dragged, then deliver the rest of the drag.

    Serving each message re-resolves the model, so the removed one is a no-op -- and the
    in-flight Tug, which holds the gone model, is closed out rather than left for the
    free-run loop to keep stepping.
    """
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        print("    (skipped: dragging needs the monomer library)")
        return

    with desktop() as app:
        app.load_file(data_path("1ubq.pdb"))
        QApplication.processEvents()
        mid = app._models[0]["id"]

        # Driven synchronously, bypassing the worker thread, so the interleaving is
        # deterministic: begin, then the model vanishes, then move and end arrive.
        app._serve_tug(mid, "begin", 0, None)
        assert app._tug is not None, "the drag should have started"
        app.remove_model(mid)

        app._serve_tug(mid, "move", 0, (1.0, 2.0, 3.0))        # must not raise
        app._serve_tug(mid, "end", 0, None)
        assert app._tug is None, "the drag was not closed out after its model was unloaded"

        QApplication.processEvents()
        assert_viewer_consistent(app)


def run():
    # Every exercise here builds a DesktopApp, which reads its defaults from QSettings --
    # so the whole file runs against a fresh install's preferences, not the user's.
    with shipped_defaults():
        for name, fn in sorted(globals().items()):
            if name.startswith("exercise"):
                print("  %s" % name)
                sys.stdout.flush()
                fn()
    print("OK")


if __name__ == "__main__":
    run()
