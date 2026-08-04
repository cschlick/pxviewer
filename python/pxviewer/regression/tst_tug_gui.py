"""The desktop wiring around a tug: scopes, continuous mode, and frame de-duplication.

The tug itself -- what the minimization does to the coordinates -- is in ``tst_tug.py``
and needs no Qt. What is here is the part between the browser and cctbx.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import sys

from pxviewer.regression.tst_utils import (
    data_path, dispose, have, qt_application, shipped_defaults, skip)

if not have("PySide6.QtWebEngineWidgets", "websockets",
            "mmtbx.geometry_restraints.reference", "numpy"):
    skip("PySide6 QtWebEngine / websockets / mmtbx restraints not available")

from pxviewer.geometry import monomer_library_available   # noqa: E402

if not monomer_library_available():
    skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

qt_application()

from pxviewer.desktop import DesktopApp             # noqa: E402

MODEL = data_path("1ubq.pdb")
ATOM = 300


@contextlib.contextmanager
def desktop_with_model():
    """A running app with 1UBQ loaded, as ``(app, model_id)``."""
    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_file(MODEL)
        yield app, app._models[0]["id"]
    finally:
        dispose(app)


def exercise_the_scope_setting_reaches_the_tug():
    """Settings' "Moves:" control sets the scope, and a drag started afterwards builds a
    Tug with it -- a single residue moves far fewer atoms than the default sphere."""
    with desktop_with_model() as (app, mid):
        app.set_tug_scope(mode="sphere", radius=8.0)
        app._serve_tug(mid, "begin", ATOM, None)
        sphere_zone = app._tug.zone_size
        app._serve_tug(mid, "end", ATOM, None)

        app.set_tug_scope(mode="residues", flank=0)
        app._serve_tug(mid, "begin", ATOM, None)
        single_zone = app._tug.zone_size
        app._serve_tug(mid, "end", ATOM, None)

        assert single_zone < sphere_zone          # one residue is fewer than a sphere
        assert app._tug is None                   # and it is cleaned up after each drag


def exercise_selection_scope_needs_a_selection():
    """With nothing picked there is no zone to build, so the drag warns and does not
    start rather than silently falling back to a sphere."""
    with desktop_with_model() as (app, mid):
        app.set_tug_scope(mode="sphere", radius=8.0)
        app._serve_tug(mid, "begin", ATOM, None)
        sphere_zone = app._tug.zone_size
        app._serve_tug(mid, "end", ATOM, None)

        app.set_tug_scope(mode="selection")
        app._serve_tug(mid, "begin", ATOM, None)
        assert app._tug is None

        # With a selection, the same drag builds a Tug bounded to it.
        app._scene_selection[mid] = list(range(295, 306))
        app._serve_tug(mid, "begin", ATOM, None)
        assert app._tug is not None
        assert 0 < app._tug.zone_size <= sphere_zone     # the picked residues, not a sphere
        app._serve_tug(mid, "end", ATOM, None)


def exercise_continuous_mode_free_runs_and_does_not_resend_a_settled_frame():
    """Two halves of the same loop.

    In continuous mode the worker keeps stepping with no new message, so a held-still
    drag settles. And a frame identical to the last is not re-sent: once a drag has truly
    converged, pushing the same conformation thirty times a second is pointless traffic.

    Counted through the session's own frame index rather than by intercepting ``push``.
    That index is what the wire protocol numbers frames with, so it moves if and only if
    something was really broadcast.
    """
    with desktop_with_model() as (app, mid):
        session = app._models[0]["session"]
        app.set_tug_continuous(True)

        start = session.model.get_sites_cart().as_numpy_array()[ATOM].copy()
        app._serve_tug(mid, "begin", ATOM, None)
        app._serve_tug(mid, "move", ATOM, (start + [3.0, 0.0, 0.0]).tolist())

        # Free-run with no new message: the model keeps moving while it has somewhere to
        # go, which is what a held-still drag does in continuous mode.
        before = session._frame_index
        for _ in range(20):
            app._tug_relax()
        assert session._frame_index > before

        settled = session._frame_index
        app._push_tug(app._tug_last.copy())          # the same coordinates again
        assert session._frame_index == settled

        app._serve_tug(mid, "end", ATOM, None)
        assert app._tug is None


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
