"""Per-atom Q-score, and colouring a model by how well it fits its map.

Q-score is the one per-atom attribute pxviewer computes from a map+model pair rather than
from geometry alone, so it exercises two things nothing else does: mapping values from
cctbx's non-hydrogen subset back onto the full model, and the threaded compute-then-colour
path in the desktop.
"""

from __future__ import absolute_import, division, print_function

import sys
import time

from pxviewer.regression.tst_utils import (
    have, qt_application, shipped_defaults, skip)

if not have("PySide6.QtWebEngineWidgets", "websockets",
            "cctbx.maptbx.qscore", "iotbx.map_model_manager", "numpy"):
    skip("PySide6 QtWebEngine / websockets / cctbx.maptbx.qscore not available")

import numpy as np                                   # noqa: E402

QAPP = qt_application()

from pxviewer.desktop import _QSCORE_COLOR, DesktopApp   # noqa: E402

#: Colouring runs on a worker thread; the map is small but reduce2-free Q-score on a real
#: structure is still tens of seconds on a slow machine.
COLOR_TIMEOUT_S = 300


def app_with_map():
    """A desktop app holding the bundled map+model demo -- a model paired with density.

    Built per exercise rather than shared: each one colours the model, and the attribute
    state that leaves behind is exactly what the next would be asserting about.
    """
    app = DesktopApp(port=0)
    app._webapp.start()
    app.load_map_model_demo(d_min=4.0)          # coarser resolution = faster to generate
    return app


def wait_for_attribute(app, entry):
    """Pump the Qt loop until the worker's result lands on the main thread."""
    deadline = time.time() + COLOR_TIMEOUT_S
    while time.time() < deadline and entry.get("attribute") is None:
        QAPP.processEvents()
        time.sleep(0.05)
    return entry.get("attribute") is not None


# -- the values themselves ----------------------------------------------------


def exercise_qscore_is_one_value_per_atom_of_the_original_model():
    """cctbx scores only non-hydrogen atoms, and strips hydrogens from the manager it is
    handed. The wrapper has to put the values back in the *full* model's atom order and
    leave the live model untouched -- an off-by-a-hydrogen shift here would silently put
    every score on the wrong atom."""
    from pxviewer.qscore import per_atom_qscore

    app = app_with_map()
    try:
        entry = app._models[0]
        mmm = app.group_mmm(entry["group"])
        model = entry["session"].model
        n_atoms = model.get_number_of_atoms()

        values = per_atom_qscore(mmm)

        assert values.shape == (n_atoms,)                 # the whole model, not the subset
        assert model.get_number_of_atoms() == n_atoms     # the live model was not stripped
        assert app.group_mmm(entry["group"]) is mmm       # nor was the manager swapped
        assert entry["session"].model is mmm.model()      # nor its model replaced

        finite = values[np.isfinite(values)]
        assert finite.size
        assert finite.max() <= 1.0                        # 1 is a textbook fit
        assert finite.mean() > 0.3                        # a model in its own map fits well
    finally:
        app.stop()


def exercise_hydrogens_come_back_missing_not_as_a_bad_fit():
    """Hydrogens are never scored, so they must come back nan -- which the attribute theme
    draws in its "missing" colour -- rather than 0, which would paint them as the
    worst-fitting atoms in the structure."""
    from pxviewer.qscore import per_atom_qscore

    app = app_with_map()
    try:
        entry = app._models[0]
        atoms = entry["session"].model.get_hierarchy().atoms()
        elements = [e.strip().upper() for e in atoms.extract_element()]

        values = per_atom_qscore(app.group_mmm(entry["group"]))

        for i, element in enumerate(elements):
            if element in ("H", "D"):
                assert np.isnan(values[i]), "hydrogen %d scored %s" % (i, values[i])
            else:
                assert np.isfinite(values[i]), "heavy atom %d did not score" % i
    finally:
        app.stop()


# -- colouring by it ----------------------------------------------------------


def exercise_colouring_by_qscore_needs_a_map():
    """With no map there is nothing to score, so the choice is refused and reverted rather
    than quietly showing something else."""
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        said = []
        app.bridge.status_changed.connect(said.append)
        mid = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "no map")

        app.set_model_color(mid, _QSCORE_COLOR)

        entry = app._model_entry(mid)
        assert entry["color"] is None                  # reverted, not left claiming Q-score
        assert entry.get("attribute") is None
        assert any("Q-score needs a map" in s for s in said)     # and it says why
    finally:
        app.stop()


def exercise_colouring_by_qscore_sends_per_atom_values():
    """It computes on a thread and colours through the attribute path, so the
    representation ends up keyed to a named per-atom attribute rather than a Mol* theme."""
    app = app_with_map()
    try:
        entry = app._models[0]
        app.set_model_color(entry["id"], _QSCORE_COLOR)
        assert wait_for_attribute(app, entry), "Q-score never landed"

        session = entry["session"]
        assert len(session._attributes[_QSCORE_COLOR]) == session._n_atoms

        spec = list(session._representations.values())[0]
        assert spec["color"] == "attribute"
        assert spec["attribute"]["name"] == _QSCORE_COLOR
        # A fixed 0-1 domain, so the same colour means the same quality in any structure.
        assert list(spec["attribute"]["domain"]) == [0.0, 1.0]
        assert spec["attribute"]["palette"] == "red-yellow-green"     # low red, high green
    finally:
        app.stop()


def exercise_leaving_qscore_drops_the_values_it_coloured_by():
    """The scores belong to one map+model pairing, so switching colour has to drop them --
    otherwise a later Q-score would have stale numbers sitting behind it."""
    app = app_with_map()
    try:
        entry = app._models[0]
        app.set_model_color(entry["id"], _QSCORE_COLOR)
        assert wait_for_attribute(app, entry), "Q-score never landed"

        app.set_model_color(entry["id"], "chain-id")

        assert entry.get("attribute") is None
        spec = list(entry["session"]._representations.values())[0]
        assert spec["color"] == "chain-id"
    finally:
        app.stop()


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
