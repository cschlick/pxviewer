"""Per-atom Q-score, and coloring a model by how well it fits its map.

Q-score is the one per-atom attribute pxviewer computes from a map+model pair rather than
from geometry alone, so it exercises two things nothing else does: mapping values from
cctbx's non-hydrogen subset back onto the full model, and the threaded compute-then-color
path in the desktop.
"""

from __future__ import absolute_import, division, print_function

import sys
import time

from pxviewer.regression.tst_utils import (
    dispose, have, process_events, qt_application, shipped_defaults, skip)

if not have("PySide6.QtWebEngineWidgets", "websockets",
            "cctbx.maptbx.qscore", "iotbx.map_model_manager", "numpy"):
    skip("PySide6 QtWebEngine / websockets / cctbx.maptbx.qscore not available")

import numpy as np                                   # noqa: E402

QAPP = qt_application()

from pxviewer.desktop import _QSCORE_COLOR, DesktopApp   # noqa: E402

#: Coloring runs on a worker thread; the map is small but reduce2-free Q-score on a real
#: structure is still tens of seconds on a slow machine.
COLOR_TIMEOUT_S = 300


def app_with_map():
    """A desktop app holding the bundled map+model demo -- a model paired with density.

    Built per exercise rather than shared: each one colors the model, and the attribute
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
        process_events()
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
        dispose(app)


def exercise_hydrogens_come_back_missing_not_as_a_bad_fit():
    """Hydrogens are never scored, so they must come back nan -- which the attribute theme
    draws in its "missing" color -- rather than 0, which would paint them as the
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
        dispose(app)


# -- coloring by it ----------------------------------------------------------


def exercise_coloring_by_qscore_needs_a_map():
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
        dispose(app)


def exercise_coloring_by_qscore_sends_per_atom_values():
    """It computes on a thread and colors through the attribute path, so the
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
        # A fixed 0-1 domain, so the same color means the same quality in any structure.
        assert list(spec["attribute"]["domain"]) == [0.0, 1.0]
        assert spec["attribute"]["palette"] == "red-yellow-green"     # low red, high green
    finally:
        dispose(app)


def exercise_leaving_qscore_drops_the_values_it_colored_by():
    """The scores belong to one map+model pairing, so switching color has to drop them --
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
        dispose(app)


def exercise_coloring_by_map_model_cc_needs_a_map():
    """Map-model CC follows Q-score's contract exactly: with no paired map the choice
    is refused and reverted, with a status line saying why."""
    from pxviewer.desktop import _CC_COLOR
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        said = []
        app.bridge.status_changed.connect(said.append)
        mid = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "no map")

        app.set_model_color(mid, _CC_COLOR)

        entry = app._model_entry(mid)
        assert entry["color"] is None
        assert entry.get("attribute") is None
        assert any("map-model CC needs a map" in s for s in said)
    finally:
        dispose(app)


def exercise_coloring_by_map_model_cc_sends_per_atom_values():
    """Per-atom CC (cctbx's mmtbx.maps.correlation) through the attribute path, on the
    correlation's own absolute [0, 1] domain with the px spectrum, plus per-residue
    means so a cartoon can show it too. On the synthetic pair — a map computed from
    the model itself — the correlation is near-perfect, which pins the atom order."""
    from pxviewer.desktop import _CC_COLOR, DesktopApp as _App

    app = app_with_map()
    try:
        entry = app._models[0]
        app.set_model_color(entry["id"], _CC_COLOR)
        assert wait_for_attribute(app, entry), "map-model CC never landed"

        session = entry["session"]
        values = np.asarray(entry["attribute"]["values"], dtype=float)
        assert values.size == session._n_atoms
        finite = values[np.isfinite(values)]
        assert finite.size and float(np.median(finite)) > 0.9   # self-map: near-perfect
        assert entry["attribute"]["residue_values"] is not None

        spec = list(session._representations.values())[0]
        assert spec["color"] == "attribute"
        assert spec["attribute"]["name"] == _CC_COLOR
        assert list(spec["attribute"]["domain"]) == [0.0, 1.0]
        assert spec["attribute"]["palette"] == _App._CC_MODEL_PALETTE
    finally:
        dispose(app)


def exercise_every_value_coloring_gets_the_same_scale_machinery():
    """The registry is the contract: *every* per-atom coloring -- refined properties
    and computed ones alike -- opens with a domain and a default_domain, keeps a range
    the user set across a re-apply, resets to its calibrated default, and shows the
    shared Range group on the pane.

    Registry-driven on purpose. The computed colorings were once kept out of the
    scale machinery deliberately, which left a correlation spanning 0.6-0.95 painted
    on a fixed 0-1 ramp with no way to see the variation; iterating the registry means
    a coloring added later cannot quietly go the same way.
    """
    from PySide6.QtWidgets import QDoubleSpinBox

    from pxviewer.desktop import _HOTSPOT_COLOR, _MODEL_VALUE_COLORS

    class Field:                      # a cached hotspot score, without the minute of probe
        def __init__(self, n):
            self.values = np.linspace(0.0, 2.0, n)

    app = app_with_map()
    try:
        entry = app._models[0]
        mid = entry["id"]
        n_atoms = entry["session"]._n_atoms
        entry["hotspots"] = Field(n_atoms)

        for color, info in _MODEL_VALUE_COLORS.items():
            entry.pop("attribute", None)
            entry["color"] = None
            app.set_model_color(mid, color)
            deadline = time.time() + COLOR_TIMEOUT_S
            while time.time() < deadline and (
                    entry.get("attribute") or {}).get("name") != color:
                process_events()
                time.sleep(0.05)
            attribute = entry.get("attribute") or {}
            assert attribute.get("name") == color, "%s never landed" % color
            assert len(attribute["values"]) == n_atoms, color

            lo, hi = attribute["domain"]
            assert hi > lo, color
            assert attribute.get("default_domain") is not None, (
                "%s has no default to Reset to" % color)

            # The range is the user's, in the same three ways for every coloring.
            app.set_model_value_domain(mid, lo, hi + 5.0)
            assert entry["attribute"]["domain"] == (lo, hi + 5.0), color
            app.set_model_value_domain(mid, hi + 5.0, lo)          # crossed: refused
            assert entry["attribute"]["domain"] == (lo, hi + 5.0), color
            app.fit_model_value_domain(mid)
            fitted = entry["attribute"]["domain"]
            assert fitted[1] > fitted[0], color
            app.reset_model_value_domain(mid)
            assert entry["attribute"]["domain"] == attribute["default_domain"], color

            # ...and it survives the coloring being re-applied (a rep change, a recompute).
            app.set_model_value_domain(mid, lo, hi + 5.0)
            getattr(app, info["method"])(mid)
            deadline = time.time() + COLOR_TIMEOUT_S
            while time.time() < deadline and \
                    entry["attribute"]["domain"] != (lo, hi + 5.0):
                process_events()
                time.sleep(0.05)
            assert entry["attribute"]["domain"] == (lo, hi + 5.0), (
                "%s clobbered the user's range on re-apply" % color)

            # The pane offers the shared Range group, labeled for this coloring.
            app._controls._update_appearance("model", mid, force=True)
            process_events()
            spins = [w for w in app._controls._appearance_box.findChildren(QDoubleSpinBox)
                     if w.objectName().startswith("value-domain-")]
            assert len(spins) == 2, "no Range group for %s" % color
            assert {round(sp.value(), 2) for sp in spins} == {
                round(lo, 2), round(hi + 5.0, 2)}, color
    finally:
        dispose(app)


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
