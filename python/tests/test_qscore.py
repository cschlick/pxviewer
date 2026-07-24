"""Per-atom Q-score, and coloring a model by its fit to the map."""

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _app_with_map():
    """A DesktopApp holding the bundled map+model demo — a model paired with density."""
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    app.load_map_model_demo(d_min=4.0)  # coarser = faster to generate
    return app


def test_qscore_is_one_value_per_atom_of_the_original_model(qapp):
    """cctbx scores only non-hydrogen atoms, and strips them from the manager it is handed.
    The wrapper puts the values back on the *full* model's atom order (nan where there is
    none) and leaves the live model alone."""
    pytest.importorskip("cctbx.maptbx.qscore")
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.qscore import per_atom_qscore

    app = _app_with_map()
    try:
        entry = app._models[0]
        mmm = app.group_mmm(entry["group"])
        model = entry["session"].model
        n_atoms = model.get_number_of_atoms()

        values = per_atom_qscore(mmm)

        assert values.shape == (n_atoms,)                  # the whole model, not the subset
        assert model.get_number_of_atoms() == n_atoms      # the live model was not stripped
        assert app.group_mmm(entry["group"]) is mmm        # nor was the manager swapped
        assert entry["session"].model is mmm.model()       # ... nor its model replaced

        finite = values[np.isfinite(values)]
        assert finite.size
        assert finite.max() <= 1.0                         # 1 is a textbook fit; nothing beats it
        assert finite.mean() > 0.3                         # a model in its own map fits well
    finally:
        app.stop()


def test_hydrogens_come_back_missing_not_as_a_bad_fit(qapp):
    """Hydrogens are never scored. They have to come back nan — which the attribute theme
    draws in its "missing" color — rather than 0, which would paint them as the worst-fitting
    atoms in the structure."""
    pytest.importorskip("cctbx.maptbx.qscore")
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.qscore import per_atom_qscore

    app = _app_with_map()
    try:
        entry = app._models[0]
        atoms = entry["session"].model.get_hierarchy().atoms()
        elements = [e.strip().upper() for e in atoms.extract_element()]

        values = per_atom_qscore(app.group_mmm(entry["group"]))

        for i, element in enumerate(elements):
            if element in ("H", "D"):
                assert np.isnan(values[i]), f"hydrogen {i} scored {values[i]}"
            else:
                assert np.isfinite(values[i]), f"heavy atom {i} did not score"
    finally:
        app.stop()


def test_coloring_by_qscore_needs_a_map(qapp):
    """Q-score measures a model against density, so with no map there is nothing to score:
    the choice is refused and reverted rather than quietly showing something else."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import _QSCORE_COLOR, DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        said = []
        app.bridge.status_changed.connect(said.append)
        mid = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "no map")

        app.set_model_color(mid, _QSCORE_COLOR)

        entry = app._model_entry(mid)
        assert entry["color"] is None                        # reverted, not left claiming
        assert entry.get("attribute") is None
        assert any("Q-score needs a map" in s for s in said)  # and it says why
    finally:
        app.stop()


def test_coloring_by_qscore_sends_per_atom_values(qapp):
    """With a map it computes on a thread and colors through the attribute path — the
    representation ends up keyed to a named per-atom attribute, not a Mol* theme."""
    pytest.importorskip("cctbx.maptbx.qscore")
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import _QSCORE_COLOR

    app = _app_with_map()
    try:
        entry = app._models[0]
        mid = entry["id"]
        app.set_model_color(mid, _QSCORE_COLOR)

        deadline = time.time() + 300
        while time.time() < deadline and entry.get("attribute") is None:
            qapp.processEvents()  # the worker applies via the main thread
            time.sleep(0.05)
        assert entry.get("attribute") is not None, "Q-score never landed"

        session = entry["session"]
        assert len(session._attributes[_QSCORE_COLOR]) == session._n_atoms
        spec = list(session._representations.values())[0]
        assert spec["color"] == "attribute"
        assert spec["attribute"]["name"] == _QSCORE_COLOR
        # A fixed 0-1 scale, so the same color means the same quality in any structure.
        assert list(spec["attribute"]["domain"]) == [0.0, 1.0]
        assert spec["attribute"]["palette"] == "red-yellow-green"  # low red, high green
    finally:
        app.stop()


def test_leaving_qscore_drops_the_values_it_colored_by(qapp):
    """The per-atom scores belong to that one map+model pairing. Switching to another color
    has to drop them, or a later Q-score would have stale numbers sitting behind it."""
    pytest.importorskip("cctbx.maptbx.qscore")
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.desktop import _QSCORE_COLOR

    app = _app_with_map()
    try:
        entry = app._models[0]
        app.set_model_color(entry["id"], _QSCORE_COLOR)
        deadline = time.time() + 300
        while time.time() < deadline and entry.get("attribute") is None:
            qapp.processEvents()
            time.sleep(0.05)
        assert entry.get("attribute") is not None

        app.set_model_color(entry["id"], "chain-id")

        assert entry.get("attribute") is None
        spec = list(entry["session"]._representations.values())[0]
        assert spec["color"] == "chain-id"
    finally:
        app.stop()
