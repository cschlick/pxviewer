"""A seeded random walk over the GUI's actions, asserting the invariant bank each step.

This is the cheap first cut of GUI fuzzing (a plain seeded loop, no Hypothesis yet). It
drives the ``DesktopApp`` action surface — load, focus, pair, restyle, remove — in a
random but *valid* order, and after every action asserts the model stays consistent (see
gui_invariants). The point is coverage of state *combinations* no hand-written test
reaches, with the invariants doing the catching.

It stays on the fast, synchronous actions on purpose: minimize, tug and make_maps run on
background threads and take seconds, so they belong in their own timed tests rather than
in a tight random loop. Everything here is a state transition the appearance pane and the
registries actually see.

Determinism: a fixed seed, RNGs seeded, and every action appended to a log so a failure
prints the exact sequence that produced it.
"""

import os

import random

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("websockets")
pytest.importorskip("PySide6.QtWebEngineWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui_invariants import assert_viewer_consistent  # noqa: E402

MODELS = ["1ubq.pdb", "1tec.pdb"]


@pytest.fixture
def guarded_modals(monkeypatch):
    """Never let a modal dialog block the run. The fuzzer drives backend methods, which
    do not open dialogs, but a stray one would hang the suite — so any that appears is
    auto-answered rather than shown."""
    from PySide6.QtWidgets import QColorDialog, QFileDialog, QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([], "")))
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: __import__(
        "PySide6.QtGui", fromlist=["QColor"]).QColor()))


class _Walk:
    """One random walk: the actions it can take, gated by what is currently loaded."""

    def __init__(self, app, rng):
        self.app = app
        self.rng = rng
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "pxviewer", "data")

    # -- helpers ----
    def _models(self):
        return list(self.app._models)

    def _volumes(self):
        return list(self.app._volumes)

    def _pick(self, seq):
        return self.rng.choice(seq) if seq else None

    # -- actions (return a label, or None if a precondition was not met) ----
    def load_model(self):
        path = os.path.join(self.data_dir, self.rng.choice(MODELS))
        self.app.load_file(path)
        return f"load_model {os.path.basename(path)}"

    def load_volume(self):
        from pxviewer.volume_io import VolumeData
        self.app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        return "load_volume"

    def load_group(self):
        self.app.load_map_model_demo(d_min=4.0)
        return "load_group (map+model)"

    def focus_object(self):
        objs = ([("model", m["id"]) for m in self._models()]
                + [("volume", v["id"]) for v in self._volumes()]
                + [("reflections", r["id"]) for r in self.app._reflections])
        pick = self._pick(objs)
        if pick is None:
            return None
        kind, ident = pick
        if kind == "model":
            self.app.set_active_model(ident)
        else:
            self.app._controls._update_appearance(kind, ident)
        return f"focus {kind} {ident}"

    def toggle_visible(self):
        m = self._pick(self._models() + self._volumes())
        if m is None:
            return None
        new = not m["visible"]
        if m in self._models():
            self.app.set_model_visible(m["id"], new)
        else:
            self.app.set_volume_visible(m["id"], new)
        return f"visible {m['id']} -> {new}"

    def restyle_volume(self):
        v = self._pick(self._volumes())
        if v is None:
            return None
        which = self.rng.choice(["color", "iso", "opacity", "style", "clip", "radius", "mask"])
        vid = v["id"]
        if which == "color":
            self.app.set_volume_color(vid, self.rng.choice(["gold", "salmon", "#3fa9f5"]))
        elif which == "iso":
            self.app.set_volume_iso(vid, round(self.rng.uniform(0.5, 6.0), 2))
        elif which == "opacity":
            self.app.set_volume_opacity(vid, round(self.rng.uniform(0.2, 1.0), 2))
        elif which == "style":
            self.app.set_volume_style(vid, self.rng.choice(["surface", "wireframe", "mesh"]))
        elif which == "clip":
            a, b = sorted((self.rng.random(), self.rng.random()))
            self.app.set_volume_clip(vid, a, b)
        elif which == "radius":
            self.app.set_volume_radius(vid, self.rng.choice([None, 10.0, 20.0]))
        elif which == "mask":
            # Only valid when paired; the method refuses otherwise, which is fine to hit.
            try:
                self.app.set_volume_mask(vid, self.rng.choice([None, 3.0]))
            except ValueError:
                pass
        return f"restyle_volume {vid} {which}"

    def restyle_model(self):
        m = self._pick(self._models())
        if m is None:
            return None
        which = self.rng.choice(["rep", "color", "clip"])
        mid = m["id"]
        if which == "rep":
            self.app.set_model_representation(mid, self.rng.choice(["cartoon", "ball-and-stick"]))
        elif which == "color":
            self.app.set_model_color(mid, self.rng.choice([None, "element-symbol", "chain-id"]))
        elif which == "clip":
            a, b = sorted((self.rng.random(), self.rng.random()))
            self.app.set_model_clip(mid, a, b)
        return f"restyle_model {mid} {which}"

    def pair(self):
        models, volumes = self.app.pairable()
        if not models or not volumes:
            return None
        self.app.pair_model_with_map(self._pick(models)["id"], self._pick(volumes)["id"])
        return "pair"

    def remove_object(self):
        m = self._pick(self._models() + self._volumes() + self.app._reflections)
        if m is None:
            return None
        if m in self._models():
            self.app.remove_model(m["id"])
        elif m in self._volumes():
            self.app.remove_volume(m["id"])
        else:
            self.app.remove_reflections(m["id"])
        return f"remove {m['id']}"

    def remove_group(self):
        gid = self._pick(list(self.app._groups))
        if gid is None:
            return None
        self.app.remove_group(gid)
        return f"remove_group {gid}"

    def reset_view(self):
        self.app.reset_view()
        return "reset_view"

    ACTIONS = [
        load_model, load_model, load_volume, load_group, focus_object, focus_object,
        toggle_visible, restyle_volume, restyle_model, pair, remove_object,
        remove_group, reset_view,
    ]

    def step(self):
        return self.rng.choice(self.ACTIONS)(self)


STEPS_PER_WALK = 150


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_gui_walk_keeps_the_model_consistent(seed, guarded_modals):
    """A run of random valid actions per seed, invariants after each. A failure prints
    the sequence that produced it, which is the minimal repro to paste into a unit test."""
    from pxviewer.desktop import DesktopApp

    rng = random.Random(seed)
    np.random.seed(seed)

    app = DesktopApp(port=0)
    app._webapp.start()
    # Show the controls: the widget invariants only bite on *visible* widgets (orphaning
    # a hidden one does not float it), so a hidden window would hide the very bug the
    # stray-window check exists for.
    app._controls.widget().show()
    walk = _Walk(app, rng)
    log = []
    try:
        for _ in range(STEPS_PER_WALK):
            label = walk.step()
            if label is None:
                continue  # a precondition was not met this time; try again next step
            log.append(label)
            QApplication.processEvents()
            try:
                assert_viewer_consistent(app)
            except AssertionError as exc:
                trail = "\n  ".join(log[-15:])
                raise AssertionError(
                    f"invariant broke after {len(log)} actions (seed {seed}):\n"
                    f"  {trail}\n-> {exc}") from exc
    finally:
        app.stop()


# -- widget monkey: real clicks through the controls -------------------------
#
# The backend walk above never touches a Qt signal; this does. It clicks buttons, cycles
# combos, toggles checkboxes and drags sliders in the real controls window, which is where
# the wiring bugs live (the colour dialog reopening on OK was one). Modals are auto-closed
# so nothing blocks, and the invariant bank runs after every interaction.
#
# The threaded buttons (Minimize/Stop/Add H + analyze) are left out: they start background
# work whose races are a separate concern from signal wiring.

# Buttons that start background work — Minimize, Stop, Add H + analyze. All icon-only now,
# so they are recognised by their tooltip prefix.
_THREADED_TOOLTIPS = (
    "Minimize the active model", "Halt the run", "Add hydrogens with reduce2")


@pytest.fixture
def modal_autocloser():
    """Reject any modal dialog that appears, so a click that opens one does not hang.

    Runs on a timer, so it fires inside the nested event loop a modal's exec() spins —
    the click handler gets its (cancelled) answer and carries on.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog

    def close_modals():
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QDialog) and w.isVisible() and w.isModal():
                w.reject()

    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(close_modals)
    timer.start()
    yield
    timer.stop()


def _interactive_widgets(ctl):
    """(label, callable) for each enabled, visible control worth poking."""
    from PySide6.QtWidgets import QAbstractSlider, QCheckBox, QComboBox, QPushButton

    root = ctl.widget()
    actions = []

    for button in root.findChildren(QPushButton):
        if not (button.isEnabled() and button.isVisibleTo(root)):
            continue
        # Threaded buttons start background work; skip them (recognised by tooltip since
        # they are icon-only).
        if button.toolTip().startswith(_THREADED_TOOLTIPS):
            continue
        # Skip menu buttons (Demos): their actions are heavy loads the backend walk
        # already covers, and triggering them in a tight loop just reloads slow demos.
        if button.menu() is not None:
            continue
        actions.append((f"click:{button.text() or button.toolTip()[:20]}", button.click))

    for combo in root.findChildren(QComboBox):
        if combo.isEnabled() and combo.isVisibleTo(root) and combo.count() > 1:
            actions.append((
                "combo", lambda c=combo: c.setCurrentIndex(
                    (c.currentIndex() + 1) % c.count())))

    for check in root.findChildren(QCheckBox):
        if check.isEnabled() and check.isVisibleTo(root):
            actions.append(("check", lambda w=check: w.toggle()))

    for slider in root.findChildren(QAbstractSlider):
        if slider.isEnabled() and slider.isVisibleTo(root):
            mid = (slider.minimum() + slider.maximum()) // 2
            actions.append(("slider", lambda w=slider, v=mid: w.setValue(v)))

    return actions


@pytest.mark.parametrize("seed", [0, 1])
def test_widget_monkey_keeps_the_model_consistent(seed, guarded_modals, modal_autocloser):
    """Set up a rich scene, then click/toggle/drag random real controls, asserting the
    invariant bank after each. A failure prints the widget it poked."""
    import os
    import random

    from PySide6.QtWidgets import QApplication

    from pxviewer.desktop import DesktopApp

    rng = random.Random(seed)
    np.random.seed(seed)
    data = os.path.join(os.path.dirname(__file__), "..", "pxviewer", "data")

    app = DesktopApp(port=0)
    app._webapp.start()
    app._controls.widget().show()
    try:
        # A scene with something of every kind to poke.
        app.load_file(os.path.join(data, "1ubq.pdb"))
        app.load_map_model_demo(d_min=4.0)
        app.load_xray_demo(d_min=3.0)
        QApplication.processEvents()

        ctl = app._controls
        poked = []
        for _ in range(80):
            # Focus a random object first, so the Appearance controls exist to poke.
            summary = app._loaded_summary()["items"]
            if summary:
                it = rng.choice(summary)
                if it["kind"] == "model":
                    app.set_active_model(it["id"])
                else:
                    ctl._update_appearance(it["kind"], it["id"])
                QApplication.processEvents()

            actions = _interactive_widgets(ctl)
            if not actions:
                continue
            label, act = rng.choice(actions)
            poked.append(label)
            act()
            QApplication.processEvents()
            try:
                assert_viewer_consistent(app)
            except AssertionError as exc:
                trail = "\n  ".join(poked[-15:])
                raise AssertionError(
                    f"invariant broke after {len(poked)} widget pokes (seed {seed}):\n"
                    f"  {trail}\n-> {exc}") from exc
    finally:
        app.stop()
