"""Smoke tests for the human-facing demos.

Demos loop forever for a human to watch; here we run each briefly against a stub
session, then stop it and assert it streamed correctly-shaped frames.
"""

import threading
import time

import numpy as np
import pytest

from pxviewer.demos import DEMOS, Player


class _StubSession:
    def __init__(self):
        self.frames = []
        self.selections = []
        self.primitives_added = []
        self.measure_modes = []
        self.interactions = None
        self.clashes = []

    def push(self, coords):
        self.frames.append(np.asarray(coords, dtype="<f4"))

    # The primitives demo drives these; record the kinds drawn.
    def add_distance(self, *a, **k):
        self.primitives_added.append("distance")

    def add_angle(self, *a, **k):
        self.primitives_added.append("angle")

    def add_dihedral(self, *a, **k):
        self.primitives_added.append("dihedral")

    def add_label(self, *a, **k):
        self.primitives_added.append("label")

    def remove_primitive(self, primitive_id):
        pass

    def clear_primitives(self):
        pass

    # The interactions demo drives these.
    def set_interactions(self, interactions, **kwargs):
        self.interactions = interactions
        return interactions

    def clear_interactions(self):
        self.interactions = None

    # The clashes demo drives these.
    def set_clashes(self, pairs, **kwargs):
        self.clashes = list(pairs)
        return self.clashes

    def clear_clashes(self):
        self.clashes = []

    # The measure demo drives these.
    def enable_measure_mode(self, kind, on_measure=None, **kwargs):
        self.measure_modes.append(kind)

    def disable_mouse_selection(self):
        pass

    # The select demo drives these; record the atom specs. select() returns a
    # length-having value like a real Selection so the demo can report a count.
    def select(self, atoms, **kwargs):
        self.selections.append(atoms)
        return atoms if hasattr(atoms, "__len__") else [atoms]

    def highlight(self, atoms, **kwargs):
        self.selections.append(atoms)
        return atoms if hasattr(atoms, "__len__") else [atoms]

    def focus(self, atoms, **kwargs):
        return atoms

    def clear_selection(self):
        self.selections.append(None)


@pytest.mark.parametrize("name", list(DEMOS))
def test_demo_streams_valid_frames_and_stops(name):
    demo = DEMOS[name]
    sites, _labels = demo.make_sites()
    n = len(sites)
    assert n >= 2

    base = np.asarray(sites, dtype="<f4")
    stub = _StubSession()
    player = Player(stub, base, fps=240)  # fast so a short run covers several steps

    thread = threading.Thread(target=demo.run, args=(player,), daemon=True)
    thread.start()
    time.sleep(0.4)
    player.stop()
    thread.join(timeout=3)

    assert not thread.is_alive(), f"demo '{name}' did not stop"
    assert stub.frames, f"demo '{name}' produced no frames"
    for frame in stub.frames[:10]:
        assert frame.shape == (n, 3)
        assert np.isfinite(frame).all()


def test_pick_demo_reacts_to_pick():
    demo = DEMOS["pick"]
    sites, _labels = demo.make_sites()
    base = np.asarray(sites, dtype="<f4")
    stub = _StubSession()
    player = Player(stub, base, fps=240)

    thread = threading.Thread(target=demo.run, args=(player,), daemon=True)
    thread.start()
    time.sleep(0.1)
    player._on_pick({"id": 1, "name": "C", "resname": "UNL", "resseq": 1, "chain": "A"})
    time.sleep(0.4)
    player.stop()
    thread.join(timeout=3)

    # Atom 0 should have moved away from its rest position at some point.
    moved = any(not np.allclose(f[0], base[0]) for f in stub.frames)
    assert moved, "picked atom never pulsed"


def test_select_demo_issues_selections():
    demo = DEMOS["select"]
    sites, _labels = demo.make_sites()
    base = np.asarray(sites, dtype="<f4")
    stub = _StubSession()
    player = Player(stub, base, fps=240)

    thread = threading.Thread(target=demo.run, args=(player,), daemon=True)
    thread.start()
    time.sleep(0.4)
    player.stop()
    thread.join(timeout=3)

    assert not thread.is_alive(), "select demo did not stop"
    assert stub.selections, "select demo issued no selections"


def test_primitives_demo_draws_measurements():
    demo = DEMOS["primitives"]
    sites, _labels = demo.make_sites()
    base = np.asarray(sites, dtype="<f4")
    stub = _StubSession()
    player = Player(stub, base, fps=240)

    thread = threading.Thread(target=demo.run, args=(player,), daemon=True)
    thread.start()
    time.sleep(0.5)
    player.stop()
    thread.join(timeout=3)

    assert not thread.is_alive(), "primitives demo did not stop"
    assert stub.primitives_added, "primitives demo drew nothing"


def test_measure_demo_enables_measure_modes():
    demo = DEMOS["measure"]
    sites, _labels = demo.make_sites()
    base = np.asarray(sites, dtype="<f4")
    stub = _StubSession()
    player = Player(stub, base, fps=240)

    thread = threading.Thread(target=demo.run, args=(player,), daemon=True)
    thread.start()
    time.sleep(0.5)
    player.stop()
    thread.join(timeout=3)

    assert not thread.is_alive(), "measure demo did not stop"
    assert stub.measure_modes, "measure demo enabled no measure mode"
    assert stub.frames, "measure demo streamed no frames"
