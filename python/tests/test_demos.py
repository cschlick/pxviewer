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

    def push(self, coords):
        self.frames.append(np.asarray(coords, dtype="<f4"))

    # The select demo drives these; record expressions, no viewer to answer.
    def select(self, expression, **kwargs):
        self.selections.append(expression)
        return None

    def highlight(self, expression, **kwargs):
        self.selections.append(expression)
        return None

    def focus(self, expression, **kwargs):
        return None

    def clear_selection(self):
        self.selections.append(None)


@pytest.mark.parametrize("name", list(DEMOS))
def test_demo_streams_valid_frames_and_stops(name):
    demo = DEMOS[name]
    atoms = demo.make_atoms()
    assert len(atoms) >= 2

    base = np.array([[a.x, a.y, a.z] for a in atoms], dtype="<f4")
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
        assert frame.shape == (len(atoms), 3)
        assert np.isfinite(frame).all()


def test_pick_demo_reacts_to_pick():
    demo = DEMOS["pick"]
    atoms = demo.make_atoms()
    base = np.array([[a.x, a.y, a.z] for a in atoms], dtype="<f4")
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
    atoms = demo.make_atoms()
    base = np.array([[a.x, a.y, a.z] for a in atoms], dtype="<f4")
    stub = _StubSession()
    player = Player(stub, base, fps=240)

    thread = threading.Thread(target=demo.run, args=(player,), daemon=True)
    thread.start()
    time.sleep(0.4)
    player.stop()
    thread.join(timeout=3)

    assert not thread.is_alive(), "select demo did not stop"
    assert any(isinstance(x, str) for x in stub.selections), "select demo issued no selections"
