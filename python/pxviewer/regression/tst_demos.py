"""The human-facing demos, run briefly instead of forever.

Each demo is an endless loop for someone to watch, so here every one is started against a
real session, given a fraction of a second at a high frame rate, and stopped. What is
checked is that it streamed well-formed frames, that it shut down when asked, and -- for
the ones that do more than move atoms -- that it actually drove the session.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import io
import sys
import threading
import time

from pxviewer.regression.tst_utils import have, skip

if not have("iotbx.data_manager", "numpy"):
    skip("iotbx.data_manager / numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer import LiveSession                     # noqa: E402
from pxviewer.demos import DEMOS, Player             # noqa: E402

FPS = 240.0            # far above the 30 a human watches at, so a short run covers steps
JOIN_S = 3.0
WAIT_S = 15.0          # generous: what is waited for normally arrives inside a second


class Recording_session(LiveSession):
    """A real session that keeps the frames and control messages it sent.

    ``LiveSession`` broadcasts to whoever is connected and keeps no history, so with
    nothing attached a demo's output goes nowhere and there is nothing to assert on.
    Recording the traffic -- rather than substituting a stand-in with the same method
    names -- means the demo drives the real selection, primitive and interaction code,
    which is what it drives in the viewer.
    """

    def __init__(self, *args, **kwargs):
        super(Recording_session, self).__init__(*args, **kwargs)
        self.frames = []
        self.control = []

    @classmethod
    def for_demo(cls, demo):
        sites, _labels = demo.make_sites()
        session = cls.from_sites(sites)
        assert isinstance(session, cls), "from_sites did not honour the subclass"
        return session, np.asarray(sites, dtype="<f4")

    def push(self, coords, changed=None):
        self.frames.append(np.asarray(coords, dtype="<f4"))
        return super(Recording_session, self).push(coords, changed)

    def _send_control(self, message):
        self.control.append(message)
        return super(Recording_session, self)._send_control(message)

    def control_types(self):
        return [m.get("type") for m in self.control]


@contextlib.contextmanager
def playing(demo, session, base):
    """Run ``demo`` on a thread for as long as the body takes, then stop it and join.

    The demos narrate to stdout for the human watching. That is tens of lines per demo
    with nothing to say about whether it worked, and there is no capture under
    ``run_tests``, so it is swallowed here.
    """
    player = Player(session, base, fps=FPS)
    thread = threading.Thread(target=demo.run, args=(player,), daemon=True)
    narration = io.StringIO()
    with contextlib.redirect_stdout(narration):
        thread.start()
        try:
            yield player
        finally:
            player.stop()
            thread.join(timeout=JOIN_S)
    assert not thread.is_alive(), "the demo ignored stop()"


def wait_until(predicate, what, timeout=WAIT_S):
    """Block until ``predicate`` holds, and say what was being waited for if it never does.

    Waiting on the thing itself rather than sleeping a fixed interval: each demo paces
    its own narration -- the clashes one opens with a 1.2 s pause before it drives the
    clusters together -- so any single sleep is either a race or slower than every demo
    needs.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), what


def exercise_every_demo_streams_well_formed_frames_and_stops():
    """Run the whole catalogue: a demo that never stops hangs the viewer's Demo menu, and
    a NaN in a frame is a structure that vanishes with nothing logged."""
    for name in sorted(DEMOS):
        demo = DEMOS[name]
        session, base = Recording_session.for_demo(demo)
        n = len(base)
        assert n >= 2, name

        with playing(demo, session, base):
            # One frame, not a run of them: the interactive demos push the resting
            # conformation once and then narrate for seconds at a time, so a count here
            # would be a statement about pacing rather than about streaming.
            wait_until(lambda: session.frames, "demo %r produced no frames" % name)

        for frame in session.frames[:10]:
            assert frame.shape == (n, 3), name
            assert np.isfinite(frame).all(), name


def exercise_the_pick_demo_reacts_to_a_pick():
    """The picked atom pulses -- the one thing in the demos driven from outside."""
    demo = DEMOS["pick"]
    session, base = Recording_session.for_demo(demo)

    with playing(demo, session, base) as player:
        wait_until(lambda: session.frames, "the pick demo never started streaming")
        player._on_pick({"id": 1, "name": "C", "resname": "UNL",
                         "resseq": 1, "chain": "A"})
        wait_until(
            lambda: any(not np.allclose(f[0], base[0]) for f in session.frames),
            "the picked atom never moved from its rest position")


def exercise_the_select_demo_issues_selections():
    """Highlight and focus go out as separate control messages -- the demo selects by
    positional index, so both resolve Python-side with no client to ask."""
    demo = DEMOS["select"]
    session, base = Recording_session.for_demo(demo)

    with playing(demo, session, base):
        wait_until(lambda: "highlight" in session.control_types(),
                   "the select demo highlighted nothing")
        wait_until(lambda: "focus" in session.control_types(),
                   "the select demo focused nothing")


def exercise_the_primitives_demo_draws_measurements():
    """The primitives land in the session's own registry, so a client connecting late is
    sent them -- that is the state the demo is really building."""
    demo = DEMOS["primitives"]
    session, base = Recording_session.for_demo(demo)

    with playing(demo, session, base):
        wait_until(lambda: session._primitives, "the primitives demo drew nothing")
        kinds = set(m.get("kind") for m in session._primitives.values())

    assert kinds <= {"distance", "angle", "dihedral", "label"}, kinds


def exercise_the_interactions_demo_sets_contacts():
    demo = DEMOS["interactions"]
    session, base = Recording_session.for_demo(demo)

    with playing(demo, session, base):
        wait_until(lambda: session._interactions_contacts, "no contacts were set")


def exercise_the_clashes_demo_sets_clashing_pairs():
    """The demo drives two clusters through each other and recomputes clashes per frame,
    so pairs appear only once the sweep starts -- after its opening pause."""
    demo = DEMOS["clashes"]
    session, base = Recording_session.for_demo(demo)

    with playing(demo, session, base):
        wait_until(lambda: session._clashes, "no clashing pairs were set")


def exercise_the_measure_demo_enables_measure_modes():
    """Measure mode is session state rather than a one-off message: a client connecting
    mid-demo is told which mode is live."""
    demo = DEMOS["measure"]
    session, base = Recording_session.for_demo(demo)

    def modes():
        return [m.get("mode") for m in session.control if m.get("type") == "click-mode"]

    with playing(demo, session, base):
        wait_until(modes, "the measure demo enabled no measure mode")
        wait_until(lambda: session.frames, "the measure demo streamed no frames")

    assert set(modes()) <= {"distance", "angle", "dihedral", "label", "off"}, modes()


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
