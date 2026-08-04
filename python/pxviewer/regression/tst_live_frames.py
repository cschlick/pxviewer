"""The live session's coordinate stream: topology, frames, and deltas.

A client connects, is sent the topology once, and then receives conformations. The part
worth pinning is what happens at the edges of that: a delta is only meaningful to someone
who already holds a conformation to patch, so a client joining mid-drag must be sent a
whole frame, and a delta covering most of the model is not worth sending at all.
"""

from __future__ import absolute_import, division, print_function

import json
import sys

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import have, skip

if not have("websockets", "numpy"):
    skip("websockets / numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer.regression.live_harness import (       # noqa: E402
    TAG_FRAME, TAG_FRAME_DELTA, client, delta_frame, eventually, frame_coords,
    run_client, session)

#: Enough atoms that a small change is worth sending as a delta -- see ``LiveSession.push``,
#: which falls back to a whole frame when the index list would cost more than it saves.
BIG = 20


def grid(n=BIG):
    return np.array([[i, i, i] for i in range(n)], dtype=float)


def exercise_a_client_receives_the_topology_then_a_frame():
    with session() as live:
        async def scenario():
            async with client(live) as ws:      # the topology is consumed on connect
                live.push(np.array([[0, 1, 0], [1, 2, 0], [2, 3, 0], [3, 4, 0]],
                                   dtype=float))
                frame = await ws.recv()
                import struct
                tag, index = struct.unpack_from("<II", frame, 0)
                assert tag == TAG_FRAME
                assert index == 0                # frames are numbered from zero
                coords = frame_coords(frame)
                assert coords.shape == (4, 3)
                assert approx_equal(coords[1].tolist(), [1.0, 2.0, 0.0])

        run_client(scenario)


def exercise_push_sends_only_the_atoms_that_changed():
    """``changed`` sends a delta: just those atoms, at their absolute positions.

    A drag moves a fixed zone whatever the structure's size, so a whole-conformation frame
    spends O(model) to say O(zone) of news.
    """
    with session(BIG) as live:
        async def scenario():
            async with client(live) as ws:
                full = grid()
                live.push(full)
                await ws.recv()                  # the whole frame

                moved = full.copy()
                moved[[1, 3]] += 5.0
                live.push(moved, changed=[1, 3])
                frame = await ws.recv()

                import struct
                assert struct.unpack_from("<I", frame, 0)[0] == TAG_FRAME_DELTA
                indices, coords = delta_frame(frame)
                assert indices.tolist() == [1, 3]
                # Absolute positions rather than offsets, so applying only the newest
                # delta is right even if an earlier one was missed.
                assert approx_equal(coords[0].tolist(), [6.0, 6.0, 6.0])
                assert approx_equal(coords[1].tolist(), [8.0, 8.0, 8.0])
                assert len(frame) < 8 + full.size * 4        # smaller than what it replaces

        run_client(scenario)


def exercise_push_falls_back_to_a_whole_frame_when_most_of_it_changed():
    """A delta covering most of the model is the same bytes plus an index per atom."""
    import struct

    with session(BIG) as live:
        async def scenario():
            async with client(live) as ws:
                live.push(grid(), changed=list(range(BIG)))       # every atom
                frame = await ws.recv()
                assert struct.unpack_from("<I", frame, 0)[0] == TAG_FRAME

        run_client(scenario)


def exercise_a_client_joining_mid_drag_gets_a_whole_frame():
    """A delta is meaningless without a conformation to patch, so replay stays whole."""
    import struct

    with session(BIG) as live:
        coords = grid()
        live.push(coords, changed=[1])           # a delta, broadcast before anyone connects

        async def scenario():
            async with client(live) as ws:
                frame = await ws.recv()
                assert struct.unpack_from("<I", frame, 0)[0] == TAG_FRAME
                assert approx_equal(frame_coords(frame), coords)

        run_client(scenario)


def exercise_a_late_client_gets_the_last_frame():
    with session() as live:
        live.push([[9, 9, 9], [8, 8, 8], [7, 7, 7], [6, 6, 6]])

        async def scenario():
            async with client(live) as ws:
                import struct
                frame = await ws.recv()
                assert struct.unpack_from("<I", frame, 0)[0] == TAG_FRAME
                assert approx_equal(frame_coords(frame)[0].tolist(), [9.0, 9.0, 9.0])

        run_client(scenario)


def exercise_a_frame_of_the_wrong_length_is_rejected():
    """Silently padding or truncating would draw a structure that is not the model."""
    with session() as live:
        with raises(ValueError):
            live.push([[0, 0, 0], [1, 1, 1]])    # 2 atoms; the topology has 4


def exercise_a_pick_reaches_its_handler():
    """The browser reports what was clicked; what it means is decided in Python."""
    with session() as live:
        received = []
        live.on_pick(received.append)

        async def scenario():
            async with client(live) as ws:
                atom = {"id": 2, "name": "C", "resname": "UNL",
                        "resseq": 1, "chain": "A"}
                await ws.send(json.dumps(
                    {"type": "pick", "empty": False, "atom": atom}))
                # The handler runs on the session's own loop, so it is not ordered
                # against anything readable from here.
                assert await eventually(lambda: received)

        run_client(scenario)
        assert received[0]["id"] == 2


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
