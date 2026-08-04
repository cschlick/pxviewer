"""Connecting to a ``LiveSession`` and reading its wire, for the ``tst_live_*`` scripts.

Every exercise in those files has the same shape: start a session, connect a websocket,
consume the topology frame that always arrives first, then provoke something and read what
comes back. Written out each time that is six lines of asyncio scaffolding around one
assertion, sixty times over, which buries the assertion.

Two details this encodes, both of which were open-coded inconsistently before:

- **Text and binary messages interleave.** A coordinate frame can arrive between a command
  and its echo, so waiting for "the next message" is a race. :func:`next_text` and
  :func:`next_binary` skip what they are not looking for.
- **The topology always comes first.** :func:`client` consumes it, so an exercise that is
  not about topology never mentions it.
"""

from __future__ import absolute_import, division, print_function

import asyncio
import contextlib
import json
import struct

#: Wire tags, mirrored from ``pxviewer.live``. Repeated rather than imported so an
#: accidental renumbering there is caught here rather than silently agreed with.
TAG_TOPOLOGY = 0
TAG_FRAME = 1
TAG_DOTS = 3
TAG_MAP = 4
TAG_FRAME_DELTA = 5

TIMEOUT_S = 5.0

#: How long to keep looking for the message an exercise wants. Generous: the session
#: replays its state on connect, so several may precede the one being waited for.
MAX_MESSAGES = 40


def sites(n=4):
    """``n`` atoms in a line, which is all most of these exercises need of a structure."""
    return [[float(i), 0.0, 0.0] for i in range(n)]


@contextlib.contextmanager
def session(n=4):
    """A started ``LiveSession`` over :func:`sites`, stopped afterwards.

    A **fresh** one per exercise: a session accumulates the state it replays to late
    clients -- interactions, clashes, primitives, click mode -- so a shared one would let
    an earlier exercise's leftovers arrive in a later one's stream.
    """
    from pxviewer import LiveSession

    live = LiveSession.from_sites(sites(n))
    live.start(port=0)
    try:
        yield live
    finally:
        live.stop()


def url_for(live):
    return "ws://%s:%d" % (live.host, live.port)


@contextlib.asynccontextmanager
async def client(live, expect_topology=True):
    """A connected websocket, with the topology frame already consumed."""
    import websockets

    async with websockets.connect(url_for(live)) as ws:
        if expect_topology:
            topology = await asyncio.wait_for(ws.recv(), TIMEOUT_S)
            assert isinstance(topology, (bytes, bytearray))
            assert struct.unpack_from("<I", topology, 0)[0] == TAG_TOPOLOGY
        yield ws


async def next_text(ws, type=None, timeout=TIMEOUT_S, **fields):
    """The next JSON control message, optionally the next one of a given ``type``.

    Extra keyword arguments narrow it further -- ``next_text(ws, "click-mode",
    mode="off")`` skips a duplicate that a coinciding connect handshake can produce.
    Binary frames in between are skipped.
    """
    for _ in range(MAX_MESSAGES):
        message = await asyncio.wait_for(ws.recv(), timeout)
        if not isinstance(message, str):
            continue
        event = json.loads(message)
        if type is not None and event.get("type") != type:
            continue
        if any(event.get(k) != v for k, v in fields.items()):
            continue
        return event
    raise AssertionError(
        "no %s message arrived" % (type or "control"))


async def next_binary(ws, tag=None, timeout=TIMEOUT_S):
    """The next binary payload, optionally the next one carrying a given tag."""
    for _ in range(MAX_MESSAGES):
        message = await asyncio.wait_for(ws.recv(), timeout)
        if isinstance(message, str):
            continue
        if tag is not None and struct.unpack_from("<I", message, 0)[0] != tag:
            continue
        return message
    raise AssertionError("no binary message with tag %r arrived" % tag)


async def eventually(predicate, timeout=TIMEOUT_S):
    """Wait for something the server does on its own loop, off the wire.

    A handler runs on the session's loop rather than this one, so its effect is not
    ordered against anything received here.
    """
    deadline = 0.0
    while deadline < timeout:
        if predicate():
            return True
        await asyncio.sleep(0.05)
        deadline += 0.05
    return predicate()


def frame_coords(payload):
    """The (N, 3) conformation in a whole-frame payload."""
    import numpy as np

    return np.frombuffer(payload[8:], dtype="<f4").reshape(-1, 3)


def delta_frame(payload):
    """``(indices, coords)`` from a delta payload -- absolute positions, not offsets."""
    import numpy as np

    _tag, _index, n = struct.unpack_from("<III", payload, 0)
    indices = np.frombuffer(payload[12:12 + 4 * n], dtype="<u4")
    coords = np.frombuffer(payload[12 + 4 * n:], dtype="<f4").reshape(-1, 3)
    return indices, coords


def decode_index_set(atoms):
    """A wire index set -- ``{"runs": ...}`` or ``{"list": ...}`` -- as a flat list."""
    if "runs" in atoms:
        out = []
        for start, end in atoms["runs"]:
            out.extend(range(start, end + 1))
        return out
    return atoms.get("list", [])


def run_client(coro_fn):
    """Run one client scenario to completion.

    Not called ``run``: every ``tst_*.py`` defines its own ``run()`` as the entry point,
    and shadowing it here would be a quiet way to break the convention.
    """
    return asyncio.run(coro_fn())
