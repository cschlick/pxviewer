"""Tests for the live coordinate-streaming session.

These exercise the wire protocol end to end: a client connects, receives the
topology, receives streamed frames, and sends a pick event back.
"""

import asyncio
import struct

import numpy as np
import pytest

from pxviewer import Atom, LiveSession

websockets = pytest.importorskip("websockets")

_TAG_TOPOLOGY = 0
_TAG_FRAME = 1


def _atoms(n=4):
    return [
        Atom(id=i + 1, element="C", name="C", resname="UNL", resseq=1, chain="A", x=float(i), y=0.0, z=0.0)
        for i in range(n)
    ]


@pytest.fixture
def session():
    s = LiveSession(_atoms())
    s.start(port=0)
    try:
        yield s
    finally:
        s.stop()


def test_client_receives_topology_and_frame(session):
    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            topo = await ws.recv()
            assert isinstance(topo, (bytes, bytearray))
            assert struct.unpack_from("<I", topo, 0)[0] == _TAG_TOPOLOGY
            assert len(topo) > 4  # msgpack-encoded BinaryCIF payload follows the tag

            session.push(np.array([[0, 1, 0], [1, 2, 0], [2, 3, 0], [3, 4, 0]], dtype=float))
            frame = await asyncio.wait_for(ws.recv(), timeout=5)
            tag, index = struct.unpack_from("<II", frame, 0)
            assert tag == _TAG_FRAME
            assert index == 0
            coords = np.frombuffer(frame[8:], dtype="<f4").reshape(-1, 3)
            assert coords.shape == (4, 3)
            assert coords[1].tolist() == pytest.approx([1.0, 2.0, 0.0])

    asyncio.run(scenario())


def test_pick_event_reaches_handler(session):
    received = []
    done = asyncio.Event()

    def on_pick(info):
        received.append(info)

    session.on_pick(on_pick)
    # The handler runs on the server loop; signal completion from there too.
    session.on_pick(lambda _info: session._loop.call_soon_threadsafe(done.set))

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            atom = {"id": 2, "name": "C", "resname": "UNL", "resseq": 1, "chain": "A"}
            await ws.send('{"type": "pick", "empty": false, "atom": %s}' % _json(atom))
            for _ in range(50):
                if received:
                    break
                await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert received and received[0]["id"] == 2


def test_late_client_gets_last_frame(session):
    session.push([[9, 9, 9], [8, 8, 8], [7, 7, 7], [6, 6, 6]])

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            frame = await asyncio.wait_for(ws.recv(), timeout=5)
            tag, _ = struct.unpack_from("<II", frame, 0)
            assert tag == _TAG_FRAME
            coords = np.frombuffer(frame[8:], dtype="<f4").reshape(-1, 3)
            assert coords[0].tolist() == pytest.approx([9.0, 9.0, 9.0])

    asyncio.run(scenario())


def test_frame_length_mismatch_rejected(session):
    with pytest.raises(ValueError):
        session.push([[0, 0, 0], [1, 1, 1]])  # only 2 atoms, topology has 4


def test_set_axis_command_reaches_client(session):
    async def scenario():
        import json

        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            session.set_axis(False)
            message = await asyncio.wait_for(ws.recv(), timeout=5)
            assert isinstance(message, str)
            event = json.loads(message)
            assert event == {"type": "axis", "visible": False}

    asyncio.run(scenario())


def _json(d):
    import json

    return json.dumps(d)
