"""Tests for the live coordinate-streaming session.

These exercise the wire protocol end to end: a client connects, receives the
topology, receives streamed frames, and sends a pick event back.
"""

import asyncio
import json
import struct

import numpy as np
import pytest

from pxviewer import Atom, LiveSession
from pxviewer.live import _encode_index_set

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


def test_highlight_message_reaches_client(session):
    """highlight() broadcasts an index-set with no round-trip; select returns synchronously."""

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            sel = session.highlight([1, 2])  # fire-and-forget, returns immediately
            assert sel.indices == [1, 2]
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg["type"] == "highlight"
            assert msg["atoms"] == _encode_index_set([1, 2])
            assert _decode(msg["atoms"]) == [1, 2]

    asyncio.run(scenario())


def test_focus_message_reaches_client(session):
    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            session.focus(session.select_by(indices=[3]))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg["type"] == "focus"
            assert _decode(msg["atoms"]) == [3]

    asyncio.run(scenario())


def test_clear_selection_message(session):
    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            session.clear_selection()
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg["type"] == "highlight"
            assert _decode(msg["atoms"]) == []

    asyncio.run(scenario())


def test_set_volume_color_command_reaches_client(session):
    async def scenario():
        import json

        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            session.set_volume_color("vol1", "green")
            message = await asyncio.wait_for(ws.recv(), timeout=5)
            assert isinstance(message, str)
            event = json.loads(message)
            assert event == {"type": "volume_color", "ref": "vol1", "color": "green"}

    asyncio.run(scenario())


def test_set_volume_opacity_command_reaches_client(session):
    async def scenario():
        import json

        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            session.set_volume_opacity("vol2", 0.25)
            message = await asyncio.wait_for(ws.recv(), timeout=5)
            assert isinstance(message, str)
            event = json.loads(message)
            assert event == {"type": "volume_opacity", "ref": "vol2", "opacity": 0.25}

    asyncio.run(scenario())


def test_highlight_replayed_to_late_client(session):
    """A viewer connecting after a highlight is caught up to the active selection."""
    session.highlight([1, 3])

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            replay = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert replay["type"] == "highlight"
            assert _decode(replay["atoms"]) == [1, 3]

    asyncio.run(scenario())


def test_add_angle_message_reaches_client(session):
    """add_angle broadcasts a primitive-add message with the atom-index groups."""

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            prim = session.add_angle(0, 1, 2, label=False)
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert message["type"] == "primitive"
            assert message["action"] == "add"
            assert message["kind"] == "angle"
            assert message["id"] == prim.id
            assert message["groups"] == [[0], [1], [2]]
            assert message["options"] == {"opacity": pytest.approx(0.35), "label": False}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "call,kind,n_groups",
    [
        (lambda s: s.add_distance(0, 1), "distance", 2),
        (lambda s: s.add_angle(0, 1, 2), "angle", 3),
        (lambda s: s.add_dihedral(0, 1, 2, 3), "dihedral", 4),
        (lambda s: s.add_label(0, "hi"), "label", 1),
    ],
)
def test_each_primitive_kind_reaches_client(session, call, kind, n_groups):
    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            call(session)
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert message["kind"] == kind
            assert len(message["groups"]) == n_groups

    asyncio.run(scenario())


def test_remove_and_clear_messages_reach_client(session):
    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            prim = session.add_angle(0, 1, 2)
            await asyncio.wait_for(ws.recv(), timeout=5)  # the add
            session.remove_primitive(prim.id)
            rem = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert rem == {"type": "primitive", "action": "remove", "id": prim.id}
            session.clear_primitives()
            clr = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert clr == {"type": "primitive", "action": "clear"}

    asyncio.run(scenario())


def test_primitives_replayed_to_late_client(session):
    """A viewer connecting after primitives are added receives them all."""
    session.add_angle(0, 1, 2, id="a1")
    session.add_distance(0, 1, id="d1")

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            seen = {}
            for _ in range(2):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                assert msg["type"] == "primitive" and msg["action"] == "add"
                seen[msg["id"]] = msg["kind"]
            assert seen == {"a1": "angle", "d1": "distance"}

    asyncio.run(scenario())


def test_enable_mouse_selection_message_reaches_client(session):
    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            session.enable_mouse_selection()
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg == {"type": "mouse-selection-mode", "enabled": True}
            session.disable_mouse_selection()
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg == {"type": "mouse-selection-mode", "enabled": False}

    asyncio.run(scenario())


def test_mouse_selection_reported_to_python(session):
    """A click-built selection from the viewer updates mouse_selection and fires the callback."""
    got = []
    session.on_selection(lambda sel: got.append(sel.indices))
    session.enable_mouse_selection()

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            await ws.recv()  # mouse-selection-mode replay
            await ws.send(json.dumps({"type": "mouse-selection", "indices": [3, 1]}))
            for _ in range(50):
                if got:
                    break
                await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert got and got[0] == [1, 3]  # sorted
    assert session.mouse_selection.indices == [1, 3]
    assert session.mouse_selection.ids == [2, 4]


def test_wait_for_selection_blocks_until_change(session):
    session.enable_mouse_selection()

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            await ws.recv()  # mode replay
            fut = asyncio.get_event_loop().run_in_executor(None, lambda: session.wait_for_selection(timeout=5))
            await asyncio.sleep(0.1)  # let the worker reach the wait
            await ws.send(json.dumps({"type": "mouse-selection", "indices": [2]}))
            return await asyncio.wait_for(fut, timeout=5)

    sel = asyncio.run(scenario())
    assert sel is not None and sel.indices == [2]


def test_mouse_selection_mode_replayed_to_late_client(session):
    session.enable_mouse_selection()

    async def scenario():
        url = f"ws://{session.host}:{session.port}"
        async with websockets.connect(url) as ws:
            await ws.recv()  # topology
            replay = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert replay == {"type": "mouse-selection-mode", "enabled": True}

    asyncio.run(scenario())


def _decode(atoms):
    """Decode a wire index-set ({list} or {runs}) back to a flat index list."""
    if "runs" in atoms:
        out = []
        for s, e in atoms["runs"]:
            out.extend(range(s, e + 1))
        return out
    return atoms.get("list", [])


def _json(d):
    import json

    return json.dumps(d)
