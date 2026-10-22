"""Selecting, measuring, and the click modes that decide what a click means.

A click can select an atom, complete a measurement, or start a drag that moves the model,
and only one of those at a time. The mode lives on the session rather than in the browser,
so a viewport reload cannot leave the two disagreeing about what the next click will do --
which is why every mode here is also checked for being replayed on connect.
"""

from __future__ import absolute_import, division, print_function

import asyncio
import json
import sys

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import _exercise_selected, have, skip

if not have("websockets", "numpy"):
    skip("websockets / numpy not available")

from pxviewer.live import _encode_index_set          # noqa: E402

from pxviewer.regression.live_harness import (       # noqa: E402
    client, decode_index_set, eventually, next_text, run_client, session)


# -- highlight and focus ------------------------------------------------------


def exercise_highlight_broadcasts_an_index_set():
    """Fire-and-forget: unlike a browser-evaluated selection there is no round trip, so
    it returns the Selection immediately."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                selection = live.highlight([1, 2])
                assert selection.indices == [1, 2]

                message = await next_text(ws, "highlight")
                assert message["atoms"] == _encode_index_set([1, 2])
                assert decode_index_set(message["atoms"]) == [1, 2]

        run_client(scenario)


def exercise_focus_broadcasts_an_index_set():
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.focus(live.select_by(indices=[3]))
                message = await next_text(ws, "focus")
                assert decode_index_set(message["atoms"]) == [3]
                assert "distance" not in message and "frame" not in message

        run_client(scenario)


def exercise_focus_distance_and_frame_ride_the_same_message():
    """focus(sel, distance=…) / focus(sel, frame=True) dolly the eye as well as
    aiming — the client needs the opt-in on the wire, and bare focus must not
    change behavior."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                sel = live.select_by(indices=[3])
                live.focus(sel, distance=30)
                message = await next_text(ws, "focus")
                assert decode_index_set(message["atoms"]) == [3]
                assert message["distance"] == 30.0
                assert "frame" not in message

                live.focus(sel, frame=True)
                message = await next_text(ws, "focus")
                assert message["frame"] is True
                assert "distance" not in message

        run_client(scenario)


def exercise_a_reported_camera_is_replayed_to_a_reconnect():
    """The page volunteers its settled camera (debounced client-side); a viewport
    reload is a new client, so the stored state goes back out as a camera-set."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live._on_message(json.dumps({"type": "camera-report", "state": {
                    "target": [1, 2, 3], "position": [0, 0, 40],
                    "up": [0, 1, 0], "radius": 15, "radiusMax": 60, "fov": 45}}))
                live._on_message('{"type": "camera-report", "state": "junk"}')
                # A ~zero slab is the never-fitted camera, not a view — storing it
                # replays a blank frame onto the next client (the volume-only load).
                live._on_message(json.dumps({"type": "camera-report", "state": {
                    "target": [0, 0, 0], "position": [0, 0, 10],
                    "up": [0, 1, 0], "radius": 0.01, "radiusMax": 60, "fov": 45}}))
                assert live._last_camera_state["radius"] == 15  # the good view stands
            async with client(live) as ws:
                message = await next_text(ws, "camera-set")
                assert message["state"]["position"] == [0, 0, 40]
                assert message["state"]["radius"] == 15

        run_client(scenario)


def exercise_commands_during_a_reload_queue_for_the_new_client():
    """``begin_reload`` says the page this socket feeds is navigating away: a
    command sent while it still nominally holds a client must not reach the
    wire (the dying page would drop it) — it queues for the next client."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.begin_reload()
                live.clear_selection()   # issued mid-reload, socket still open
                try:
                    await asyncio.wait_for(ws.recv(), 0.5)
                    raise AssertionError("a command reached the dying page")
                except asyncio.TimeoutError:
                    pass
            # The old client is gone; the new one replays stored state, then
            # the queued command lands on top of it.
            async with client(live) as ws2:
                event = await next_text(ws2, "highlight")
                assert decode_index_set(event["atoms"]) == []

        run_client(scenario)


def exercise_a_request_sent_mid_reload_is_answered_by_the_new_page():
    """A request/response message (screenshot, camera-state) issued during the
    reload window queues too — its waiter is usually still blocked on the
    timeout, so the reloaded page answering is better than a guaranteed miss."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.begin_reload()
                live._send_control({"type": "camera-state", "reqId": 77})
            async with client(live) as ws2:
                event = await next_text(ws2, "camera-state")
                assert event["reqId"] == 77

        run_client(scenario)


def exercise_selection_prose_never_reaches_the_wire():
    """The whole selection flow resolves prose in Python via cctbx; every control
    message the session can emit must carry index sets, never a query string."""
    with session() as live:
        sent = []
        original = live._send_control
        live._send_control = lambda m: (sent.append(m), original(m))[1]
        sel = live.select_by(selection="resseq 1")
        live.select(sel)
        live.clear_selection()
        live.focus(sel.indices)
        for message in sent:
            assert "expression" not in message, message
            assert message["type"] != "select", message
            if "atoms" in message:
                decode_index_set(message["atoms"])  # must be an index set


def exercise_viewing_specs_reject_prose():
    """select/highlight/focus take resolved identity only — a bare string has no
    grammar, so it must raise rather than guess one."""
    with session() as live:
        for method in (live.select, live.highlight, live.focus):
            with raises(TypeError):
                method("resseq 1")


def exercise_non_finite_camera_payloads_are_rejected():
    """NaN/Infinity cannot cross a JSON wire: they either serialize as invalid JSON
    the page drops silently, or poison the camera outright. The wire boundary
    refuses them instead."""
    with session() as live:
        nan = float("nan")
        with raises(ValueError):
            live.orient_camera([nan, 0, 0], [0, 1, 0], [0, 0, 1], 10.0)
        with raises(ValueError):
            live.orient_camera([0, 0, 0], [0, 1, 0], [0, 0, 1], float("inf"))
        with raises(ValueError):
            live.orient_camera([0, 0, 0], [0, 1, 0], [0, 0, 1], 10.0,
                               extents=[1.0, nan, 1.0])
        with raises(ValueError):
            live.set_camera({"target": [0, 0, 0], "position": [0, 0, nan],
                             "up": [0, 1, 0], "radius": 10.0, "fov": 45.0})
        with raises(ValueError):
            live.set_camera({"radiusMax": float("inf")})


def exercise_clear_selection_sends_an_empty_highlight():
    """Clearing also drops the replayed state, so a late client stays clear."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.clear_selection()
                message = await next_text(ws, "highlight")
                assert decode_index_set(message["atoms"]) == []

        run_client(scenario)


def exercise_a_highlight_is_replayed_to_a_late_client():
    """A viewer connecting after a highlight is caught up to the active selection."""
    with session() as live:
        live.highlight([1, 3])

        async def scenario():
            async with client(live) as ws:
                replay = await next_text(ws, "highlight")
                assert decode_index_set(replay["atoms"]) == [1, 3]

        run_client(scenario)


# -- primitives ---------------------------------------------------------------


def exercise_adding_an_angle_reaches_the_client():
    """The message carries the atom-index groups, so the viewer draws the notation
    against atoms rather than against fixed points that would not follow a drag."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                primitive = live.add_angle(0, 1, 2, label=False)
                message = await next_text(ws, "primitive")

                assert message["action"] == "add"
                assert message["kind"] == "angle"
                assert message["id"] == primitive.id
                assert message["groups"] == [[0], [1], [2]]
                assert message["options"]["label"] is False
                assert approx_equal(message["options"]["opacity"], 0.35)

        run_client(scenario)


def exercise_every_primitive_kind_reaches_the_client():
    """One group per atom the measurement needs: two for a distance, four for a
    dihedral. A kind arriving with the wrong arity draws nothing."""
    cases = [
        ("distance", 2, lambda live: live.add_distance(0, 1)),
        ("angle", 3, lambda live: live.add_angle(0, 1, 2)),
        ("dihedral", 4, lambda live: live.add_dihedral(0, 1, 2, 3)),
        ("label", 1, lambda live: live.add_label(0, "hi")),
    ]
    for kind, n_groups, add in cases:
        with session() as live:
            async def scenario(add=add, kind=kind, n_groups=n_groups):
                async with client(live) as ws:
                    add(live)
                    message = await next_text(ws, "primitive")
                    assert message["kind"] == kind
                    assert len(message["groups"]) == n_groups

            run_client(scenario)


def exercise_removing_and_clearing_primitives_reach_the_client():
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                primitive = live.add_angle(0, 1, 2)
                await next_text(ws, "primitive", action="add")

                live.remove_primitive(primitive.id)
                removed = await next_text(ws, "primitive", action="remove")
                assert removed["id"] == primitive.id

                live.clear_primitives()
                assert await next_text(ws, "primitive", action="clear") == {
                    "type": "primitive", "action": "clear"}

        run_client(scenario)


def exercise_primitives_are_replayed_to_a_late_client():
    with session() as live:
        live.add_angle(0, 1, 2, id="a1")
        live.add_distance(0, 1, id="d1")

        async def scenario():
            async with client(live) as ws:
                seen = {}
                for _ in range(2):
                    message = await next_text(ws, "primitive", action="add")
                    seen[message["id"]] = message["kind"]
                assert seen == {"a1": "angle", "d1": "distance"}

        run_client(scenario)


# -- click modes --------------------------------------------------------------


def exercise_mouse_selection_mode_reaches_the_client():
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.enable_mouse_selection()
                await next_text(ws, "click-mode", mode="select")
                live.disable_mouse_selection()
                await next_text(ws, "click-mode", mode="off")

        run_client(scenario)


def exercise_the_click_mode_is_replayed_to_a_late_client():
    with session() as live:
        live.enable_mouse_selection()

        async def scenario():
            async with client(live) as ws:
                assert await next_text(ws, "click-mode") == {
                    "type": "click-mode", "mode": "select"}

        run_client(scenario)


def exercise_refine_drag_mode_reaches_and_is_replayed():
    with session() as live:
        async def change():
            async with client(live) as ws:
                live.set_tug_mode(True)
                assert await next_text(ws, "tug-mode") == {
                    "type": "tug-mode", "enabled": True}

        run_client(change)

        async def late():
            async with client(live) as ws:
                assert await next_text(ws, "tug-mode") == {
                    "type": "tug-mode", "enabled": True}

        run_client(late)


def exercise_focus_surroundings_reaches_and_is_replayed():
    with session() as live:
        async def change():
            async with client(live) as ws:
                live.set_focus_surroundings(True)
                assert await next_text(ws, "focus-surroundings") == {
                    "type": "focus-surroundings", "enabled": True}

        run_client(change)

        async def late():
            async with client(live) as ws:
                assert await next_text(ws, "focus-surroundings") == {
                    "type": "focus-surroundings", "enabled": True}

        run_client(late)


# -- selections made in the viewer --------------------------------------------


def exercise_a_click_built_selection_is_reported_to_python():
    with session() as live:
        got = []
        live.on_selection(lambda sel: got.append(sel.indices))
        live.enable_mouse_selection()

        async def scenario():
            async with client(live) as ws:
                await ws.send(json.dumps(
                    {"type": "mouse-selection", "indices": [3, 1]}))
                assert await eventually(lambda: got)

        run_client(scenario)

        assert got[0] == [1, 3]                       # sorted on the way in
        assert live.mouse_selection.indices == [1, 3]
        assert live.mouse_selection.ids == [2, 4]     # id == index + 1


def exercise_wait_for_selection_blocks_until_one_arrives():
    """The script-driving case: ask the user to pick something, and wait for it."""
    with session() as live:
        live.enable_mouse_selection()

        async def scenario():
            async with client(live) as ws:
                await next_text(ws, "click-mode", mode="select")     # the replay
                loop = asyncio.get_event_loop()
                waiting = loop.run_in_executor(
                    None, lambda: live.wait_for_selection(timeout=5))
                await asyncio.sleep(0.1)              # let the worker reach the wait
                await ws.send(json.dumps({"type": "mouse-selection", "indices": [2]}))
                return await asyncio.wait_for(waiting, timeout=5)

        selection = run_client(scenario)

    assert selection is not None
    assert selection.indices == [2]


# -- measure mode -------------------------------------------------------------


def exercise_measure_mode_reaches_the_client():
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.enable_measure_mode("angle")
                await next_text(ws, "click-mode", mode="angle")

        run_client(scenario)


def exercise_measure_mode_rejects_an_unknown_kind():
    with session() as live:
        with raises(ValueError):
            live.enable_measure_mode("banana")


def exercise_a_click_built_measurement_is_drawn_and_reported():
    """A completed set of clicks becomes a primitive on the server as well as a drawing,
    so it replays to a late client and can be removed like any other."""
    with session() as live:
        drawn = []
        live.enable_measure_mode("angle", on_measure=lambda p: drawn.append(p.kind))

        async def scenario():
            async with client(live) as ws:
                await next_text(ws, "click-mode", mode="angle")      # the replay
                await ws.send(json.dumps(
                    {"type": "measure", "kind": "angle", "atoms": [0, 1, 2]}))

                message = await next_text(ws, "primitive")
                assert message["kind"] == "angle"
                assert message["groups"] == [[0], [1], [2]]
                assert await eventually(lambda: drawn)

        run_client(scenario)

        assert drawn == ["angle"]
        assert len(live._primitives) == 1             # recorded, so it replays and removes


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise") and _exercise_selected(name):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
