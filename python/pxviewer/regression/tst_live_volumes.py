"""Volume appearance, clipping and screenshots over the wire.

A volume's colour, style and level are baked into the MVSJ scene, so they survive a
reload on their own and only need a live command to avoid one. Three things are *not* in
the scene and so have to be replayed on connect: the clip, which is recomputed from the
camera as it moves; the scroll target, which says which volume the wheel contours; and
hidden state, which is a render skip. All three went dead after any scene change before
they were replayed -- for hidden state that meant every viewport reload silently redrew
every hidden map, which is how a "hidden" local-resolution map ended up on screen as a
giant featureless blob.
"""

from __future__ import absolute_import, division, print_function

import asyncio
import base64
import json
import sys
import threading

from pxviewer.regression.tst_utils import have, skip

if not have("websockets", "numpy"):
    skip("websockets / numpy not available")

from pxviewer.regression.live_harness import (       # noqa: E402
    client, eventually, next_text, run_client, session, url_for)


# -- appearance commands ------------------------------------------------------


def exercise_volume_appearance_commands_reach_the_client():
    """Each is addressed by ref, since one session carries every volume's commands."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.set_volume_color("vol1", "green")
                assert await next_text(ws, "volume_color") == {
                    "type": "volume_color", "ref": "vol1", "color": "green"}

                live.set_volume_opacity("vol2", 0.25)
                assert await next_text(ws, "volume_opacity") == {
                    "type": "volume_opacity", "ref": "vol2", "opacity": 0.25}

                live.set_volume_style("vol3", "mesh")
                assert await next_text(ws, "volume_style") == {
                    "type": "volume_style", "ref": "vol3", "style": "mesh"}

                live.set_volume_iso("vol5", 2.5)
                assert await next_text(ws, "volume_iso") == {
                    "type": "volume_iso", "ref": "vol5", "value": 2.5}

                live.set_volume_position("vol4", (1.0, 2.0, 3.0))
                assert await next_text(ws, "volume_position") == {
                    "type": "volume_position", "ref": "vol4",
                    "position": [1.0, 2.0, 3.0]}

        run_client(scenario)


def exercise_the_scroll_target_is_replayed_to_a_late_client():
    """A volume's style, colour and level survive a reload because the scene carries
    them. The scroll target is not part of the scene, so it must be replayed on connect
    -- otherwise wheel contouring goes dead after any scene change."""
    with session() as live:
        live.set_volume_scroll_target("vol6")

        async def scenario():
            async with client(live) as ws:
                assert await next_text(ws, "volume_scroll_target") == {
                    "type": "volume_scroll_target", "ref": "vol6"}

        run_client(scenario)


def exercise_a_contour_changed_in_the_viewport_reaches_a_handler():
    """Wheel contouring is applied in the viewer and echoed back, which is how the
    controls hear about a level they did not set."""
    with session() as live:
        seen = []
        live.on_volume_iso(lambda ref, value: seen.append((ref, value)))

        async def scenario():
            async with client(live) as ws:
                await ws.send(json.dumps(
                    {"type": "volume-iso-changed", "ref": "vol7", "value": 3.25}))
                assert await eventually(lambda: seen)

        run_client(scenario)
        assert seen == [("vol7", 3.25)]


# -- clipping -----------------------------------------------------------------


def exercise_a_clip_addresses_one_representation():
    """With a ref it clips that volume; without one, the session's own model. That is
    what lets density be cut open while the model inside it stays whole."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.set_clip(0.25, 0.75, ref="vol8")
                assert await next_text(ws, "clip") == {
                    "type": "clip", "ref": "vol8", "front": 0.25, "back": 0.75,
                    "radius": None}

                live.set_clip(0.0, 1.0)                       # no ref: this model
                assert await next_text(ws, "clip") == {
                    "type": "clip", "ref": None, "front": 0.0, "back": 1.0,
                    "radius": None}

                # A radius rides the same message: to the viewer the slab and the radius
                # are one clip, so either changing re-sends both.
                live.set_clip(0.0, 1.0, radius=12.0, ref="vol8")
                assert await next_text(ws, "clip") == {
                    "type": "clip", "ref": "vol8", "front": 0.0, "back": 1.0,
                    "radius": 12.0}

        run_client(scenario)


def exercise_a_clip_is_replayed_to_a_late_client():
    """A clip is worked out from the camera and re-aimed as it moves, so unlike a colour
    or a level it cannot be baked into the scene. The session has to replay it, or every
    viewport reload silently drops it."""
    with session() as live:
        live.set_clip(0.2, 0.8, radius=12.0, ref="vol9")       # before anyone connects

        async def scenario():
            async with client(live) as ws:
                assert await next_text(ws, "clip") == {
                    "type": "clip", "ref": "vol9", "front": 0.2, "back": 0.8,
                    "radius": 12.0}

        run_client(scenario)


def exercise_a_hidden_volume_stays_hidden_for_a_late_client():
    """Hiding is a broadcast render skip, and a viewport reload connects a new client:
    without replay, the reload redraws every hidden map."""
    with session() as live:
        live.set_volume_visible("vol20", False)                # before anyone connects

        async def scenario():
            async with client(live) as ws:
                assert await next_text(ws, "volume_visible") == {
                    "type": "volume_visible", "ref": "vol20", "value": False}

        run_client(scenario)


def exercise_a_reshown_volume_is_not_replayed():
    """Visible is the scene's default; replaying it would be a no-op message per map."""
    with session() as live:
        live.set_volume_visible("vol21", False)
        live.set_volume_visible("vol21", True)                 # shown again
        live.set_volume_visible("vol22", False)                # the one that should replay

        async def scenario():
            async with client(live) as ws:
                message = await next_text(ws, "volume_visible")
                assert message["ref"] == "vol22", "replayed a volume that is shown: %r" % message

        run_client(scenario)


def exercise_an_open_clip_is_not_replayed():
    """Restoring a clip that clips nothing is shader cost for no effect."""
    with session() as live:
        live.set_clip(0.2, 0.8, ref="vol10")
        live.set_clip(0.0, 1.0, ref="vol10")                   # opened again

        async def scenario():
            async with client(live) as ws:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    return                                     # nothing replayed: correct
                raise AssertionError("an open clip was replayed: %r" % (message,))

        run_client(scenario)


# -- screenshots --------------------------------------------------------------


def request_screenshot(live, into, timeout):
    """Call ``screenshot`` from another thread, since it blocks for the answer."""
    def ask():
        into["png"] = live.screenshot(timeout=timeout)

    caller = threading.Thread(target=ask)
    caller.start()
    return caller


def exercise_a_screenshot_round_trips():
    """The scene only exists in the browser, so the picture is taken there and comes
    back over the wire -- which is what makes this work for a remote viewer too."""
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n fake").decode()
    result = {}

    with session() as live:
        async def scenario():
            async with client(live) as ws:
                caller = request_screenshot(live, result, timeout=5)
                request = await next_text(ws, "screenshot")
                assert "reqId" in request              # answers are matched by request id
                await ws.send(json.dumps({
                    "type": "screenshot-result", "reqId": request["reqId"],
                    "dataUri": "data:image/png;base64,%s" % png}))
                caller.join(timeout=5)

        run_client(scenario)

    assert result["png"] == b"\x89PNG\r\n\x1a\n fake"          # decoded from the data URI


def exercise_a_screenshot_survives_a_large_image():
    """A real screenshot is megabytes: a 1640x1280 PNG is around 700 kB, and base64 in
    JSON inflates it by a third.

    The websockets default caps a message at 1 MiB and answers an oversized one by
    closing the connection with 1009 -- which killed the whole live session, not just the
    picture.
    """
    import websockets

    big = base64.b64encode(b"\x89PNG" + b"\x00" * (3 * 1024 * 1024)).decode()
    result = {}

    with session() as live:
        async def scenario():
            # max_size=None on the *client* too: this end has the same default cap.
            async with websockets.connect(url_for(live), max_size=None) as ws:
                await ws.recv()                                # topology
                caller = request_screenshot(live, result, timeout=15)
                request = await next_text(ws, "screenshot")
                await ws.send(json.dumps({
                    "type": "screenshot-result", "reqId": request["reqId"],
                    "dataUri": "data:image/png;base64,%s" % big}))
                caller.join(timeout=15)
                assert ws.state.name == "OPEN"                 # still usable afterwards

        run_client(scenario)

    assert result["png"] is not None
    assert len(result["png"]) > 3 * 1024 * 1024


def exercise_a_screenshot_returns_none_when_nobody_answers():
    """No viewer connected means no picture, rather than hanging forever."""
    with session() as live:
        assert live.screenshot(timeout=0.3) is None


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
