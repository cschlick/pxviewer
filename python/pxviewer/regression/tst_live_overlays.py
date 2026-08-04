"""Overlays drawn over the structure: interactions, clashes and probe dots.

All three share a shape worth pinning once. Each is validated in Python before it reaches
the wire -- an index out of range or an unknown interaction kind is a clear error rather
than an overlay that silently draws nothing -- and each is **replayed to a client that
connects later**, because a viewport reload must not quietly lose what was drawn.
"""

from __future__ import absolute_import, division, print_function

import struct
import sys

from libtbx.test_utils import raises

from pxviewer.regression.tst_utils import have, skip

if not have("websockets", "numpy"):
    skip("websockets / numpy not available")

from pxviewer.regression.live_harness import (       # noqa: E402
    TAG_DOTS, client, next_text, run_client, session)


# -- the structure itself -----------------------------------------------------


def exercise_hiding_the_structure_reaches_the_client():
    """A render skip, so it is a command rather than a scene change."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.set_structure_visible(False)
                # A coordinate frame may arrive first, so take the next *control* message.
                assert await next_text(ws, "structure_visible") == {
                    "type": "structure_visible", "value": False}

        run_client(scenario)


# -- interactions -------------------------------------------------------------


def exercise_interactions_from_a_mapping_reach_the_client():
    """Aliases are normalised to canonical Mol* kinds, and indices are left alone."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.set_interactions({"h-bond": [(0, 1)], "salt-bridge": [(2, 3)]})
                event = await next_text(ws, "interactions")
                assert event["action"] == "set"
                assert {"kind": "hydrogen-bond", "a": 0, "b": 1} in event["contacts"]
                assert {"kind": "ionic", "a": 2, "b": 3} in event["contacts"]

        run_client(scenario)


def exercise_interactions_accept_tuple_and_dict_forms():
    """Both spellings normalise to the same contact, so callers need not care."""
    with session() as live:
        from_tuples = live.set_interactions([("hydrogen-bond", 0, 1, "backbone")])
        assert from_tuples == [{"kind": "hydrogen-bond", "a": 0, "b": 1,
                                "description": "backbone"}]
        from_dicts = live.set_interactions([{"kind": "hydrophobic", "a": 1, "b": 2}])
        assert from_dicts == [{"kind": "hydrophobic", "a": 1, "b": 2}]


def exercise_interactions_reject_a_bad_index_or_kind():
    """Either would reach the viewer and draw nothing, with nothing said about why."""
    with session() as live:
        with raises(ValueError) as e:
            live.set_interactions({"hydrogen-bond": [(0, 999)]})
        assert "out of range" in str(e.value)

        with raises(ValueError) as e:
            live.set_interactions([("not-a-bond", 0, 1)])
        assert "unknown interaction kind" in str(e.value)


def exercise_clearing_interactions_reaches_the_client():
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.set_interactions({"hydrogen-bond": [(0, 1)]})
                await next_text(ws, "interactions", action="set")
                live.clear_interactions()
                assert await next_text(ws, "interactions") == {
                    "type": "interactions", "action": "clear"}

        run_client(scenario)


def exercise_interactions_are_replayed_to_a_late_client():
    with session() as live:
        live.set_interactions({"hydrogen-bond": [(0, 1)]})      # before anyone connects

        async def scenario():
            async with client(live) as ws:
                event = await next_text(ws, "interactions")
                assert event["action"] == "set"
                assert event["contacts"] == [{"kind": "hydrogen-bond", "a": 0, "b": 1}]

        run_client(scenario)


def exercise_computed_interactions_are_a_toggle():
    """Mol* computes these itself, so the session only says whether to show them."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.set_computed_interactions(True)
                assert await next_text(ws, "computed-interactions") == {
                    "type": "computed-interactions", "visible": True}
                live.hide_computed_interactions()
                assert await next_text(ws, "computed-interactions") == {
                    "type": "computed-interactions", "visible": False}

        run_client(scenario)


def exercise_computed_interactions_are_replayed_to_a_late_client():
    with session() as live:
        live.show_computed_interactions()

        async def scenario():
            async with client(live) as ws:
                assert await next_text(ws, "computed-interactions") == {
                    "type": "computed-interactions", "visible": True}

        run_client(scenario)


# -- clashes ------------------------------------------------------------------


def exercise_clashes_are_validated_and_deduped():
    """A clash is unordered, so (0, 2) and (2, 0) are one pair -- drawing both would
    double the markers on every clash in the structure."""
    with session() as live:
        assert live.set_clashes([(0, 2), (2, 0), {"a": 1, "b": 3}]) == [(0, 2), (1, 3)]

        with raises(ValueError) as e:
            live.set_clashes([(0, 99)])
        assert "out of range" in str(e.value)

        with raises(ValueError) as e:
            live.set_clashes([(1, 1)])
        assert "two distinct atoms" in str(e.value)


def exercise_clashes_reach_the_client_and_can_be_cleared():
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.set_clashes([(0, 2)])
                assert await next_text(ws, "clashes") == {
                    "type": "clashes", "action": "set", "pairs": [{"a": 0, "b": 2}]}
                live.clear_clashes()
                assert await next_text(ws, "clashes") == {
                    "type": "clashes", "action": "clear"}

        run_client(scenario)


def exercise_clashes_are_replayed_to_a_late_client():
    with session() as live:
        live.set_clashes([(0, 2)])

        async def scenario():
            async with client(live) as ws:
                assert await next_text(ws, "clashes") == {
                    "type": "clashes", "action": "set", "pairs": [{"a": 0, "b": 2}]}

        run_client(scenario)


# -- probe dots ---------------------------------------------------------------


def exercise_probe_dot_channels_are_independent():
    """Contacts and clashes are two overlays keyed by channel, so showing or clearing
    one must not disturb the other -- the Clashes tab toggles them separately."""
    from pxviewer.live import PROBE_CLASHES, PROBE_CONTACTS, _TAG_DOTS

    assert _TAG_DOTS == TAG_DOTS          # the harness mirrors the tag; keep them agreed

    with session() as live:
        live.show_probe_dots([((0, 0, 0), (0, 0, 0), (0, 255, 0))],
                             channel=PROBE_CONTACTS)
        live.show_probe_dots([((1, 0, 0), (1.2, 0, 0), (255, 0, 0))],
                             channel=PROBE_CLASHES)
        assert set(live._probe_dots_payloads) == {PROBE_CONTACTS, PROBE_CLASHES}

        # The payload leads with [tag][channel], so the client can route it.
        tag, channel = struct.unpack(
            "<II", live._probe_dots_payloads[PROBE_CLASHES][:8])
        assert tag == TAG_DOTS
        assert channel == PROBE_CLASHES

        live.clear_probe_dots(channel=PROBE_CLASHES)
        assert set(live._probe_dots_payloads) == {PROBE_CONTACTS}
        live.clear_probe_dots()                                  # every channel
        assert live._probe_dots_payloads == {}


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
