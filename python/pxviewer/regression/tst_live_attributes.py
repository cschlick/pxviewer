"""Representations, and colouring atoms by a per-atom attribute.

Colouring by a value is the one path where the payload is large enough for its encoding to
matter: N floats per atom, resent whenever the values change. They go as a compact binary
message keyed by name, and the representation JSON that follows *references* that key
rather than inlining the numbers -- so a re-colour costs the values once, not once per
representation that reads them.
"""

from __future__ import absolute_import, division, print_function

import struct
import sys

from libtbx.test_utils import raises

from pxviewer.regression.tst_utils import have, skip

if not have("websockets", "numpy"):
    skip("websockets / numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer.regression.live_harness import (       # noqa: E402
    client, next_binary, next_text, run_client, session)

#: The binary attribute payload's tag, mirrored from ``pxviewer.live``.
TAG_ATTRIBUTE = 2


def decode_attribute(payload):
    """``(key, values)`` from a binary attribute message.

    The layout is ``[tag][key length][key][padding to 4][float32 values]`` -- the padding
    is what lets the values be read as a numpy view rather than copied out byte by byte.
    """
    tag, key_length = struct.unpack("<II", payload[:8])
    assert tag == TAG_ATTRIBUTE
    key = payload[8:8 + key_length].decode()
    padding = (-key_length) % 4
    values = np.frombuffer(payload[8 + key_length + padding:], dtype="<f4")
    return key, values


# -- representations ----------------------------------------------------------


def exercise_setting_a_representation_reaches_the_client():
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                rid = live.set_representation("cartoon", color="secondary-structure")
                message = await next_text(ws, "representations")
                assert message["reprs"] == [
                    {"id": rid, "type": "cartoon", "color": "secondary-structure"}]

        run_client(scenario)


def exercise_representations_are_replayed_to_a_late_client():
    with session() as live:
        live.add_representation("spacefill", color="chain-id")

        async def scenario():
            async with client(live) as ws:
                message = await next_text(ws, "representations")
                assert message["reprs"][0]["type"] == "spacefill"
                assert message["reprs"][0]["color"] == "chain-id"

        run_client(scenario)


# -- the attributes a model always has ----------------------------------------


def exercise_bfactor_and_occupancy_are_always_available():
    """``from_sites`` writes occupancy 1.0 and B 0.0, so both exist on any model and
    can be coloured by without anything being computed first."""
    with session() as live:
        assert set(live.attributes()) >= {"bfactor", "occupancy"}
        live.color_by("bfactor")
        live.color_by("occupancy")


def exercise_a_named_attribute_can_be_set_and_coloured_by():
    with session() as live:
        live.set_attribute("score", [0.1, 0.2, 0.3, 0.4])
        assert "score" in live.attributes()

        rid = live.color_by("score", domain=(0, 1))
        assert live._representations[rid]["attribute"]["name"] == "score"


def exercise_an_attribute_of_the_wrong_length_is_rejected():
    """One value per atom, or the colours land on the wrong atoms silently."""
    with session() as live:
        with raises(ValueError) as e:
            live.set_attribute("bad", [1, 2, 3])       # the topology has 4
        assert "4 atoms" in str(e.value)


def exercise_colouring_by_an_unknown_attribute_is_rejected():
    with session() as live:
        with raises(ValueError) as e:
            live.color_by("nonsense")
        assert "unknown attribute" in str(e.value)


# -- what goes on the wire ----------------------------------------------------


def exercise_color_by_sends_binary_values_then_a_representation():
    """The values first, as a compact binary message; then the representation that
    references them by key. Inlining the numbers in the JSON would resend them for every
    representation that used them."""
    with session() as live:
        async def scenario():
            async with client(live) as ws:
                live.color_by([0.0, 1.0, 2.0, 3.0], palette="viridis",
                              domain=(0, 3), type="cartoon")

                key, values = decode_attribute(await next_binary(ws, TAG_ATTRIBUTE))
                assert list(values) == [0.0, 1.0, 2.0, 3.0]

                message = await next_text(ws, "representations")
                spec = message["reprs"][0]
                assert spec["color"] == "attribute"
                assert spec["type"] == "cartoon"
                assert spec["attribute"]["key"] == key
                assert spec["attribute"]["domain"] == [0.0, 3.0]
                assert spec["attribute"]["palette"] == "viridis"
                assert "values" not in spec["attribute"]      # they went as binary

        run_client(scenario)


def exercise_color_by_is_replayed_to_a_late_client():
    """Values before representation, in that order -- the other way round the client
    would briefly hold a representation keyed to values it does not have."""
    with session() as live:
        live.color_by([4.0, 5.0, 6.0, 7.0], palette="turbo")

        async def scenario():
            async with client(live) as ws:
                _key, values = decode_attribute(await next_binary(ws, TAG_ATTRIBUTE))
                assert list(values) == [4.0, 5.0, 6.0, 7.0]

                message = await next_text(ws, "representations")
                assert message["reprs"][0]["color"] == "attribute"

        run_client(scenario)


def exercise_the_domain_is_taken_from_the_finite_values():
    with session() as live:
        rid = live.color_by([10.0, 20.0, 30.0, 40.0])
        assert live._representations[rid]["attribute"]["domain"] == [10.0, 40.0]


def exercise_a_missing_value_survives_as_nan():
    """nan means "not computed for this atom" -- Q-score leaves it on every hydrogen --
    and the theme draws those in its missing colour. Encoding it as a number would paint
    them as real values at one end of the scale."""
    with session() as live:
        live.color_by([float("nan"), 1.0, 2.0, 3.0])
        payload = next(iter(live._attribute_payloads.values()))
        _key, values = decode_attribute(payload)
        assert np.isnan(values[0])
        assert values[1] == 1.0


# -- bookkeeping --------------------------------------------------------------


def exercise_color_by_replaces_the_representations():
    """Like ``set_representation`` rather than ``add_representation``: colouring by a
    value is a statement about how the model is drawn, not another layer on it."""
    with session() as live:
        live.add_representation("spacefill", color="chain-id")
        live.color_by([0.0, 1.0, 2.0, 3.0])
        assert len(live._representations) == 1


def exercise_a_replaced_attribute_payload_is_pruned():
    """The payloads are held to replay them, so nothing must accumulate: a re-colour
    drops the values it replaced, and clearing the representations drops them all."""
    with session() as live:
        live.color_by([1.0, 2.0, 3.0, 4.0])
        live.color_by([5.0, 6.0, 7.0, 8.0])
        assert len(live._attribute_payloads) == 1

        live.clear_representations()
        assert live._attribute_payloads == {}


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
