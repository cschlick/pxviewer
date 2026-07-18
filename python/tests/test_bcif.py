"""Tests for the self-contained BinaryCIF encoder/decoder (pxviewer.bcif).

This module is meant to be copyable on its own, so its tests lean only on numpy —
they encode, decode, and check the values survive, including the fiddly bits of the
StringArray encoding (unique-string dedup, repeats, empty strings).
"""

import numpy as np

from pxviewer import bcif


def _roundtrip(categories, header="BLOCK"):
    return bcif.decode(bcif.encode(header, categories))[header]


def test_number_column_types_roundtrip():
    cat = bcif.category("_pt", 3, [
        bcif.number_column("i", [1, 2, 3], bcif.INT32),
        bcif.number_column("f", [1.5, 2.5, 3.5], bcif.FLOAT32),
    ])
    out = _roundtrip([cat])["_pt"]
    assert out["i"] == [1, 2, 3]
    np.testing.assert_allclose(out["f"], [1.5, 2.5, 3.5])


def test_string_column_dedups_and_indexes():
    # Repeats must map back to the same string; order of first appearance preserved.
    cat = bcif.category("_s", 5, [
        bcif.string_column("v", ["O", "C", "C", "N", "O"]),
    ])
    assert _roundtrip([cat])["_s"]["v"] == ["O", "C", "C", "N", "O"]


def test_string_column_handles_empty_and_none():
    cat = bcif.category("_s", 4, [
        bcif.string_column("v", ["", "A", None, "A"]),
    ])
    assert _roundtrip([cat])["_s"]["v"] == ["", "A", "", "A"]


def test_leading_underscore_is_added():
    out = _roundtrip([bcif.category("thing", 1, [bcif.number_column("n", [1], bcif.INT32)])])
    assert "_thing" in out


def test_multi_char_strings_offsets():
    # Offsets delimit variable-length uniques; a classic place to get an off-by-one.
    vals = ["ALA", "G", "TRP", "G", "ALA"]
    cat = bcif.category("_r", len(vals), [bcif.string_column("comp", vals)])
    assert _roundtrip([cat])["_r"]["comp"] == vals
