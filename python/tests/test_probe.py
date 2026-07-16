"""Tests for the probe2 contact-dot extraction + wire encoding."""

import struct
from pathlib import Path

import pytest

from pxviewer.probe import _dot_rgb, encode_dots

UBIQUITIN = Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb"


def test_dot_colour_mapping():
    assert _dot_rgb("hb", 0.0) != _dot_rgb("wc", 0.0)  # H-bonds are distinct
    assert _dot_rgb("wc", 0.4) == (0x40, 0x40, 0xFF)    # wide contact -> blue
    assert _dot_rgb("bo", -0.5) == (0xFF, 0x66, 0xB4)   # bad clash -> hotpink


def test_encode_dots_roundtrip():
    dots = [((1.0, 2.0, 3.0), (1.5, 2.0, 3.0), (255, 0, 0)),
            ((4.0, 5.0, 6.0), (4.0, 5.0, 6.0), (0, 128, 0))]
    blob = encode_dots(dots)
    assert struct.unpack("<I", blob[:4])[0] == 2
    assert len(blob) == 4 + 2 * 28  # 6 floats + 1 uint32 per dot

    lx, ly, lz, sx, sy, sz, rgb = struct.unpack("<6fI", blob[4:32])
    assert (lx, ly, lz) == (1.0, 2.0, 3.0)
    assert (sx, sy, sz) == (1.5, 2.0, 3.0)
    assert rgb == (255 << 16)  # red packed


def test_show_probe_dots_payload():
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    from pxviewer.live import LiveSession

    session = LiveSession.from_sites([[0, 0, 0], [1.5, 0, 0]])
    n = session.show_probe_dots([((0, 0, 0), (0.1, 0, 0), (255, 0, 0))])
    assert n == 1 and session._probe_dots_payloads
    tag, channel = struct.unpack("<II", session._probe_dots_payloads[0][:8])
    assert tag == 3 and channel == 0  # _TAG_DOTS, default PROBE_CONTACTS channel

    session.clear_probe_dots()
    assert session._probe_dots_payloads == {}


def test_probe_dots_on_ubiquitin():
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("mmtbx.programs.probe2")
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

    from iotbx.data_manager import DataManager

    from pxviewer.probe import probe_dots

    dm = DataManager()
    dm.process_model_file(str(UBIQUITIN))
    dots = probe_dots(dm.get_model())
    assert len(dots) > 1000  # thousands of contact dots
    loc, spike, rgb = dots[0]
    assert len(loc) == 3 and len(spike) == 3 and len(rgb) == 3
    # at least some dots are overlaps (spike differs from loc)
    assert any(l != s for l, s, _ in dots)


def test_probe_dots_split_is_subset():
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("mmtbx.programs.probe2")
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

    from iotbx.data_manager import DataManager

    from pxviewer.probe import probe_dots, probe_dots_split

    dm = DataManager()
    dm.process_model_file(str(UBIQUITIN))
    contacts, clashes = probe_dots_split(dm.get_model())
    # clashes are a strict subset of the full surface, and clashes only == probe_dots(only_clashes)
    assert 0 < len(clashes) < len(contacts)
    assert clashes == probe_dots(dm.get_model(), only_clashes=True)
