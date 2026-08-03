"""probe2 contact-dot extraction and its wire encoding."""

from __future__ import absolute_import, division, print_function

import struct
import sys

from pxviewer.probe import _dot_rgb, encode_dots
from pxviewer.regression.tst_utils import data_path, have

_cache = []


def model():
    """1UBQ, read once per process."""
    if not _cache:
        from iotbx.data_manager import DataManager

        dm = DataManager()
        dm.process_model_file(data_path("1ubq.pdb"))
        _cache.append(dm.get_model())
    return _cache[0]


def probe_runnable():
    if not have("iotbx.data_manager", "mmtbx.programs.probe2"):
        print("  skipping: data_manager / probe2 not available")
        return False
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        print("  skipping: no monomer library")
        return False
    return True


def exercise_dot_colour_mapping():
    assert _dot_rgb("hb", 0.0) != _dot_rgb("wc", 0.0)     # H-bonds are distinct
    assert _dot_rgb("wc", 0.4) == (0x40, 0x40, 0xFF)       # wide contact -> blue
    assert _dot_rgb("bo", -0.5) == (0xFF, 0x66, 0xB4)      # bad clash -> hotpink


def exercise_encode_dots_roundtrip():
    dots = [((1.0, 2.0, 3.0), (1.5, 2.0, 3.0), (255, 0, 0)),
            ((4.0, 5.0, 6.0), (4.0, 5.0, 6.0), (0, 128, 0))]
    blob = encode_dots(dots)
    assert struct.unpack("<I", blob[:4])[0] == 2
    assert len(blob) == 4 + 2 * 28          # 6 floats + 1 uint32 per dot

    lx, ly, lz, sx, sy, sz, rgb = struct.unpack("<6fI", blob[4:32])
    assert (lx, ly, lz) == (1.0, 2.0, 3.0)
    assert (sx, sy, sz) == (1.5, 2.0, 3.0)
    assert rgb == (255 << 16)               # red packed


def exercise_show_probe_dots_payload():
    if not have("iotbx.data_manager", "websockets"):
        print("  skipping: data_manager / websockets not available")
        return
    from pxviewer.live import LiveSession

    session = LiveSession.from_sites([[0, 0, 0], [1.5, 0, 0]])
    n = session.show_probe_dots([((0, 0, 0), (0.1, 0, 0), (255, 0, 0))])
    assert n == 1 and session._probe_dots_payloads
    tag, channel = struct.unpack("<II", session._probe_dots_payloads[0][:8])
    assert tag == 3 and channel == 0        # _TAG_DOTS, default PROBE_CONTACTS channel

    session.clear_probe_dots()
    assert session._probe_dots_payloads == {}


def exercise_probe_dots_on_ubiquitin():
    if not probe_runnable():
        return
    from pxviewer.probe import probe_dots

    dots = probe_dots(model())
    assert len(dots) > 1000                 # thousands of contact dots
    loc, spike, rgb = dots[0]
    assert len(loc) == 3 and len(spike) == 3 and len(rgb) == 3
    # At least some dots are overlaps (the spike differs from the location).
    assert any(l != s for l, s, _ in dots)


def exercise_probe_dots_split_is_a_subset():
    if not probe_runnable():
        return
    from pxviewer.probe import probe_dots, probe_dots_split

    contacts, clashes = probe_dots_split(model())
    # Clashes are a strict subset of the full surface, and match the clashes-only run.
    assert 0 < len(clashes) < len(contacts)
    assert clashes == probe_dots(model(), only_clashes=True)


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
