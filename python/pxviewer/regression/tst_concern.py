"""Importing bounded concern fields: the display contract, and what is refused.

Pure I/O and calibration -- no desktop shell, no live session, so this runs in a headless
cctbx build. The parts that need a viewer are in tst_hotspots_gui.py.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import json
import os
import sys

from libtbx.test_utils import raises

from pxviewer.regression.tst_utils import have, monkeypatched, skip, tmp_dir

if not have("mmtbx", "numpy"):
    skip("mmtbx/numpy not available")

import numpy as np                                    # noqa: E402


def write_manifest(work, metrics=("combined", "clash"), percentile=True, anchors=None):
    """A manifest in the generator's shape, with map files that exist but are never read
    (``VolumeData.from_map_file`` is patched by :func:`patched_map_reads`)."""
    outputs = {}
    for metric in metrics:
        name = "model_%s_hotspot.ccp4" % metric
        open(os.path.join(work, name), "w").close()
        entry = {"concern": name}
        if percentile:
            pname = "model_%s_hotspot_percentile.ccp4" % metric
            open(os.path.join(work, pname), "w").close()
            entry["color_percentile"] = pname
        outputs[metric] = entry
    payload = {"outputs": outputs, "color_scaling": {"combined": {"concern_gate": 0.05}}}
    if anchors is not None:
        payload["primary_display"] = {"color_anchors": anchors}
    manifest = os.path.join(work, "model_hotspots.json")
    with open(manifest, "w") as fh:
        json.dump(payload, fh)
    return manifest


@contextlib.contextmanager
def patched_map_reads(concern_value=0.6, percentile_value=0.96):
    """Return synthetic grids instead of reading CCP4s, yielding the list of paths opened."""
    from pxviewer.volume_io import VolumeData

    concern = VolumeData.from_numpy(
        np.full((3, 4, 5), concern_value, dtype=np.float32),
        spacing=(1.0, 1.5, 2.0), origin=(2, 3, 4), name="concern")
    percentile = VolumeData.from_numpy(
        np.full((3, 4, 5), percentile_value, dtype=np.float32),
        spacing=(1.0, 1.5, 2.0), origin=(2, 3, 4), name="percentile")
    opened = []

    # Match on the file name, not the whole path: a temp directory named after the test
    # would otherwise hand back the percentile grid for every map it reads.
    def _read(cls, path, **kwargs):
        opened.append(str(path))
        return percentile if "percentile" in os.path.basename(str(path)) else concern

    with monkeypatched(VolumeData, "from_map_file", classmethod(_read)):
        yield opened


def exercise_anchors_come_from_the_manifest():
    """The display contract travels with the data; the viewer follows it rather than
    hardcoding a ramp that silently goes stale when the generator changes."""
    from pxviewer import concern

    assert concern.display_anchors({}) == concern.DEFAULT_ANCHORS
    declared = concern.display_anchors({"primary_display": {"color_anchors": {
        "0.0": "transparent", "0.4": "yellow", "0.6": "orange", "0.9": "red"}}})
    assert declared == {"yellow": 0.4, "orange": 0.6, "red": 0.9}
    # A contract whose colours do not ascend cannot make a coherent ramp; fall back rather
    # than paint an incoherent one.
    assert concern.display_anchors({"primary_display": {"color_anchors": {
        "0.8": "yellow", "0.2": "red"}}}) == concern.DEFAULT_ANCHORS

    # The contour colour is read off the same anchors the density is painted with, so the
    # two styles cannot state the same level in different colours.
    assert concern.concern_color(0.4, declared) == "#FFD400"    # exactly the yellow anchor
    assert concern.concern_color(0.6, declared) == "#F46D43"    # exactly the orange anchor
    assert concern.concern_color(0.9, declared) == "#B2182B"    # exactly the red anchor
    assert concern.concern_color(0.5, declared) == "#FAA022"    # interpolated between them
    assert concern.concern_color(0.1) == "#FFD400"              # below yellow, clamped


def exercise_a_bare_map_finds_its_pair_and_names_the_metric():
    """Opening either half of a pair works, and neither names the metric after the file."""
    from pxviewer import concern
    from pxviewer.volume_io import VolumeData

    with tmp_dir() as work, patched_map_reads():
        for name in ("1tec_rama_hotspot.map.gz", "1tec_rama_hotspot_percentile.map.gz"):
            open(os.path.join(work, name), "w").close()

        # Opening the concern map picks up the sibling; opening the percentile map works back
        # to the concern map. Compound extensions survive, and "_hotspot"/"_percentile" are
        # not mistaken for part of the metric name.
        for opened in ("1tec_rama_hotspot.map.gz", "1tec_rama_hotspot_percentile.map.gz"):
            imported = concern.read_fields(os.path.join(work, opened), VolumeData)
            assert list(imported.fields) == ["1tec_rama"], opened
            assert imported.fields["1tec_rama"].percentile is not None

        # A concern map with no companion still imports: percentile is optional.
        open(os.path.join(work, "solo_hotspot.ccp4"), "w").close()
        solo = concern.read_fields(os.path.join(work, "solo_hotspot.ccp4"), VolumeData)
        assert list(solo.fields) == ["solo"] and solo.fields["solo"].percentile is None
        # With no manifest there is no declared contract, so the default applies.
        assert solo.anchors == concern.DEFAULT_ANCHORS


def exercise_an_unbounded_map_is_refused():
    """A severity map opened as concern is refused, not silently rescaled."""
    from pxviewer import concern
    from pxviewer.volume_io import VolumeData

    severity = VolumeData.from_numpy(
        np.full((3, 4, 5), 2.5, dtype=np.float32), spacing=1.0, name="severity")

    with tmp_dir() as work, monkeypatched(
            VolumeData, "from_map_file",
            classmethod(lambda cls, path, **kwargs: severity)):
        path = os.path.join(work, "model_combined_hotspot.ccp4")
        open(path, "w").close()
        with raises(ValueError) as e:
            concern.read_fields(path, VolumeData)
        assert "bounded to" in str(e.value)


def exercise_a_manifest_imports_every_field():
    """Reading a manifest keeps concern and percentile separate, and combined leads."""
    from pxviewer import concern
    from pxviewer.volume_io import VolumeData

    with tmp_dir() as work, patched_map_reads() as opened:
        manifest = write_manifest(work)
        imported = concern.read_fields(manifest, VolumeData)

        assert len(opened) == 4        # two metrics, concern + percentile each
        assert set(imported.fields) == set(["combined", "clash"])
        assert imported.primary == "combined"
        assert abs(imported.fields["combined"].values[1, 1, 1] - 0.6) < 1e-6
        assert abs(imported.fields["combined"].percentile_values[1, 1, 1] - 0.96) < 1e-6
        assert imported.anchors == concern.DEFAULT_ANCHORS


def exercise_percentile_is_optional():
    """The generator states determines_visibility: false; a viewer that *required* the
    companion map would make that contract untrue."""
    from pxviewer import concern
    from pxviewer.volume_io import VolumeData

    with tmp_dir() as work, patched_map_reads() as opened:
        manifest = write_manifest(work, metrics=("rama",), percentile=False)
        imported = concern.read_fields(manifest, VolumeData)

        assert len(opened) == 1        # only the concern map was read
        assert imported.fields["rama"].percentile is None


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
