"""Importing bounded concern fields: the display contract, and what is refused.

Pure I/O and calibration -- no desktop shell, no live session, so this runs in a headless
cctbx build. The parts that need a viewer are in tst_hotspots_gui.py.

Maps here are **written and read back for real**, through cctbx, rather than faked. That is
the cctbx convention and it is also better coverage: the CCP4 round trip is part of what
these tests are about, since NXSTART placement and anisotropic pixel sizes have to survive
it for the viewer to draw a field where the model is.
"""

from __future__ import absolute_import, division, print_function

import json
import os
import sys

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import have, skip, tmp_dir

if not have("mmtbx", "numpy"):
    skip("mmtbx/numpy not available")

import numpy as np                                    # noqa: E402

#: Distinguishable fill values, so a test can tell which map it was handed.
CONCERN_VALUE = 0.6
PERCENTILE_VALUE = 0.96

#: A small anisotropic grid at a nonzero origin. Both matter: the origin exercises CCP4
#: NXSTART placement and the unequal pixel sizes catch an axis mix-up that a cube hides.
GRID = (3, 4, 5)
SPACING = (1.0, 1.5, 2.0)
ORIGIN = (2, 3, 4)


def write_map(path, value, grid=GRID, spacing=SPACING, origin=ORIGIN):
    """Write a real CCP4 of constant ``value`` through cctbx."""
    from pxviewer.volume_io import VolumeData

    VolumeData.from_numpy(np.full(grid, value, dtype=np.float32),
                          spacing=spacing, origin=origin,
                          name=os.path.basename(path)).write_map(path)
    return path


def write_manifest(work, metrics=("combined", "clash"), percentile=True, anchors=None):
    """A manifest in the generator's shape, with the maps it names actually written."""
    outputs = {}
    for metric in metrics:
        name = "model_%s_hotspot.ccp4" % metric
        write_map(os.path.join(work, name), CONCERN_VALUE)
        entry = {"concern": name}
        if percentile:
            pname = "model_%s_hotspot_percentile.ccp4" % metric
            write_map(os.path.join(work, pname), PERCENTILE_VALUE)
            entry["color_percentile"] = pname
        outputs[metric] = entry
    payload = {"outputs": outputs, "color_scaling": {"combined": {"concern_gate": 0.05}}}
    if anchors is not None:
        payload["primary_display"] = {"color_anchors": anchors}
    manifest = os.path.join(work, "model_hotspots.json")
    with open(manifest, "w") as fh:
        json.dump(payload, fh)
    return manifest


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


def exercise_map_names_split_on_compound_extensions():
    """``.map.gz`` is one extension, and "_hotspot"/"_percentile" are decoration rather than
    part of the metric name.

    Tested directly on the splitter rather than through a file: a name is a string, so
    writing a real ``.map.gz`` would say nothing extra about how it was parsed -- and cctbx
    does not write one anyway. Splitting on every dot (``Path.suffixes``) would mangle a name
    that merely contains one, which is what this pins.
    """
    from pathlib import Path

    from pxviewer import concern

    for name, stem, suffix in (
        ("1tec_rama_hotspot.map.gz", "1tec_rama_hotspot", ".map.gz"),
        ("1tec_rama_hotspot.ccp4", "1tec_rama_hotspot", ".ccp4"),
        ("1tec_rama_hotspot_percentile.mrc.gz", "1tec_rama_hotspot_percentile", ".mrc.gz"),
        ("1tec.v2_hotspot.ccp4", "1tec.v2_hotspot", ".ccp4"),   # a dot in the stem survives
        ("plain.mrc", "plain", ".mrc"),
    ):
        assert concern._split_map_name(Path(name)) == (stem, suffix), name


def exercise_a_bare_map_finds_its_pair_and_names_the_metric():
    """Opening either half of a pair works, and neither names the metric after the file."""
    from pxviewer import concern
    from pxviewer.volume_io import VolumeData

    with tmp_dir() as work:
        write_map(os.path.join(work, "1tec_rama_hotspot.ccp4"), CONCERN_VALUE)
        write_map(os.path.join(work, "1tec_rama_hotspot_percentile.ccp4"), PERCENTILE_VALUE)

        # Opening the concern map picks up the sibling; opening the percentile map works
        # back to the concern map. Either way the metric is named for the metric.
        for opened in ("1tec_rama_hotspot.ccp4", "1tec_rama_hotspot_percentile.ccp4"):
            imported = concern.read_fields(os.path.join(work, opened), VolumeData)
            assert list(imported.fields) == ["1tec_rama"], opened
            field = imported.fields["1tec_rama"]
            assert field.percentile is not None
            # The right map landed in the right slot, whichever one was named.
            assert approx_equal(float(field.values.max()), CONCERN_VALUE, eps=1e-5)
            assert approx_equal(float(field.percentile_values.max()), PERCENTILE_VALUE,
                                eps=1e-5)

        # A concern map with no companion still imports: percentile is optional.
        write_map(os.path.join(work, "solo_hotspot.ccp4"), CONCERN_VALUE)
        solo = concern.read_fields(os.path.join(work, "solo_hotspot.ccp4"), VolumeData)
        assert list(solo.fields) == ["solo"] and solo.fields["solo"].percentile is None
        # With no manifest there is no declared contract, so the default applies.
        assert solo.anchors == concern.DEFAULT_ANCHORS


def exercise_an_unbounded_map_is_refused():
    """A severity map opened as concern is refused, not silently rescaled."""
    from pxviewer import concern
    from pxviewer.volume_io import VolumeData

    with tmp_dir() as work:
        path = write_map(os.path.join(work, "model_combined_hotspot.ccp4"), 2.5)
        with raises(ValueError) as e:
            concern.read_fields(path, VolumeData)
        assert "bounded to" in str(e.value)


def exercise_a_manifest_imports_every_field():
    """Reading a manifest keeps concern and percentile separate, and combined leads.

    Also pins the CCP4 round trip the viewer depends on: a nonzero NXSTART origin at
    anisotropic pixel sizes has to come back out placed where it went in, or the field draws
    somewhere the model is not.
    """
    from pxviewer import concern
    from pxviewer.volume_io import VolumeData, grid_affine

    with tmp_dir() as work:
        manifest = write_manifest(work)
        imported = concern.read_fields(manifest, VolumeData)

        assert set(imported.fields) == set(["combined", "clash"])
        assert imported.primary == "combined"
        combined = imported.fields["combined"]
        assert approx_equal(float(combined.values[1, 1, 1]), CONCERN_VALUE, eps=1e-5)
        assert approx_equal(float(combined.percentile_values[1, 1, 1]), PERCENTILE_VALUE,
                            eps=1e-5)
        assert imported.anchors == concern.DEFAULT_ANCHORS

        assert combined.values.shape == GRID
        assert combined.concern.origin == ORIGIN
        assert approx_equal(tuple(combined.concern.pixel_sizes), SPACING, eps=1e-5)
        # Grid origin (2,3,4) at spacing (1.0,1.5,2.0) lands here in Cartesian space.
        cart, _steps = grid_affine(combined.concern.map_manager)
        assert approx_equal(tuple(float(v) for v in cart), (2.0, 4.5, 8.0), eps=1e-5)


def exercise_percentile_is_optional():
    """The generator states determines_visibility: false; a viewer that *required* the
    companion map would make that contract untrue."""
    from pxviewer import concern
    from pxviewer.volume_io import VolumeData

    with tmp_dir() as work:
        manifest = write_manifest(work, metrics=("rama",), percentile=False)
        imported = concern.read_fields(manifest, VolumeData)

        assert imported.fields["rama"].percentile is None
        assert imported.fields["rama"].percentile_values is None
        # ... and the concern field itself came through unaffected.
        assert approx_equal(float(imported.fields["rama"].values.max()), CONCERN_VALUE,
                            eps=1e-5)


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
