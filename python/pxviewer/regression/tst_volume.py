"""The volume data helpers: MRC round trips and the MVSJ scene a Volume builds."""

from __future__ import absolute_import, division, print_function

import json
import os
import sys

from libtbx.test_utils import approx_equal

from pxviewer.regression.tst_utils import have, skip, tmp_dir

if not have("numpy"):
    skip("numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer import read_volume, write_volume       # noqa: E402


def exercise_write_volume_round_trip():
    """Write a volume as MRC and read it back."""
    with tmp_dir() as work:
        data = np.zeros((10, 12, 14), dtype=np.float32)
        data[5, 6, 7] = 5.0
        path = os.path.join(work, "density.mrc")
        # cctbx snaps the origin to whole voxels, so use a grid-aligned angstrom origin.
        write_volume(data, path, voxel_size=(1.0, 2.0, 3.0), origin=(4.0, 6.0, 6.0))

        read = read_volume(path)
        assert read["shape"] == (10, 12, 14)
        assert approx_equal(read["voxel_size"], (1.0, 2.0, 3.0))
        assert approx_equal(read["origin"], (4.0, 6.0, 6.0))
        assert approx_equal(read["data"][5, 6, 7], 5.0)


def exercise_write_volume_xyz_order():
    """Data given in data[x, y, z] order must come back in MRC order."""
    with tmp_dir() as work:
        data_xyz = np.zeros((14, 12, 10), dtype=np.float32)
        data_xyz[7, 6, 5] = 3.0
        path = os.path.join(work, "density.mrc")
        write_volume(data_xyz, path, voxel_size=1.0, data_order="xyz")

        read = read_volume(path)
        assert read["shape"] == (10, 12, 14)
        assert approx_equal(read["data"][5, 6, 7], 3.0)


def exercise_write_volume_grid_origin():
    """An origin given in grid cells rather than angstroms."""
    with tmp_dir() as work:
        data = np.zeros((8, 8, 8), dtype=np.float32)
        data[1, 2, 3] = 9.0
        path = os.path.join(work, "density.mrc")
        write_volume(data, path, voxel_size=1.0, origin=(3, 2, 1), origin_units="grid")

        read = read_volume(path)
        assert read["shape"] == (8, 8, 8)
        assert approx_equal(read["data"][1, 2, 3], 9.0)


def exercise_float64_is_cast_to_float32():
    """MRC is float32, so float64 input is cast rather than refused."""
    with tmp_dir() as work:
        path = os.path.join(work, "density.mrc")
        write_volume(np.ones((4, 4, 4), dtype=np.float64) * 1.5, path, voxel_size=1.0)

        read = read_volume(path)
        assert read["data"].dtype == np.float32
        assert approx_equal(read["data"][0, 0, 0], 1.5)


def scene_tree(volume):
    """The MVSJ node tree for one Volume, flattened depth-first."""
    import molviewspec as mvs

    from pxviewer.volume import _build_volume

    builder = mvs.create_builder()
    _build_volume(builder, volume, volume.ref)
    state = json.loads(builder.get_state().dumps())

    found = []

    def walk(node):
        found.append(node)
        for child in node.get("children") or []:
            walk(child)

    walk(state["root"])
    return found


def exercise_a_difference_map_is_drawn_at_both_signs():
    """A difference map is only readable as a pair -- green where the density wants more
    than the model has, red where it wants less. Both contours hang off one volume node, so
    the map is downloaded and parsed once, and one level drives both."""
    if not have("molviewspec"):
        print("  skipping: molviewspec not available")
        return
    from pxviewer.volume import Volume

    nodes = scene_tree(Volume(url="d.map", ref="v1", isosurface_value=3.0,
                              color="green", negative_color="red"))
    reprs = [n for n in nodes if n["kind"] == "volume_representation"]
    assert len(reprs) == 2
    assert [r["params"]["relative_isovalue"] for r in reprs] == [3.0, -3.0]
    assert [n["ref"] for n in reprs] == ["v1-repr", "v1-repr-neg"]
    # One download and one parse feed both.
    assert len([n for n in nodes if n["kind"] == "download"]) == 1
    assert len([n for n in nodes if n["kind"] == "volume"]) == 1
    # The colours differ; that is the entire point of drawing both.
    colors = [c["params"]["color"] for r in reprs
              for c in (r.get("children") or []) if c["kind"] == "color"]
    assert colors == ["green", "red"]


def exercise_a_regular_map_has_one_contour():
    """Only difference maps have a negative side worth drawing; a 2Fo-Fc map's would be
    noise, and a second isosurface is not free."""
    if not have("molviewspec"):
        print("  skipping: molviewspec not available")
        return
    from pxviewer.volume import Volume

    nodes = scene_tree(Volume(url="d.map", ref="v1", isosurface_value=1.5,
                              color="dodgerblue"))
    assert len([n for n in nodes if n["kind"] == "volume_representation"]) == 1


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
