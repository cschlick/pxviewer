"""Tests and examples for the pxviewer volume data helpers."""

import numpy as np
import pytest

from pxviewer import read_volume, write_volume


def test_write_volume_round_trip(tmp_path):
    """Write a volume as MRC and read it back."""
    data = np.zeros((10, 12, 14), dtype=np.float32)
    data[5, 6, 7] = 5.0
    path = tmp_path / "density.mrc"
    # cctbx snaps the origin to whole voxels, so use a grid-aligned Angstrom origin.
    write_volume(data, path, voxel_size=(1.0, 2.0, 3.0), origin=(4.0, 6.0, 6.0))

    read = read_volume(path)
    assert read["shape"] == (10, 12, 14)
    assert read["voxel_size"] == pytest.approx((1.0, 2.0, 3.0))
    assert read["origin"] == pytest.approx((4.0, 6.0, 6.0))
    assert read["data"][5, 6, 7] == pytest.approx(5.0)


def test_write_volume_xyz_order(tmp_path):
    """Write volume given in data[x, y, z] order and verify it comes back as MRC order."""
    data_xyz = np.zeros((14, 12, 10), dtype=np.float32)
    data_xyz[7, 6, 5] = 3.0
    path = tmp_path / "density.mrc"
    write_volume(data_xyz, path, voxel_size=1.0, data_order="xyz")

    read = read_volume(path)
    assert read["shape"] == (10, 12, 14)
    assert read["data"][5, 6, 7] == pytest.approx(3.0)


def test_write_volume_grid_origin(tmp_path):
    """Write a volume with a grid-cell origin and read it back."""
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[1, 2, 3] = 9.0
    path = tmp_path / "density.mrc"
    write_volume(data, path, voxel_size=1.0, origin=(3, 2, 1), origin_units="grid")

    read = read_volume(path)
    assert read["shape"] == (8, 8, 8)
    assert read["data"][1, 2, 3] == pytest.approx(9.0)


def test_write_volume_float64_cast_to_float32(tmp_path):
    """Float64 data is cast to float32 because MRC uses float32."""
    data = np.ones((4, 4, 4), dtype=np.float64) * 1.5
    path = tmp_path / "density.mrc"
    write_volume(data, path, voxel_size=1.0)

    read = read_volume(path)
    assert read["data"].dtype == np.float32
    assert read["data"][0, 0, 0] == pytest.approx(1.5)


def _scene_tree(volume):
    """The MVSJ node tree for one Volume, as (kind, params) pairs by depth."""
    import json

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


def test_a_difference_map_is_drawn_at_both_signs():
    """A difference map is only readable as a pair — green where the density wants more
    than the model has, red where it wants less. Both contours hang off one volume node,
    so the map is downloaded and parsed once, and one level drives both."""
    from pxviewer.volume import Volume

    nodes = _scene_tree(Volume(url="d.map", ref="v1", isosurface_value=3.0,
                               color="green", negative_color="red"))
    reprs = [n for n in nodes if n["kind"] == "volume_representation"]
    assert len(reprs) == 2
    assert [r["params"]["relative_isovalue"] for r in reprs] == [3.0, -3.0]
    assert [n["ref"] for n in reprs] == ["v1-repr", "v1-repr-neg"]
    # One download and one parse feed both.
    assert len([n for n in nodes if n["kind"] == "download"]) == 1
    assert len([n for n in nodes if n["kind"] == "volume"]) == 1
    # The colours differ; that is the entire point of drawing both.
    colours = [c["params"]["color"] for r in reprs
               for c in (r.get("children") or []) if c["kind"] == "color"]
    assert colours == ["green", "red"]


def test_a_regular_map_has_one_contour():
    """Only difference maps have a negative side worth drawing; a 2Fo-Fc map's would be
    noise, and a second isosurface is not free."""
    from pxviewer.volume import Volume

    nodes = _scene_tree(Volume(url="d.map", ref="v1", isosurface_value=1.5,
                               color="dodgerblue"))
    assert len([n for n in nodes if n["kind"] == "volume_representation"]) == 1
