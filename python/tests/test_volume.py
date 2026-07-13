"""Tests and examples for the pxviewer volume API."""

import json

import numpy as np
import pytest

from pxviewer import create_volume_view, create_volume_view_from_data, read_volume, write_volume


def test_write_volume_round_trip(tmp_path):
    """Write a volume as MRC and read it back."""
    data = np.zeros((10, 12, 14), dtype=np.float32)
    data[5, 6, 7] = 5.0
    path = tmp_path / "density.mrc"
    write_volume(data, path, voxel_size=(1.0, 2.0, 3.0), origin=(4.0, 5.0, 6.0))

    read = read_volume(path)
    assert read["shape"] == (10, 12, 14)
    assert read["voxel_size"] == pytest.approx((1.0, 2.0, 3.0))
    assert read["origin"] == pytest.approx((4.0, 5.0, 6.0))
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


def test_create_volume_view_builds_map_node():
    """Build an MVSJ scene that loads a volume from a URL."""
    mvsj = create_volume_view(
        "density.mrc",
        isosurface_value=3.0,
        isosurface_kind="absolute",
        color="red",
        opacity=0.5,
    )
    state = json.loads(mvsj)
    assert state["kind"] == "single"
    root = state["root"]
    download = root["children"][0]
    assert download["kind"] == "download"
    assert download["params"]["url"] == "density.mrc"

    parse_node = download["children"][0]
    assert parse_node["kind"] == "parse"
    assert parse_node["params"]["format"] == "map"

    volume = parse_node["children"][0]
    assert volume["kind"] == "volume"

    repr = volume["children"][0]
    assert repr["kind"] == "volume_representation"
    assert repr["params"]["type"] == "isosurface"
    assert repr["params"]["absolute_isovalue"] == pytest.approx(3.0)


def test_create_volume_view_from_data(tmp_path):
    """Write a volume and MVSJ in one call."""
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[5, 5, 5] = 10.0
    mrc_path = tmp_path / "model.mrc"
    mvsj_path = tmp_path / "model.mvsj"

    mvsj = create_volume_view_from_data(
        data,
        mrc_path=mrc_path,
        mvsj_path=mvsj_path,
        write_kwargs={"voxel_size": 2.0},
        view_kwargs={"isosurface_value": 2.0, "isosurface_kind": "relative"},
    )

    assert mrc_path.exists()
    assert mvsj_path.exists()

    state = json.loads(mvsj)
    assert "model.mrc" in state["root"]["children"][0]["params"]["url"]

    read = read_volume(mrc_path)
    assert read["voxel_size"] == pytest.approx((2.0, 2.0, 2.0))
    assert read["data"][5, 5, 5] == pytest.approx(10.0)


def test_write_volume_float64_cast_to_float32(tmp_path):
    """Float64 data is cast to float32 because MRC uses float32."""
    data = np.ones((4, 4, 4), dtype=np.float64) * 1.5
    path = tmp_path / "density.mrc"
    write_volume(data, path, voxel_size=1.0)

    read = read_volume(path)
    assert read["data"].dtype == np.float32
    assert read["data"][0, 0, 0] == pytest.approx(1.5)
