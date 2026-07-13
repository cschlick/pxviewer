"""Tests and examples for the pxviewer volume data helpers."""

import numpy as np
import pytest

from pxviewer import read_volume, write_volume


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
