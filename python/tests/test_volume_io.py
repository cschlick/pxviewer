"""Tests for cctbx-native volume I/O (VolumeData + map_model_manager grouping)."""

import numpy as np
import pytest

pytest.importorskip("iotbx.data_manager")
pytest.importorskip("iotbx.map_model_manager")

from pxviewer.volume_io import (  # noqa: E402
    VolumeData,
    map_model_manager_from_files,
    split_map_model_manager,
)


@pytest.fixture(scope="module")
def synthetic_mmm():
    """A small in-memory map+model group from cctbx's own synthetic example."""
    from iotbx.map_model_manager import map_model_manager

    mmm = map_model_manager()
    mmm.generate_map()
    return mmm


def test_volumedata_from_map_manager_metadata(synthetic_mmm):
    vol = VolumeData.from_map_manager(synthetic_mmm.map_manager())

    assert vol.grid == (30, 40, 32)
    assert vol.array.shape == (30, 40, 32)
    assert vol.array.dtype == np.float64
    assert vol.origin == (0, 0, 0)
    assert len(vol.unit_cell) == 6
    assert vol.unit_cell_grid == (30, 40, 32)
    assert len(vol.pixel_sizes) == 3
    assert isinstance(vol.space_group, str)

    stats = vol.stats()
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert vol.suggested_iso() > 0


def test_volumedata_array_is_lazy_and_cached(synthetic_mmm):
    vol = VolumeData.from_map_manager(synthetic_mmm.map_manager())
    assert vol._array is None  # not materialised until asked for
    first = vol.array
    assert vol._array is first  # cached
    assert vol.array is first


def test_split_map_model_manager_groups_model_and_maps(synthetic_mmm):
    model_data, volumes = split_map_model_manager(synthetic_mmm, name="demo")

    assert model_data is not None
    assert model_data.n_atoms > 0
    # One VolumeData per map cctbx holds, keeping their ids.
    assert [v.map_id for v in volumes] == list(synthetic_mmm.map_id_list())
    assert "map_manager" in [v.map_id for v in volumes]
    by_id = {v.map_id: v for v in volumes}
    assert by_id["map_manager"].name == "demo:map_manager"
    assert by_id["map_manager"].grid == (30, 40, 32)


def test_roundtrip_through_files_rebuilds_the_group(synthetic_mmm, tmp_path):
    """Write map+model, reload as a group via DataManager, and split it back."""
    map_path = tmp_path / "map.mrc"
    model_path = tmp_path / "model.pdb"
    synthetic_mmm.map_manager().write_map(str(map_path))
    model_path.write_text(synthetic_mmm.model().model_as_pdb())

    # A single map file on its own -> one VolumeData.
    vol = VolumeData.from_map_file(str(map_path))
    assert vol.grid == (30, 40, 32)
    assert vol.name == "map.mrc"

    # Model + map together -> cctbx builds the group; we split it.
    mmm = map_model_manager_from_files(model_file=str(model_path), map_files=[str(map_path)])
    model_data, volumes = split_map_model_manager(mmm)
    assert model_data is not None and model_data.n_atoms > 0
    assert len(volumes) == 1 and volumes[0].grid == (30, 40, 32)


def test_write_map_is_reloadable(synthetic_mmm, tmp_path):
    vol = VolumeData.from_map_manager(synthetic_mmm.map_manager())
    out = tmp_path / "out.mrc"
    vol.write_map(str(out))
    assert out.exists()

    reloaded = VolumeData.from_map_file(str(out))
    assert reloaded.grid == vol.grid
    np.testing.assert_allclose(reloaded.array, vol.array, atol=1e-4)
