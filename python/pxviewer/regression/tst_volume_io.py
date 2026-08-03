"""cctbx-native volume I/O: VolumeData and map_model_manager grouping."""

from __future__ import absolute_import, division, print_function

import os
import sys

from pxviewer.regression.tst_utils import have, skip, tmp_dir

if not have("numpy", "iotbx.data_manager", "iotbx.map_model_manager"):
    skip("iotbx data_manager / map_model_manager not available")

import numpy as np                                   # noqa: E402

from pxviewer.volume_io import (VolumeData, map_model_manager_from_files,   # noqa: E402
                                masked_map_copy, split_map_model_manager)

_cache = []


def synthetic_mmm():
    """A small in-memory map+model group from cctbx's own synthetic example.

    Built once per process. Shared, so exercises that would mutate it make their own.
    """
    if not _cache:
        from iotbx.map_model_manager import map_model_manager

        mmm = map_model_manager()
        mmm.generate_map()
        _cache.append(mmm)
    return _cache[0]


def exercise_metadata_from_a_map_manager():
    vol = VolumeData.from_map_manager(synthetic_mmm().map_manager())

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


def exercise_the_array_is_lazy_and_cached():
    """Metadata and re-writing the map need only the map_manager, so the flex->numpy copy
    is deferred until something actually asks for values."""
    vol = VolumeData.from_map_manager(synthetic_mmm().map_manager())
    assert vol._array is None            # not materialised until asked for
    first = vol.array
    assert vol._array is first           # cached
    assert vol.array is first


def exercise_splitting_groups_the_model_and_its_maps():
    model_data, volumes = split_map_model_manager(synthetic_mmm(), name="demo")

    assert model_data is not None
    assert model_data.n_atoms > 0
    # One VolumeData per map cctbx holds, keeping their ids.
    assert [v.map_id for v in volumes] == list(synthetic_mmm().map_id_list())
    assert "map_manager" in [v.map_id for v in volumes]
    by_id = dict((v.map_id, v) for v in volumes)
    assert by_id["map_manager"].name == "demo:map_manager"
    assert by_id["map_manager"].grid == (30, 40, 32)


def exercise_a_roundtrip_through_files_rebuilds_the_group():
    """Write map+model, reload as a group via DataManager, and split it back."""
    mmm = synthetic_mmm()
    with tmp_dir() as work:
        map_path = os.path.join(work, "map.mrc")
        model_path = os.path.join(work, "model.pdb")
        mmm.map_manager().write_map(map_path)
        with open(model_path, "w") as fh:
            fh.write(mmm.model().model_as_pdb())

        # A single map file on its own -> one VolumeData.
        vol = VolumeData.from_map_file(map_path)
        assert vol.grid == (30, 40, 32)
        assert vol.name == "map.mrc"

        # Model + map together -> cctbx builds the group; we split it.
        group = map_model_manager_from_files(model_file=model_path,
                                             map_files=[map_path])
        model_data, volumes = split_map_model_manager(group)
        assert model_data is not None and model_data.n_atoms > 0
        assert len(volumes) == 1 and volumes[0].grid == (30, 40, 32)


def exercise_write_map_is_reloadable():
    vol = VolumeData.from_map_manager(synthetic_mmm().map_manager())
    with tmp_dir() as work:
        out = os.path.join(work, "out.mrc")
        vol.write_map(out)
        assert os.path.exists(out)

        reloaded = VolumeData.from_map_file(out)
        assert reloaded.grid == vol.grid
        assert np.allclose(reloaded.array, vol.array, atol=1e-4)


def exercise_masking_leaves_the_real_map_alone():
    """The map the viewer draws and the map it refines against are the same object, so
    masking must copy. cctbx's mask_all_maps_around_atoms masks in place -- using it here
    would quietly put holes in the density minimization is fitting to."""
    from iotbx.map_model_manager import map_model_manager

    # Its own manager: this one is about mutation, so it must not share the cached group.
    mmm = map_model_manager()
    mmm.generate_map()
    before = mmm.map_manager().map_data().as_numpy_array().copy()
    ids_before = set(mmm.map_id_list())

    masked = masked_map_copy(mmm, "map_manager", 3.0)

    after = mmm.map_manager().map_data().as_numpy_array()
    assert np.array_equal(before, after)              # the real map is whole
    assert set(mmm.map_id_list()) == ids_before       # scratch maps cleaned up

    # The copy has lost the density away from the model.
    kept = masked.map_data().as_numpy_array()
    occupied = lambda d: float((np.abs(d) > 1e-4).mean())      # noqa: E731
    assert occupied(kept) < 0.5 * occupied(before)


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
