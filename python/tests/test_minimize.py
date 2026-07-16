"""Tests for geometry minimization and its live intermediate states."""

from pathlib import Path

import numpy as np
import pytest

UBIQUITIN = Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb"


def _shaken_model(distance=0.3):
    """1UBQ with its coordinates shaken, so there is geometry to relax."""
    from iotbx.data_manager import DataManager

    dm = DataManager()
    dm.process_model_file(str(UBIQUITIN))
    model = dm.get_model()
    xrs = model.get_xray_structure().deep_copy_scatterers()
    xrs.shake_sites_in_place(mean_distance=distance)
    model.set_sites_cart(xrs.sites_cart())
    return model


def _require_restraints():
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("mmtbx.refinement.geometry_minimization")
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")


def test_minimize_geometry_improves_and_updates_model():
    _require_restraints()
    from pxviewer.minimize import minimize_geometry

    model = _shaken_model()
    before = model.get_sites_cart().as_numpy_array().copy()
    stats = minimize_geometry(model)

    # The shaken geometry is pulled back onto its restraints.
    assert stats["bonds_after"] < stats["bonds_before"]
    assert stats["angles_after"] < stats["angles_before"]
    assert stats["bonds_after"] < 0.05  # near-ideal bonds
    # and the model itself is minimized in place, so tables/validation/Write follow.
    assert not np.array_equal(before, model.get_sites_cart().as_numpy_array())


def test_minimize_streams_intermediate_states():
    """cctbx hands each intermediate conformation to the states_collector — that is
    what makes the run watchable rather than a jump to the answer."""
    _require_restraints()
    from pxviewer.minimize import minimize_geometry

    model = _shaken_model()
    frames = []
    stats = minimize_geometry(model, on_state=frames.append)

    assert len(frames) > 10  # a run, not a single jump
    assert stats["n_sent"] == len(frames) - 1  # every state, plus the forced final one
    assert all(f.shape == (model.get_number_of_atoms(), 3) for f in frames)
    assert not np.array_equal(frames[0], frames[-1])  # it moved
    # It must land on the real answer, not one step short.
    assert np.allclose(frames[-1], model.get_sites_cart().as_numpy_array())


def _shaken_model_and_map(distance=0.3, d_min=3.0):
    """A model with a density computed from it, in a common frame, then shaken."""
    from iotbx.data_manager import DataManager
    from iotbx.map_model_manager import map_model_manager

    dm = DataManager()
    dm.process_model_file(str(UBIQUITIN))
    mmm = map_model_manager(model=dm.get_model())
    mmm.generate_map(d_min=d_min)
    model = mmm.model()
    xrs = model.get_xray_structure().deep_copy_scatterers()
    xrs.shake_sites_in_place(mean_distance=distance)
    model.set_sites_cart(xrs.sites_cart())
    return model, mmm.map_manager().map_data()


def test_minimize_to_map_improves_geometry_and_streams():
    _require_restraints()
    from pxviewer.minimize import minimize_to_map

    model, map_data = _shaken_model_and_map()
    frames = []
    stats = minimize_to_map(model, map_data, on_state=frames.append)

    assert stats["bonds_after"] < stats["bonds_before"]
    assert stats["bonds_after"] < 0.05
    assert stats["weight"] > 0  # cctbx derived the map-vs-restraints balance
    assert len(frames) > 10  # a watchable run, not a jump to the answer
    assert np.allclose(frames[-1], model.get_sites_cart().as_numpy_array())


def test_minimize_to_map_pulls_the_model_back_towards_the_density():
    """The point of the map target: the shaken model returns towards the coordinates
    the density was computed from, which restraints alone cannot know about."""
    _require_restraints()
    from pxviewer.minimize import minimize_geometry, minimize_to_map

    truth = _shaken_model_and_map(distance=0.0)[0].get_sites_cart().as_numpy_array()

    with_map, map_data = _shaken_model_and_map()
    minimize_to_map(with_map, map_data)
    map_rmsd = np.sqrt(
        ((with_map.get_sites_cart().as_numpy_array() - truth) ** 2).sum(axis=1).mean())

    no_map = _shaken_model_and_map()[0]
    minimize_geometry(no_map)
    geom_rmsd = np.sqrt(
        ((no_map.get_sites_cart().as_numpy_array() - truth) ** 2).sum(axis=1).mean())

    assert map_rmsd < geom_rmsd  # the density carries information the restraints do not


def test_minimize_to_map_can_be_halted():
    """The map minimizer has no stop hook of its own, so the states collector unwinds
    it — and the model still lands on the conformation it reached."""
    _require_restraints()
    from pxviewer.minimize import minimize_to_map

    model, map_data = _shaken_model_and_map()
    frames = []
    stats = minimize_to_map(
        model, map_data, on_state=frames.append, should_stop=lambda: len(frames) >= 15)

    assert stats["stopped"] is True
    assert stats["n_states"] < 100  # cut short; a full run takes ~150
    assert stats["bonds_after"] < stats["bonds_before"]  # progress kept, not discarded
    assert np.allclose(frames[-1], model.get_sites_cart().as_numpy_array())


def test_minimize_dispatches_on_whether_a_map_is_given():
    _require_restraints()
    from pxviewer.minimize import minimize

    assert minimize(_shaken_model())["weight"] is None  # restraints only
    model, map_data = _shaken_model_and_map()
    assert minimize(model, map_data=map_data)["weight"] > 0  # map target


def test_minimize_can_be_halted_and_keeps_the_progress_so_far():
    """Stop is a shorter run, not a discarded one: the model stays at the conformation
    it reached. scitbx.lbfgs halts when callback_after_step returns True."""
    _require_restraints()
    from pxviewer.minimize import minimize_geometry

    model = _shaken_model()
    frames = []
    stats = minimize_geometry(
        model, on_state=frames.append, should_stop=lambda: len(frames) >= 20)

    assert stats["stopped"] is True
    assert stats["n_states"] < 100  # cut short; a full run takes ~500
    # Partway down the hill, but definitively on the way: better than the shaken start,
    # and short of what a full run reaches.
    assert stats["bonds_after"] < stats["bonds_before"]
    assert np.allclose(frames[-1], model.get_sites_cart().as_numpy_array())


def test_minimize_runs_to_completion_when_never_asked_to_stop():
    _require_restraints()
    from pxviewer.minimize import minimize_geometry

    stats = minimize_geometry(_shaken_model(), should_stop=lambda: False)
    assert stats["stopped"] is False
    assert stats["bonds_after"] < 0.05  # converged, not cut short


def test_minimize_stride_thins_the_stream_but_keeps_the_final_state():
    _require_restraints()
    from pxviewer.minimize import minimize_geometry

    model = _shaken_model()
    frames = []
    stats = minimize_geometry(model, on_state=frames.append, stride=10)

    assert stats["n_sent"] < stats["n_states"] / 5  # meaningfully thinned
    assert np.allclose(frames[-1], model.get_sites_cart().as_numpy_array())
