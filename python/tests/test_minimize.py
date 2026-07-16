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


def test_minimize_stride_thins_the_stream_but_keeps_the_final_state():
    _require_restraints()
    from pxviewer.minimize import minimize_geometry

    model = _shaken_model()
    frames = []
    stats = minimize_geometry(model, on_state=frames.append, stride=10)

    assert stats["n_sent"] < stats["n_states"] / 5  # meaningfully thinned
    assert np.allclose(frames[-1], model.get_sites_cart().as_numpy_array())
