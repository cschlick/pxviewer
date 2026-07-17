"""Tests for dragging an atom with the model giving way live."""

from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb"


def _require_restraints():
    pytest.importorskip("mmtbx.geometry_restraints.reference")
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")


def _model():
    from pxviewer.cctbx_io import read_model

    return read_model(str(MODEL))


def test_a_tug_pulls_it_does_not_teleport():
    """The atom arrives where the geometry lets it, not where the pointer is — that is
    the difference between dragging a model and editing coordinates."""
    _require_restraints()
    from pxviewer.tug import Tug

    model = _model()
    tug = Tug(model, 300)
    start = model.get_sites_cart().as_numpy_array().copy()  # after process(), see below
    target = start[300] + np.array([3.0, 0.0, 0.0])
    for i in range(20):  # as a pointer moves: in steps
        tug.move_to(start[300] + np.array([3.0 * (i + 1) / 20, 0.0, 0.0]))
    tug.finish()

    now = model.get_sites_cart().as_numpy_array()
    moved = np.linalg.norm(now - start, axis=1)
    assert 1.0 < moved[300] < 3.0            # it followed, but the geometry argued
    assert np.linalg.norm(now[300] - target) > 0.01
    assert (moved > 0.05).sum() > 10         # the neighbourhood gave way with it

    # And the model is still a model: strained, not torn.
    energies = model.get_restraints_manager().geometry.energies_sites(
        sites_cart=model.get_sites_cart(), compute_gradients=False)
    assert energies.bond_deviations()[2] < 0.1


def test_only_the_zone_moves_and_it_stays_attached():
    """Two things at once. The zone is what makes this interactive at all — its cost is
    its own size, not the model's — and grm.select drops every restraint reaching out of
    it, so without pinned boundary atoms the zone drifts off, edges first."""
    _require_restraints()
    from pxviewer.tug import Tug

    model = _model()
    tug = Tug(model, 300, radius=8.0)
    start = model.get_sites_cart().as_numpy_array().copy()
    assert tug.zone_size < len(start) / 4  # a fraction of the model, not all of it

    for i in range(20):
        tug.move_to(start[300] + np.array([3.0 * (i + 1) / 20, 0.0, 0.0]))
    tug.finish()

    now = model.get_sites_cart().as_numpy_array()
    outside = ~np.isin(np.arange(len(start)), tug._indices)
    assert np.linalg.norm(now[outside] - start[outside], axis=1).max() == 0.0

    # The zone stayed put rather than sailing off with the atom.
    drift = np.linalg.norm(now[tug._indices].mean(axis=0) - start[tug._indices].mean(axis=0))
    assert drift < 0.5


def test_density_is_what_makes_a_tug_correct_something():
    """Geometry cannot know where the atoms belong; density can. The map term is also
    the one that silently does nothing if you get it wrong — that lbfgs refines a copy
    and rebinds it, so handing it sites and hoping leaves them untouched."""
    _require_restraints()
    pytest.importorskip("iotbx.map_model_manager")
    from iotbx.map_model_manager import map_model_manager
    from scitbx.array_family import flex

    from pxviewer.tug import Tug

    mmm = map_model_manager(model=_model())
    mmm.generate_map(d_min=2.0)
    truth = mmm.model().get_sites_cart().as_numpy_array().copy()
    map_data = mmm.map_manager().map_data()

    shaken = _model()
    xrs = shaken.get_xray_structure().deep_copy_scatterers()
    xrs.shake_sites_in_place(mean_distance=0.4)
    shaken_sites = xrs.sites_cart().as_numpy_array().copy()

    def jiggle(use_map):
        """The same drag, from the same start, with and without density."""
        model = _model()
        model.set_sites_cart(flex.vec3_double(shaken_sites))
        tug = Tug(model, 300, map_data=map_data if use_map else None, map_weight=50.0)
        zone = tug._indices
        rmsd = lambda: float(np.sqrt(
            ((model.get_sites_cart().as_numpy_array()[zone] - truth[zone]) ** 2).sum(axis=1).mean()))
        before = rmsd()
        here = model.get_sites_cart().as_numpy_array()[300]
        for i in range(25):
            tug.move_to(here + np.array([0.15 * np.sin(i / 3), 0.0, 0.0]))
        tug.finish()
        return before, rmsd()

    before_geom, after_geom = jiggle(False)
    before_map, after_map = jiggle(True)
    assert before_geom == pytest.approx(before_map)  # the same start, or this proves nothing

    # Geometry alone cannot improve on the truth it cannot see; density moves toward it.
    assert after_map < before_map - 0.05
    assert after_map < after_geom - 0.05
