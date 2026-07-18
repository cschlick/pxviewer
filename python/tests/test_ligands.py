"""Tests for building a monomer-library ligand (pxviewer.ligands).

The fit itself (explode-and-refine) is slow and stochastic, so it is proven by hand
rather than here; these cover the fast, deterministic build path — reading geostd's ideal
coordinates and turning them into a restraint-ready, correctly-placed model.
"""

import numpy as np
import pytest

pytest.importorskip("iotbx.data_manager")

from pxviewer import ligands  # noqa: E402


def test_availability():
    assert ligands.available("GOL")
    assert ligands.available("gol")          # case-insensitive
    assert not ligands.available("NOTACODE")  # no such component
    assert not ligands.available("")


def test_ideal_atoms_have_coordinates():
    names, elements, xyz = ligands.ideal_atoms("GOL")
    assert len(names) == len(elements) == xyz.shape[0] == 14
    assert xyz.shape[1] == 3


def test_build_model_is_centred_and_restraint_ready():
    m = ligands.build_ligand_model("GOL", (12.0, 8.0, 20.0))
    assert m.get_number_of_atoms() == 14
    assert np.allclose(m.get_sites_cart().mean(), (12.0, 8.0, 20.0), atol=1e-6)
    assert m.restraints_manager_available()
    assert {ag.resname for ag in m.get_hierarchy().atom_groups()} == {"GOL"}
    # restraints resolved from the same library the coordinates came from
    assert m.get_restraints_manager().geometry.pair_proxies().bond_proxies.simple.size() > 0


def test_unknown_code_raises():
    with pytest.raises(ValueError):
        ligands.ideal_atoms("NOTACODE")


def test_coarse_orient_recovers_a_bad_orientation():
    """The rotational pre-search rotates a mis-oriented rigid ligand back toward its
    density — deterministic and fast, so it belongs in the suite (unlike the full fit)."""
    from cctbx import crystal
    from cctbx.array_family import flex
    from scitbx.math import euler_angles

    cs = crystal.symmetry(unit_cell=(40, 40, 40, 90, 90, 90), space_group_symbol="P1")
    target = ligands.build_ligand_model("GOL", (20, 20, 20), crystal_symmetry=cs)
    tgt = target.get_sites_cart().as_numpy_array()
    map_data = target.get_xray_structure().structure_factors(d_min=2.0).f_calc().fft_map(
        resolution_factor=0.25).apply_sigma_scaling().real_map_unpadded()

    mis = ligands.build_ligand_model("GOL", (20, 20, 20), crystal_symmetry=cs)
    rot = np.array(euler_angles.xyz_matrix(120, 80, 40)).reshape(3, 3)
    c = mis.get_sites_cart().as_numpy_array().mean(0)
    mis.set_sites_cart(flex.vec3_double(
        np.ascontiguousarray((mis.get_sites_cart().as_numpy_array() - c) @ rot.T + c)))

    before = np.sqrt(((mis.get_sites_cart().as_numpy_array() - tgt) ** 2).sum(1).mean())
    ligands.coarse_orient(mis, map_data, step_deg=30)
    after = np.sqrt(((mis.get_sites_cart().as_numpy_array() - tgt) ** 2).sum(1).mean())
    assert after < before / 2, f"pre-search did not improve orientation: {before:.2f} -> {after:.2f}"
