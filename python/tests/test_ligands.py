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
