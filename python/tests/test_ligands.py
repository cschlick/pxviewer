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


def test_build_from_smiles_is_centred_and_restraint_ready():
    """A ligand not in the library, built from SMILES: rdkit embeds a conformer whose
    geometry both places the atoms and supplies the on-the-fly restraints, so the model
    comes out centred and with a real (bond + angle) geometry restraints manager."""
    pytest.importorskip("rdkit")
    from cctbx import crystal

    cs = crystal.symmetry(unit_cell=(40, 40, 40, 90, 90, 90), space_group_symbol="P1")
    m = ligands.build_ligand_from_smiles("c1ccccc1O", "IPH", (12.0, 8.0, 20.0),
                                         crystal_symmetry=cs)
    assert m.get_number_of_atoms() == 13  # phenol C6H5OH, hydrogens included
    assert np.allclose(m.get_sites_cart().mean(), (12.0, 8.0, 20.0), atol=1e-6)
    assert {ag.resname for ag in m.get_hierarchy().atom_groups()} == {"IPH"}
    geo = m.get_restraints_manager().geometry
    assert geo.pair_proxies().bond_proxies.simple.size() == 13
    assert geo.angle_proxies.size() > 0


def test_build_from_smiles_rejects_junk():
    pytest.importorskip("rdkit")
    with pytest.raises(ValueError):
        ligands.build_ligand_from_smiles("not a smiles!!!", "LIG", (0, 0, 0))
    with pytest.raises(ValueError):
        ligands.build_ligand_from_smiles("", "LIG", (0, 0, 0))


def test_smiles_ligand_carries_a_geostd_cif_with_rdkit_provenance():
    """The SMILES ligand keeps the exact restraint CIF that built it — a geostd-style
    monomer file that reparses, and records its rdkit provenance (source SMILES, canonical
    SMILES / InChIKey, and the program) so a saved file says where it came from."""
    pytest.importorskip("rdkit")
    import iotbx.cif

    m = ligands.build_ligand_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "AIN", (0, 0, 0))
    cif = ligands.restraints_cif_text(m)
    assert cif is not None

    # It is a real monomer CIF: reparses, and its comp block carries the restraint loops.
    blocks = iotbx.cif.reader(input_string=cif).model()
    assert "comp_list" in blocks and "comp_AIN" in blocks
    comp = blocks["comp_AIN"]
    assert "_chem_comp_bond.value_dist" in comp and "_chem_comp_angle.value_angle" in comp

    # Provenance: the source SMILES, the standard descriptor block, and the program.
    assert "CC(=O)Oc1ccccc1C(=O)O" in cif
    assert "_pdbx_chem_comp_descriptor" in cif and "SMILES_CANONICAL" in cif
    assert "RDKit" in cif
    # aspirin's InChIKey — proof the recorded structure is the actual molecule
    assert "BSYNRYMUTXBXSQ" in cif


def test_library_ligand_carries_its_geostd_cif():
    """A library ligand carries the geostd file it came from, so it can be saved too."""
    if not ligands.available("GOL"):
        pytest.skip("no monomer library (GOL) available")
    m = ligands.build_ligand_model("GOL", (0, 0, 0))
    cif = ligands.restraints_cif_text(m)
    assert cif is not None and "comp_GOL" in cif.replace("data_comp_GOL", "comp_GOL")


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
