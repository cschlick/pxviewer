"""Building a monomer-library ligand (pxviewer.ligands).

The fit itself (explode-and-refine) is slow and stochastic, so it is proven by hand rather
than here; these cover the fast, deterministic build path -- reading geostd's ideal
coordinates and turning them into a restraint-ready, correctly-placed model.
"""

from __future__ import absolute_import, division, print_function

import sys

from libtbx.test_utils import raises

from pxviewer.regression.tst_utils import have, skip

if not have("iotbx.data_manager", "numpy"):
    skip("iotbx.data_manager / numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer import ligands                         # noqa: E402


def exercise_availability():
    assert ligands.available("GOL")
    assert ligands.available("gol")             # case-insensitive
    assert not ligands.available("NOTACODE")    # no such component
    assert not ligands.available("")


def exercise_ideal_atoms_have_coordinates():
    names, elements, xyz = ligands.ideal_atoms("GOL")
    assert len(names) == len(elements) == xyz.shape[0] == 14
    assert xyz.shape[1] == 3


def exercise_build_model_is_centred_and_restraint_ready():
    m = ligands.build_ligand_model("GOL", (12.0, 8.0, 20.0))
    assert m.get_number_of_atoms() == 14
    assert np.allclose(m.get_sites_cart().mean(), (12.0, 8.0, 20.0), atol=1e-6)
    assert m.restraints_manager_available()
    assert set(ag.resname for ag in m.get_hierarchy().atom_groups()) == set(["GOL"])
    # Restraints resolved from the same library the coordinates came from.
    assert m.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size() > 0


def exercise_an_unknown_code_raises():
    with raises(ValueError):
        ligands.ideal_atoms("NOTACODE")


def exercise_build_from_smiles_is_centred_and_restraint_ready():
    """A ligand not in the library, built from SMILES: rdkit embeds a conformer whose
    geometry both places the atoms and supplies the on-the-fly restraints, so the model
    comes out centred and with a real (bond + angle) geometry restraints manager."""
    if not have("rdkit"):
        print("  skipping: rdkit not available")
        return
    from cctbx import crystal

    cs = crystal.symmetry(unit_cell=(40, 40, 40, 90, 90, 90), space_group_symbol="P1")
    m = ligands.build_ligand_from_smiles("c1ccccc1O", "IPH", (12.0, 8.0, 20.0),
                                         crystal_symmetry=cs)
    assert m.get_number_of_atoms() == 13        # phenol C6H5OH, hydrogens included
    assert np.allclose(m.get_sites_cart().mean(), (12.0, 8.0, 20.0), atol=1e-6)
    assert set(ag.resname for ag in m.get_hierarchy().atom_groups()) == set(["IPH"])
    geo = m.get_restraints_manager().geometry
    assert geo.pair_proxies().bond_proxies.simple.size() == 13
    assert geo.angle_proxies.size() > 0


def exercise_build_from_smiles_rejects_junk():
    if not have("rdkit"):
        print("  skipping: rdkit not available")
        return
    with raises(ValueError):
        ligands.build_ligand_from_smiles("not a smiles!!!", "LIG", (0, 0, 0))
    with raises(ValueError):
        ligands.build_ligand_from_smiles("", "LIG", (0, 0, 0))


def exercise_a_smiles_ligand_carries_a_cif_with_rdkit_provenance():
    """The SMILES ligand keeps the exact restraint CIF that built it -- a geostd-style
    monomer file that reparses, and records its rdkit provenance (source SMILES, canonical
    SMILES / InChIKey, and the program) so a saved file says where it came from."""
    if not have("rdkit"):
        print("  skipping: rdkit not available")
        return
    import iotbx.cif

    m = ligands.build_ligand_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "AIN", (0, 0, 0))
    cif = ligands.restraints_cif_text(m)
    assert cif is not None

    # A real monomer CIF: it reparses, and its comp block carries the restraint loops.
    blocks = iotbx.cif.reader(input_string=cif).model()
    assert "comp_list" in blocks and "comp_AIN" in blocks
    comp = blocks["comp_AIN"]
    assert "_chem_comp_bond.value_dist" in comp
    assert "_chem_comp_angle.value_angle" in comp

    # Provenance: the source SMILES, the standard descriptor block, and the program.
    assert "CC(=O)Oc1ccccc1C(=O)O" in cif
    assert "_pdbx_chem_comp_descriptor" in cif and "SMILES_CANONICAL" in cif
    assert "RDKit" in cif
    # Aspirin's InChIKey -- proof the recorded structure is the actual molecule.
    assert "BSYNRYMUTXBXSQ" in cif


def exercise_a_library_ligand_carries_its_geostd_cif():
    """A library ligand carries the geostd file it came from, so it can be saved too."""
    if not ligands.available("GOL"):
        print("  skipping: no monomer library (GOL) available")
        return
    m = ligands.build_ligand_model("GOL", (0, 0, 0))
    cif = ligands.restraints_cif_text(m)
    assert cif is not None and "comp_GOL" in cif.replace("data_comp_GOL", "comp_GOL")


def exercise_coarse_orient_recovers_a_bad_orientation():
    """The rotational pre-search rotates a mis-oriented rigid ligand back toward its
    density -- deterministic and fast, so it belongs here (unlike the full fit)."""
    from cctbx import crystal
    from cctbx.array_family import flex
    from scitbx.math import euler_angles

    cs = crystal.symmetry(unit_cell=(40, 40, 40, 90, 90, 90), space_group_symbol="P1")
    target = ligands.build_ligand_model("GOL", (20, 20, 20), crystal_symmetry=cs)
    tgt = target.get_sites_cart().as_numpy_array()
    map_data = target.get_xray_structure().structure_factors(
        d_min=2.0).f_calc().fft_map(
        resolution_factor=0.25).apply_sigma_scaling().real_map_unpadded()

    mis = ligands.build_ligand_model("GOL", (20, 20, 20), crystal_symmetry=cs)
    rot = np.array(euler_angles.xyz_matrix(120, 80, 40)).reshape(3, 3)
    c = mis.get_sites_cart().as_numpy_array().mean(0)
    mis.set_sites_cart(flex.vec3_double(np.ascontiguousarray(
        (mis.get_sites_cart().as_numpy_array() - c) @ rot.T + c)))

    before = np.sqrt(((mis.get_sites_cart().as_numpy_array() - tgt) ** 2).sum(1).mean())
    ligands.coarse_orient(mis, map_data, step_deg=30)
    after = np.sqrt(((mis.get_sites_cart().as_numpy_array() - tgt) ** 2).sum(1).mean())
    assert after < before / 2, \
        "pre-search did not improve orientation: %.2f -> %.2f" % (before, after)


# -- ligands the monomer library does not know --------------------------------


def _unknown_ligand_model(smiles="CC(=O)Oc1ccccc1C(=O)O", code="L01", scale=1.0):
    """A model of ``smiles`` filed under a residue name cctbx has no dictionary for.

    Built rather than shipped as a file: the point is a residue the monomer library cannot
    type, and a fixture that stopped being unknown -- because the code was added to
    geostd -- would quietly stop testing anything.
    """
    import tempfile
    import os

    from pxviewer.cctbx_io import read_model

    model = ligands.build_ligand_from_smiles(smiles, "AIN", (0, 0, 0))
    hierarchy = model.get_hierarchy()
    for group in hierarchy.atom_groups():
        group.resname = code
    if scale != 1.0:                     # a badly built ligand: every bond stretched
        for atom in hierarchy.atoms():
            atom.set_xyz(tuple(scale * np.array(atom.xyz)))
    directory = tempfile.mkdtemp(prefix="pxviewer_unknown_")
    path = os.path.join(directory, "unknown.pdb")
    with open(path, "w") as handle:
        handle.write(hierarchy.as_pdb_string(crystal_symmetry=model.crystal_symmetry()))
    return read_model(path)


def _restraints_ready():
    from pxviewer.geometry import monomer_library_available

    return have("rdkit", "mmtbx.monomer_library.pdb_interpretation") \
        and monomer_library_available()


def exercise_an_unrecognised_ligand_is_reported_with_all_of_its_atoms():
    """cctbx flags only the atoms it could not type -- often the heavy ones, their
    hydrogens having typed fine -- but a dictionary has to describe the whole residue, and
    perceiving chemistry from part of a molecule fails outright. So the report widens each
    flagged atom to its residue.
    """
    if not _restraints_ready():
        print("  skipping: rdkit / monomer library not available")
        return
    model = _unknown_ligand_model()
    found = ligands.unknown_ligands(model)
    assert len(found) == 1
    assert found[0]["code"] == "L01"
    assert found[0]["n_atoms"] == model.get_number_of_atoms()   # all 21, not the 14 flagged


def exercise_a_model_cctbx_understands_reports_nothing():
    if not _restraints_ready():
        print("  skipping: rdkit / monomer library not available")
        return
    from pxviewer.regression.tst_utils import data_path
    from pxviewer.cctbx_io import read_model

    assert ligands.unknown_ligands(read_model(data_path("zn_site.pdb"))) == []


def exercise_restraints_are_read_back_out_of_the_coordinates():
    if not _restraints_ready():
        print("  skipping: rdkit / monomer library not available")
        return
    model = _unknown_ligand_model()
    found = ligands.unknown_ligands(model)
    cif_text, smiles = ligands.restraints_from_residue(
        model, found[0]["i_seqs"], found[0]["code"])

    from rdkit import Chem
    assert smiles == Chem.CanonSmiles("CC(=O)Oc1ccccc1C(=O)O")   # aspirin, recovered
    assert "comp_L01" in cif_text
    # The header must say the SMILES was *perceived*, not supplied: a reader deciding
    # whether to trust these restraints needs to know it is a guess.
    assert "Perceived SMILES" in cif_text
    assert "perceived from the modelled coordinates" in cif_text
    assert "Source SMILES" not in cif_text


def exercise_the_ideals_come_from_clean_geometry_not_the_model():
    """The property that makes this worth doing at all.

    A ligand needing restraints is usually one that is modelled badly. Measuring its
    ideals off its own coordinates would restrain it to the distortion -- the restraints
    would hold the error in place instead of pulling it out. So the ideals are measured
    from a fresh conformer of the perceived chemistry.
    """
    if not _restraints_ready():
        print("  skipping: rdkit / monomer library not available")
        return
    import iotbx.cif

    model = _unknown_ligand_model(smiles="CCO", code="L02", scale=1.25)
    found = ligands.unknown_ligands(model)
    cif_text, _smiles = ligands.restraints_from_residue(
        model, found[0]["i_seqs"], found[0]["code"])

    positions = {a.name.strip(): np.array(a.xyz) for a in model.get_hierarchy().atoms()}
    modelled = float(np.linalg.norm(positions["C1"] - positions["C2"]))
    assert modelled > 1.8                          # stretched by a quarter, as built

    block = iotbx.cif.reader(input_string=cif_text).model()["comp_L02"]
    ideal = next(float(d) for a1, a2, d in zip(block["_chem_comp_bond.atom_id_1"],
                                               block["_chem_comp_bond.atom_id_2"],
                                               block["_chem_comp_bond.value_dist"])
                 if {a1, a2} == {"C1", "C2"})
    assert 1.45 < ideal < 1.60                     # an ordinary C-C, not the 1.89 modelled


def exercise_a_generated_dictionary_makes_the_model_buildable():
    """The whole point: a model that had no restraints at all now has them, and the
    ligand stops being reported as unknown."""
    if not _restraints_ready():
        print("  skipping: rdkit / monomer library not available")
        return
    from libtbx.utils import Sorry

    model = _unknown_ligand_model()
    try:
        model.process(make_restraints=True)
    except Sorry:
        pass
    else:
        raise AssertionError("the fixture ligand was not unknown after all")

    found = ligands.unknown_ligands(model)
    cif_text, _smiles = ligands.restraints_from_residue(
        model, found[0]["i_seqs"], found[0]["code"])
    ligands.apply_generated_restraints(model, {found[0]["code"]: cif_text})
    model.process(make_restraints=True)

    geometry = model.get_restraints_manager().geometry
    assert geometry.pair_proxies().bond_proxies.simple.size() > 15
    assert geometry.angle_proxies.size() > 15
    # And the detector agrees, which it only can if it registers the carried dictionary.
    assert ligands.unknown_ligands(model) == []


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
