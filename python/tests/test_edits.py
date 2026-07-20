"""Read/write/apply of cctbx geometry_restraints.edits (custom restraints)."""

import math

import pytest

from pxviewer import edits


def test_geometry_value_distance_angle_dihedral():
    # a right angle at the origin, unit arms — no cctbx needed
    assert edits.geometry_value("bond", [(0, 0, 0), (3, 4, 0)]) == pytest.approx(5.0)
    assert edits.geometry_value("angle", [(1, 0, 0), (0, 0, 0), (0, 1, 0)]) == pytest.approx(90.0)
    # a classic +90 deg dihedral
    d = edits.geometry_value("dihedral", [(1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)])
    assert abs(abs(d) - 90.0) < 1e-6


def test_serialize_parse_round_trip():
    pytest.importorskip("iotbx.phil")
    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")

    original = [
        {"kind": "bond", "action": "add",
         "selections": ["chain A and resseq 1 and name SG", "chain B and resname LIG and name C7"],
         "ideal": 1.81, "sigma": 0.02},
        {"kind": "angle", "action": "add",
         "selections": ["chain A and name NE2", "chain A and name ZN", "chain A and name ND1"],
         "ideal": 109.5, "sigma": 3.0},
        {"kind": "dihedral", "action": "add",
         "selections": ["name C1", "name C2", "name C3", "name C4"],
         "ideal": 180.0, "sigma": 20.0, "periodicity": 2},
    ]
    text = edits.edits_to_phil(original)
    assert "geometry_restraints.edits" in text
    parsed, unsupported = edits.parse_edits(text)
    assert unsupported == 0
    assert [e["kind"] for e in parsed] == ["bond", "angle", "dihedral"]
    assert parsed[0]["selections"] == original[0]["selections"]
    assert parsed[0]["ideal"] == pytest.approx(1.81)
    assert parsed[2]["periodicity"] == 2


def test_parse_tolerates_phenix_refinement_prefix_and_flags_unsupported():
    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")
    text = """
    refinement.geometry_restraints.edits {
      bond {
        atom_selection_1 = "name A"
        atom_selection_2 = "name B"
        distance_ideal = 2.0
        sigma = 0.02
      }
      planarity {
        atom_selection = "name A or name B"
        sigma = 0.02
      }
    }
    """
    parsed, unsupported = edits.parse_edits(text)
    assert len(parsed) == 1 and parsed[0]["kind"] == "bond"
    assert unsupported == 1  # the planarity edit is counted, not silently dropped


def test_build_restraints_applies_a_custom_bond():
    pytest.importorskip("rdkit")
    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")
    from pxviewer import ligands
    from pxviewer.geometry import monomer_library_available
    if not monomer_library_available():
        pytest.skip("no monomer library")

    model = ligands.build_ligand_from_smiles("CCO", "EOH", (0, 0, 0))
    grm = model.get_restraints_manager().geometry
    n0 = grm.pair_proxies().bond_proxies.simple.size()

    # C1 and O1 are 1-3 (not bonded); add a bond between them
    names = [a.name.strip() for a in model.get_hierarchy().atoms()]
    sels = [edits.selection_for_atom(model, names.index("C1")),
            edits.selection_for_atom(model, names.index("O1"))]
    # each selection names exactly one atom
    cache = model.get_atom_selection_cache()
    assert cache.selection(sels[0]).count(True) == 1

    edits.set_edits(model, [{"kind": "bond", "selections": sels, "ideal": 2.4, "sigma": 0.02}])
    edits.build_restraints(model, force=True)  # force: an edit changed
    n1 = model.get_restraints_manager().geometry.pair_proxies().bond_proxies.simple.size()
    assert n1 == n0 + 1

    # a plain (unforced) build reuses that manager — the edit stays applied
    edits.build_restraints(model)
    assert model.get_restraints_manager().geometry.pair_proxies().bond_proxies.simple.size() == n1

    # clearing (forced) restores the plain build
    edits.set_edits(model, [])
    edits.build_restraints(model, force=True)
    assert model.get_restraints_manager().geometry.pair_proxies().bond_proxies.simple.size() == n0
