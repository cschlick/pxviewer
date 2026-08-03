"""Read/write/apply of cctbx geometry_restraints.edits (custom restraints)."""

from __future__ import absolute_import, division, print_function

import sys

from libtbx.test_utils import approx_equal

from pxviewer import edits
from pxviewer.regression.tst_utils import have


def exercise_geometry_value_distance_angle_dihedral():
    """A right angle at the origin with unit arms -- no cctbx needed."""
    assert approx_equal(edits.geometry_value("bond", [(0, 0, 0), (3, 4, 0)]), 5.0)
    assert approx_equal(
        edits.geometry_value("angle", [(1, 0, 0), (0, 0, 0), (0, 1, 0)]), 90.0)
    # A classic +90 degree dihedral.
    d = edits.geometry_value("dihedral",
                             [(1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)])
    assert abs(abs(d) - 90.0) < 1e-6


def exercise_serialize_parse_round_trip():
    if not have("iotbx.phil", "mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: iotbx.phil / pdb_interpretation not available")
        return

    original = [
        {"kind": "bond", "action": "add",
         "selections": ["chain A and resseq 1 and name SG",
                        "chain B and resname LIG and name C7"],
         "ideal": 1.81, "sigma": 0.02},
        {"kind": "angle", "action": "add",
         "selections": ["chain A and name NE2", "chain A and name ZN",
                        "chain A and name ND1"],
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
    assert approx_equal(parsed[0]["ideal"], 1.81)
    assert parsed[2]["periodicity"] == 2


def exercise_parse_tolerates_a_refinement_prefix_and_flags_unsupported():
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: pdb_interpretation not available")
        return
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
    assert unsupported == 1     # the planarity edit is counted, not silently dropped


def exercise_build_restraints_applies_a_custom_bond():
    if not have("rdkit", "mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: rdkit / pdb_interpretation not available")
        return
    from pxviewer import ligands
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        print("  skipping: no monomer library")
        return

    model = ligands.build_ligand_from_smiles("CCO", "EOH", (0, 0, 0))
    grm = model.get_restraints_manager().geometry
    n0 = grm.pair_proxies().bond_proxies.simple.size()

    # C1 and O1 are 1-3 (not bonded); add a bond between them.
    names = [a.name.strip() for a in model.get_hierarchy().atoms()]
    sels = [edits.selection_for_atom(model, names.index("C1")),
            edits.selection_for_atom(model, names.index("O1"))]
    cache = model.get_atom_selection_cache()
    assert cache.selection(sels[0]).count(True) == 1    # each names exactly one atom

    edits.set_edits(model, [{"kind": "bond", "selections": sels,
                             "ideal": 2.4, "sigma": 0.02}])
    edits.build_restraints(model, force=True)           # force: an edit changed
    n1 = model.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size()
    assert n1 == n0 + 1

    # A plain (unforced) build reuses that manager -- the edit stays applied.
    edits.build_restraints(model)
    assert model.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size() == n1

    # Clearing (forced) restores the plain build.
    edits.set_edits(model, [])
    edits.build_restraints(model, force=True)
    assert model.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size() == n0


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
