"""Read/write/apply of cctbx geometry_restraints.edits (custom restraints)."""

from __future__ import absolute_import, division, print_function

import sys

from libtbx.test_utils import approx_equal

from pxviewer import edits
from pxviewer.regression.tst_utils import data_path, have


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


# -- a real PHIL file, the way a user supplies one -----------------------------
#
# Everything above works on text built in the test or on a synthetic ligand. These start
# from a file on disk, because that is how the feature is actually used: someone writes a
# PHIL, points the app at it, and expects the restraint to exist afterwards.


def _restraints_ready():
    """The monomer library and pdb_interpretation, without which none of this can build."""
    if not have("mmtbx.monomer_library.pdb_interpretation", "iotbx.data_manager"):
        return False
    from pxviewer.geometry import monomer_library_available

    return monomer_library_available()


def exercise_the_shipped_edits_phil_parses():
    """``zn_site_edits.phil`` ships next to the model it belongs to and is offered by the
    Examples menu, so a typo in it would be a broken demo rather than a broken test."""
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: pdb_interpretation not available")
        return
    with open(data_path("zn_site_edits.phil")) as fh:
        parsed, unsupported = edits.parse_edits(fh.read())
    assert unsupported == 0
    assert len(parsed) == 1
    edit = parsed[0]
    assert edit["kind"] == "bond"
    assert approx_equal(edit["ideal"], 2.1)
    assert approx_equal(edit["sigma"], 0.05)
    assert all("ZN" in s or "name O" in s for s in edit["selections"])


def exercise_the_shipped_edits_phil_applies_to_the_model_it_names():
    """The selections have to match *that* file's atoms. A selection naming nothing is
    not a quiet no-op -- cctbx raises -- so this also proves the pair belong together."""
    if not _restraints_ready():
        print("  skipping: no monomer library")
        return
    from pxviewer.cctbx_io import read_model

    model = read_model(data_path("zn_site.pdb"))
    edits.build_restraints(model, force=True)
    before = model.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size()

    with open(data_path("zn_site_edits.phil")) as fh:
        parsed, _unsupported = edits.parse_edits(fh.read())
    edits.set_edits(model, parsed)
    edits.build_restraints(model, force=True)

    after = model.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size()
    assert after == before + 1


def exercise_a_restraint_from_a_phil_is_marked_as_user_supplied():
    """cctbx tags it with the ``edits`` origin, which is what lets the Geometry tab pull
    the user's own restraints out from under the thousands the library contributes."""
    if not _restraints_ready():
        print("  skipping: no monomer library")
        return
    from pxviewer.cctbx_io import read_model
    from pxviewer.geometry import GeometryRestraints, origin_name

    model = read_model(data_path("zn_site.pdb"))
    with open(data_path("zn_site_edits.phil")) as fh:
        parsed, _ = edits.parse_edits(fh.read())
    edits.set_edits(model, parsed)
    edits.build_restraints(model, force=True)

    geo = GeometryRestraints(model)
    origins = dict((oid, count) for oid, _name, count in geo.origins("bond"))
    edits_origin = _edits_origin_id()
    assert origins.get(edits_origin) == 1
    assert "user-defined" in origin_name(edits_origin)

    # And it is reachable as exactly one row.
    indices = geo.indices_with_origin("bond", edits_origin)
    assert len(indices) == 1
    _i_seqs, values = geo.row("bond", indices[0])
    assert approx_equal(values["ideal"], 2.1)


def _edits_origin_id():
    from cctbx.geometry_restraints.linking_class import linking_class

    return linking_class().get_origin_id("edits")


def exercise_a_selection_that_names_no_atom_is_refused():
    """The commonest mistake in a hand-written PHIL. It must fail loudly: a restraint the
    user believes is holding two atoms together, silently absent, is worse than an error.
    """
    if not _restraints_ready():
        print("  skipping: no monomer library")
        return
    from pxviewer.cctbx_io import read_model

    model = read_model(data_path("zn_site.pdb"))
    edits.set_edits(model, [{"kind": "bond", "sigma": 0.02, "ideal": 2.0,
                             "selections": ["name NOSUCHATOM", "name ZN"]}])
    try:
        edits.build_restraints(model, force=True)
    except Exception as exc:
        assert "NOSUCHATOM" in str(exc)          # says which selection, not just "failed"
    else:
        raise AssertionError("a selection matching no atom was accepted")


def exercise_malformed_phil_is_reported_rather_than_ignored():
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: pdb_interpretation not available")
        return
    try:
        edits.parse_edits("this is not phil {{{")
    except Exception as exc:
        assert "Syntax error" in str(exc)
    else:
        raise AssertionError("malformed PHIL parsed without complaint")


def exercise_a_misspelled_scope_yields_nothing_without_complaining():
    """PHIL ignores a scope it does not recognise, so ``edit`` for ``edits`` parses
    cleanly and produces no restraints at all -- no exception, no warning.

    Pinned because it is silent, and silence here is expensive: the user believes they
    supplied a restraint. What saves them is one layer up -- ``DesktopApp.load_edits``
    treats "nothing parsed and nothing skipped" as an error and says so.
    """
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: pdb_interpretation not available")
        return
    assert edits.parse_edits(
        "geometry_restraints.edit { bond { distance_ideal = 2.0 } }") == ([], 0)


def exercise_a_bond_without_a_sigma_gets_the_default_one():
    """Not a rejection -- a default, and a fairly tight one at 0.02 A.

    Worth stating explicitly because the alternative guess is that the edit is dropped.
    It is not: the restraint exists and is enforced at a weight the user never chose, so
    anyone reading a PHIL to find out what was restrained needs to know the sigma may not
    be written down in it.
    """
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: pdb_interpretation not available")
        return
    parsed, unsupported = edits.parse_edits("""
    geometry_restraints.edits {
      bond {
        atom_selection_1 = "name A"
        atom_selection_2 = "name B"
        distance_ideal = 2.0
      }
    }
    """)
    assert unsupported == 0
    assert len(parsed) == 1
    assert approx_equal(parsed[0]["ideal"], 2.0)
    assert approx_equal(parsed[0]["sigma"], 0.02)


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
