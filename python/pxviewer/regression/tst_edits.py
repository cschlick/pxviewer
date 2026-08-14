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


def exercise_a_phil_round_trips_through_cctbx_unchanged():
    """Read a file, write it back, read it again: same restraints, same values.

    Both directions go through cctbx's own fetch and format, so this also covers the
    fields pxviewer never looks at -- the point of not having an intermediate of our own.
    """
    if not have("iotbx.phil", "mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: iotbx.phil / pdb_interpretation not available")
        return

    text = """
    geometry_restraints.edits {
      bond {
        atom_selection_1 = "chain A and resseq 1 and name SG"
        atom_selection_2 = "chain B and resname LIG and name C7"
        distance_ideal = 1.81
        sigma = 0.02
      }
      angle {
        atom_selection_1 = "chain A and name NE2"
        atom_selection_2 = "chain A and name ZN"
        atom_selection_3 = "chain A and name ND1"
        angle_ideal = 109.5
        sigma = 3.0
      }
      dihedral {
        atom_selection_1 = "name C1"
        atom_selection_2 = "name C2"
        atom_selection_3 = "name C3"
        atom_selection_4 = "name C4"
        angle_ideal = 180.0
        sigma = 20.0
        periodicity = 2
      }
    }
    """
    scope = edits.edits_from_phil(text)
    assert [kind for kind, _obj in edits.entries(scope)] == ["bond", "angle", "dihedral"]

    written = edits.edits_as_phil(scope)
    assert "geometry_restraints" in written
    again = edits.edits_from_phil(written)

    assert edits.count(again) == edits.count(scope)
    assert again.bond[0].atom_selection_1 == "chain A and resseq 1 and name SG"
    assert approx_equal(again.bond[0].distance_ideal, 1.81)
    assert approx_equal(again.angle[0].angle_ideal, 109.5)
    assert again.dihedral[0].periodicity == 2


def exercise_fields_pxviewer_has_no_opinion_about_survive():
    """The reason the PHIL is handed to cctbx rather than parsed into something local.

    ``symmetry_operation`` names a bond to a symmetry mate -- ordinary for a metal site
    at a crystal contact. An intermediate with no field for it silently restrained the
    wrong pair of atoms, and ``slack``, ``limit`` and ``top_out`` vanished the same way.
    """
    if not have("iotbx.phil", "mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: iotbx.phil / pdb_interpretation not available")
        return

    scope = edits.edits_from_phil("""
    geometry_restraints.edits {
      bond {
        atom_selection_1 = "name ZN"
        atom_selection_2 = "name O"
        symmetry_operation = -x-1,-y,z
        distance_ideal = 2.1
        sigma = 0.05
        slack = 0.1
        top_out = True
      }
    }
    """)
    bond = scope.bond[0]
    assert bond.symmetry_operation == "-x-1,-y,z"
    assert approx_equal(bond.slack, 0.1)
    assert bond.top_out is True

    # ... and they are still there after a write/read cycle.
    again = edits.edits_from_phil(edits.edits_as_phil(scope))
    assert again.bond[0].symmetry_operation == "-x-1,-y,z"
    assert approx_equal(again.bond[0].slack, 0.1)


def exercise_planarity_and_parallelity_are_kept_not_counted():
    """They used to be tallied as "unsupported" and dropped. cctbx supports them, so
    handing it the file means pxviewer does too, without writing any code for them."""
    if not have("iotbx.phil", "mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: iotbx.phil / pdb_interpretation not available")
        return

    scope = edits.edits_from_phil("""
    geometry_restraints.edits {
      planarity {
        atom_selection = "chain S and resseq 1"
        sigma = 0.02
      }
      parallelity {
        atom_selection_1 = "chain S and resseq 1"
        atom_selection_2 = "chain S and resseq 2"
        sigma = 0.05
      }
    }
    """)
    kinds = [kind for kind, _obj in edits.entries(scope)]
    assert kinds == ["planarity", "parallelity"]
    assert edits.count(scope) == 2


def exercise_the_refinement_prefix_phenix_writes_is_accepted():
    """phenix.refine writes ``refinement.geometry_restraints.edits``; the master carries
    the matching alias, so the same file works either way round."""
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: pdb_interpretation not available")
        return
    scope = edits.edits_from_phil("""
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
    """)
    assert [kind for kind, _obj in edits.entries(scope)] == ["bond", "planarity"]


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

    scope = edits.empty_edits(model)
    edits.add_entry(
        scope, edits.new_entry(model, "bond", sels, ideal=2.4, sigma=0.02), "bond")
    edits.set_edits(model, scope)
    edits.build_restraints(model, force=True)           # force: an edit changed
    n1 = model.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size()
    assert n1 == n0 + 1

    # A plain (unforced) build reuses that manager -- the edit stays applied.
    edits.build_restraints(model)
    assert model.get_restraints_manager().geometry.pair_proxies(
        ).bond_proxies.simple.size() == n1

    # Clearing (forced) restores the plain build.
    edits.set_edits(model, edits.empty_edits(model))
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
        scope = edits.edits_from_phil(fh.read())
    listing = edits.entries(scope)
    assert len(listing) == 1
    kind, obj = listing[0]
    assert kind == "bond"
    assert approx_equal(obj.distance_ideal, 2.1)
    assert approx_equal(obj.sigma, 0.05)
    assert all("ZN" in s or "name O" in s for s in edits.selections_of(kind, obj))


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
        edits.set_edits(model, edits.edits_from_phil(fh.read(), model))
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
        edits.set_edits(model, edits.edits_from_phil(fh.read(), model))
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
    scope = edits.empty_edits(model)
    edits.add_entry(scope, edits.new_entry(
        model, "bond", ["name NOSUCHATOM", "name ZN"], ideal=2.0, sigma=0.02), "bond")
    edits.set_edits(model, scope)
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
        edits.edits_from_phil("this is not phil {{{")
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
    assert edits.count(edits.edits_from_phil(
        "geometry_restraints.edit { bond { distance_ideal = 2.0 } }")) == 0


def exercise_an_edit_missing_its_sigma_is_refused_by_cctbxs_own_validator():
    """cctbx will not restrain at a weight nobody chose -- but it does not refuse on its
    own either. ``model.process`` **silently skips** an incomplete edit, so the user ends
    up believing a restraint exists that does not.

    ``pdb_interpretation.validate_geometry_edits_params`` is the check phenix runs, and it
    has to be called deliberately. :func:`edits.validate` calls it, and
    :func:`edits.build_restraints` calls that before every build. Using cctbx's function
    rather than a copy of its rules is the whole point -- a copy would drift.
    """
    if not _restraints_ready():
        print("  skipping: no monomer library")
        return
    from pxviewer.cctbx_io import read_model

    model = read_model(data_path("zn_site.pdb"))
    scope = edits.edits_from_phil("""
    geometry_restraints.edits {
      bond {
        atom_selection_1 = "chain S and resseq 1 and name ZN"
        atom_selection_2 = "chain S and resseq 2 and name O"
        distance_ideal = 2.1
      }
    }
    """, model)
    assert edits.count(scope) == 1          # well-formed PHIL: it parses

    try:
        edits.validate(scope)
    except Exception as exc:
        assert "sigma" in str(exc)
        assert "ZN" in str(exc)             # which edit, not just that one was bad
    else:
        raise AssertionError("an edit with no sigma passed validation")

    # And the build refuses rather than skipping it.
    edits.set_edits(model, scope)
    try:
        edits.build_restraints(model, force=True)
    except Exception as exc:
        assert "sigma" in str(exc)
    else:
        raise AssertionError("an edit with no sigma was built")


def exercise_an_edit_missing_its_ideal_value_is_refused_too():
    """The same validator covers a missing ideal distance/angle, which fails the same
    silent way -- so it is worth knowing it is covered without writing the check here."""
    if not _restraints_ready():
        print("  skipping: no monomer library")
        return
    scope = edits.edits_from_phil("""
    geometry_restraints.edits {
      bond {
        atom_selection_1 = "name ZN"
        atom_selection_2 = "name O"
        sigma = 0.05
      }
    }
    """)
    try:
        edits.validate(scope)
    except Exception as exc:
        assert "ideal distance" in str(exc)
    else:
        raise AssertionError("an edit with no ideal distance passed validation")


def exercise_the_old_dict_shape_is_refused_rather_than_ignored():
    """This module once took a list of dicts. A caller still passing that shape got no
    error and no restraint: the value was stored, ``entries`` found no ``bond`` attribute
    on a list, and the build applied nothing.

    That is exactly how it escaped -- a test in another file went on asserting a custom
    bond existed while checking nothing, and only a full-registry run found it.
    """
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("  skipping: pdb_interpretation not available")
        return
    from pxviewer.cctbx_io import read_model

    model = read_model(data_path("zn_site.pdb"))
    try:
        edits.set_edits(model, [{"kind": "bond", "selections": ["name ZN", "name O"],
                                 "ideal": 2.1, "sigma": 0.05}])
    except TypeError as exc:
        assert "scope" in str(exc)
        assert "edits_from_phil" in str(exc)      # says what to do instead
    else:
        raise AssertionError("a list of dicts was accepted as an edits scope")


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
