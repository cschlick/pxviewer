"""Geometry-restraints extraction (needs cctbx and a monomer library)."""

from __future__ import absolute_import, division, print_function

import os
import sys

from libtbx.test_utils import approx_equal

from pxviewer.regression.tst_utils import data_path, have, skip, tmp_dir

if not have("iotbx.data_manager"):
    skip("iotbx.data_manager not available")

from pxviewer.geometry import (CATEGORIES, GeometryRestraints,      # noqa: E402
                               build_geometry, geostd_monomer_path,
                               monomer_library_available, monomer_library_root)

_cache = []


def model():
    """1UBQ, read once per process. Shared, so do not mutate it."""
    if not _cache:
        from iotbx.data_manager import DataManager

        dm = DataManager()
        dm.process_model_file(data_path("1ubq.pdb"))
        _cache.append(dm.get_model())
    return _cache[0]


def have_monlib():
    if not monomer_library_available():
        print("  skipping: no monomer library "
              "(set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")
        return False
    return True


def exercise_build_geometry_refuses_nothing_to_build_from():
    """The guard's reachable half: no model, no restraints, and no exception.

    The pytest original also covered the *other* half -- that a missing monomer library
    yields None -- by deleting two environment variables and replacing
    ``geometry._chem_data_geostd`` with a stub. That branch is not reproducible here
    without patching: chem_data is installed and importable, so clearing the environment
    alone still finds geostd. Rather than reintroduce patching for one case, the
    unreachable branch is left uncovered and said so. `geostd_monomer_path(None, ...)`
    below covers the closest thing that can be exercised honestly.
    """
    assert build_geometry(None) is None


def exercise_restraint_counts_and_categories():
    if not have_monlib():
        return
    geo = GeometryRestraints(model())
    counts = dict((cat, geo.count(cat)) for cat, _, _ in CATEGORIES)
    # 1UBQ (660 atoms): sensible, nonzero restraint counts in every category.
    assert counts["bond"] > 500
    assert counts["angle"] > counts["bond"]        # more angles than bonds
    assert counts["dihedral"] > 0
    assert counts["chirality"] > 0
    assert counts["planarity"] > 0


def exercise_bond_row_values_are_physical():
    if not have_monlib():
        return
    geo = GeometryRestraints(model())
    iseqs, vals = geo.row("bond", 0)
    assert len(iseqs) == 2                          # a bond is two atoms
    assert 1.0 < vals["ideal"] < 2.0                # a covalent bond length, angstrom
    assert 1.0 < vals["model"] < 2.0
    assert vals["sigma"] > 0
    # cctbx convention: delta = ideal - model
    assert approx_equal(vals["delta"], vals["ideal"] - vals["model"], eps=1e-4)


def exercise_geostd_monomer_path_resolves():
    if not have_monlib():
        return
    root = monomer_library_root()
    ala = geostd_monomer_path(root, "ALA")
    assert ala is not None and ala.endswith("a/data_ALA.cif") and os.path.isfile(ala)
    assert geostd_monomer_path(root, "MET").endswith("m/data_MET.cif")
    # No file in an empty root, and no library at all -> None.
    with tmp_dir() as work:
        assert geostd_monomer_path(work, "ALA") is None
    assert geostd_monomer_path(None, "ALA") is None


def exercise_indices_within_selection():
    if not have_monlib():
        return
    geo = GeometryRestraints(model())
    selected = set(geo.row("bond", 0)[0])           # the two atoms of the first bond

    idx = geo.indices_within("bond", selected)
    assert 0 in idx                                  # that bond is within its own atoms
    for i in idx:                                    # all atoms of every hit are selected
        assert all(s in selected for s in geo.row("bond", i)[0])
    assert geo.indices_within("bond", set()) == []   # an empty selection matches nothing


def exercise_row_arities_match_restraint_type():
    if not have_monlib():
        return
    geo = GeometryRestraints(model())
    assert len(geo.row("angle", 0)[0]) == 3
    assert len(geo.row("dihedral", 0)[0]) == 4
    assert len(geo.row("chirality", 0)[0]) == 4
    assert len(geo.row("planarity", 0)[0]) >= 4      # a plane is >= 4 atoms
    # planarity exposes rms/max deltas rather than an ideal/model pair
    assert set(geo.row("planarity", 0)[1]) == set(["rms_delta", "max_delta", "residual"])


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
