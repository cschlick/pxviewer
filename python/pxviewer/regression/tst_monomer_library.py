"""The monomer library resolves *both* halves of chem_data, not just geostd.

chem_data ships two monomer libraries: ``geostd`` (~54k monomers) and ``mon_lib`` (~140
CCP4-derived ones geostd does not duplicate). They carry *different*
``list/mon_lib_list.cif`` indices, and cctbx's ``find_mon_lib_file`` consults
``MMTBX_CCP4_MONOMER_LIB`` — a single-directory redirect — before its own cascade. So
exporting that variable to geostd makes cctbx read geostd's index and never look in
mon_lib: ALA and 54k other monomers keep working while HEM silently stops resolving.
pxviewer used to export exactly that, from an ``activate.d`` hook.

The checks below are deliberately two-pronged, because either one alone can pass while
the path is broken:

* HEM must resolve and build restraints. This is the user-visible symptom.
* mon_lib must be on the search path. If HEM is ever moved into geostd upstream, the
  first check would pass over a still-broken path — every *other* mon_lib-only monomer
  would remain unreachable — so the path itself is asserted directly.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

from pxviewer.regression.tst_utils import data_path, have, skip

if not have("iotbx.data_manager"):
    skip("iotbx.data_manager not available")

import pxviewer  # noqa: F401, E402  (import configures the monomer library)
from pxviewer.geometry import (build_geometry,                       # noqa: E402
                               monomer_cif_path, monomer_library_available,
                               monomer_library_roots)


def have_monlib():
    if not monomer_library_available():
        print("  skipping: no monomer library (chem_data not importable)")
        return False
    return True


def using_chem_data():
    """Whether the roots come from chem_data rather than an external override.

    An external ``MMTBX_CCP4_MONOMER_LIB`` (a bare geostd checkout, say) legitimately
    has no mon_lib, so the path assertions below do not apply to it.
    """
    return any(os.path.join("chem_data", "") in r for r in monomer_library_roots())


# --- prong 1: HEM resolves -----------------------------------------------------------

def exercise_hem_is_in_the_monomer_library():
    """The server finds HEM's monomer definition, wherever it ships."""
    if not have_monlib():
        return
    from mmtbx.monomer_library import server

    srv = server.server()
    assert srv.get_comp_comp_id_direct("ALA") is not None, "geostd unreachable"
    assert srv.get_comp_comp_id_direct("HEM") is not None, (
        "HEM did not resolve -- the mon_lib half of chem_data is not being searched")


def exercise_hem_has_a_cif_on_disk():
    if not have_monlib():
        return
    path = monomer_cif_path("HEM")
    assert path and os.path.isfile(path), "no HEM cif found in any monomer-library root"
    assert monomer_cif_path("ALA"), "no ALA cif found"
    # The two libraries name their files differently (data_ALA.cif vs HEM.cif); a lookup
    # that only understood geostd's convention would miss this even with mon_lib on path.
    assert os.path.basename(path) in ("HEM.cif", "data_HEM.cif")


def exercise_restraints_build_for_a_heme_fragment():
    """End to end: the symptom a user actually hits.

    Pinned to geostd this raises "unknown nonbonded energy type symbols: 43" rather
    than returning something empty, so a partial result cannot pass for success.
    """
    if not have_monlib():
        return
    from iotbx.data_manager import DataManager

    dm = DataManager()
    dm.process_model_file(data_path("heme_site.pdb"))
    model = dm.get_model()
    assert model.get_number_of_atoms() == 43

    geo = build_geometry(model)
    assert geo is not None, "no geometry restraints for the heme fragment"
    assert geo.count("bond") > 40, "too few bonds: %d" % geo.count("bond")
    assert geo.count("angle") > 60, "too few angles: %d" % geo.count("angle")


# --- prong 2: mon_lib is on the search path ------------------------------------------

def exercise_mon_lib_is_among_the_roots():
    """Asserted independently of HEM, so moving HEM upstream cannot mask a bad path."""
    if not have_monlib() or not using_chem_data():
        return
    roots = monomer_library_roots()
    names = [os.path.basename(r.rstrip(os.sep)) for r in roots]
    assert "geostd" in names, "geostd missing from the roots: %r" % (names,)
    assert "mon_lib" in names, "mon_lib missing from the roots: %r" % (names,)
    for root in roots:
        assert os.path.isdir(root), "root does not exist: %s" % root


def exercise_cctbx_reaches_mon_lib_itself():
    """cctbx's own resolution -- not just pxviewer's view of it -- must see mon_lib.

    pxviewer can list mon_lib in its roots and cctbx still miss it, because
    pdb_interpretation resolves paths through ``find_mon_lib_file``, not through us.
    """
    if not have_monlib() or not using_chem_data():
        return
    from mmtbx.monomer_library import server

    found = server.find_mon_lib_file(relative_path_components=["h", "HEM.cif"])
    assert found and os.path.isfile(found), "cctbx cannot resolve a file inside mon_lib"
    assert "mon_lib" in found, "resolved outside mon_lib: %s" % found


def exercise_the_search_is_not_pinned_to_one_directory():
    """The regression guard: no redirect narrowing cctbx to a single chem_data dir.

    Importing pxviewer must leave MMTBX_CCP4_MONOMER_LIB either unset or pointing at a
    genuine external library -- never at chem_data's own geostd, which is what the old
    activate.d hook wrote and what hides mon_lib.
    """
    if not have_monlib() or not using_chem_data():
        return
    for var in ("MMTBX_CCP4_MONOMER_LIB", "CLIBD_MON"):
        value = os.environ.get(var)
        if not value:
            continue
        assert os.path.join("chem_data", "geostd") not in value, (
            "%s pins the search to chem_data's geostd, hiding mon_lib: %s" % (var, value))


def exercise_a_stale_redirect_does_not_survive_configuration():
    """An env still carrying the old hook self-heals rather than staying half-broken."""
    if not have_monlib() or not using_chem_data():
        return
    from pxviewer.geometry import configure_monomer_library

    saved = os.environ.get("MMTBX_CCP4_MONOMER_LIB")
    try:
        os.environ["MMTBX_CCP4_MONOMER_LIB"] = "/nonexistent/python3.13/chem_data/geostd"
        configure_monomer_library()
        assert os.environ.get("MMTBX_CCP4_MONOMER_LIB") != \
            "/nonexistent/python3.13/chem_data/geostd", "stale redirect left in place"
        assert monomer_cif_path("HEM"), "HEM unreachable after a stale redirect"
    finally:
        if saved is None:
            os.environ.pop("MMTBX_CCP4_MONOMER_LIB", None)
        else:
            os.environ["MMTBX_CCP4_MONOMER_LIB"] = saved


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
