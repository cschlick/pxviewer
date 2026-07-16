"""Tests for geometry-restraints extraction (needs cctbx + a monomer library)."""

from pathlib import Path

import pytest

pytest.importorskip("iotbx.data_manager")

from pxviewer.geometry import (  # noqa: E402
    CATEGORIES,
    GeometryRestraints,
    build_geometry,
    monomer_library_available,
)

UBIQUITIN = Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb"

needs_monlib = pytest.mark.skipif(
    not monomer_library_available(),
    reason="no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)",
)


def _model():
    from iotbx.data_manager import DataManager

    dm = DataManager()
    dm.process_model_file(str(UBIQUITIN))
    return dm.get_model()


def test_build_geometry_returns_none_without_monomer_library(monkeypatch):
    monkeypatch.delenv("MMTBX_CCP4_MONOMER_LIB", raising=False)
    monkeypatch.delenv("CLIBD_MON", raising=False)
    assert not monomer_library_available()
    assert build_geometry(object()) is None


@needs_monlib
def test_restraint_counts_and_categories():
    geo = GeometryRestraints(_model())
    counts = {cat: geo.count(cat) for cat, _, _ in CATEGORIES}
    # 1UBQ (660 atoms): sensible, nonzero restraint counts in every category.
    assert counts["bond"] > 500
    assert counts["angle"] > counts["bond"]  # more angles than bonds
    assert counts["dihedral"] > 0
    assert counts["chirality"] > 0
    assert counts["planarity"] > 0


@needs_monlib
def test_bond_row_values_are_physical():
    geo = GeometryRestraints(_model())
    iseqs, vals = geo.row("bond", 0)
    assert len(iseqs) == 2  # a bond is two atoms
    assert 1.0 < vals["ideal"] < 2.0  # a covalent bond length, Angstrom
    assert 1.0 < vals["model"] < 2.0
    assert vals["sigma"] > 0
    # cctbx convention: delta = ideal - model
    assert vals["delta"] == pytest.approx(vals["ideal"] - vals["model"], abs=1e-4)


@needs_monlib
def test_indices_within_selection():
    geo = GeometryRestraints(_model())
    selected = set(geo.row("bond", 0)[0])  # the two atoms of the first bond

    idx = geo.indices_within("bond", selected)
    assert 0 in idx  # that bond is within its own atoms
    # every returned restraint has all its atoms in the selection
    for i in idx:
        assert all(s in selected for s in geo.row("bond", i)[0])
    # an empty selection matches nothing
    assert geo.indices_within("bond", set()) == []


@needs_monlib
def test_row_arities_match_restraint_type():
    geo = GeometryRestraints(_model())
    assert len(geo.row("angle", 0)[0]) == 3
    assert len(geo.row("dihedral", 0)[0]) == 4
    assert len(geo.row("chirality", 0)[0]) == 4
    assert len(geo.row("planarity", 0)[0]) >= 4  # a plane is >= 4 atoms
    # planarity exposes rms/max deltas rather than an ideal/model pair
    assert set(geo.row("planarity", 0)[1]) == {"rms_delta", "max_delta", "residual"}
