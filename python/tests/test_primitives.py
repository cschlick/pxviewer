"""Unit tests for graphics primitives (angle/distance/dihedral/label) and the
Selection plumbing they build on. These are pure Python — no viewer required:
``add_*`` computes the measured value locally and records the wire message.
"""

import math

import numpy as np
import pytest

from pxviewer import LiveSession, Primitive, Selection


def _session(coords):
    return LiveSession.from_sites(coords)


@pytest.fixture
def session():
    # A right-angle "L" plus a fourth atom lifted in +z for a clean 90° dihedral.
    return _session([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)])


# -- measured values -----------------------------------------------------

def test_distance_value(session):
    p = session.add_distance(0, 1)
    assert p.kind == "distance"
    assert p.value == pytest.approx(1.0)
    assert p.distance == pytest.approx(1.0)
    assert p.degrees is None


def test_angle_right(session):
    p = session.add_angle(0, 1, 2)
    assert p.kind == "angle"
    assert p.value == pytest.approx(90.0)
    assert p.degrees == pytest.approx(90.0)
    assert p.distance is None


@pytest.mark.parametrize("deg", [30.0, 60.0, 120.0, 150.0])
def test_angle_arbitrary(deg):
    r = math.radians(deg)
    s = _session([(1, 0, 0), (0, 0, 0), (math.cos(r), math.sin(r), 0)])
    # Coordinates round-trip through cctbx as float32, so allow that precision.
    assert s.add_angle(0, 1, 2).value == pytest.approx(deg, abs=0.01)


def test_dihedral_right(session):
    assert session.add_dihedral(0, 1, 2, 3).value == pytest.approx(90.0)


def test_dihedral_sign():
    # Mirror the fourth atom through the plane -> the dihedral flips sign.
    pos = _session([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)])
    neg = _session([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, -1)])
    assert pos.add_dihedral(0, 1, 2, 3).value == pytest.approx(90.0)
    assert neg.add_dihedral(0, 1, 2, 3).value == pytest.approx(-90.0)


def test_label_has_no_value(session):
    p = session.add_label(2, "atom two")
    assert p.kind == "label"
    assert p.text == "atom two"
    assert p.value is None


def test_angle_degenerate_returns_none():
    # Two coincident points make the angle undefined.
    s = _session([(0, 0, 0), (0, 0, 0), (1, 0, 0)])
    assert s.add_angle(0, 1, 2).value is None


def test_group_centroid_used(session):
    # distance from atom0 to the centroid of atoms {2,3} = midpoint (1,1,0.5).
    p = session.add_distance(0, [2, 3])
    assert p.value == pytest.approx(math.sqrt(1 + 1 + 0.25))


def test_value_reflects_latest_frame(session):
    # Push a new conformation; the measurement uses the latest coordinates.
    session.push([[0, 0, 0], [2, 0, 0], [2, 2, 0], [2, 2, 2]])
    assert session.add_distance(0, 1).value == pytest.approx(2.0)


# -- Selection construction & coercion -----------------------------------

def test_select_by_indices(session):
    sel = session.select_by(indices=[0, 2])
    assert isinstance(sel, Selection)
    assert sel.indices == [0, 2]
    assert sel.ids == [1, 3]


def test_select_by_ids(session):
    sel = session.select_by(ids=[2, 4])
    assert sel.indices == [1, 3]
    assert sel.ids == [2, 4]


def test_select_by_requires_exactly_one(session):
    with pytest.raises(ValueError):
        session.select_by()
    with pytest.raises(ValueError):
        session.select_by(indices=[0], ids=[1])


def test_select_by_out_of_range(session):
    with pytest.raises(ValueError):
        session.select_by(indices=[99])


def test_select_by_unknown_id(session):
    with pytest.raises(ValueError):
        session.select_by(ids=[999])


def test_coercion_accepts_selection_int_and_list(session):
    a = session.add_angle(session.select_by(indices=[0]), 1, [2, 3])
    assert session._primitives[a.id]["groups"] == [[0], [1], [2, 3]]


def test_coercion_rejects_bool(session):
    with pytest.raises(TypeError):
        session.add_label(True, "nope")


def test_str_spec_resolved_via_cctbx(session):
    # A string is a cctbx selection; the fixture is model-backed, so it resolves.
    a = session.add_distance("resseq 1", "resseq 2")  # one atom per residue
    assert session._primitives[a.id]["groups"] == [[0], [1]]


def test_select_by_mask(session):
    m = np.zeros(4, dtype=bool)
    m[[1, 3]] = True
    assert session.select_by(mask=m).indices == [1, 3]


def test_select_by_mask_wrong_shape(session):
    with pytest.raises(ValueError):
        session.select_by(mask=np.ones(3, dtype=bool))


def test_coercion_accepts_mask(session):
    a = session.add_distance(np.array([True, False, False, False]), [1, 2, 3])
    assert session._primitives[a.id]["groups"] == [[0], [1, 2, 3]]


def test_indices_sorted_and_deduped(session):
    assert session.select_by(indices=[3, 1, 1, 0]).indices == [0, 1, 3]


def test_empty_group_rejected(session):
    with pytest.raises(ValueError):
        session.add_distance(session.select_by(indices=[]), 1)


# -- primitive bookkeeping ----------------------------------------------

def test_ids_unique_and_kinded(session):
    d = session.add_distance(0, 1)
    a = session.add_angle(0, 1, 2)
    assert d.id.startswith("distance-")
    assert a.id.startswith("angle-")
    assert d.id != a.id


def test_custom_id_used(session):
    p = session.add_angle(0, 1, 2, id="myangle")
    assert p.id == "myangle"
    assert "myangle" in session._primitives


def test_remove_and_clear(session):
    a = session.add_angle(0, 1, 2)
    b = session.add_distance(0, 1)
    session.remove_primitive(a.id)
    assert a.id not in session._primitives and b.id in session._primitives
    session.clear_primitives()
    assert session._primitives == {}


def test_add_returns_primitive_dataclass(session):
    p = session.add_angle(0, 1, 2)
    assert isinstance(p, Primitive)
    assert [s.indices for s in p.selections] == [[0], [1], [2]]
