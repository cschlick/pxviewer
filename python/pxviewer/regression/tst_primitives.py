"""Graphics primitives -- distance, angle, dihedral, label -- and the Selection plumbing.

No viewer is involved: ``add_*`` measures against the session's current coordinates and
records a wire message, so the measured value is available immediately whether or not
anything is connected. That is what makes these usable from a script.
"""

from __future__ import absolute_import, division, print_function

import math
import sys

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import have, skip

if not have("iotbx.data_manager", "numpy"):
    skip("iotbx.data_manager / numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer import LiveSession, Primitive, Selection   # noqa: E402


def session(coords=None):
    """A **fresh** session each call.

    Primitives accumulate on a session and ``push`` replaces its coordinates, so sharing
    one between exercises would let an earlier measurement change a later one. The pytest
    fixture this replaces was function-scoped for the same reason.

    The default is a right-angle "L" with a fourth atom lifted in +z, which gives a clean
    90 degrees for both the angle and the dihedral.
    """
    if coords is None:
        coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    return LiveSession.from_sites(coords)


# -- measured values ----------------------------------------------------------


def exercise_distance_value():
    p = session().add_distance(0, 1)
    assert p.kind == "distance"
    assert approx_equal(p.value, 1.0)
    assert approx_equal(p.distance, 1.0)
    assert p.degrees is None            # a distance has no angular reading


def exercise_angle_right():
    p = session().add_angle(0, 1, 2)
    assert p.kind == "angle"
    assert approx_equal(p.value, 90.0)
    assert approx_equal(p.degrees, 90.0)
    assert p.distance is None


def exercise_angle_over_a_range_of_openings():
    for deg in [30.0, 60.0, 120.0, 150.0]:
        r = math.radians(deg)
        s = session([(1, 0, 0), (0, 0, 0), (math.cos(r), math.sin(r), 0)])
        # Coordinates round-trip through cctbx as float32, so allow that precision.
        assert approx_equal(s.add_angle(0, 1, 2).value, deg, eps=0.01), deg


def exercise_angle_degenerate_returns_none():
    """Two coincident points leave the angle undefined -- report nothing, not zero."""
    s = session([(0, 0, 0), (0, 0, 0), (1, 0, 0)])
    assert s.add_angle(0, 1, 2).value is None


def exercise_dihedral_right():
    assert approx_equal(session().add_dihedral(0, 1, 2, 3).value, 90.0)


def exercise_dihedral_is_signed():
    """Mirroring the fourth atom through the plane flips the sign, and must -- an
    unsigned dihedral cannot tell the two chiralities apart."""
    pos = session([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)])
    neg = session([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, -1)])
    assert approx_equal(pos.add_dihedral(0, 1, 2, 3).value, 90.0)
    assert approx_equal(neg.add_dihedral(0, 1, 2, 3).value, -90.0)


def exercise_label_has_no_value():
    p = session().add_label(2, "atom two")
    assert p.kind == "label"
    assert p.text == "atom two"
    assert p.value is None


def exercise_a_group_measures_from_its_centroid():
    # atom 0 to the centroid of {2, 3}, which is the midpoint (1, 1, 0.5).
    p = session().add_distance(0, [2, 3])
    assert approx_equal(p.value, math.sqrt(1 + 1 + 0.25))


def exercise_value_reflects_the_latest_frame():
    """A measurement added after ``push`` uses the pushed conformation, not the original."""
    s = session()
    s.push([[0, 0, 0], [2, 0, 0], [2, 2, 0], [2, 2, 2]])
    assert approx_equal(s.add_distance(0, 1).value, 2.0)


# -- building and coercing selections -----------------------------------------


def exercise_select_by_indices():
    sel = session().select_by(indices=[0, 2])
    assert isinstance(sel, Selection)
    assert sel.indices == [0, 2]
    assert sel.ids == [1, 3]                # id == index + 1


def exercise_select_by_ids():
    sel = session().select_by(ids=[2, 4])
    assert sel.indices == [1, 3]
    assert sel.ids == [2, 4]


def exercise_select_by_mask():
    m = np.zeros(4, dtype=bool)
    m[[1, 3]] = True
    assert session().select_by(mask=m).indices == [1, 3]


def exercise_indices_are_sorted_and_deduped():
    assert session().select_by(indices=[3, 1, 1, 0]).indices == [0, 1, 3]


def exercise_select_by_requires_exactly_one_argument():
    s = session()
    with raises(ValueError):
        s.select_by()
    with raises(ValueError):
        s.select_by(indices=[0], ids=[1])


def exercise_out_of_range_selections_are_rejected():
    """Silently dropping an unknown index would measure the wrong atoms."""
    s = session()
    with raises(ValueError):
        s.select_by(indices=[99])
    with raises(ValueError):
        s.select_by(ids=[999])
    with raises(ValueError):
        s.select_by(mask=np.ones(3, dtype=bool))     # wrong length for a 4-atom session


def exercise_coercion_accepts_selections_ints_and_lists():
    s = session()
    p = s.add_angle(s.select_by(indices=[0]), 1, [2, 3])
    assert s._primitives[p.id]["groups"] == [[0], [1], [2, 3]]


def exercise_coercion_accepts_a_mask():
    s = session()
    p = s.add_distance(np.array([True, False, False, False]), [1, 2, 3])
    assert s._primitives[p.id]["groups"] == [[0], [1, 2, 3]]


def exercise_coercion_resolves_a_cctbx_selection_string():
    """A session built from sites is still model-backed, so strings resolve."""
    s = session()
    p = s.add_distance("resseq 1", "resseq 2")       # one atom per residue
    assert s._primitives[p.id]["groups"] == [[0], [1]]


def exercise_coercion_rejects_a_bool():
    """``True`` is an int in Python, so it would silently mean atom 1."""
    with raises(TypeError):
        session().add_label(True, "nope")


def exercise_an_empty_group_is_rejected():
    s = session()
    with raises(ValueError):
        s.add_distance(s.select_by(indices=[]), 1)


# -- bookkeeping --------------------------------------------------------------


def exercise_ids_are_unique_and_carry_the_kind():
    s = session()
    d = s.add_distance(0, 1)
    a = s.add_angle(0, 1, 2)
    assert d.id.startswith("distance-")
    assert a.id.startswith("angle-")
    assert d.id != a.id


def exercise_a_custom_id_is_used_as_given():
    s = session()
    p = s.add_angle(0, 1, 2, id="myangle")
    assert p.id == "myangle"
    assert "myangle" in s._primitives


def exercise_remove_and_clear():
    s = session()
    a = s.add_angle(0, 1, 2)
    b = s.add_distance(0, 1)
    s.remove_primitive(a.id)
    assert a.id not in s._primitives
    assert b.id in s._primitives         # removing one leaves the rest alone
    s.clear_primitives()
    assert s._primitives == {}


def exercise_add_returns_a_primitive():
    p = session().add_angle(0, 1, 2)
    assert isinstance(p, Primitive)
    assert [sel.indices for sel in p.selections] == [[0], [1], [2]]


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
