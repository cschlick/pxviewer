"""Selection and representation integration.

Selections are resolved by cctbx's own atom-selection machinery (the full Phenix selection
language), so a session must be model-backed. Representations map MVS types and colours onto
Mol*'s vocabulary.
"""

from __future__ import absolute_import, division, print_function

import sys

from libtbx.test_utils import raises

from pxviewer.regression.tst_utils import have, skip

if not have("iotbx.data_manager", "numpy"):
    skip("iotbx.data_manager / numpy not available")

import numpy as np                                   # noqa: E402

from pxviewer import LiveSession                     # noqa: E402


def model_pdb():
    """12 atoms: chain A resseq 1-3, chain B resseq 4-6; each residue an N and a CA.

    cctbx canonicalises atom order within a residue (N before CA), so i_seq -- and therefore
    the wire index -- follows that order, not the file's line order.
    """
    lines = []
    serial = 1
    for chain, residues in [("A", [1, 2, 3]), ("B", [4, 5, 6])]:
        for rs in residues:
            for nm, el in [("CA", "C"), ("N", "N")]:
                x = float(serial - 1)
                lines.append(
                    "ATOM  %5d %-4s ALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
                    % (serial, (" " + nm).ljust(4), chain, rs, x, 0.0, 0.0, el))
                serial += 1
    return "\n".join(lines) + "\n"


def session():
    """A **fresh** session each call.

    Not cached: representations accumulate on a session, so sharing one would let an earlier
    exercise's add_representation change what a later one sees. The pytest fixture this
    replaces was function-scoped for the same reason.
    """
    from iotbx.data_manager import DataManager

    dm = DataManager()
    model = dm.get_model(dm.process_model_str("test", model_pdb()))
    return LiveSession.from_cctbx_model(model)


# -- cctbx selection strings --------------------------------------------------


def exercise_selection_by_chain():
    assert session().select_by(selection="chain A").indices == [0, 1, 2, 3, 4, 5]


def exercise_selection_by_residue_range():
    assert session().select_by(selection="resseq 1:2").indices == [0, 1, 2, 3]


def exercise_selection_by_element():
    # cctbx orders N before CA, so the nitrogens are the even indices.
    assert session().select_by(selection="element N").indices == [0, 2, 4, 6, 8, 10]


def exercise_selection_by_name():
    assert session().select_by(selection="name CA").indices == [1, 3, 5, 7, 9, 11]


def exercise_selection_conjunction():
    assert session().select_by(selection="chain A and element N").indices == [0, 2, 4]


def exercise_selection_union():
    got = session().select_by(selection="chain B or element N").indices
    assert got == [0, 2, 4, 6, 7, 8, 9, 10, 11]


def exercise_selection_result_carries_labels():
    sel = session().select_by(selection="chain A and name CA")
    assert sel.names == ["CA", "CA", "CA"]
    assert sel.chains == ["A", "A", "A"]


def exercise_selection_columnar_accessors():
    sel = session().select_by(selection="chain A")
    assert sel.indices == [0, 1, 2, 3, 4, 5]      # cctbx orders N before CA
    assert sel.ids == [1, 2, 3, 4, 5, 6]          # id == i_seq + 1
    assert sel.names == ["N", "CA", "N", "CA", "N", "CA"]
    assert sel.elements == ["N", "C", "N", "C", "N", "C"]
    assert sel.resseqs == [1, 1, 2, 2, 3, 3]
    assert sel.chains == ["A"] * 6
    assert not hasattr(sel, "atoms")              # columnar only -- no per-atom objects


def exercise_coercion_accepts_a_selection_string():
    """Anything that takes a Selection also takes a cctbx selection string."""
    assert session()._as_selection("name CA").indices == [1, 3, 5, 7, 9, 11]


def exercise_a_bad_selection_string_raises():
    with raises(Exception):
        session().select_by(selection="chain A and blorp 3")


# -- positional selection -----------------------------------------------------


def exercise_selection_by_indices_ids_and_mask():
    s = session()
    assert s.select_by(indices=[5, 3]).indices == [3, 5]     # sorted, deduped
    assert s.select_by(ids=[1]).indices == [0]                # id 1 == i_seq 0
    mask = np.zeros(12, dtype=bool)
    mask[[2, 7]] = True
    assert s.select_by(mask=mask).indices == [2, 7]


def exercise_select_by_requires_exactly_one():
    s = session()
    with raises(ValueError):
        s.select_by()
    with raises(ValueError):
        s.select_by(indices=[0], selection="chain A")


def exercise_selection_to_component_expression():
    exprs = session().select_by(indices=[3, 1]).to_component_expression()
    assert [e.atom_index for e in exprs] == [1, 3]           # sorted


# -- representations on MVS types ---------------------------------------------


def exercise_repr_type_aliases_and_molstar_mapping():
    s = session()

    def rtype(kind):
        return s._representations[s.add_representation(kind)]["type"]

    assert rtype("sphere") == "spacefill"
    assert rtype("ribbon") == "cartoon"
    assert rtype("surface") == "molecular-surface"
    assert rtype("ball_and_stick") == "ball-and-stick"
    assert rtype("ball-and-stick") == "ball-and-stick"


def exercise_repr_unknown_type_rejected():
    with raises(ValueError):
        session().add_representation("putty")


def exercise_repr_named_colour_is_uniform():
    s = session()
    spec = s._representations[s.add_representation("spacefill", color="orange")]
    assert spec["color"] == "uniform" and spec["colorValue"] == "orange"


def exercise_repr_theme_colour_stays_a_theme():
    s = session()
    spec = s._representations[
        s.add_representation("ball_and_stick", color="element-symbol")]
    assert spec["color"] == "element-symbol" and "colorValue" not in spec


def exercise_repr_colour_value_forces_uniform():
    s = session()
    spec = s._representations[s.add_representation("cartoon", color_value="red")]
    assert spec["color"] == "uniform" and spec["colorValue"] == "red"


def exercise_repr_subset_via_a_selection_string():
    s = session()
    spec = s._representations[s.add_representation("spacefill", on="chain A")]
    assert spec["on"] == {"runs": [[0, 5]]}      # chain A = contiguous indices 0-5


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
