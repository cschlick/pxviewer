"""Selection + representation integration.

Selections are resolved by cctbx's own atom-selection machinery (the full Phenix
selection language), so a session must be model-backed. Representations map MVS
types/colours onto Mol*'s vocabulary.
"""

import pytest

from pxviewer import Atom, LiveSession

pytest.importorskip("iotbx.data_manager")

from iotbx.data_manager import DataManager  # noqa: E402


def _model_pdb() -> str:
    # 12 atoms: chain A resseq 1-3, chain B resseq 4-6; each residue an N + CA.
    # cctbx canonicalises atom order within a residue (N before CA), so i_seq (and
    # therefore the wire index) follows that order, not the file's line order.
    lines = []
    serial = 1
    for chain, residues in [("A", [1, 2, 3]), ("B", [4, 5, 6])]:
        for rs in residues:
            for nm, el in [("CA", "C"), ("N", "N")]:
                x = float(serial - 1)
                lines.append(
                    "ATOM  %5d %-4s ALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
                    % (serial, (" " + nm).ljust(4), chain, rs, x, 0.0, 0.0, el)
                )
                serial += 1
    return "\n".join(lines) + "\n"


@pytest.fixture
def session():
    dm = DataManager()
    model = dm.get_model(dm.process_model_str("test", _model_pdb()))
    return LiveSession.from_cctbx_model(model)


# -- cctbx selection strings -------------------------------------------------

def test_selection_by_chain(session):
    assert session.select_by(selection="chain A").indices == [0, 1, 2, 3, 4, 5]


def test_selection_by_residue_range(session):
    assert session.select_by(selection="resseq 1:2").indices == [0, 1, 2, 3]


def test_selection_by_element(session):
    # cctbx orders N before CA, so the nitrogens are the even indices.
    assert session.select_by(selection="element N").indices == [0, 2, 4, 6, 8, 10]


def test_selection_by_name(session):
    assert session.select_by(selection="name CA").indices == [1, 3, 5, 7, 9, 11]


def test_selection_conjunction(session):
    assert session.select_by(selection="chain A and element N").indices == [0, 2, 4]


def test_selection_union(session):
    got = session.select_by(selection="chain B or element N").indices
    assert got == [0, 2, 4, 6, 7, 8, 9, 10, 11]


def test_selection_result_carries_labels(session):
    sel = session.select_by(selection="chain A and name CA")
    assert [a.name for a in sel.atoms] == ["CA", "CA", "CA"]
    assert all(a.chain == "A" for a in sel.atoms)


def test_coercion_accepts_selection_string(session):
    # anything that takes a Selection also takes a cctbx selection string
    assert session._as_selection("name CA").indices == [1, 3, 5, 7, 9, 11]


def test_bad_selection_string_raises(session):
    with pytest.raises(Exception):
        session.select_by(selection="chain A and blorp 3")


def test_selection_string_needs_a_model():
    # A session with no cctbx model can't resolve selection strings.
    bare = LiveSession([Atom(id=1, element="C", name="C", resname="UNL", resseq=1, chain="A", x=0, y=0, z=0)])
    with pytest.raises(ValueError, match="model-backed"):
        bare.select_by(selection="chain A")


# -- positional selection (works with or without a model) --------------------

def test_selection_by_indices_ids_mask(session):
    assert session.select_by(indices=[5, 3]).indices == [3, 5]  # sorted, deduped
    assert session.select_by(ids=[1]).indices == [0]  # id 1 == i_seq 0
    import numpy as np
    m = np.zeros(12, dtype=bool)
    m[[2, 7]] = True
    assert session.select_by(mask=m).indices == [2, 7]


def test_select_by_requires_exactly_one(session):
    with pytest.raises(ValueError):
        session.select_by()
    with pytest.raises(ValueError):
        session.select_by(indices=[0], selection="chain A")


def test_selection_to_component_expression(session):
    exprs = session.select_by(indices=[3, 1]).to_component_expression()
    assert [e.atom_index for e in exprs] == [1, 3]  # sorted


# -- representations on MVS types --------------------------------------------

def test_repr_type_aliases_and_molstar_mapping(session):
    def rtype(type):
        return session._representations[session.add_representation(type)]["type"]
    assert rtype("sphere") == "spacefill"
    assert rtype("ribbon") == "cartoon"
    assert rtype("surface") == "molecular-surface"
    assert rtype("ball_and_stick") == "ball-and-stick"
    assert rtype("ball-and-stick") == "ball-and-stick"


def test_repr_unknown_type_rejected(session):
    with pytest.raises(ValueError):
        session.add_representation("putty")


def test_repr_named_color_is_uniform(session):
    spec = session._representations[session.add_representation("spacefill", color="orange")]
    assert spec["color"] == "uniform" and spec["colorValue"] == "orange"


def test_repr_theme_color_stays_theme(session):
    spec = session._representations[session.add_representation("ball_and_stick", color="element-symbol")]
    assert spec["color"] == "element-symbol" and "colorValue" not in spec


def test_repr_color_value_forces_uniform(session):
    spec = session._representations[session.add_representation("cartoon", color_value="red")]
    assert spec["color"] == "uniform" and spec["colorValue"] == "red"


def test_repr_subset_via_selection_string(session):
    spec = session._representations[session.add_representation("spacefill", on="chain A")]
    assert spec["on"] == {"runs": [[0, 5]]}  # chain A = contiguous indices 0-5
