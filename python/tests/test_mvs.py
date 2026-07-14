"""MVS (MolViewSpec) integration.

Phase 1: selections are built on MVS ``ComponentExpression`` — pxviewer resolves
them to positional indices against the topology, and a ``Selection`` can emit them.
"""

import pytest

from pxviewer import Atom, ComponentExpression, LiveSession


def _atoms():
    # 12 atoms: alternating CA(C)/N, residues 1..6, chain A (0-5) then B (6-11).
    return [
        Atom(
            id=i + 1,
            element=("N" if i % 2 else "C"),
            name=("N" if i % 2 else "CA"),
            resname="ALA",
            resseq=(i // 2) + 1,
            chain=("A" if i < 6 else "B"),
            x=float(i), y=0.0, z=0.0,
        )
        for i in range(12)
    ]


@pytest.fixture
def session():
    return LiveSession(_atoms())


def test_expression_by_chain(session):
    assert session.select_by(expression=ComponentExpression(label_asym_id="A")).indices == [0, 1, 2, 3, 4, 5]


def test_expression_by_residue_range(session):
    got = session.select_by(expression=ComponentExpression(beg_label_seq_id=1, end_label_seq_id=2)).indices
    assert got == [0, 1, 2, 3]


def test_expression_by_element(session):
    assert session.select_by(expression=ComponentExpression(type_symbol="N")).indices == [1, 3, 5, 7, 9, 11]


def test_expression_by_atom_index(session):
    assert session.select_by(expression=ComponentExpression(atom_index=5)).indices == [5]


def test_expression_by_atom_id(session):
    assert session.select_by(expression=ComponentExpression(atom_id=1)).indices == [0]


def test_expression_fields_are_conjunction(session):
    # multiple fields in one expression AND together
    assert session.select_by(expression=ComponentExpression(label_asym_id="A", type_symbol="N")).indices == [1, 3, 5]


def test_expression_list_is_union(session):
    got = session.select_by(
        expression=[ComponentExpression(label_asym_id="B"), ComponentExpression(type_symbol="N")]
    ).indices
    assert got == [1, 3, 5, 6, 7, 8, 9, 10, 11]


def test_coercion_accepts_component_expression(session):
    # anything that takes a Selection also takes an expression (or a list of them)
    assert session._as_selection(ComponentExpression(label_atom_id="CA")).indices == [0, 2, 4, 6, 8, 10]


def test_selection_to_component_expression(session):
    exprs = session.select_by(indices=[3, 1]).to_component_expression()
    assert [e.atom_index for e in exprs] == [1, 3]  # sorted


def test_expression_unsupported_field_raises(session):
    with pytest.raises(ValueError):
        session.select_by(expression=ComponentExpression(residue_index=0))


def test_select_by_requires_exactly_one(session):
    with pytest.raises(ValueError):
        session.select_by()
    with pytest.raises(ValueError):
        session.select_by(indices=[0], expression=ComponentExpression(atom_index=0))


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


def test_repr_hex_color_is_uniform(session):
    spec = session._representations[session.add_representation("spacefill", color="#112233")]
    assert spec["color"] == "uniform" and spec["colorValue"] == "#112233"


def test_repr_theme_color_stays_theme(session):
    spec = session._representations[session.add_representation("ball_and_stick", color="element-symbol")]
    assert spec["color"] == "element-symbol" and "colorValue" not in spec


def test_repr_color_value_forces_uniform(session):
    spec = session._representations[session.add_representation("cartoon", color_value="red")]
    assert spec["color"] == "uniform" and spec["colorValue"] == "red"


def test_repr_subset_via_component_expression(session):
    spec = session._representations[
        session.add_representation("spacefill", on=ComponentExpression(label_asym_id="A"))
    ]
    assert spec["on"] == {"runs": [[0, 5]]}  # chain A = contiguous indices 0-5
