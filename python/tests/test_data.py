"""Tests for the columnar BinaryCIF encoder (encode_bcif_arrays).

Decoded with :func:`pxviewer.bcif.decode`, the same module that wrote them — a
round trip. (Cross-compatibility with an independent decoder is Mol* itself, at the
other end of the wire.) Category names come back with the leading underscore.
"""

import numpy as np
import pytest

from pxviewer import AtomArrays, bcif, encode_bcif_arrays


def _protein_arrays(n=6):
    return AtomArrays(
        element=["C"] * n,
        name=["CA"] * n,
        resname=["ALA"] * n,
        chain=["A"] * n,
        resseq=list(range(1, n + 1)),
        x=np.arange(n, dtype=float),
        y=np.zeros(n),
        z=np.zeros(n),
    )


def _block(raw):
    return bcif.decode(raw)["PXVIEWER"]


def test_encode_has_atom_site_cell_and_symmetry():
    block = _block(encode_bcif_arrays(_protein_arrays()))
    assert "_atom_site" in block and "_cell" in block and "_symmetry" in block
    assert block["_symmetry"]["space_group_name_H-M"][0] == "P 1"
    assert len(block["_atom_site"]["id"]) == 6


def test_atom_site_columns_map_from_arrays():
    arrays = AtomArrays(
        element=["N", "C"], name=["N", "CA"], resname=["ALA", "ALA"], chain=["A", "A"],
        resseq=[1, 1], x=[1.0, 4.0], y=[2.0, 5.0], z=[3.0, 6.0],
    )
    site = _block(encode_bcif_arrays(arrays))["_atom_site"]
    assert site["type_symbol"][0] == "N"
    assert site["label_atom_id"][1] == "CA"
    assert site["Cartn_x"][0] == pytest.approx(1.0)
    assert site["Cartn_y"][1] == pytest.approx(5.0)


def test_optional_b_and_occupancy_columns():
    arrays = _protein_arrays(2)
    arrays.b = np.array([11.0, 22.0], dtype=np.float32)
    arrays.occ = np.array([1.0, 0.5], dtype=np.float32)
    site = _block(encode_bcif_arrays(arrays))["_atom_site"]
    assert "B_iso_or_equiv" in site and "occupancy" in site
    assert site["B_iso_or_equiv"][1] == pytest.approx(22.0)


def test_polymer_flag_emits_entity():
    block = _block(encode_bcif_arrays(_protein_arrays(), polymer=True))
    assert "_entity" in block and "_entity_poly" in block
    assert block["_entity_poly"]["type"][0] == "polypeptide(L)"
    assert "label_entity_id" in block["_atom_site"]


def test_default_is_not_polymer():
    block = _block(encode_bcif_arrays(_protein_arrays()))
    assert "_entity" not in block
    assert "label_entity_id" not in block["_atom_site"]


def test_secondary_structure_categories():
    block = _block(encode_bcif_arrays(
        _protein_arrays(),
        secondary_structure=[("A", 1, 3, "helix"), ("A", 4, 6, "sheet")],
    ))
    assert "_entity_poly" in block  # SS implies polymer
    conf = block["_struct_conf"]
    assert len(conf["id"]) == 1
    assert conf["conf_type_id"][0] == "HELX_P"
    assert int(conf["beg_label_seq_id"][0]) == 1
    assert int(conf["end_label_seq_id"][0]) == 3
    sheet = block["_struct_sheet_range"]
    assert len(sheet["id"]) == 1
    assert int(sheet["beg_label_seq_id"][0]) == 4


def test_secondary_structure_bad_kind():
    with pytest.raises(ValueError):
        encode_bcif_arrays(_protein_arrays(), secondary_structure=[("A", 1, 3, "coil")])
