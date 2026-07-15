"""Tests for the columnar BinaryCIF encoder (encode_bcif_arrays)."""

import ciftools.serialization as cif_io
import numpy as np
import pytest

from pxviewer import AtomArrays, encode_bcif_arrays


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


def test_encode_has_atom_site_cell_and_symmetry():
    block = cif_io.loads(encode_bcif_arrays(_protein_arrays()), lazy=False)[0]
    assert "atom_site" in block and "cell" in block and "symmetry" in block
    assert block["symmetry"]["space_group_name_H-M"].get_string(0) == "P 1"
    assert block["atom_site"].n_rows == 6


def test_atom_site_columns_map_from_arrays():
    arrays = AtomArrays(
        element=["N", "C"], name=["N", "CA"], resname=["ALA", "ALA"], chain=["A", "A"],
        resseq=[1, 1], x=[1.0, 4.0], y=[2.0, 5.0], z=[3.0, 6.0],
    )
    site = cif_io.loads(encode_bcif_arrays(arrays), lazy=False)[0]["atom_site"]
    assert site["type_symbol"].get_string(0) == "N"
    assert site["label_atom_id"].get_string(1) == "CA"
    assert site["Cartn_x"].as_ndarray()[0] == pytest.approx(1.0)
    assert site["Cartn_y"].as_ndarray()[1] == pytest.approx(5.0)


def test_optional_b_and_occupancy_columns():
    arrays = _protein_arrays(2)
    arrays.b = np.array([11.0, 22.0], dtype=np.float32)
    arrays.occ = np.array([1.0, 0.5], dtype=np.float32)
    fields = list(cif_io.loads(encode_bcif_arrays(arrays), lazy=False)[0]["atom_site"].field_names)
    assert "B_iso_or_equiv" in fields and "occupancy" in fields


def test_polymer_flag_emits_entity():
    block = cif_io.loads(encode_bcif_arrays(_protein_arrays(), polymer=True), lazy=False)[0]
    assert "entity" in block and "entity_poly" in block
    assert block["entity_poly"]["type"].get_string(0) == "polypeptide(L)"
    assert "label_entity_id" in list(block["atom_site"].field_names)


def test_default_is_not_polymer():
    block = cif_io.loads(encode_bcif_arrays(_protein_arrays()), lazy=False)[0]
    assert "entity" not in block
    assert "label_entity_id" not in list(block["atom_site"].field_names)


def test_secondary_structure_categories():
    block = cif_io.loads(
        encode_bcif_arrays(_protein_arrays(), secondary_structure=[("A", 1, 3, "helix"), ("A", 4, 6, "sheet")]),
        lazy=False,
    )[0]
    assert "entity_poly" in block  # SS implies polymer
    conf = block["struct_conf"]
    assert conf.n_rows == 1
    assert conf["conf_type_id"].get_string(0) == "HELX_P"
    assert int(conf["beg_label_seq_id"].as_ndarray()[0]) == 1
    assert int(conf["end_label_seq_id"].as_ndarray()[0]) == 3
    sheet = block["struct_sheet_range"]
    assert sheet.n_rows == 1
    assert int(sheet["beg_label_seq_id"].as_ndarray()[0]) == 4


def test_secondary_structure_bad_kind():
    with pytest.raises(ValueError):
        encode_bcif_arrays(_protein_arrays(), secondary_structure=[("A", 1, 3, "coil")])
