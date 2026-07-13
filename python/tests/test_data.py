"""Tests and examples for the pxviewer BinaryCIF data API."""

import ciftools.serialization as cif_io
import pytest

from pxviewer import Atom, read_atoms, write_bcif


def test_write_bcif_round_trip(tmp_path):
    """Example of writing a small model to BCIF and reading the atoms back."""
    atoms = [
        Atom(id=1, element="N", name="N", resname="ALA", resseq=1, chain="A", x=1.0, y=2.0, z=3.0),
        Atom(id=2, element="C", name="CA", resname="ALA", resseq=1, chain="A", x=4.0, y=5.0, z=6.0),
    ]
    path = tmp_path / "model.bcif"
    write_bcif(atoms, path)

    read = read_atoms(path)
    assert len(read) == 2
    assert read[0].id == 1
    assert read[0].element == "N"
    assert read[0].x == pytest.approx(1.0)
    assert read[1].y == pytest.approx(5.0)


def test_write_bcif_has_cell_and_symmetry(tmp_path):
    """Example of reading raw BinaryCIF categories with ciftools."""
    atoms = [Atom(id=1, element="H", name="H", x=0.0, y=0.0, z=0.0)]
    path = tmp_path / "model.bcif"
    write_bcif(atoms, path)

    with open(path, "rb") as f:
        file = cif_io.loads(f.read(), lazy=False)

    block = file[0]
    assert "atom_site" in block
    assert "cell" in block
    assert "symmetry" in block
    assert block["symmetry"]["space_group_name_H-M"].get_string(0) == "P 1"


def _protein_atoms(n=6):
    return [
        Atom(id=i + 1, element="C", name="CA", resname="ALA", resseq=i + 1, chain="A",
             x=float(i), y=0.0, z=0.0)
        for i in range(n)
    ]


def test_polymer_flag_emits_entity(tmp_path):
    from pxviewer.data import encode_bcif

    block = cif_io.loads(encode_bcif(_protein_atoms(), polymer=True), lazy=False)[0]
    assert "entity" in block and "entity_poly" in block
    assert block["entity_poly"]["type"].get_string(0) == "polypeptide(L)"
    assert "label_entity_id" in list(block["atom_site"].field_names)


def test_default_is_not_polymer(tmp_path):
    from pxviewer.data import encode_bcif

    block = cif_io.loads(encode_bcif(_protein_atoms()), lazy=False)[0]
    assert "entity" not in block
    assert "label_entity_id" not in list(block["atom_site"].field_names)


def test_secondary_structure_categories(tmp_path):
    from pxviewer.data import encode_bcif

    block = cif_io.loads(
        encode_bcif(_protein_atoms(), secondary_structure=[("A", 1, 3, "helix"), ("A", 4, 6, "sheet")]),
        lazy=False,
    )[0]
    # SS implies polymer
    assert "entity_poly" in block
    conf = block["struct_conf"]
    assert conf.n_rows == 1
    assert conf["conf_type_id"].get_string(0) == "HELX_P"
    assert int(conf["beg_label_seq_id"].as_ndarray()[0]) == 1
    assert int(conf["end_label_seq_id"].as_ndarray()[0]) == 3
    sheet = block["struct_sheet_range"]
    assert sheet.n_rows == 1
    assert int(sheet["beg_label_seq_id"].as_ndarray()[0]) == 4


def test_secondary_structure_bad_kind():
    from pxviewer.data import encode_bcif

    with pytest.raises(ValueError):
        encode_bcif(_protein_atoms(), secondary_structure=[("A", 1, 3, "coil")])
