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
