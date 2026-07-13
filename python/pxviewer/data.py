"""Create and read BinaryCIF files for small atom models."""

import dataclasses
import os
from typing import Any, List

import numpy as np
from ciftools.models.writer import CIFCategoryDesc, CIFFieldDesc
from ciftools.serialization import create_binary_writer


@dataclasses.dataclass
class Atom:
    """A single atom for a minimal atom-site table."""

    id: int
    element: str
    x: float
    y: float
    z: float
    name: str = "C"
    resname: str = "UNL"
    resseq: int = 1
    chain: str = "A"


class AtomSiteCategory(CIFCategoryDesc):
    """CIF category for _atom_site."""

    @property
    def name(self) -> str:
        return "atom_site"

    @staticmethod
    def get_row_count(atoms: List[Atom]) -> int:
        return len(atoms)

    @staticmethod
    def get_field_descriptors(atoms: List[Atom]) -> List[CIFFieldDesc]:
        return [
            CIFFieldDesc.number_array(
                name="id",
                dtype=np.int32,
                array=lambda a: np.array([atom.id for atom in a], dtype=np.int32),
            ),
            CIFFieldDesc.string_array(
                name="type_symbol",
                array=lambda a: [atom.element for atom in a],
            ),
            CIFFieldDesc.string_array(
                name="label_atom_id",
                array=lambda a: [atom.name for atom in a],
            ),
            CIFFieldDesc.string_array(
                name="label_comp_id",
                array=lambda a: [atom.resname for atom in a],
            ),
            CIFFieldDesc.number_array(
                name="label_seq_id",
                dtype=np.int32,
                array=lambda a: np.array([atom.resseq for atom in a], dtype=np.int32),
            ),
            CIFFieldDesc.string_array(
                name="label_asym_id",
                array=lambda a: [atom.chain for atom in a],
            ),
            CIFFieldDesc.number_array(
                name="Cartn_x",
                dtype=np.float32,
                array=lambda a: np.array([atom.x for atom in a], dtype=np.float32),
            ),
            CIFFieldDesc.number_array(
                name="Cartn_y",
                dtype=np.float32,
                array=lambda a: np.array([atom.y for atom in a], dtype=np.float32),
            ),
            CIFFieldDesc.number_array(
                name="Cartn_z",
                dtype=np.float32,
                array=lambda a: np.array([atom.z for atom in a], dtype=np.float32),
            ),
        ]


@dataclasses.dataclass
class _Cell:
    length_a: float = 1.0
    length_b: float = 1.0
    length_c: float = 1.0
    angle_alpha: float = 90.0
    angle_beta: float = 90.0
    angle_gamma: float = 90.0
    Z_PDB: int = 1


class CellCategory(CIFCategoryDesc):
    """CIF category for _cell."""

    @property
    def name(self) -> str:
        return "cell"

    @staticmethod
    def get_row_count(_cell: _Cell) -> int:
        return 1

    @staticmethod
    def get_field_descriptors(_cell: _Cell) -> List[CIFFieldDesc]:
        return [
            CIFFieldDesc.number_array(
                name="length_a",
                dtype=np.float32,
                array=lambda c: np.array([c.length_a], dtype=np.float32),
            ),
            CIFFieldDesc.number_array(
                name="length_b",
                dtype=np.float32,
                array=lambda c: np.array([c.length_b], dtype=np.float32),
            ),
            CIFFieldDesc.number_array(
                name="length_c",
                dtype=np.float32,
                array=lambda c: np.array([c.length_c], dtype=np.float32),
            ),
            CIFFieldDesc.number_array(
                name="angle_alpha",
                dtype=np.float32,
                array=lambda c: np.array([c.angle_alpha], dtype=np.float32),
            ),
            CIFFieldDesc.number_array(
                name="angle_beta",
                dtype=np.float32,
                array=lambda c: np.array([c.angle_beta], dtype=np.float32),
            ),
            CIFFieldDesc.number_array(
                name="angle_gamma",
                dtype=np.float32,
                array=lambda c: np.array([c.angle_gamma], dtype=np.float32),
            ),
            CIFFieldDesc.number_array(
                name="Z_PDB",
                dtype=np.int32,
                array=lambda c: np.array([c.Z_PDB], dtype=np.int32),
            ),
        ]


@dataclasses.dataclass
class _Symmetry:
    space_group_name_H_M: str = "P 1"


class SymmetryCategory(CIFCategoryDesc):
    """CIF category for _symmetry."""

    @property
    def name(self) -> str:
        return "symmetry"

    @staticmethod
    def get_row_count(_symmetry: _Symmetry) -> int:
        return 1

    @staticmethod
    def get_field_descriptors(_symmetry: _Symmetry) -> List[CIFFieldDesc]:
        return [
            CIFFieldDesc.string_array(
                name="space_group_name_H-M",
                array=lambda s: [s.space_group_name_H_M],
            ),
        ]


def write_bcif(atoms: List[Atom], path: str | os.PathLike, *, block_header: str = "PXVIEWER") -> None:
    """Write a minimal BinaryCIF file from a list of atoms."""
    writer = create_binary_writer()
    writer.start_data_block(block_header)
    writer.write_category(AtomSiteCategory(), [atoms])
    writer.write_category(CellCategory(), [_Cell()])
    writer.write_category(SymmetryCategory(), [_Symmetry()])
    with open(path, "wb") as f:
        f.write(writer.encode())


def read_atoms(path: str | os.PathLike) -> List[Atom]:
    """Read atoms back from a BinaryCIF file."""
    import ciftools.serialization as cif_io

    with open(path, "rb") as f:
        data = f.read()

    file = cif_io.loads(data, lazy=False)
    block = file[0]
    cat = block["atom_site"]
    n = cat.n_rows

    ids = cat["id"].as_ndarray().astype(int)
    elements = cat["type_symbol"].as_ndarray()
    names = cat["label_atom_id"].as_ndarray()
    resnames = cat["label_comp_id"].as_ndarray()
    resseqs = cat["label_seq_id"].as_ndarray().astype(int)
    chains = cat["label_asym_id"].as_ndarray()
    xs = cat["Cartn_x"].as_ndarray().astype(float)
    ys = cat["Cartn_y"].as_ndarray().astype(float)
    zs = cat["Cartn_z"].as_ndarray().astype(float)

    return [
        Atom(
            id=int(ids[i]),
            element=str(elements[i]),
            x=float(xs[i]),
            y=float(ys[i]),
            z=float(zs[i]),
            name=str(names[i]),
            resname=str(resnames[i]),
            resseq=int(resseqs[i]),
            chain=str(chains[i]),
        )
        for i in range(n)
    ]
