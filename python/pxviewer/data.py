"""Encode BinaryCIF topology from columnar atom data.

Atom data is always columnar (:class:`AtomArrays`) — there is no per-atom object.
The columns come from a cctbx model (see :mod:`pxviewer.cctbx_io`); this module
maps them straight onto a minimal BinaryCIF ``_atom_site`` for the browser.
"""

import dataclasses
from typing import List

import numpy as np
from ciftools.models.writer import CIFCategoryDesc, CIFFieldDesc
from ciftools.serialization import create_binary_writer


@dataclasses.dataclass
class AtomArrays:
    """A structure's atom-site data as parallel columns rather than per-atom objects.

    This is the efficient hand-off from a vectorised source (e.g. a cctbx
    hierarchy's ``extract_xyz``/``extract_element`` arrays) to BinaryCIF: the
    columns go straight into the CIF fields with no per-atom Python. ``x/y/z`` and
    ``resseq`` are numpy arrays; the string columns are plain lists. ``id`` defaults
    to a 1-based serial. ``altloc``, ``b`` and ``occ`` are optional enrichments.
    """

    element: List[str]
    name: List[str]
    resname: List[str]
    chain: List[str]
    resseq: "np.ndarray"
    x: "np.ndarray"
    y: "np.ndarray"
    z: "np.ndarray"
    id: "np.ndarray | None" = None
    altloc: "List[str] | None" = None
    b: "np.ndarray | None" = None
    occ: "np.ndarray | None" = None

    def __post_init__(self) -> None:
        self.resseq = np.asarray(self.resseq, dtype=np.int32)
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)
        self.z = np.asarray(self.z, dtype=np.float32)
        n = len(self.element)
        if not (len(self.name) == len(self.resname) == len(self.chain) == n
                == self.resseq.shape[0] == self.x.shape[0] == self.y.shape[0] == self.z.shape[0]):
            raise ValueError("AtomArrays columns must all have the same length")
        if self.id is None:
            self.id = np.arange(1, n + 1, dtype=np.int32)
        else:
            self.id = np.asarray(self.id, dtype=np.int32)
        if self.b is not None:
            self.b = np.asarray(self.b, dtype=np.float32)
        if self.occ is not None:
            self.occ = np.asarray(self.occ, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.element)

    @property
    def xyz(self) -> "np.ndarray":
        """The coordinates as an ``(N, 3)`` float32 array (streaming base frame)."""
        return np.stack([self.x, self.y, self.z], axis=1).astype("<f4")


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


class _RowsCategory(CIFCategoryDesc):
    """Base for categories whose data is a list of row-tuples."""

    @staticmethod
    def get_row_count(rows: list) -> int:
        return len(rows)


class EntityCategory(_RowsCategory):
    """_entity: rows = [(id, type), ...]."""

    @property
    def name(self) -> str:
        return "entity"

    @staticmethod
    def get_field_descriptors(rows: list) -> List[CIFFieldDesc]:
        return [
            CIFFieldDesc.string_array(name="id", array=lambda rs: [r[0] for r in rs]),
            CIFFieldDesc.string_array(name="type", array=lambda rs: [r[1] for r in rs]),
        ]


class EntityPolyCategory(_RowsCategory):
    """_entity_poly: rows = [(entity_id, type), ...]."""

    @property
    def name(self) -> str:
        return "entity_poly"

    @staticmethod
    def get_field_descriptors(rows: list) -> List[CIFFieldDesc]:
        return [
            CIFFieldDesc.string_array(name="entity_id", array=lambda rs: [r[0] for r in rs]),
            CIFFieldDesc.string_array(name="type", array=lambda rs: [r[1] for r in rs]),
        ]


class StructConfCategory(_RowsCategory):
    """_struct_conf (helices): rows = [(id, chain, beg_seq, end_seq), ...]."""

    @property
    def name(self) -> str:
        return "struct_conf"

    @staticmethod
    def get_field_descriptors(rows: list) -> List[CIFFieldDesc]:
        return [
            CIFFieldDesc.string_array(name="id", array=lambda rs: [r[0] for r in rs]),
            CIFFieldDesc.string_array(name="conf_type_id", array=lambda rs: ["HELX_P" for _ in rs]),
            CIFFieldDesc.string_array(name="beg_label_asym_id", array=lambda rs: [r[1] for r in rs]),
            CIFFieldDesc.number_array(name="beg_label_seq_id", dtype=np.int32, array=lambda rs: np.array([r[2] for r in rs], dtype=np.int32)),
            CIFFieldDesc.string_array(name="end_label_asym_id", array=lambda rs: [r[1] for r in rs]),
            CIFFieldDesc.number_array(name="end_label_seq_id", dtype=np.int32, array=lambda rs: np.array([r[3] for r in rs], dtype=np.int32)),
        ]


class StructSheetRangeCategory(_RowsCategory):
    """_struct_sheet_range (strands): rows = [(sheet_id, id, chain, beg_seq, end_seq), ...]."""

    @property
    def name(self) -> str:
        return "struct_sheet_range"

    @staticmethod
    def get_field_descriptors(rows: list) -> List[CIFFieldDesc]:
        return [
            CIFFieldDesc.string_array(name="sheet_id", array=lambda rs: [r[0] for r in rs]),
            CIFFieldDesc.string_array(name="id", array=lambda rs: [r[1] for r in rs]),
            CIFFieldDesc.string_array(name="beg_label_asym_id", array=lambda rs: [r[2] for r in rs]),
            CIFFieldDesc.number_array(name="beg_label_seq_id", dtype=np.int32, array=lambda rs: np.array([r[3] for r in rs], dtype=np.int32)),
            CIFFieldDesc.string_array(name="end_label_asym_id", array=lambda rs: [r[2] for r in rs]),
            CIFFieldDesc.number_array(name="end_label_seq_id", dtype=np.int32, array=lambda rs: np.array([r[4] for r in rs], dtype=np.int32)),
        ]


_HELIX_KINDS = {"helix", "h", "helx", "helx_p"}
_SHEET_KINDS = {"sheet", "strand", "e", "s", "beta"}


def _normalize_ss(secondary_structure) -> tuple:
    """Split [(chain, beg, end, kind)] into _struct_conf and _struct_sheet_range rows."""
    helices, sheets = [], []
    for entry in secondary_structure:
        chain, beg, end, kind = entry
        k = str(kind).lower()
        if k in _HELIX_KINDS:
            helices.append((f"H{len(helices) + 1}", str(chain), int(beg), int(end)))
        elif k in _SHEET_KINDS:
            n = len(sheets) + 1
            sheets.append((str(n), str(n), str(chain), int(beg), int(end)))
        else:
            raise ValueError(f"unknown secondary-structure kind {kind!r}; use 'helix' or 'sheet'")
    return helices, sheets


class AtomSiteArraysCategory(CIFCategoryDesc):
    """_atom_site built directly from :class:`AtomArrays` columns (no per-atom loop)."""

    def __init__(self, polymer: bool = False):
        self._polymer = polymer

    @property
    def name(self) -> str:
        return "atom_site"

    @staticmethod
    def get_row_count(arrays: "AtomArrays") -> int:
        return len(arrays)

    def get_field_descriptors(self, arrays: "AtomArrays") -> List[CIFFieldDesc]:
        fields = [
            CIFFieldDesc.number_array(name="id", dtype=np.int32, array=lambda a: a.id),
            CIFFieldDesc.string_array(name="type_symbol", array=lambda a: a.element),
            CIFFieldDesc.string_array(name="label_atom_id", array=lambda a: a.name),
            CIFFieldDesc.string_array(name="label_comp_id", array=lambda a: a.resname),
            CIFFieldDesc.number_array(name="label_seq_id", dtype=np.int32, array=lambda a: a.resseq),
            CIFFieldDesc.string_array(name="label_asym_id", array=lambda a: a.chain),
            CIFFieldDesc.string_array(name="auth_asym_id", array=lambda a: a.chain),
            CIFFieldDesc.number_array(name="auth_seq_id", dtype=np.int32, array=lambda a: a.resseq),
            CIFFieldDesc.number_array(name="Cartn_x", dtype=np.float32, array=lambda a: a.x),
            CIFFieldDesc.number_array(name="Cartn_y", dtype=np.float32, array=lambda a: a.y),
            CIFFieldDesc.number_array(name="Cartn_z", dtype=np.float32, array=lambda a: a.z),
        ]
        # cctbx gives these cheaply; they enable alt-conf handling and b-factor /
        # occupancy colouring in Mol*.
        if any(alt for alt in (arrays.altloc or [])):
            fields.append(CIFFieldDesc.string_array(name="label_alt_id", array=lambda a: a.altloc))
        if arrays.occ is not None:
            fields.append(CIFFieldDesc.number_array(name="occupancy", dtype=np.float32, array=lambda a: a.occ))
        if arrays.b is not None:
            fields.append(CIFFieldDesc.number_array(name="B_iso_or_equiv", dtype=np.float32, array=lambda a: a.b))
        if self._polymer:
            fields.append(CIFFieldDesc.string_array(name="label_entity_id", array=lambda a: ["1"] * len(a)))
        return fields


def encode_bcif_arrays(
    arrays: "AtomArrays",
    *,
    block_header: str = "PXVIEWER",
    polymer: bool = False,
    secondary_structure=None,
) -> bytes:
    """Encode :class:`AtomArrays` as BinaryCIF, mapping columns straight to CIF fields.

    Nothing is iterated per atom in Python — the numpy/list columns become the CIF
    field arrays directly. With ``polymer=True`` (implied when
    ``secondary_structure`` is given) the atoms are declared a polypeptide entity so
    Mol* enables cartoon / secondary-structure rendering; ``secondary_structure`` is
    a list of ``(chain, beg_resseq, end_resseq, kind)`` with ``kind`` ``"helix"`` or
    ``"sheet"``.
    """
    if secondary_structure:
        polymer = True
    writer = create_binary_writer()
    writer.start_data_block(block_header)
    writer.write_category(AtomSiteArraysCategory(polymer=polymer), [arrays])
    if polymer:
        writer.write_category(EntityCategory(), [[("1", "polymer")]])
        writer.write_category(EntityPolyCategory(), [[("1", "polypeptide(L)")]])
    if secondary_structure:
        helices, sheets = _normalize_ss(secondary_structure)
        if helices:
            writer.write_category(StructConfCategory(), [helices])
        if sheets:
            writer.write_category(StructSheetRangeCategory(), [sheets])
    writer.write_category(CellCategory(), [_Cell()])
    writer.write_category(SymmetryCategory(), [_Symmetry()])
    return writer.encode()
