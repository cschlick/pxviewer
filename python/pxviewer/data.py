"""Encode BinaryCIF topology from columnar atom data.

Atom data is always columnar (:class:`AtomArrays`) — there is no per-atom object.
The columns come from a cctbx model (see :mod:`pxviewer.cctbx_io`); this module
maps them straight onto a minimal BinaryCIF ``_atom_site`` (plus the entity and
secondary-structure categories Mol* needs for cartoon rendering) for the browser.

The BinaryCIF writing itself is :mod:`pxviewer.bcif`, a self-contained encoder; this
module is only the schema — which categories and columns a topology has.
"""

import dataclasses
from typing import List, Sequence

import numpy as np

from . import bcif


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


def _label_asym_ids(chains: "Sequence[str]") -> List[str]:
    """A distinct ``label_asym_id`` for each *contiguous* run of the author chain id.

    An author chain id is not a chain in the mmCIF sense: a PDB routinely reuses one for
    several blocks — a protein chain, then that chain's waters, then its ligands — which
    are separate asyms. Writing the author id straight into ``label_asym_id`` tells the
    reader those blocks are one chain, and Mol* duly gathers each label's atoms together.
    That *reorders* them relative to the file, which silently breaks the contract this
    whole protocol rests on: that streamed coordinates are positionally aligned to the
    topology (see ATOM_IDENTITY_CONTRACT in live.py). Atoms after the first repeat land
    on the wrong positions — 1TEC (chains E, I, E-waters, I-waters) moves by up to 58 A.

    So label the runs A, B, C, ... in file order, which is what a real mmCIF does and what
    keeps the reader's atom order equal to ours. ``auth_asym_id`` still carries the author
    id, so the user sees the chain they expect.
    """
    labels: List[str] = []
    current = None
    index = -1
    for chain in chains:
        if chain != current:
            current = chain
            index += 1
        n, name = index, ""
        while True:                      # A..Z, AA..AZ, ... so the count is never a limit
            name = chr(ord("A") + n % 26) + name
            n = n // 26 - 1
            if n < 0:
                break
        labels.append(name)
    return labels


def _ss_label_lookup(arrays: "AtomArrays", label_asym: "Sequence[str]"):
    """``(author chain, residue number) -> label_asym_id``, for retargeting SS ranges.

    Secondary structure arrives keyed by *author* chain, but ``_struct_conf`` and
    ``_struct_sheet_range`` are matched on ``label_asym_id`` — so once the labels stop
    being the author ids the ranges have to follow, or Mol* finds no residues in them and
    the cartoon loses every helix and strand.

    First occurrence wins. A repeated author chain is the polymer block followed by its
    waters and ligands, and it is the polymer block that carries the secondary structure.
    """
    lookup: dict = {}
    for chain, resseq, label in zip(arrays.chain, arrays.resseq, label_asym):
        lookup.setdefault((str(chain), int(resseq)), label)
    return lookup


def _retarget_ss(rows: list, chain_at: int, beg_at: int, lookup: dict) -> list:
    """Replace the author chain in each SS row with the label_asym_id it belongs to."""
    out = []
    for row in rows:
        row = list(row)
        label = lookup.get((str(row[chain_at]), int(row[beg_at])))
        if label is None:
            continue  # a range naming residues this topology does not have
        row[chain_at] = label
        out.append(tuple(row))
    return out


def _atom_site_category(arrays: "AtomArrays", polymer: bool, label_asym: "Sequence[str]"):
    """``_atom_site`` built directly from the columns — no per-atom Python."""
    cols = [
        bcif.number_column("id", arrays.id, bcif.INT32),
        bcif.string_column("type_symbol", arrays.element),
        bcif.string_column("label_atom_id", arrays.name),
        bcif.string_column("label_comp_id", arrays.resname),
        bcif.number_column("label_seq_id", arrays.resseq, bcif.INT32),
        bcif.string_column("label_asym_id", label_asym),
        bcif.string_column("auth_asym_id", arrays.chain),
        bcif.number_column("auth_seq_id", arrays.resseq, bcif.INT32),
        bcif.number_column("Cartn_x", arrays.x, bcif.FLOAT32),
        bcif.number_column("Cartn_y", arrays.y, bcif.FLOAT32),
        bcif.number_column("Cartn_z", arrays.z, bcif.FLOAT32),
    ]
    # cctbx gives these cheaply; they enable alt-conf handling and b-factor /
    # occupancy colouring in Mol*.
    if any(alt for alt in (arrays.altloc or [])):
        cols.append(bcif.string_column("label_alt_id", arrays.altloc))
    if arrays.occ is not None:
        cols.append(bcif.number_column("occupancy", arrays.occ, bcif.FLOAT32))
    if arrays.b is not None:
        cols.append(bcif.number_column("B_iso_or_equiv", arrays.b, bcif.FLOAT32))
    if polymer:
        cols.append(bcif.string_column("label_entity_id", ["1"] * len(arrays)))
    return bcif.category("_atom_site", len(arrays), cols)


def _struct_conf_category(helices: list):
    """``_struct_conf`` — helix ranges. Rows are ``(id, chain, beg, end)``."""
    return bcif.category("_struct_conf", len(helices), [
        bcif.string_column("id", [r[0] for r in helices]),
        bcif.string_column("conf_type_id", ["HELX_P"] * len(helices)),
        bcif.string_column("beg_label_asym_id", [r[1] for r in helices]),
        bcif.number_column("beg_label_seq_id", [r[2] for r in helices], bcif.INT32),
        bcif.string_column("end_label_asym_id", [r[1] for r in helices]),
        bcif.number_column("end_label_seq_id", [r[3] for r in helices], bcif.INT32),
    ])


def _struct_sheet_range_category(sheets: list):
    """``_struct_sheet_range`` — strands. Rows are ``(sheet_id, id, chain, beg, end)``."""
    return bcif.category("_struct_sheet_range", len(sheets), [
        bcif.string_column("sheet_id", [r[0] for r in sheets]),
        bcif.string_column("id", [r[1] for r in sheets]),
        bcif.string_column("beg_label_asym_id", [r[2] for r in sheets]),
        bcif.number_column("beg_label_seq_id", [r[3] for r in sheets], bcif.INT32),
        bcif.string_column("end_label_asym_id", [r[2] for r in sheets]),
        bcif.number_column("end_label_seq_id", [r[4] for r in sheets], bcif.INT32),
    ])


def _cell_category():
    """A placeholder P1 unit cell — Mol* expects the category to exist."""
    return bcif.category("_cell", 1, [
        bcif.number_column("length_a", [1.0], bcif.FLOAT32),
        bcif.number_column("length_b", [1.0], bcif.FLOAT32),
        bcif.number_column("length_c", [1.0], bcif.FLOAT32),
        bcif.number_column("angle_alpha", [90.0], bcif.FLOAT32),
        bcif.number_column("angle_beta", [90.0], bcif.FLOAT32),
        bcif.number_column("angle_gamma", [90.0], bcif.FLOAT32),
        bcif.number_column("Z_PDB", [1], bcif.INT32),
    ])


def _symmetry_category():
    return bcif.category("_symmetry", 1, [
        bcif.string_column("space_group_name_H-M", ["P 1"]),
    ])


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
    label_asym = _label_asym_ids(arrays.chain)
    cats = [_atom_site_category(arrays, polymer, label_asym)]
    if polymer:
        cats.append(bcif.category("_entity", 1, [
            bcif.string_column("id", ["1"]),
            bcif.string_column("type", ["polymer"]),
        ]))
        cats.append(bcif.category("_entity_poly", 1, [
            bcif.string_column("entity_id", ["1"]),
            bcif.string_column("type", ["polypeptide(L)"]),
        ]))
    if secondary_structure:
        helices, sheets = _normalize_ss(secondary_structure)
        lookup = _ss_label_lookup(arrays, label_asym)
        # Rows are (id, chain, beg, end) and (sheet_id, id, chain, beg, end).
        helices = _retarget_ss(helices, chain_at=1, beg_at=2, lookup=lookup)
        sheets = _retarget_ss(sheets, chain_at=2, beg_at=3, lookup=lookup)
        if helices:
            cats.append(_struct_conf_category(helices))
        if sheets:
            cats.append(_struct_sheet_range_category(sheets))
    cats.append(_cell_category())
    cats.append(_symmetry_category())
    return bcif.encode(block_header, cats, encoder="pxviewer")
