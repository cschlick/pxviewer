"""Read atomic models with cctbx and map them onto pxviewer's transfer format.

pxviewer treats cctbx as the single source of truth for model I/O: a file is read
by cctbx's :class:`~iotbx.data_manager.DataManager` into an
``mmtbx.model.manager``, and everything the viewer shows is derived from that
model's ``pdb_hierarchy`` — no structure is ever parsed in the browser.

The hierarchy exposes its columns as vectorised arrays (``extract_xyz`` etc.), so
we lift them straight into an :class:`~pxviewer.data.AtomArrays` and hand that to
the BinaryCIF encoder, mapping cctbx to the wire with no per-atom Python on the
hot path (only the residue/chain labels need a single ordered walk).

cctbx is imported lazily inside the functions so ``import pxviewer`` stays fast
and the rest of the package works without cctbx installed.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from .data import Atom, AtomArrays

__all__ = [
    "ModelData",
    "CctbxModel",
    "read_model",
    "first_model",
    "model_to_arrays",
    "model_secondary_structure",
    "model_is_polymer",
    "load_model",
    "model_from_sites",
]


def _column(value: Any, n: int, default: Any) -> List[Any]:
    """Broadcast a scalar to length ``n``, or pass a per-atom sequence through."""
    if value is None:
        return [default] * n
    if isinstance(value, str) or not hasattr(value, "__len__"):
        return [value] * n
    seq = list(value)
    if len(seq) != n:
        raise ValueError(f"column length {len(seq)} does not match {n} atoms")
    return seq


def model_from_sites(
    sites: Any,
    *,
    elements: Any = None,
    names: Any = None,
    chains: Any = None,
    resseqs: Any = None,
    resnames: Any = None,
    label: str = "pxviewer",
) -> Any:
    """Build a cctbx model from coordinate + label arrays via a generated mmCIF.

    This is the string route the demos and tests use instead of hand-constructing
    a hierarchy: it writes a minimal ``_atom_site`` loop and loads it through the
    DataManager. ``sites`` is an ``(N, 3)`` array; each label argument is a scalar
    (broadcast) or a per-atom sequence. Defaults make a chain of carbons, one per
    residue, so positional index == residue == i_seq.
    """
    sites = np.asarray(sites, dtype=float).reshape(-1, 3)
    n = sites.shape[0]
    elements = _column(elements, n, "C")
    names = _column(names, n, "CA")
    chains = _column(chains, n, "A")
    resnames = _column(resnames, n, "ALA")
    resseqs = _column(resseqs, n, None)
    if resseqs[0] is None:
        resseqs = list(range(1, n + 1))

    cols = [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
        "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
        "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv",
        "auth_seq_id", "auth_asym_id", "pdbx_PDB_model_num",
    ]
    out = ["data_" + label, "loop_"] + ["_atom_site." + c for c in cols]
    for i in range(n):
        x, y, z = sites[i]
        out.append(
            "ATOM %d %s %s . %s %s 1 %d %.3f %.3f %.3f 1.00 0.00 %d %s 1"
            % (i + 1, elements[i], names[i], resnames[i], chains[i], int(resseqs[i]),
               x, y, z, int(resseqs[i]), chains[i])
        )
    cif = "\n".join(out) + "\n"

    DataManager = _require_data_manager()
    dm = DataManager()
    return dm.get_model(dm.process_model_str(label, cif))


def _require_data_manager():
    try:
        from iotbx.data_manager import DataManager
    except ImportError as exc:  # pragma: no cover - exercised only without cctbx
        raise ImportError(
            "pxviewer's model loading needs cctbx. Install it with:\n"
            "    conda install -c conda-forge cctbx-base"
        ) from exc
    return DataManager


def read_model(path: str | Path) -> Any:
    """Read a model file (PDB or mmCIF) and return its ``mmtbx.model.manager``."""
    DataManager = _require_data_manager()
    dm = DataManager()
    dm.process_model_file(str(path))
    return dm.get_model()


def first_model(model: Any) -> Any:
    """Reduce a multi-MODEL (NMR ensemble) to its first model; else return as-is.

    We stream a single fixed topology, so a file with several MODEL records is
    collapsed to model 1 (the common expectation). The reduced model's atoms keep
    a contiguous i_seq order, so it stays consistent with the extracted columns and
    with cctbx selections. (Treating the models as trajectory frames is a possible
    future opt-in — they share a topology.)
    """
    models = model.get_hierarchy().models()
    if len(models) <= 1:
        return model
    return model.select(models[0].atoms().extract_i_seq())


def model_to_arrays(model: Any) -> AtomArrays:
    """Lift a cctbx model's hierarchy into :class:`AtomArrays`.

    Coordinates, element, name, B and occupancy come from the hierarchy's
    vectorised ``extract_*`` arrays; residue name, chain id, residue number and
    altloc need one ordered pass over ``atoms_with_labels`` (cctbx exposes no
    vectorised accessor for those). Both walks follow the same atom order.
    """
    hierarchy = model.get_hierarchy()
    atoms = hierarchy.atoms()

    xyz = atoms.extract_xyz().as_numpy_array()  # (N, 3) float64
    element = [e.strip() for e in atoms.extract_element()]
    name = [n.strip() for n in atoms.extract_name()]
    b = atoms.extract_b().as_numpy_array()
    occ = atoms.extract_occ().as_numpy_array()

    n = len(element)
    resname: List[str] = [""] * n
    chain: List[str] = [""] * n
    resseq = np.empty(n, dtype=np.int32)
    altloc: List[str] = [""] * n
    for i, a in enumerate(hierarchy.atoms_with_labels()):
        resname[i] = a.resname.strip()
        chain[i] = a.chain_id.strip()
        resseq[i] = a.resseq_as_int()
        altloc[i] = a.altloc.strip()

    return AtomArrays(
        element=element,
        name=name,
        resname=resname,
        chain=chain,
        resseq=resseq,
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        altloc=altloc,
        b=b,
        occ=occ,
    )


def model_secondary_structure(model: Any) -> List[Tuple[str, int, int, str]]:
    """Extract secondary structure as ``(chain, beg_resseq, end_resseq, kind)`` rows.

    Reads the model's SS annotation (from HELIX/SHEET records or a stored
    assignment); ``kind`` is ``"helix"`` or ``"sheet"``, ready for the BinaryCIF
    encoder so Mol* can render cartoon. Returns ``[]`` when no annotation exists.
    """
    try:
        annotation = model.get_ss_annotation()
    except Exception:  # pragma: no cover - defensive; some models have no SS machinery
        annotation = None
    if annotation is None:
        return []

    rows: List[Tuple[str, int, int, str]] = []
    for helix in annotation.helices:
        rows.append(
            (helix.start_chain_id.strip(), helix.get_start_resseq_as_int(),
             helix.get_end_resseq_as_int(), "helix")
        )
    for sheet in annotation.sheets:
        for strand in sheet.strands:
            rows.append(
                (strand.start_chain_id.strip(), strand.get_start_resseq_as_int(),
                 strand.get_end_resseq_as_int(), "sheet")
            )
    return rows


def model_is_polymer(model: Any) -> bool:
    """Whether the model contains a polymer (protein or nucleic acid) — enables cartoon."""
    try:
        return bool(model.contains_protein() or model.contains_nucleic_acid())
    except Exception:  # pragma: no cover - defensive
        return False


class ModelData:
    """The session's atom source: numpy columns plus (optionally) the native model.

    The columns (:class:`AtomArrays`) are kept so we never re-derive them from flex
    on every access. When present, ``model`` is the authority for **identity and
    selection**: `select("...")` goes through cctbx's own atom-selection machinery
    rather than any reimplementation, and `diff()` catches the cached columns
    drifting from the model. cctbx calls are serialised under a lock, since the
    session may touch the model from its WebSocket thread.
    """

    def __init__(self, arrays: AtomArrays, model: Any = None):
        self.arrays = arrays
        self.model = model
        self._cache: Any = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.arrays)

    @property
    def n_atoms(self) -> int:
        return len(self.arrays)

    @property
    def elements(self) -> List[str]:
        return self.arrays.element

    @property
    def coords(self) -> np.ndarray:
        """Base coordinates as ``(N, 3)`` float32 (topology frame)."""
        return self.arrays.xyz

    def has_model(self) -> bool:
        return self.model is not None

    def selection_indices(self, expression: str) -> np.ndarray:
        """Resolve a cctbx atom-selection string to positional (i_seq) indices.

        Uses `hierarchy.atom_selection_cache()` — the full Phenix selection language
        — so nothing is reimplemented here. i_seq == our positional/wire index.
        """
        if self.model is None:
            raise ValueError(
                "cctbx selection strings require a model-backed session "
                "(build it via LiveSession.from_model_file / from_cctbx_model)"
            )
        with self._lock:
            if self._cache is None:
                self._cache = self.model.get_hierarchy().atom_selection_cache()
            bsel = self._cache.selection(expression)  # flex.bool
        return bsel.iselection().as_numpy_array()

    def diff(self, tol: float = 1e-3) -> Optional[str]:
        """None if the cached columns still match the model, else a drift message.

        Guards against the held model being mutated underneath the cache (e.g. a
        refinement step moved atoms): compares atom count and coordinates against
        ``model.get_sites_cart()``. Cheap enough to call before relying on the cache.
        """
        if self.model is None:
            return None
        with self._lock:
            sites = self.model.get_sites_cart().as_numpy_array()
        if sites.shape[0] != self.n_atoms:
            return f"atom-count drift: model has {sites.shape[0]}, cache has {self.n_atoms}"
        cached = np.stack([self.arrays.x, self.arrays.y, self.arrays.z], axis=1)
        dev = float(np.abs(sites - cached).max())
        return None if dev <= tol else f"coordinate drift: max |delta| = {dev:.4f} A"


@dataclasses.dataclass
class CctbxModel:
    """A model reduced to what pxviewer streams: columns, polymer flag, SS, model."""

    arrays: AtomArrays
    polymer: bool
    secondary_structure: List[Tuple[str, int, int, str]]
    model: Any


def load_model(path: str | Path) -> CctbxModel:
    """Read a model file, reduce to model 1, and bundle it for streaming."""
    model = first_model(read_model(path))
    return CctbxModel(
        arrays=model_to_arrays(model),
        polymer=model_is_polymer(model),
        secondary_structure=model_secondary_structure(model),
        model=model,
    )
