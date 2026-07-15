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
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from .data import AtomArrays

__all__ = [
    "CctbxModel",
    "read_model",
    "model_to_arrays",
    "model_secondary_structure",
    "model_is_polymer",
    "load_model",
]


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


@dataclasses.dataclass
class CctbxModel:
    """A model reduced to what pxviewer streams: columns, polymer flag, and SS."""

    arrays: AtomArrays
    polymer: bool
    secondary_structure: List[Tuple[str, int, int, str]]


def load_model(path: str | Path) -> CctbxModel:
    """Read a model file and reduce it to a :class:`CctbxModel` for streaming."""
    model = read_model(path)
    return CctbxModel(
        arrays=model_to_arrays(model),
        polymer=model_is_polymer(model),
        secondary_structure=model_secondary_structure(model),
    )
