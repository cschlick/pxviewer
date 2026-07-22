"""Geometry restraints extraction for the desktop Geometry tables.

Builds a cctbx geometry restraints manager for a model and exposes its restraint
objects (bonds, angles, dihedrals, chirality, planarity) for display. Nothing is
copied into new data structures: we hold the cctbx proxy arrays directly and, for
each row the table actually paints, compute the restraint's value on demand with
``geometry_restraints.bond/angle/...`` against the model's sites — so it stays
cheap even for very large restraint sets.

Building restraints needs the CCP4/geostd monomer library; when it isn't set up
(:func:`monomer_library_available`) the desktop shows :data:`MONOMER_LIBRARY_HELP`
instead of the tables.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

MONOMER_LIBRARY_HELP = (
    "Geometry restraints need the CCP4 / geostd monomer library.\n\n"
    "It ships with the `chem_data` conda package — install it alongside pxviewer:\n"
    "    conda install -c chem_data chem_data\n\n"
    "or point MMTBX_CCP4_MONOMER_LIB at a geostd checkout:\n"
    "    git clone https://github.com/phenix-project/geostd\n"
    "    export MMTBX_CCP4_MONOMER_LIB=/path/to/geostd\n\n"
    "then reopen the model."
)


def _chem_data_geostd() -> Optional[str]:
    """The geostd directory shipped by the ``chem_data`` conda package, if importable.

    ``chem_data`` is a plain importable package, so this resolves regardless of the
    Python version or platform layout (no hard-coded ``site-packages`` path) — which
    is what lets the conda package find the monomer library without an activation hook.
    """
    try:
        import chem_data
    except Exception:
        return None
    path = os.path.join(os.path.dirname(chem_data.__file__), "geostd")
    return path if os.path.isdir(path) else None


def monomer_library_root() -> Optional[str]:
    """The monomer-library (geostd) directory cctbx will use, or None.

    An explicit ``MMTBX_CCP4_MONOMER_LIB`` / ``CLIBD_MON`` wins; otherwise fall back to
    the geostd the ``chem_data`` package ships and export ``MMTBX_CCP4_MONOMER_LIB`` so
    cctbx's own pdb_interpretation — which reads that variable directly — finds it too.
    """
    for var in ("MMTBX_CCP4_MONOMER_LIB", "CLIBD_MON"):
        path = os.environ.get(var)
        if path and os.path.isdir(path):
            return path
    geostd = _chem_data_geostd()
    if geostd:
        os.environ.setdefault("MMTBX_CCP4_MONOMER_LIB", geostd)
        return geostd
    return None


def monomer_library_available() -> bool:
    """Whether cctbx can find a monomer library to build restraints from."""
    return monomer_library_root() is not None


def geostd_monomer_path(root: Optional[str], resname: str) -> Optional[str]:
    """Path to a monomer's geostd CIF, or None.

    geostd buckets monomers by lowercased first character and names each file
    ``data_<CODE>.cif`` (e.g. ``a/data_ALA.cif``, ``m/data_MET.cif``).
    """
    if not root or not resname:
        return None
    candidate = os.path.join(root, resname[0].lower(), f"data_{resname}.cif")
    return candidate if os.path.isfile(candidate) else None


def _sigma(weight: float) -> float:
    return 1.0 / math.sqrt(weight) if weight and weight > 0 else float("nan")


# Each restraint category: how to fetch its proxy array from the geometry manager,
# the value columns it exposes, and how to turn one proxy into (i_seqs, values).
# The value objects come straight from cctbx.geometry_restraints.


def _bond_row(gr, sites, p):
    v = gr.bond(sites, p)
    return tuple(p.i_seqs), {
        "ideal": v.distance_ideal, "model": v.distance_model,
        "delta": v.delta, "sigma": _sigma(p.weight), "residual": v.residual(),
    }


def _angle_row(gr, sites, p):
    v = gr.angle(sites, p)
    return tuple(p.i_seqs), {
        "ideal": v.angle_ideal, "model": v.angle_model,
        "delta": v.delta, "sigma": _sigma(p.weight), "residual": v.residual(),
    }


def _dihedral_row(gr, sites, p):
    v = gr.dihedral(sites, p)
    return tuple(p.i_seqs), {
        "ideal": v.angle_ideal, "model": v.angle_model,
        "delta": v.delta, "sigma": _sigma(p.weight), "residual": v.residual(),
    }


def _chirality_row(gr, sites, p):
    v = gr.chirality(sites, p)
    return tuple(p.i_seqs), {
        "ideal": v.volume_ideal, "model": v.volume_model,
        "delta": v.delta, "sigma": _sigma(p.weight), "residual": v.residual(),
    }


def _planarity_row(gr, sites, p):
    v = gr.planarity(sites, p)
    deltas = list(v.deltas())
    return tuple(p.i_seqs), {
        "rms_delta": v.rms_deltas(),
        "max_delta": max((abs(d) for d in deltas), default=0.0),
        "residual": v.residual(),
    }


# category key -> (label, value columns, proxy accessor name/kind, row function)
_ANGLE_LIKE = ["ideal", "model", "delta", "sigma", "residual"]

CATEGORIES: List[Tuple[str, str, List[str]]] = [
    ("bond", "Bonds", _ANGLE_LIKE),
    ("angle", "Angles", _ANGLE_LIKE),
    ("dihedral", "Dihedrals", _ANGLE_LIKE),
    ("chirality", "Chirality", _ANGLE_LIKE),
    ("planarity", "Planarity", ["rms_delta", "max_delta", "residual"]),
]

_ROW_FUNCS: Dict[str, Callable] = {
    "bond": _bond_row, "angle": _angle_row, "dihedral": _dihedral_row,
    "chirality": _chirality_row, "planarity": _planarity_row,
}


class GeometryRestraints:
    """A model's geometry restraints, read straight from cctbx proxy arrays.

    Builds restraints on the cctbx model if they aren't already present, then
    serves per-category counts and lazily-computed row values. The proxy arrays
    and the sites are references into the model — nothing is materialised per
    restraint.
    """

    def __init__(self, model: Any):
        restraints = model.get_restraints_manager()
        if restraints is None:
            from . import edits
            edits.build_restraints(model)  # one build path, one lock (see edits._BUILD_LOCK)
            restraints = model.get_restraints_manager()
        self.model = model
        self.geometry = restraints.geometry
        self.sites = model.get_sites_cart()
        self._proxy_cache: Dict[str, Any] = {}

    def _proxies(self, category: str):
        if category not in self._proxy_cache:
            g = self.geometry
            if category == "bond":
                proxies = g.get_all_bond_proxies()[0]  # simple (covalent) bonds
            elif category == "angle":
                proxies = g.get_all_angle_proxies()
            elif category == "dihedral":
                proxies = g.get_dihedral_proxies()
            elif category == "chirality":
                proxies = g.chirality_proxies
            elif category == "planarity":
                proxies = g.planarity_proxies
            else:
                raise ValueError(f"unknown restraint category {category!r}")
            self._proxy_cache[category] = proxies
        return self._proxy_cache[category]

    def count(self, category: str) -> int:
        proxies = self._proxies(category)
        return int(proxies.size()) if proxies is not None else 0

    def row(self, category: str, index: int) -> Tuple[Tuple[int, ...], Dict[str, float]]:
        """``(i_seqs, {column: value})`` for one restraint, computed on demand."""
        import cctbx.geometry_restraints as gr

        proxy = self._proxies(category)[index]
        return _ROW_FUNCS[category](gr, self.sites, proxy)

    def indices_within(self, category: str, selected) -> List[int]:
        """Indices of restraints whose atoms are all in ``selected`` (a set of i_seqs).

        Reads each proxy's ``i_seqs`` directly — no value objects built — so it's a
        cheap O(restraints) scan used to filter a category to the current selection.
        """
        proxies = self._proxies(category)
        if proxies is None or not selected:
            return []
        selected = set(selected)
        out: List[int] = []
        for i in range(proxies.size()):
            if all(s in selected for s in proxies[i].i_seqs):
                out.append(i)
        return out


def build_geometry(model: Any) -> Optional[GeometryRestraints]:
    """Build restraints for a cctbx model, or None if the monomer library is absent."""
    if model is None or not monomer_library_available():
        return None
    return GeometryRestraints(model)
