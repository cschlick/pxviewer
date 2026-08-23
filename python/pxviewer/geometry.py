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
    "then reopen the model.\n\n"
    "Note that a bare geostd checkout carries no mon_lib, so the monomers that ship\n"
    "only there (HEM among them) will not resolve. The chem_data package has both."
)


#: The directories ``chem_data`` ships, in the order cctbx's own cascade searches them.
#: Both are needed. geostd holds the bulk of the ligands (~54k), but a small CCP4-derived
#: set lives only in mon_lib (~140, HEM among them) — and, decisively, the two ship
#: *different* ``list/mon_lib_list.cif`` indices. Pointing cctbx at one directory makes it
#: read that directory's index and never consult the other.
_CHEM_DATA_SUBDIRS = ("geostd", "mon_lib")


def _chem_data_subdir(name: str) -> Optional[str]:
    """A directory shipped inside the importable ``chem_data`` package, or None.

    ``chem_data`` is a plain importable package, so this resolves regardless of the
    Python version or platform layout (no hard-coded ``site-packages`` path) — which
    is what lets the conda package find the monomer library without an activation hook.
    """
    try:
        import chem_data
    except Exception:
        return None
    path = os.path.join(os.path.dirname(chem_data.__file__), name)
    return path if os.path.isdir(path) else None


def _is_chem_data_subdir(path: str) -> bool:
    """Whether a path is one of chem_data's own library directories.

    A redirect naming chem_data's own geostd is not a user preference: it is what
    pxviewer's former ``activate.d`` hook wrote, and it is precisely the value that
    hides mon_lib. Recognising it lets an env that still has that hook self-heal
    instead of staying half-broken until someone deletes the file.
    """
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    for name in _CHEM_DATA_SUBDIRS:
        own = _chem_data_subdir(name)
        if own and os.path.realpath(own) == real:
            return True
    return False


def _explicit_monomer_library() -> Optional[str]:
    """A monomer library the user deliberately pointed at, if it exists on disk.

    Ignores a redirect that merely names chem_data's own directories (see
    :func:`_is_chem_data_subdir`) — cctbx searches those anyway, and honouring it as an
    override would pin the search to one of the two.
    """
    for var in ("MMTBX_CCP4_MONOMER_LIB", "CLIBD_MON"):
        path = os.environ.get(var)
        if path and os.path.isdir(path) and not _is_chem_data_subdir(path):
            return path
    return None


def _cctbx_resolves_chem_data() -> bool:
    """Whether cctbx can reach chem_data through its own repository cascade.

    ``mmtbx.monomer_library.server.find_mon_lib_file`` searches ``chem_data/geostd`` and
    ``chem_data/mon_lib`` relative to libtbx's repository paths, which on a conda install
    include site-packages. When that works, cctbx finds *both* directories by itself and
    we must not set ``MMTBX_CCP4_MONOMER_LIB`` — see :func:`configure_monomer_library`.
    """
    try:
        import libtbx.load_env  # noqa: F401  (populates libtbx.env)
        import libtbx

        return any(
            libtbx.env.find_in_repositories(relative_path="chem_data/" + name)
            for name in _CHEM_DATA_SUBDIRS
        )
    except Exception:
        return False


def monomer_library_roots() -> Tuple[str, ...]:
    """Every monomer-library root that will be searched, in precedence order.

    An explicit ``MMTBX_CCP4_MONOMER_LIB`` / ``CLIBD_MON`` wins outright and is used
    alone, matching cctbx. Otherwise this is both directories ``chem_data`` ships.
    """
    explicit = _explicit_monomer_library()
    if explicit:
        return (explicit,)
    return tuple(
        path for path in (_chem_data_subdir(n) for n in _CHEM_DATA_SUBDIRS) if path
    )


def configure_monomer_library() -> Optional[str]:
    """Make cctbx able to find a monomer library, and report the root in effect.

    Deliberately does **not** export ``MMTBX_CCP4_MONOMER_LIB`` when cctbx can already
    reach chem_data on its own. That variable is a single-directory redirect that
    ``find_mon_lib_file`` consults *before* its own cascade, so exporting it pins the
    search to one directory: with it set to geostd, cctbx loads geostd's
    ``mon_lib_list.cif`` and HEM — which lives only in mon_lib — stops resolving, while
    ALA and the other 54k geostd monomers keep working. The breakage is partial, which
    is what makes it hard to spot. Leaving the variable unset lets cctbx cascade through
    both directories, which is what it is designed to do.

    A stale value (an old ``activate.d`` hook naming a Python version that no longer
    exists) is cleared rather than left in place: ``find_mon_lib_file`` reads the raw
    variable, so a path that fails our ``isdir`` check would still be handed to cctbx.
    """
    explicit = _explicit_monomer_library()
    if explicit:
        return explicit

    # Drop any redirect that is not a real override -- a stale path, or one naming
    # chem_data's own directories. find_mon_lib_file reads these variables raw and
    # consults them before its cascade, so leaving one set would narrow the search.
    for var in ("MMTBX_CCP4_MONOMER_LIB", "CLIBD_MON"):
        if os.environ.get(var):
            del os.environ[var]

    roots = monomer_library_roots()
    if not roots:
        return None
    if _cctbx_resolves_chem_data():
        # cctbx finds both directories itself; setting the variable would only narrow it.
        return roots[0]
    # chem_data is importable but outside libtbx's repository paths, so the cascade
    # cannot see it. A single-directory redirect is then better than nothing, even
    # though it costs the monomers that live only in the other directory.
    os.environ["MMTBX_CCP4_MONOMER_LIB"] = roots[0]
    return roots[0]


def monomer_library_root() -> Optional[str]:
    """The primary monomer-library (geostd) directory, or None.

    Kept for callers that want a single directory to look a file up in; use
    :func:`monomer_library_roots` when a monomer may live in either directory.
    """
    roots = monomer_library_roots()
    return roots[0] if roots else None


def monomer_library_available() -> bool:
    """Whether cctbx can find a monomer library to build restraints from."""
    return bool(monomer_library_roots())


def geostd_monomer_path(root: Optional[str], resname: str) -> Optional[str]:
    """Path to a monomer's CIF under one library root, or None.

    Both libraries bucket monomers by lowercased first character but name the files
    differently: geostd uses ``data_<CODE>.cif`` (``a/data_ALA.cif``), mon_lib uses
    ``<CODE>.cif`` (``h/HEM.cif``). Try both, so one root argument works for either.
    """
    if not root or not resname:
        return None
    bucket = os.path.join(root, resname[0].lower())
    for name in (f"data_{resname}.cif", f"{resname}.cif"):
        candidate = os.path.join(bucket, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def monomer_cif_path(resname: str) -> Optional[str]:
    """Path to a monomer's CIF in whichever library root ships it, or None.

    Searches every root :func:`monomer_library_roots` reports, so a monomer carried
    only by mon_lib (HEM) is found as readily as one carried only by geostd.
    """
    for root in monomer_library_roots():
        path = geostd_monomer_path(root, resname)
        if path:
            return path
    return None


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


#: Origins worth naming more helpfully than cctbx does. Everything else uses cctbx's own
#: key verbatim, which is already descriptive ("metal coordination", "hydrogen bonds").
#: ``edits`` earns a gloss because it is the one origin the user creates themselves, and
#: "edits" alone does not say so.
_ORIGIN_GLOSS = {
    0: "covalent geometry (monomer library)",
    4: "edits (user-defined)",
}


def origin_name(origin_id: int) -> str:
    """A short description of a restraint origin, e.g. ``"edits (user-defined)"``.

    Falls back to the bare number for an origin this cctbx does not know, rather than
    raising: the id came out of a proxy, so it is real whether or not it is nameable.
    """
    gloss = _ORIGIN_GLOSS.get(origin_id)
    if gloss:
        return gloss
    try:
        from cctbx.geometry_restraints.linking_class import linking_class

        key = linking_class().get_origin_key(origin_id)
    except Exception:  # pragma: no cover - defensive; unknown cctbx build
        key = None
    return key or "origin %d" % origin_id


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

    def origins(self, category: str) -> List[Tuple[int, str, int]]:
        """``(origin_id, name, count)`` for each restraint origin present, id order.

        Only origins actually present are returned. cctbx defines well over a hundred --
        one per link type -- and a dropdown listing all of them, nearly every entry empty,
        would bury the two or three that a given model actually has.
        """
        proxies = self._proxies(category)
        if proxies is None or not proxies.size():
            return []
        counts: Dict[int, int] = {}
        for i in range(proxies.size()):
            oid = getattr(proxies[i], "origin_id", 0)
            counts[oid] = counts.get(oid, 0) + 1
        return [(oid, origin_name(oid), counts[oid]) for oid in sorted(counts)]

    def indices_with_origin(self, category: str, origin_id: Optional[int]) -> Optional[List[int]]:
        """Indices of restraints from one origin, or ``None`` for "no origin filter"."""
        if origin_id is None:
            return None
        proxies = self._proxies(category)
        if proxies is None:
            return []
        return [i for i in range(proxies.size())
                if getattr(proxies[i], "origin_id", 0) == origin_id]

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
