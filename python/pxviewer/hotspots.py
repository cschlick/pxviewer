"""Validation hotspots — one per-atom severity field aggregating several validation metrics.

The point is navigation: colour the model by "how much does this atom deserve a look", so
the eye goes to the places worth rebuilding instead of cross-referencing six separate lists.

The design, and the alternatives it was chosen over, are written up in ``HOTSPOTS.md`` at the
repo root. The rules that matter for reading this file:

* **Severity is calibrated to the community outlier threshold** — 1.0 *is* the MolProbity cut
  for that metric, whatever its native units. So no weights are invented here: the relative
  importance of the metrics is inherited from the reference distributions that set those cuts,
  and a severity means the same thing in every structure.
* **Metrics are assigned to atoms topologically** (:data:`_ASSIGNMENT` and
  :func:`_component_severities`) — a rotamer outlier implicates the side chain and *not* the
  backbone, so the picture says what to fix rather than only where.
* **Aggregation is a p-norm, in two stages** — worst instance within a metric, then
  :data:`P_NORM` across metrics. It preserves severity (a mean would dilute one bad clash into
  an orange smudge) and needs no denominator, which is what lets atoms carrying different
  numbers of applicable metrics stay comparable.

That last property is also why this works without a map. The map-fit term is simply absent,
and since 0 is the p-norm's identity, an atom whose fit was clean scores identically with or
without one. Geometry-only scores sit on the same absolute scale as map-inclusive scores —
they are missing a term, not rescaled. See :func:`score`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

#: Exponent for the p-norm that combines metrics. ``p -> inf`` is a max (pure severity,
#: no credit for corroboration); ``p = 1`` is a sum (triple-counts one error seen by three
#: metrics). 4 behaves like a max while giving a modest bonus when several metrics fire on
#: the same atom. See HOTSPOTS.md "Rule 4".
P_NORM = 4.0

#: Severity is unbounded in principle (a percentage can go to zero). Cap it so one pathological
#: atom cannot dominate every summary statistic.
SEVERITY_CAP = 4.0

# -- calibration anchors: severity 1.0 == the community outlier cut --------------------

#: MolProbity's Ramachandran outlier contour, as a percentage of the reference density.
RAMA_OUTLIER_PCT = 0.05
#: MolProbity's rotamer outlier contour, likewise.
ROTA_OUTLIER_PCT = 0.3
#: MolProbity counts a "serious clash" at >= 0.4 A of van der Waals overlap.
CLASH_OUTLIER_A = 0.4

#: Ramalyze/rotalyze report a percentage, which can be a hard 0 for a severe outlier. Clamp
#: it before the log so severity stays finite.
_MIN_PCT = 1e-4

# Map-fit anchors. Unlike the three above these are *conventions, not calibrated cuts* —
# there is no community outlier threshold for a per-atom Q-score or CC. They are the weakest
# link in the calibration; see the open questions in HOTSPOTS.md.
#: Q-score at/above which an atom is considered a clean fit (severity 0).
QSCORE_GOOD = 0.7
#: Q-score treated as equivalent to an outlier (severity 1.0).
QSCORE_OUTLIER = 0.3
#: Local map-model correlation at/above which the fit is clean (severity 0).
CC_GOOD = 0.8
#: Local map-model correlation treated as equivalent to an outlier (severity 1.0).
CC_OUTLIER = 0.4

#: Display range: 0 is clean, 1.0 is exactly at the outlier cut, 2.0+ is severe. Fixed rather
#: than stretched to the structure's own range, so a colour means the same thing everywhere —
#: the same reasoning that puts Q-score on a fixed 0-1 domain.
DOMAIN = (0.0, 2.0)

#: Green (clean) through yellow at the outlier cut to red (severe). Explicit rather than a
#: Mol* colour-list name because the built-in ramps run the wrong way for a badness scale.
PALETTE = ["#1A9850", "#A6D96A", "#FFD400", "#F46D43", "#B2182B"]

#: Backbone atoms of the residue whose phi/psi produced a Ramachandran score.
_RAMA_ATOMS = frozenset({"N", "CA", "C", "O"})
#: Excluded from the side chain, so a rotamer outlier does not implicate its own backbone.
_MAINCHAIN = frozenset({"N", "CA", "C", "O", "OXT"})

_HYDROGEN = frozenset({"H", "D"})

#: Human-readable component names, in the order they are shown.
COMPONENT_TITLES = {
    "ramachandran": "Ramachandran",
    "rotamer": "Rotamer",
    "clash": "Clash",
    "fit": "Map fit",
}

#: The selectable map-fit terms. ``none`` keeps the score working with no map at all.
FIT_CHOICES = ("qscore", "cc", "none")
FIT_TITLES = {
    "qscore": "Q-score",
    "cc": "Local map-model CC",
    "none": "None (geometry only)",
}


@dataclass
class Hotspots:
    """A computed severity field: the combined per-atom score plus the parts it came from.

    ``values`` is the p-norm combination, one float per atom in the model's atom order (the
    same order the viewer streams, so it can go straight down the attribute path).
    ``components`` holds each metric's own per-atom severity, which is what makes the score
    diagnosable rather than a number to be taken on faith — the aggregate navigates, the
    components say what is actually wrong.
    """

    values: np.ndarray
    components: Dict[str, np.ndarray]
    fit: str
    summary: str
    missing: List[str] = field(default_factory=list)

    def worst_component(self, index: int) -> Optional[str]:
        """Which metric is responsible for atom ``index``'s score, or None if it is clean."""
        best, best_key = 0.0, None
        for key, values in self.components.items():
            if values[index] > best:
                best, best_key = float(values[index]), key
        return best_key


def _surprisal_severity(percent: np.ndarray, outlier_pct: float) -> np.ndarray:
    """Severity from a reference-density percentage, scaled so the outlier cut is 1.0.

    ``percent`` is what ramalyze/rotalyze report: the probability of a conformation this
    likely or better, in percent. Turning it into ``-log10(p)`` gives a surprisal — decades of
    surprise under "this structure is correct" — and dividing by the surprisal *at the cut*
    puts every metric on one scale without choosing a weight.
    """
    p = np.clip(np.asarray(percent, dtype=float), _MIN_PCT, 100.0) / 100.0
    at_cut = -np.log10(outlier_pct / 100.0)
    return np.clip(-np.log10(p) / at_cut, 0.0, SEVERITY_CAP)


def _deficit_severity(value: np.ndarray, good: float, outlier: float) -> np.ndarray:
    """Severity from a fit measure that runs high-is-good, 0 at ``good`` and 1.0 at ``outlier``.

    Linear, because unlike the geometry metrics there is no reference distribution to take a
    tail probability against — these anchors are conventions, and pretending otherwise by
    dressing them as a surprisal would overstate how calibrated they are.
    """
    value = np.asarray(value, dtype=float)
    severity = (good - value) / (good - outlier)
    severity = np.where(np.isfinite(value), severity, 0.0)  # unscored atoms are not outliers
    return np.clip(severity, 0.0, SEVERITY_CAP)


def _residue_atoms(hierarchy: Any) -> Dict[tuple, Dict[str, List[int]]]:
    """``(chain, resid) -> {altloc: [atom index]}``, in the hierarchy's own atom order.

    The per-residue validators report a residue by chain/resid/altloc, so this is the join
    that turns "residue 42 of chain A is a rotamer outlier" into atom indices to paint.
    """
    index: Dict[tuple, Dict[str, List[int]]] = {}
    for i, atom in enumerate(hierarchy.atoms_with_labels()):
        key = (atom.chain_id.strip(), atom.resid().strip())
        index.setdefault(key, {}).setdefault(atom.altloc.strip(), []).append(i)
    return index


def _atoms_for(index: Dict[tuple, Dict[str, List[int]]], chain: str, resid: str,
               altloc: str = "") -> List[int]:
    """Atom indices of one residue. A validator's altloc selects that conformer plus the
    atoms shared by all of them (blank altloc); no altloc takes the whole residue."""
    by_alt = index.get((chain.strip(), resid.strip()))
    if not by_alt:
        return []
    altloc = (altloc or "").strip()
    if not altloc:
        return [i for group in by_alt.values() for i in group]
    return sorted(by_alt.get("", []) + by_alt.get(altloc, []))


def _atom_names(hierarchy: Any) -> List[str]:
    return [n.strip() for n in hierarchy.atoms().extract_name()]


def _hydrogen_parents(hierarchy: Any) -> Dict[int, int]:
    """Each hydrogen's parent heavy atom — the nearest one in its own atom group.

    Needed because Probe finds clashes *through* hydrogens, but hydrogens carry no Q-score and
    are not drawn in ribbon or heavy-atom views. Without this the clash signal would vanish
    exactly when hydrogens are hidden, which is the default. See HOTSPOTS.md "Rule 5".
    """
    parents: Dict[int, int] = {}
    atoms = hierarchy.atoms()
    xyz = atoms.extract_xyz().as_numpy_array()
    elements = [e.strip().upper() for e in atoms.extract_element()]
    for atom_group in hierarchy.atom_groups():
        indices = [a.i_seq for a in atom_group.atoms()]
        heavy = [i for i in indices if elements[i] not in _HYDROGEN]
        if not heavy:
            continue
        heavy_xyz = xyz[heavy]
        for i in indices:
            if elements[i] in _HYDROGEN:
                parents[i] = heavy[int(np.argmin(((heavy_xyz - xyz[i]) ** 2).sum(axis=1)))]
    return parents


def ramachandran_severity(model: Any, n_atoms: int) -> np.ndarray:
    """Per-atom Ramachandran severity, on the backbone N/CA/C/O of the scored residue.

    Assigned narrowly: phi/psi involve atoms from three residues, but implicating the
    neighbours would smear one residue's problem onto two innocent ones.
    """
    from mmtbx.validation.ramalyze import ramalyze

    values = np.zeros(n_atoms, dtype=float)
    hierarchy = model.get_hierarchy()
    index = _residue_atoms(hierarchy)
    names = _atom_names(hierarchy)
    result = ramalyze(pdb_hierarchy=hierarchy, outliers_only=False)
    for res in result.results:
        if res.score is None:
            continue
        severity = float(_surprisal_severity(np.array([res.score]), RAMA_OUTLIER_PCT)[0])
        if severity <= 0:
            continue
        for i in _atoms_for(index, res.chain_id, res.resid, getattr(res, "altloc", "")):
            if names[i] in _RAMA_ATOMS:
                values[i] = max(values[i], severity)
    return values


def rotamer_severity(model: Any, n_atoms: int) -> np.ndarray:
    """Per-atom rotamer severity, on side-chain atoms only.

    A rotamer outlier is a statement about chi angles, so it says nothing about that residue's
    own backbone — painting the whole residue would assert that it does.
    """
    from mmtbx.validation.rotalyze import rotalyze

    values = np.zeros(n_atoms, dtype=float)
    hierarchy = model.get_hierarchy()
    index = _residue_atoms(hierarchy)
    names = _atom_names(hierarchy)
    result = rotalyze(pdb_hierarchy=hierarchy, outliers_only=False)
    for res in result.results:
        if res.score is None:
            continue
        severity = float(_surprisal_severity(np.array([res.score]), ROTA_OUTLIER_PCT)[0])
        if severity <= 0:
            continue
        for i in _atoms_for(index, res.chain_id, res.resid, getattr(res, "altloc", "")):
            if names[i] not in _MAINCHAIN:
                values[i] = max(values[i], severity)
    return values


def clash_severity(model: Any, n_atoms: int, *, data_manager: Any = None) -> np.ndarray:
    """Per-atom clash severity from Probe's all-atom contacts.

    Both atoms of a clashing pair carry the full severity — neither is the innocent party. An
    atom in several clashes keeps its *worst*, not their sum: a badly placed atom typically
    hits three neighbours at once, and that is one mistake seen three times.
    """
    from .probe import run_probe_dots

    values = np.zeros(n_atoms, dtype=float)
    hierarchy = model.get_hierarchy()
    by_key: Dict[tuple, int] = {}
    for i, atom in enumerate(hierarchy.atoms_with_labels()):
        by_key[(atom.chain_id.strip(), atom.resid().strip(),
                atom.name.strip(), atom.altloc.strip())] = i

    def lookup(side: dict) -> Optional[int]:
        resid = f"{side.get('resID', '')}{side.get('iCode', '') or ''}".strip()
        return by_key.get((str(side.get("chainID", "")).strip(), resid,
                           str(side.get("atomName", "")).strip(),
                           str(side.get("alt", "")).strip()))

    for row in run_probe_dots(model, data_manager=data_manager):
        # Probe reports a signed gap; an overlap is negative. Severity is anchored so that
        # MolProbity's 0.4 A "serious clash" is exactly 1.0.
        overlap = -float(row.get("gap", 0.0))
        if overlap <= 0:
            continue
        severity = min(overlap / CLASH_OUTLIER_A, SEVERITY_CAP)
        for side in (row.get("src"), row.get("target")):
            if side is None:
                continue
            i = lookup(side)
            if i is not None:
                values[i] = max(values[i], severity)

    # Rule 5: a heavy atom inherits the worst of its hydrogens, so the signal survives views
    # that do not draw them. Max, not sum, or an atom with several clashing H is inflated.
    for h, parent in _hydrogen_parents(hierarchy).items():
        values[parent] = max(values[parent], values[h])
    return values


def fit_severity(mmm: Any, n_atoms: int, kind: str) -> np.ndarray:
    """Per-atom map-fit severity, from Q-score or from local map-model correlation."""
    if kind == "qscore":
        from .qscore import per_atom_qscore

        return _deficit_severity(per_atom_qscore(mmm), QSCORE_GOOD, QSCORE_OUTLIER)
    if kind == "cc":
        return _deficit_severity(per_atom_cc(mmm), CC_GOOD, CC_OUTLIER)
    raise ValueError(f"unknown fit term {kind!r}")


def per_atom_cc(mmm: Any, *, radius: float = 2.0) -> np.ndarray:
    """Local map-model correlation at every atom: the experimental map against a map computed
    from the model, correlated in a ``radius`` sphere around each atom.

    Cheaper than Q-score and the familiar real-space measure, but it is a *local* correlation,
    so it is blind to a peak being in the wrong place as long as the shapes agree there.
    """
    import mmtbx.maps.correlation as correlation

    model = mmm.model()
    map_data = mmm.map_manager().map_data()
    d_min = mmm.resolution()
    xrs = model.get_xray_structure()
    maps = correlation.from_map_and_xray_structure_or_fmodel(
        xray_structure=xrs, map_data=map_data, d_min=d_min)
    # Not maps.cc(per_atom=True): that branch reaches through self.fmodel, which is None when
    # the object was built from an xray_structure, so it raises. The underlying function takes
    # exactly what we already have.
    values = correlation.from_map_map_atoms_per_atom(
        map_1=maps.map_data, map_2=maps.map_model, sites_cart=maps.sites_cart,
        unit_cell=xrs.unit_cell(), radius=radius)
    return np.asarray(values, dtype=float).reshape(-1)


def combine(components: Dict[str, np.ndarray], *, p: float = P_NORM) -> np.ndarray:
    """Combine per-metric severities into one field with a p-norm.

    No denominator, and 0 is the identity — which is what keeps atoms comparable when they
    carry different numbers of applicable metrics (a carbonyl O has no rotamer term at all),
    and what lets the map term be dropped entirely without moving the scale.
    """
    if not components:
        raise ValueError("no components to combine")
    stacked = np.vstack([np.asarray(v, dtype=float) for v in components.values()])
    return np.clip((stacked ** p).sum(axis=0) ** (1.0 / p), 0.0, SEVERITY_CAP)


def score(model: Any, *, mmm: Any = None, fit: str = "qscore", p: float = P_NORM,
          data_manager: Any = None) -> Hotspots:
    """Compute the hotspot field for ``model``.

    ``fit`` selects the map term — ``'qscore'``, ``'cc'``, or ``'none'`` for geometry only.
    Without ``mmm`` the map term is dropped whatever ``fit`` says, and the result records that
    in :attr:`Hotspots.missing`.

    Dropping it does not rescale anything: severity 1.0 still means "at the outlier cut" for
    every metric that *is* present, so a geometry-only score is directly comparable to a
    map-inclusive one — it is the same scale with one fewer term, and identical for any atom
    whose fit was clean.
    """
    if fit not in FIT_CHOICES:
        raise ValueError(f"fit must be one of {FIT_CHOICES}, not {fit!r}")
    n_atoms = model.get_number_of_atoms()
    components: Dict[str, np.ndarray] = {}
    missing: List[str] = []

    components["ramachandran"] = ramachandran_severity(model, n_atoms)
    components["rotamer"] = rotamer_severity(model, n_atoms)
    components["clash"] = clash_severity(model, n_atoms, data_manager=data_manager)

    if fit == "none":
        missing.append("map fit (geometry only)")
    elif mmm is None:
        missing.append(f"map fit ({FIT_TITLES[fit]} needs a map paired with this model)")
    else:
        components["fit"] = fit_severity(mmm, n_atoms, fit)

    values = combine(components, p=p)
    n_over = int((values >= 1.0).sum())
    parts = ", ".join(
        f"{COMPONENT_TITLES[k]} {int((v >= 1.0).sum())}" for k, v in components.items())
    summary = (f"{n_over} atoms at or past an outlier threshold "
               f"(worst {values.max():.2f}) — {parts}")
    if missing:
        summary += f"; no {'; no '.join(missing)}"
    return Hotspots(values=values, components=components,
                    fit=fit if "fit" in components else "none",
                    summary=summary, missing=missing)


def residue_broadcast(model: Any, values: np.ndarray) -> np.ndarray:
    """``values`` with every atom of a residue raised to that residue's worst value.

    Per-atom severity is the right thing to *compute* — a rotamer outlier implicates the side
    chain and not the backbone, and saying so is the whole point of Rule 1. But a cartoon or
    ribbon draws no side chains, so on those representations the rotamer component would be
    invisible: the atoms carrying it are not on screen. Broadcasting the residue max is what
    a ribbon can actually show, and it loses nothing the ribbon could have drawn anyway.

    Used for display only; the per-atom field is what the table and the components report.
    """
    values = np.asarray(values, dtype=float)
    out = values.copy()
    by_residue: Dict[tuple, List[int]] = {}
    for i, atom in enumerate(model.get_hierarchy().atoms_with_labels()):
        by_residue.setdefault((atom.chain_id, atom.resid(), atom.altloc), []).append(i)
    for indices in by_residue.values():
        worst = max(values[i] for i in indices)
        for i in indices:
            out[i] = worst
    return out


# -- the spatial field ----------------------------------------------------------------

#: Voxel size for the severity field. Coarse on purpose: the field is smoothed over several
#: angstroms, so there is nothing finer to resolve and a fine grid only costs memory.
FIELD_SPACING = 1.0

#: Kernel width. Chosen as the *action* scale — a residue plus its environment — rather than
#: the map resolution, because the claim a contour makes is "there is work to do here", and
#: that is the granularity at which work happens.
FIELD_SIGMA = 2.5

#: Kernel support, past which the weight is negligible.
FIELD_CUTOFF = 3.0 * FIELD_SIGMA

#: Atoms below this contribute nothing worth splatting. Most atoms in a decent model are
#: exactly 0, so skipping them is most of why this is cheap.
FIELD_FLOOR = 0.05

#: The contour to open at: the calibrated outlier cut, so the shell has an exact meaning
#: rather than being a tuned opacity ramp. Raise the level to 2.0 for severe-only.
FIELD_ISO = 1.0

#: Fixed, and deliberately unlike any map color, so a severity shell is never misread as
#: density sitting in the same scene.
FIELD_COLOR = "#FF2D95"


def severity_field(model: Any, values: np.ndarray, *, spacing: float = FIELD_SPACING,
                   sigma: float = FIELD_SIGMA, p: float = P_NORM,
                   floor: float = FIELD_FLOOR):
    """Smear the per-atom severity into a grid, for contouring in 3-D.

    Returns ``(array, spacing, origin)`` where ``origin`` is the integer grid offset — what
    :meth:`VolumeData.from_numpy` wants — so the box lands on the model's own coordinates.

    Per-atom coloring only shows the surface, so a buried hotspot stays hidden until you
    rotate or clip into it, and it costs the atom-color channel. A contour of this field is
    visible through the structure and leaves that channel free.

    The combination is the same distance-weighted p-norm used across metrics, **not a sum**:
    summing would light up the core for no better reason than having more atoms in it, which
    would make the field a map of where the protein is rather than of where the trouble is.

    The weight is 1 within a voxel's reach of an atom, so the field at its nearest grid point
    is *at least* that atom's severity — a contour at 1.0 therefore always encloses every
    outlier atom, which a bare Gaussian would not guarantee. Neighbours can
    only add, so a cluster of merely-poor atoms can also reach 1.0 with none of them
    individually past the cut. That is the regional aggregation the field is for, and it is
    the one place the field says something the per-atom coloring does not; read the shell as
    "there is work in here", and the table for which residue it is.
    """
    values = np.asarray(values, dtype=float)
    xyz = model.get_hierarchy().atoms().extract_xyz().as_numpy_array()
    hot = np.flatnonzero(np.nan_to_num(values) > floor)

    # Half a voxel diagonal: the furthest an atom can be from the nearest grid point.
    plateau = spacing * np.sqrt(3.0) / 2.0

    pad = FIELD_CUTOFF
    lo = (xyz.min(axis=0) - pad) if len(xyz) else np.zeros(3)
    hi = (xyz.max(axis=0) + pad) if len(xyz) else np.ones(3)
    origin = np.floor(lo / spacing).astype(int)
    shape = tuple(int(np.ceil(h / spacing)) - o + 1 for h, o in zip(hi, origin))

    # Accumulate the p-th powers, then take the p-th root once at the end — that *is* the
    # p-norm, and doing it per voxel would be the same arithmetic done many more times.
    acc = np.zeros(shape, dtype=float)
    if hot.size:
        reach = int(np.ceil(FIELD_CUTOFF / spacing))
        axes = [np.arange(n) for n in shape]
        for i in hot:
            centre = xyz[i] / spacing - origin
            slices, coords = [], []
            for axis in range(3):
                start = max(0, int(np.floor(centre[axis])) - reach)
                stop = min(shape[axis], int(np.ceil(centre[axis])) + reach + 1)
                if start >= stop:
                    break
                slices.append(slice(start, stop))
                coords.append((axes[axis][start:stop] - centre[axis]) * spacing)
            if len(slices) < 3:
                continue  # the atom's support falls entirely outside the box
            d2 = (coords[0][:, None, None] ** 2 + coords[1][None, :, None] ** 2
                  + coords[2][None, None, :] ** 2)
            # Flat-topped within a voxel's reach of the atom. An atom sits between grid
            # points, so the nearest voxel samples the kernel up to half a voxel diagonal
            # away; with a bare Gaussian that lands *below* the atom's own severity and a
            # contour at 1.0 could miss an atom the table lists as an outlier. The plateau
            # makes "the shell encloses every outlier" exact rather than nearly true.
            d = np.sqrt(d2) - plateau
            weight = np.exp(-np.maximum(d, 0.0) ** 2 / (2.0 * sigma * sigma))
            acc[tuple(slices)] += (weight * values[i]) ** p

    field = np.clip(acc ** (1.0 / p), 0.0, SEVERITY_CAP)
    return field, spacing, tuple(int(o) for o in origin)


def residue_columns(hotspots: Hotspots) -> List[str]:
    """Columns of :func:`residue_rows` — the score, then every component behind it.

    The breakdown is not decoration: the aggregate is only allowed to *rank*, and the
    components are what say what is actually wrong. A severity with no visible provenance is
    the thing that gets screenshotted and refined against.
    """
    return ["chain", "resid", "res", "severity"] + [
        COMPONENT_TITLES[k] for k in hotspots.components]


def residue_rows(model: Any, hotspots: Hotspots, *, limit: int = 200) -> List[list]:
    """The hotspot field rolled up per residue, worst first — the list you work down.

    The roll-up is a **max** over the residue's atoms, never a sum: Ramachandran is assigned to
    four backbone atoms, so summing would count one phi/psi four times, and it would rank a
    TRP above a GLY for no reason but size.
    """
    hierarchy = model.get_hierarchy()
    by_residue: Dict[tuple, List[int]] = {}
    for i, atom in enumerate(hierarchy.atoms_with_labels()):
        key = (atom.chain_id.strip(), atom.resid().strip(), atom.resname.strip())
        by_residue.setdefault(key, []).append(i)

    scored = []
    for key, indices in by_residue.items():
        value = max(float(hotspots.values[i]) for i in indices)
        if value <= 0:
            continue  # clean; nothing to work on
        parts = [max(float(component[i]) for i in indices)
                 for component in hotspots.components.values()]
        scored.append((value, key, parts))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [[chain, resid, resname, f"{value:.2f}"]
            + [f"{part:.2f}" if part > 0 else "" for part in parts]
            for value, (chain, resid, resname), parts in scored[:limit]]
