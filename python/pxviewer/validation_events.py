"""Shared validation extraction: one definition of *what is wrong and where*.

This module is deliberately **standalone and copyable**. It imports nothing from pxviewer
and uses no relative imports, so the same file can sit in the pxviewer package and in the
sibling ``hotspots/`` generator, and both then localize validation identically. If you copy
it, copy it whole and do not fork the localization rules — the point of the file is that a
disagreement between the two projects becomes impossible rather than merely unlikely.

**It carries native values, not calibrated ones.** A Ramachandran result travels as its
probability *percentage*, a rotamer result as its percentage, a clash as its *overlap in
angstroms* — plus the validator's own ``outlier`` boolean. Turning those into a score is the
caller's business, and the two callers deliberately disagree:

* pxviewer maps them to unbounded **surprisal severity** on a ``[0, 4]`` scale where 1.0 is
  the community cut (see ``pxviewer/hotspots.py`` and HOTSPOTS.md);
* the generator maps them to bounded **concern** in ``[0, 1]`` with its own log
  interpolation between a "good" and a class-specific "bad" percentage.

Both are correct for their purpose and neither is a rescaling of the other. What must *not*
differ is which residue a result belongs to, which atoms it implicates, and whether the
validator called it an outlier. That is what lives here.

The other half of the file is :func:`check_field_agreement`, the sanity check a field
generator should run on its own output: take the field back to the atoms, and confirm the
hot places are the places validation actually complained about.

Requires cctbx/mmtbx (``ramalyze``, ``rotalyze``, ``probe2``) and numpy. Run it directly for
a self-test on any model:

    libtbx.python validation_events.py MODEL.pdb
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field as _field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# -- localization rules -------------------------------------------------------
#
# These sets are the topological assignment: which atoms a result is *about*. They are
# shared because a field generator and a per-atom scorer disagreeing here would put the
# same problem in two different places.

#: Backbone atoms of the residue whose phi/psi produced a Ramachandran score. Narrow on
#: purpose: phi/psi involve three residues, but implicating the neighbours smears one
#: residue's problem onto two innocent ones.
RAMA_ATOMS = frozenset({"N", "CA", "C", "O"})

#: Excluded from the side chain, so a rotamer outlier does not implicate its own backbone.
#: A rotamer result is a statement about chi angles and says nothing about where the
#: backbone sits.
MAINCHAIN = frozenset({"N", "CA", "C", "O", "OXT"})

_HYDROGEN = frozenset({"H", "D"})

#: MolProbity's reporting boundary for a serious clash, in angstroms of overlap. Recorded
#: here as provenance for the ``outlier`` flag only — neither project's calibration curve
#: is defined in this file.
CLASH_OUTLIER_A = 0.40

Xyz = Tuple[float, float, float]


@dataclass(frozen=True)
class ResidueKey:
    """Canonical residue identity, agreed between projects.

    ``resseq`` is kept as the integer cctbx reports and ``icode`` separately, rather than the
    formatted ``resid`` string, because the two projects were formatting it differently and
    string keys silently failed to join.
    """

    chain: str
    resseq: int
    icode: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.chain}{self.resseq}{self.icode}".strip()


@dataclass
class ValidationEvent:
    """One thing a validator complained about, with its native value and its atoms.

    ``value`` is in ``units`` and is **not** calibrated: percentages stay percentages and a
    clash overlap stays angstroms. ``outlier`` is the validator's own boolean, not a
    re-derivation from ``value`` — so a caller can always recover exactly the set MolProbity
    would report, whatever its own scoring does.
    """

    metric: str                       # 'rama' | 'rota' | 'clash'
    value: float                      # native: percent for rama/rota, A overlap for clash
    units: str                        # 'percent' | 'angstrom'
    outlier: bool                     # the validator's own call
    residue: Optional[ResidueKey] = None
    atom_indices: Tuple[int, ...] = ()   # indices into hierarchy.atoms() order
    atoms_xyz: Tuple[Xyz, ...] = ()
    detail: Dict[str, Any] = _field(default_factory=dict)


# -- residue/atom indexing ----------------------------------------------------


def residue_atom_index(hierarchy: Any) -> Dict[ResidueKey, Dict[str, List[int]]]:
    """``ResidueKey -> {altloc: [atom index, ...]}`` over ``hierarchy.atoms()`` order.

    Altloc is kept as a second level rather than folded into the key so a validator result
    that names a conformer can select that conformer plus the atoms shared by all of them
    (blank altloc), which is what :func:`atoms_of` does.
    """
    index: Dict[ResidueKey, Dict[str, List[int]]] = {}
    for model in hierarchy.models():
        for chain in model.chains():
            for rg in chain.residue_groups():
                key = ResidueKey(chain.id.strip(), rg.resseq_as_int(), rg.icode.strip())
                by_alt = index.setdefault(key, {})
                for atom_group in rg.atom_groups():
                    alt = atom_group.altloc.strip()
                    slot = by_alt.setdefault(alt, [])
                    for atom in atom_group.atoms():
                        slot.append(atom.i_seq)
    return index


def atoms_of(index: Dict[ResidueKey, Dict[str, List[int]]], key: ResidueKey,
             altloc: str = "") -> List[int]:
    """Atom indices of one residue. A named altloc selects that conformer plus the atoms
    shared by all of them; no altloc takes the whole residue."""
    by_alt = index.get(key)
    if not by_alt:
        return []
    altloc = (altloc or "").strip()
    if not altloc:
        return sorted(i for group in by_alt.values() for i in group)
    return sorted(by_alt.get("", []) + by_alt.get(altloc, []))


def hydrogen_parents(hierarchy: Any) -> Dict[int, int]:
    """Each hydrogen's parent heavy atom — the nearest one in its own atom group.

    Probe finds clashes *through* hydrogens, but hydrogens are absent from a heavy-atom
    model and undrawn in most representations, so the signal has to be handed to a heavy
    atom or it disappears exactly when hydrogens are hidden.
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


def _names(hierarchy: Any) -> List[str]:
    return [n.strip() for n in hierarchy.atoms().extract_name()]


def _xyz(hierarchy: Any) -> np.ndarray:
    return hierarchy.atoms().extract_xyz().as_numpy_array()


def _key_of(result: Any) -> ResidueKey:
    return ResidueKey(result.chain_id.strip(), result.resseq_as_int(), result.icode.strip())


# -- extraction ---------------------------------------------------------------


def extract_ramachandran(hierarchy: Any, *, index=None, ramalyze_result=None
                         ) -> List[ValidationEvent]:
    """Every Ramachandran result, on residue *i*'s own N/CA/C/O.

    ``ramalyze_result`` lets a caller pass a run it already paid for (pxviewer shares one
    across its Validation and Hotspots tabs).
    """
    from mmtbx.validation import ramalyze

    if ramalyze_result is None:
        ramalyze_result = ramalyze.ramalyze(pdb_hierarchy=hierarchy, outliers_only=False)
    index = residue_atom_index(hierarchy) if index is None else index
    names, xyz = _names(hierarchy), _xyz(hierarchy)

    events = []
    for r in ramalyze_result.results:
        if r.score is None:
            continue
        key = _key_of(r)
        picked = [i for i in atoms_of(index, key, getattr(r, "altloc", ""))
                  if names[i] in RAMA_ATOMS]
        events.append(ValidationEvent(
            metric="rama", value=float(r.score), units="percent",
            outlier=bool(r.is_outlier()), residue=key,
            atom_indices=tuple(picked),
            atoms_xyz=tuple(tuple(float(c) for c in xyz[i]) for i in picked) or
                      ((tuple(float(c) for c in r.xyz),) if r.xyz is not None else ()),
            detail={"id": r.id_str().strip(), "res_type": r.res_type,
                    "rama_type": r.rama_type}))
    return events


def extract_rotamer(hierarchy: Any, *, index=None, rotalyze_result=None
                    ) -> List[ValidationEvent]:
    """Every rotamer result, on side-chain atoms only (never the residue's own backbone)."""
    from mmtbx.validation import rotalyze

    if rotalyze_result is None:
        rotalyze_result = rotalyze.rotalyze(pdb_hierarchy=hierarchy, outliers_only=False)
    index = residue_atom_index(hierarchy) if index is None else index
    names, xyz = _names(hierarchy), _xyz(hierarchy)

    events = []
    for t in rotalyze_result.results:
        if t.score is None:
            continue
        key = _key_of(t)
        picked = [i for i in atoms_of(index, key, getattr(t, "altloc", ""))
                  if names[i] not in MAINCHAIN]
        events.append(ValidationEvent(
            metric="rota", value=float(t.score), units="percent",
            outlier=bool(t.is_outlier()), residue=key,
            atom_indices=tuple(picked),
            atoms_xyz=tuple(tuple(float(c) for c in xyz[i]) for i in picked) or
                      ((tuple(float(c) for c in t.xyz),) if t.xyz is not None else ()),
            detail={"id": t.id_str().strip(), "rotamer": t.rotamer_name}))
    return events


def probe2_dots(model: Any, *, data_manager: Any = None) -> List[dict]:
    """Run cctbx's **probe2** in memory and return its raw ``flat_results`` dots.

    Deliberately probe2 and not ``mmtbx.validation.clashscore``: clashscore shells out to
    the classic Duke ``probe`` binary, which is frequently not installed (it is not, in the
    pxviewer conda environment), whereas probe2 ships with cctbx and is the same MolProbity
    contact analysis reimplemented. Using it here is what lets the clash channel work at all
    in environments that only have cctbx.

    No disk round-trip: the model goes into a DataManager and probe2's JSON is captured from
    ``run()``'s return value with ``output.write_files=False``.
    """
    import iotbx.phil
    from iotbx.data_manager import DataManager
    from libtbx.program_template import ProgramTemplate
    from libtbx.utils import null_out
    from mmtbx.programs import probe2

    dm = data_manager
    if dm is None:
        dm = DataManager()
        dm.set_overwrite(True)
    dm.add_model("model", model)

    master = iotbx.phil.parse(probe2.Program.master_phil_str, process_includes=True)
    master.adopt_scope(iotbx.phil.parse(ProgramTemplate.output_phil_str))
    params = master.extract()
    params.approach = "self"
    params.source_selection = "all"
    params.output.format = "json"
    params.output.write_files = False
    params.output.filename = "probe.json"   # set but never written; keeps validate() quiet
    elements = {e.strip().upper() for e in model.get_hierarchy().atoms().extract_element()}
    if not (elements & _HYDROGEN):
        params.ignore_lack_of_explicit_hydrogens = True

    task = probe2.Program(dm, params, master_phil=master, logger=null_out())
    task.validate()
    _results, out_string = task.run()
    return json.loads(out_string)["flat_results"]


def extract_clashes(model: Any, *, dots=None, data_manager: Any = None,
                    inherit_to_heavy: bool = True) -> List[ValidationEvent]:
    """Every steric overlap probe2 reports, on **both** atoms of the pair.

    Neither atom of a clashing pair is the innocent party, so both carry it. ``dots`` accepts
    an already-computed probe2 ``flat_results`` (pxviewer caches one per model, since probe2
    is the most expensive step in the stack).

    ``inherit_to_heavy`` hands each hydrogen's clash to its parent heavy atom *in addition*
    to the hydrogen, so the signal survives a heavy-atom view — see :func:`hydrogen_parents`.
    ``model`` here must be the same model the dots were computed from.
    """
    hierarchy = model.get_hierarchy()
    if dots is None:
        dots = probe2_dots(model, data_manager=data_manager)

    atoms = list(hierarchy.atoms_with_labels())
    xyz = _xyz(hierarchy)

    def key(atom) -> tuple:
        return (atom.chain_id.strip(), atom.resid().strip(),
                atom.name.strip(), atom.altloc.strip())

    by_key = {key(a): i for i, a in enumerate(atoms)}
    parents = hydrogen_parents(hierarchy) if inherit_to_heavy else {}

    def side_index(side: dict) -> Optional[int]:
        resid = f"{side.get('resID', '')}{side.get('iCode', '') or ''}".strip()
        return by_key.get((str(side.get("chainID", "")).strip(), resid,
                           str(side.get("atomName", "")).strip(),
                           str(side.get("alt", "")).strip()))

    # probe2 reports one row per surface *dot*, so a single contact arrives dozens of times.
    # Collapse to one event per atom pair, keeping its worst overlap: an event means "these
    # two atoms are too close", and a consumer that deposits one kernel per event would
    # otherwise weight a contact by how much of it happens to be dotted.
    worst: Dict[Tuple[int, int], float] = {}
    for row in dots:
        # probe2 reports a signed gap; an overlap is negative, so flip it.
        overlap = -float(row.get("gap", 0.0))
        if overlap <= 0:
            continue
        a = side_index(row.get("src") or {})
        b = side_index(row.get("target") or {})
        pair = tuple(sorted(i for i in (a, b) if i is not None))
        if not pair:
            continue
        if overlap > worst.get(pair, 0.0):
            worst[pair] = overlap

    events: List[ValidationEvent] = []
    for pair, overlap in sorted(worst.items()):
        picked = set(pair)
        for i in pair:
            parent = parents.get(i)
            if parent is not None:
                picked.add(parent)
        picked = tuple(sorted(picked))
        first = atoms[picked[0]]
        events.append(ValidationEvent(
            metric="clash", value=float(overlap), units="angstrom",
            outlier=overlap >= CLASH_OUTLIER_A,
            residue=ResidueKey(first.chain_id.strip(),
                               int(first.resseq_as_int()), first.icode.strip()),
            atom_indices=picked,
            atoms_xyz=tuple(tuple(float(c) for c in xyz[i]) for i in picked),
            detail={"pair": [int(i) for i in pair]}))
    return events


def extract_all(model: Any, *, dots=None, data_manager: Any = None,
                ramalyze_result=None, rotalyze_result=None,
                metrics: Sequence[str] = ("rama", "rota", "clash")) -> List[ValidationEvent]:
    """Every event for ``metrics``. ``model`` is an mmtbx model manager.

    Clash needs a model (probe2 runs on one); rama/rota need only its hierarchy. Pass any
    result you already have so an expensive validator is not run twice.
    """
    hierarchy = model.get_hierarchy()
    index = residue_atom_index(hierarchy)
    events: List[ValidationEvent] = []
    if "rama" in metrics:
        events += extract_ramachandran(hierarchy, index=index,
                                       ramalyze_result=ramalyze_result)
    if "rota" in metrics:
        events += extract_rotamer(hierarchy, index=index, rotalyze_result=rotalyze_result)
    if "clash" in metrics:
        events += extract_clashes(model, dots=dots, data_manager=data_manager)
    return events


# -- rolling up ---------------------------------------------------------------


def per_atom(events: Iterable[ValidationEvent], n_atoms: int, *, metric: Optional[str] = None,
             transform=None, outliers_only: bool = False) -> np.ndarray:
    """Roll events onto atoms with a **max**, never a sum.

    Max because one physical mistake is routinely seen several times — a badly placed atom
    clashes with three neighbours, and a phi/psi is assigned to four backbone atoms — and
    summing would count it once per sighting and rank a large residue above a small one for
    no reason but size.

    ``transform(event) -> float`` converts a native value to whatever scale the caller uses;
    the default records the native value itself.
    """
    out = np.zeros(int(n_atoms), dtype=float)
    for event in events:
        if metric is not None and event.metric != metric:
            continue
        if outliers_only and not event.outlier:
            continue
        value = float(event.value) if transform is None else float(transform(event))
        if value <= 0:
            continue
        for i in event.atom_indices:
            if 0 <= i < n_atoms and value > out[i]:
                out[i] = value
    return out


def outlier_atoms(events: Iterable[ValidationEvent], *, metric: Optional[str] = None,
                  predicate=None) -> set:
    """Indices of every atom implicated by an event matching ``predicate``.

    Defaults to the validator's own ``outlier`` boolean. Pass a predicate to widen it — a
    continuous field is usually built from *every* result, not just the flagged ones, so
    "which atoms could legitimately carry signal" is a broader question than "which atoms
    are outliers". See :func:`worse_than_percent`.
    """
    predicate = (lambda e: e.outlier) if predicate is None else predicate
    picked = set()
    for event in events:
        if metric is not None and event.metric != metric:
            continue
        if predicate(event):
            picked.update(event.atom_indices)
    return picked


def worse_than_percent(cut: float):
    """Predicate: a rama/rota result at or below ``cut`` percent.

    The generator's concern curve starts rising at its "good" percentage (2.0% for both rama
    and rotamer) rather than at the outlier cut, so a field built from it legitimately marks
    residues MolProbity does not flag. Pass ``worse_than_percent(2.0)`` as
    ``concerning=`` to judge such a field against what actually fed it.
    """
    return lambda e: e.units == "percent" and float(e.value) <= float(cut)


# -- the sanity check ---------------------------------------------------------


@dataclass
class AgreementReport:
    """Does a spatial field actually mark the places validation complained about?"""

    metric: str
    hot_threshold: float
    tolerance_a: float
    n_outlier_atoms: int
    n_covered: int                    # outlier atoms the field marks
    n_hot_atoms: int
    n_hot_explained: int              # hot atoms at/inside tolerance of an outlier atom
    missed: List[dict] = _field(default_factory=list)      # outliers the field does not mark
    unexplained: List[dict] = _field(default_factory=list)  # hot atoms with no outlier near

    @property
    def recall(self) -> float:
        """Fraction of outlier atoms the field marks. **This is the check that matters**:
        a validation hotspot field that misses a real outlier is wrong."""
        return 1.0 if not self.n_outlier_atoms else self.n_covered / self.n_outlier_atoms

    @property
    def explained(self) -> float:
        """Fraction of hot atoms attributable to some outlier within ``tolerance_a``."""
        return 1.0 if not self.n_hot_atoms else self.n_hot_explained / self.n_hot_atoms

    @property
    def ok(self) -> bool:
        return not self.missed and not self.unexplained

    def summary(self) -> str:
        return (f"{self.metric}: recall {self.recall:.3f} "
                f"({self.n_covered}/{self.n_outlier_atoms} outlier atoms marked at "
                f">= {self.hot_threshold:g}), explained {self.explained:.3f} "
                f"({self.n_hot_explained}/{self.n_hot_atoms} hot atoms within "
                f"{self.tolerance_a:g} A of an outlier)"
                + ("" if self.ok else
                   f"; {len(self.missed)} missed, {len(self.unexplained)} unexplained"))


def check_field_agreement(events: Iterable[ValidationEvent], sampled: Sequence[float],
                          sites_cart: Any, *, metric: str, hot_threshold: float,
                          tolerance_a: float = 4.0, concerning=None,
                          limit: int = 20) -> AgreementReport:
    """Map a field back to atoms and confirm the hot places are the bad places.

    ``sampled`` is the field read at each atom (same order as the hierarchy), ``sites_cart``
    their coordinates. Two questions are asked, and they are deliberately not symmetric:

    * **Recall** — does every atom the validator called an *outlier* reach ``hot_threshold``?
      A miss is a real defect: the field failed to mark a problem it was built from. This
      side always uses the outlier boolean, because "never lose a real outlier" is the one
      guarantee such a field owes.
    * **Explained** — is every hot atom near *something* that could have put signal there?
      ``concerning`` says what counts, defaulting to outliers. **For a continuous field,
      widen it** — pass ``worse_than_percent(2.0)`` for rama/rota — because the concern curve
      starts rising well before the outlier cut, so the field legitimately marks residues
      MolProbity never flags. Judging it against outliers alone reports correct behaviour as
      failure.

    A hot atom that is not itself concerning is still **expected**: these fields splat each
    observation with a Gaussian wider than the ~3.8 A between adjacent CA atoms, so a residue
    beside a bad one genuinely sits in its density. That is why this side allows a distance
    ``tolerance_a`` instead of demanding identity — tightening it to 0 would measure the
    kernel, not the field. What is worth investigating is a hot atom with nothing concerning
    anywhere near it: that is signal appearing where nothing was deposited.
    """
    sampled = np.asarray(sampled, dtype=float).ravel()
    sites = np.asarray(sites_cart, dtype=float).reshape(-1, 3)
    required = sorted(outlier_atoms(events, metric=metric))
    flagged = sorted(outlier_atoms(events, metric=metric, predicate=concerning))
    hot = np.flatnonzero(sampled >= float(hot_threshold))

    missed = [{"atom": int(i), "sampled": float(sampled[i])}
              for i in required if sampled[i] < hot_threshold]

    unexplained: List[dict] = []
    if len(hot):
        if flagged:
            flagged_xyz = sites[flagged]
            for i in hot:
                d = float(np.sqrt(((flagged_xyz - sites[i]) ** 2).sum(axis=1)).min())
                if d > tolerance_a:
                    unexplained.append({"atom": int(i), "sampled": float(sampled[i]),
                                        "nearest_outlier_a": round(d, 2)})
        else:
            unexplained = [{"atom": int(i), "sampled": float(sampled[i]),
                            "nearest_outlier_a": None} for i in hot]

    return AgreementReport(
        metric=metric, hot_threshold=float(hot_threshold), tolerance_a=float(tolerance_a),
        n_outlier_atoms=len(required), n_covered=len(required) - len(missed),
        n_hot_atoms=int(len(hot)), n_hot_explained=int(len(hot)) - len(unexplained),
        missed=sorted(missed, key=lambda m: m["sampled"])[:limit],
        unexplained=sorted(unexplained, key=lambda m: -m["sampled"])[:limit])


if __name__ == "__main__":  # pragma: no cover - manual self-test
    import sys

    import iotbx.pdb
    from mmtbx.model import manager as model_manager

    path = sys.argv[1]
    inp = iotbx.pdb.input(file_name=path)
    model = model_manager(model_input=inp, log=None)
    events = extract_all(model)
    n = model.get_number_of_atoms()
    print(f"{path}: {n} atoms, {len(events)} events")
    for metric in ("rama", "rota", "clash"):
        subset = [e for e in events if e.metric == metric]
        flagged = outlier_atoms(subset, metric=metric)
        print(f"  {metric:6s} {len(subset):5d} events, "
              f"{sum(1 for e in subset if e.outlier):4d} outliers, "
              f"{len(flagged):5d} atoms implicated")
