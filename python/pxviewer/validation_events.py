"""Shared validation extraction: one definition of *what is wrong and where*.

This module is deliberately **standalone and copyable**. It imports nothing from pxviewer
and uses no relative imports, so the same file can sit in the pxviewer package and in the
sibling ``hotspots/`` generator, and both then localize validation identically. If you copy
it, copy it whole and do not fork the localization rules — the point of the file is that a
disagreement between the two projects becomes impossible rather than merely unlikely.

**Consumers (three now).** Alongside pxviewer and ``hotspots/``, the cryo-EM per-atom RSR
project (map-model) uses this file. Two families of revision, both additive and backward-
compatible (existing ``extract_all`` callers see identical results):

* GEOMETRY: bond/angle deviation events (:func:`extract_bonds` / :func:`extract_angles` —
  the covalent channel, opt-in, not in the default ``metrics``), and region roll-ups
  (:func:`restrict`, :func:`summarize`) that turn events over an atom subset into the
  standard MolProbity aggregates (clashscore, rota/rama %, bond/angle RMSD).
* MAP-FIT (:func:`extract_fit` and friends): how well the model fits the *map*, as colorable
  per-atom/residue fields — ``cc_mapmodel`` / ``cc_half`` / ``cc_star`` / ``cc_gap`` (the
  local CC* ceiling & gap), ``qscore``, ``local_resolution``, ``rsr`` (real-space R-value).
  These are the fields the visualization consumer colors a surface by. They need a boxed
  ``map_model_manager`` (maps) and heavier deps — a separate dependency tier, documented at
  the MAP-FIT section header, with all imports function-local so a geometry-only consumer
  never touches them.

The invariant is unchanged and now spans both families: this file carries **native values
and one localization** (which residue, which atoms, and — for a validator — its outlier
boolean); turning a value into a score or a color is the caller's business. That is why a
disagreement between the three projects becomes impossible rather than merely unlikely.

**The two families do not share a roll-up.** Geometry values are badness measured up from
zero, so :func:`per_atom` filters non-positives and fills unmarked atoms with ``0.0`` — for
that family both are true statements. Map-fit values are a continuous scale where negatives
are the *worst* atoms and zero is a real reading, so they use :func:`per_atom_field`, which
keeps every value and fills ``nan``. Getting this wrong is silent rather than loud — a cc gap
is normally negative, so severity defaults turn a well-fit structure into a field of zeros —
so :func:`per_atom` refuses map-fit events outright unless the caller says what it means.

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

#: probe2 dot types that are hydrogen bonds, not steric clashes. They sit well inside the
#: vdW sum by construction, so they carry a negative gap and would be counted as clashes.
_HBOND_DOT_TYPES = frozenset({"hb"})

#: MolProbity flags a covalent bond or angle as an outlier at |Z| >= 4 sigma from its
#: restraint ideal. Recorded as provenance for the ``outlier`` flag only; the native
#: deviation (angstrom / degree) travels as the event ``value``, the Z in ``detail``.
BOND_OUTLIER_SIGMA = 4.0
ANGLE_OUTLIER_SIGMA = 4.0

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

    metric: str                       # geometry: 'rama'|'rota'|'clash'|'bond'|'angle'
                                      # map-fit:  'cc_mapmodel'|'cc_half'|'cc_star'|
                                      #           'cc_gap'|'qscore'|'local_resolution'|'rsr'
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


#: CaBLAM contour cuts (fractions, not percentages; the comparisons are strict `<`).
CABLAM_OUTLIER = 0.01        # cablam score below this is an outlier
CABLAM_DISFAVORED = 0.05     # below this is "disfavored" — the allowed-region analogue
CA_GEOM_OUTLIER = 0.005      # c_alpha_geom below this is a CA-geometry outlier

#: MolProbity's C-beta deviation cut, in angstroms. Note `>=`, not `>`.
CBETA_OUTLIER_A = 0.25


def extract_cablam(hierarchy: Any, *, index=None, cablam_result=None
                   ) -> List[ValidationEvent]:
    """CaBLAM backbone conformation, as **two** metrics on residue *i*'s own CA/O/N.

    ``cablam`` is the backbone-conformation contour and ``ca_geom`` the CA-trace geometry
    contour; they are different quantities with different cuts, so they travel as separate
    events rather than one blended number. Both are contour fractions in ``[0, 1]`` where
    **lower is worse**, the same shape as a Ramachandran percentage.

    CaBLAM sees five consecutive CAs, so the score is unavailable at chain ends (``None``,
    skipped here) and it is informed by residues *i±2*. It is still assigned narrowly to
    residue *i*, for the reason Ramachandran is: implicating the neighbours would smear one
    residue's problem across five.
    """
    from libtbx.utils import null_out
    from mmtbx.validation.cablam import cablamalyze

    if cablam_result is None:
        cablam_result = cablamalyze(pdb_hierarchy=hierarchy, outliers_only=False,
                                    out=null_out(), quiet=True)
    index = residue_atom_index(hierarchy) if index is None else index
    names, xyz = _names(hierarchy), _xyz(hierarchy)

    events = []
    for r in cablam_result.results:
        key = _key_of(r)     # cablam right-pads chain_id to 2 chars; _key_of strips it
        picked = [i for i in atoms_of(index, key, getattr(r, "altloc", ""))
                  if names[i] in ("CA", "O", "N")]
        seat = dict(residue=key, atom_indices=tuple(picked),
                    atoms_xyz=tuple(tuple(float(c) for c in xyz[i]) for i in picked))
        if r.scores.cablam is not None:
            events.append(ValidationEvent(
                metric="cablam", value=float(r.scores.cablam), units="fraction",
                outlier=bool(r.feedback.cablam_outlier), **seat,
                detail={"id": r.mp_id().strip(),
                        "disfavored": bool(r.feedback.cablam_disfavored)}))
        if r.scores.c_alpha_geom is not None:
            events.append(ValidationEvent(
                metric="ca_geom", value=float(r.scores.c_alpha_geom), units="fraction",
                outlier=bool(r.feedback.c_alpha_geom_outlier), **seat,
                detail={"id": r.mp_id().strip()}))
    return events


def extract_cbeta(hierarchy: Any, *, index=None, cbetadev_result=None
                  ) -> List[ValidationEvent]:
    """C-beta deviation, in angstroms, on the CB atom whose position is off.

    Narrow on purpose, like Ramachandran: the deviation *is* a statement about where CB
    sits relative to the backbone that positions it, so it implicates CB and not the
    residue at large. MolProbity's cut is 0.25 A.
    """
    from mmtbx.validation.cbetadev import cbetadev

    if cbetadev_result is None:
        cbetadev_result = cbetadev(pdb_hierarchy=hierarchy, outliers_only=False)
    index = residue_atom_index(hierarchy) if index is None else index
    names, xyz = _names(hierarchy), _xyz(hierarchy)

    events = []
    for r in cbetadev_result.results:
        if r.deviation is None:
            continue
        key = _key_of(r)
        picked = [i for i in atoms_of(index, key, getattr(r, "altloc", ""))
                  if names[i] == "CB"]
        events.append(ValidationEvent(
            metric="cbeta", value=float(r.deviation), units="angstrom",
            outlier=bool(r.outlier), residue=key,
            atom_indices=tuple(picked),
            atoms_xyz=tuple(tuple(float(c) for c in xyz[i]) for i in picked),
            detail={"id": r.id_str().strip(), "dihedral_NABB": r.dihedral_NABB}))
    return events


def extract_omega(hierarchy: Any, *, index=None, omegalyze_result=None
                  ) -> List[ValidationEvent]:
    """Peptide omega dihedral, in degrees, on the backbone atoms that define it.

    ``value`` is the **twist away from the nearest ideal** — 0 for a cis peptide, 180 for a
    trans one — rather than omega itself, so that a larger number is always worse, as with
    every other channel. omega itself is in ``detail``, along with omegalyze's own category:
    a cis non-proline or a twisted peptide is flagged, a cis-proline is not.

    The dihedral spans two residues; the reported residue's own N and CA carry it. The
    preceding carbonyl is within a bond length, which is well inside any field kernel.
    """
    from mmtbx.validation import omegalyze as _om

    if omegalyze_result is None:
        omegalyze_result = _om.omegalyze(pdb_hierarchy=hierarchy, nontrans_only=False,
                                         out=None, quiet=True)
    index = residue_atom_index(hierarchy) if index is None else index
    names, xyz = _names(hierarchy), _xyz(hierarchy)

    kinds = {_om.OMEGALYZE_TRANS: "trans", _om.OMEGALYZE_CIS: "cis",
             _om.OMEGALYZE_TWISTED: "twisted"}
    events = []
    for r in omegalyze_result.results:
        if r.omega is None:
            continue
        omega = float(r.omega)
        twist = min(abs(omega), abs(180.0 - abs(omega)))   # distance to cis or to trans
        key = _key_of(r)
        picked = [i for i in atoms_of(index, key, getattr(r, "altloc", ""))
                  if names[i] in ("N", "CA")]
        events.append(ValidationEvent(
            metric="omega", value=twist, units="degree",
            # omegalyze flags *any* non-trans peptide, so a perfectly ordinary cis-proline
            # arrives with outlier=True. `is_proline` is what lets a consumer apply the
            # MolProbity reading: cis-Pro is common, cis-nonPro and twisted are the problems.
            outlier=bool(r.outlier), residue=key,
            atom_indices=tuple(picked),
            atoms_xyz=tuple(tuple(float(c) for c in xyz[i]) for i in picked),
            detail={"id": r.id_str().strip(), "omega": omega,
                    "kind": kinds.get(r.omega_type, str(r.omega_type)),
                    "is_proline": bool(r.res_type == _om.OMEGA_PRO),
                    "resname": r.resname.strip(),
                    "prev_resname": str(getattr(r, "prev_resname", "")).strip()}))
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
    # Also keep where the worst dot sat. A consumer that deposits a kernel per event may
    # legitimately prefer the contact point -- the interface itself -- to the two atom
    # centres; carrying it means that choice stays the consumer's rather than being decided
    # here by omission.
    worst: Dict[Tuple[int, int], float] = {}
    contact: Dict[Tuple[int, int], Optional[Xyz]] = {}
    for row in dots:
        # A hydrogen bond has the donor and acceptor well inside the vdW sum, so it shows a
        # negative gap and would otherwise be counted as a steric clash. MolProbity subtracts
        # H-bonded pairs before computing clashscore, and probe2 labels them, so drop them
        # here rather than let every consumer rediscover the inflation.
        if str(row.get("type", "")).strip() in _HBOND_DOT_TYPES:
            continue
        # probe2 reports a signed gap; an overlap is negative, so flip it.
        overlap = -float(row.get("gap", 0.0))
        if overlap <= 0:
            continue
        a = side_index(row.get("src") or {})
        b = side_index(row.get("target") or {})
        # Both ends must resolve. A row where only one side maps is not a pair, and admitting
        # it as a one-element "pair" invents a contact and inflates the clash count.
        if a is None or b is None or a == b:
            continue
        pair = (a, b) if a < b else (b, a)
        if overlap > worst.get(pair, 0.0):
            worst[pair] = overlap
            loc = row.get("loc")
            contact[pair] = tuple(float(c) for c in loc) if loc else None

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
            detail={"pair": [int(i) for i in pair],
                    "contact_xyz": contact.get(pair)}))
    return events


# -- bond / angle geometry (the covalent channel) -----------------------------
#
# Bond and angle deviations complete the geometry set: rama/rota cover torsions,
# clash the steric channel, these the covalent one. Unlike rama/rota (hierarchy)
# and clash (probe2), they read the RESTRAINTS. The event value is the native
# deviation (angstrom for a bond, degree for an angle); the outlier boolean is
# |Z| >= the sigma cut, with Z and sigma in ``detail`` -- same split as everywhere
# else.
#
# PASS ``geometry=`` IF YOUR PROJECT OWNS A RESTRAINT BUILD. The fallback below calls
# ``model.process(make_restraints=True)`` on the caller's model. That is a mutation, and it
# is not a quiet one -- measured on 1TEC, asking for the covalent channel changes the
# *rotamer* answer (26 outliers -> 27, same 2737 atoms), because process() sorts the
# hierarchy's atoms in place and resets their serials. Three consequences a host will care
# about:
#
#   * validation results move, as above;
#   * every atom index recorded before the build points at a different atom (``extract_all``
#     defends against this by building first, but a caller holding its own earlier indices
#     cannot be);
#   * the build is not serialised, and a plain ``process()`` ignores any custom bond/angle
#     edits the host carries on the model. Since an existing restraints manager is reused
#     rather than rebuilt, that edit-less manager is then inherited by whatever runs next --
#     in pxviewer, silently dropping a user's custom restraints from minimize and drag, which
#     is why ``edits.build_restraints`` exists (one build path, one lock).
#
# Injecting geometry is the same "pass what you already have" contract as ``dots=`` and
# ``ramalyze_result=`` elsewhere in this file, and it avoids all three.


def covalent_origin_ids() -> frozenset:
    """Restraint origins that count as covalent geometry for the bond/angle channels.

    A restraints manager can also carry secondary-structure **hydrogen-bond** distance and
    angle restraints in the very same proxy arrays; counting those as covalent geometry
    inflates ``bond_rmsd`` / ``angle_rmsd`` against any phenix or MolProbity number.

    cctbx's own ``get_covalent_bond_proxies`` keeps origin ``'covalent geometry'`` alone,
    which is too narrow here: a restraint the user **declared** (origin ``'edits'`` — a custom
    bond or angle a host application let them add) is a real covalent restraint they expect to
    see, and silently dropping it makes the channel disagree with the model being refined.
    So both are kept and everything else is excluded.
    """
    from cctbx.geometry_restraints.linking_class import linking_class

    lc = linking_class()
    allowed = set()
    for name in ("covalent geometry", "edits"):
        try:
            allowed.add(lc.get_origin_id(name))
        except Exception:  # pragma: no cover - origin missing in this cctbx
            pass
    return frozenset(allowed or {0})


def _restraints_geometry(model: Any):
    """Fallback restraint build for a standalone consumer. See the section note above:
    hosts with their own build path should pass ``geometry=`` instead of relying on this."""
    if getattr(model, "restraints_manager", None) is None:
        model.add_crystal_symmetry_if_necessary()
        model.process(make_restraints=True)
    return model.get_restraints_manager().geometry


def _covalent_events(model, metric, proxies, restraint_ctor, ideal_attr, model_attr,
                     value_units, sigma_outlier):
    from cctbx import geometry_restraints as gr  # noqa: F401  (restraint_ctor lives here)
    hierarchy = model.get_hierarchy()
    sites = model.get_sites_cart()
    xyz = _xyz(hierarchy)
    atoms = list(hierarchy.atoms_with_labels())
    events: List[ValidationEvent] = []
    for proxy in proxies:
        r = restraint_ctor(sites_cart=sites, proxy=proxy)
        sigma = (1.0 / proxy.weight ** 0.5) if proxy.weight > 0 else float("nan")
        z = abs(r.delta) / sigma if (sigma == sigma and sigma > 0) else 0.0
        iseqs = tuple(int(i) for i in proxy.i_seqs)
        first = atoms[iseqs[0]]
        events.append(ValidationEvent(
            metric=metric, value=abs(float(r.delta)), units=value_units,
            outlier=bool(z >= sigma_outlier),
            residue=ResidueKey(first.chain_id.strip(), int(first.resseq_as_int()),
                               first.icode.strip()),
            atom_indices=iseqs,
            atoms_xyz=tuple(tuple(float(c) for c in xyz[i]) for i in iseqs),
            detail={"z": float(z), "sigma": float(sigma),
                    "ideal": float(getattr(proxy, ideal_attr)),
                    "model": float(getattr(r, model_attr)), "delta": float(r.delta)}))
    return events


def extract_bonds(model: Any, *, geometry=None, sigma_outlier: float = BOND_OUTLIER_SIGMA
                  ) -> List[ValidationEvent]:
    """Every covalent bond deviation, as its |delta| in angstroms, on its two atoms.

    ``geometry`` is a ``geometry_restraints`` manager the caller already built — pass it if
    your project owns a restraint-build path (see the section note above); omitted, restraints
    are built on ``model`` as a fallback. ``outlier`` is |Z| >= 4 sigma.
    """
    from cctbx import geometry_restraints as gr
    geo = _restraints_geometry(model) if geometry is None else geometry
    sites = model.get_sites_cart()
    # Covalent (and user-declared) only -- see covalent_origin_ids.
    #
    # NOTE: the asu half is excluded. Symmetry-related covalent bonds (a cross-symmetry
    # disulfide, a metal link) are reported there with i_seq/j_seq rather than i_seqs and
    # need the asu_mappings overload, so they are absent from these events and from
    # bond_rmsd. Fine for a boxed/P1 model; state it if you rely on this for a crystal form.
    allowed = covalent_origin_ids()
    simple = [p for p in geo.pair_proxies(sites_cart=sites).bond_proxies.simple
              if p.origin_id in allowed]
    return _covalent_events(model, "bond", simple, gr.bond,
                            "distance_ideal", "distance_model", "angstrom", sigma_outlier)


def extract_angles(model: Any, *, geometry=None, sigma_outlier: float = ANGLE_OUTLIER_SIGMA
                   ) -> List[ValidationEvent]:
    """Every covalent bond-angle deviation, as its |delta| in degrees, on its three atoms.

    ``geometry`` as in :func:`extract_bonds`. ``outlier`` is |Z| >= 4 sigma.
    """
    from cctbx import geometry_restraints as gr
    geo = _restraints_geometry(model) if geometry is None else geometry
    # Covalent (and user-declared) only, as in extract_bonds: secondary-structure restraints
    # add H-bond angle proxies to the same array.
    allowed = covalent_origin_ids()
    proxies = [p for p in geo.angle_proxies if p.origin_id in allowed]
    return _covalent_events(model, "angle", proxies, gr.angle,
                            "angle_ideal", "angle_model", "degree", sigma_outlier)


def extract_all(model: Any, *, dots=None, data_manager: Any = None,
                ramalyze_result=None, rotalyze_result=None, geometry=None,
                metrics: Sequence[str] = ("rama", "rota", "clash")) -> List[ValidationEvent]:
    """Every event for ``metrics``. ``model`` is an mmtbx model manager.

    Clash needs a model (probe2 runs on one); rama/rota/cablam/ca_geom/cbeta/omega need
    only its hierarchy; bond/angle read the restraints. ``metrics`` defaults to the three original
    channels -- pass ``metrics=("rama","rota","clash","bond","angle")`` for the full
    covalent set. Pass any result you already have (``dots``, ``ramalyze_result``,
    ``rotalyze_result``, ``geometry``) so an expensive step is not repeated -- and, for
    ``geometry``, so this file does not build restraints behind your project's back.
    """
    # Restraints FIRST when the covalent channels are wanted. Building them runs
    # model.process(), which sorts the hierarchy's atoms in place and resets their serials —
    # so every atom index recorded before that point would silently start pointing at a
    # different atom. Doing it up front means one atom numbering for the whole call.
    if geometry is None and ("bond" in metrics or "angle" in metrics):
        geometry = _restraints_geometry(model)

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
    if "cablam" in metrics or "ca_geom" in metrics:
        both = extract_cablam(hierarchy, index=index)
        events += [e for e in both if e.metric in metrics]
    if "cbeta" in metrics:
        events += extract_cbeta(hierarchy, index=index)
    if "omega" in metrics:
        events += extract_omega(hierarchy, index=index)
    if "bond" in metrics:
        events += extract_bonds(model, geometry=geometry)
    if "angle" in metrics:
        events += extract_angles(model, geometry=geometry)
    return events


# =============================================================================
# MAP-FIT channels -- how well the model fits the MAP, here, as colorable fields.
#
# These are not "what is wrong" (a validator's outlier) but "how good is the fit"
# (a continuous per-atom/residue quantity): the visualization consumer colors a
# surface by them, and a scorer thresholds them. Same ValidationEvent container,
# same localization and roll-ups -- ``value`` stays native (a correlation, an
# angstrom, an R-value) and ``outlier`` is used only where a metric has a real
# outlier notion (the gap's overfit: cc_mapmodel > cc_star).
#
# DEPENDENCY TIER (heavier than the geometry channels). These need a boxed
# ``map_model_manager`` carrying the full map (``map_manager``) and, for most,
# both half maps (``map_manager_1`` / ``map_manager_2``) + a model -- plus the
# ``local_cc_star`` cctbx branch (cc*/gap) and ``cctbx.maptbx.qscore``. All imports
# are function-local, so a consumer that only wants geometry never touches them.
# The convention (radius, fsc_cutoff, shells) is passed in, never imported, so the
# file stays standalone; a caller supplies its own frozen convention.
# =============================================================================

#: The map-fit metrics. These are **continuous fields**, not badness-from-zero: a value may
#: legitimately be negative (a correlation, a cc gap), and zero is a real point on the scale
#: rather than "nothing here". :func:`per_atom`'s severity defaults would corrupt them, so
#: they are named here and :func:`per_atom_field` is the roll-up that suits them.
MAP_FIT_METRICS = frozenset({
    "cc_mapmodel", "cc_half", "cc_star", "cc_gap", "qscore", "local_resolution", "rsr"})

#: Q-score radial shells (angstrom), 0.0..2.0 by 0.1 -- the standard shell set.
#: Q-score probe geometry. These are cctbx's own defaults (``cctbx.programs.qscore``'s phil:
#: 20 shells from 0.1 to 2.0 A, 32 probes, rtol 0.9), and matching them is what makes the
#: result *the* Q-score ``phenix.qscore`` reports rather than a lookalike computed some other
#: way. The shells and probe count are part of the metric's definition, not tuning knobs.
#:
#: A shell at radius 0.0 in particular is not a free choice: every probe collapses onto the
#: atom centre, the rejection mask keeps them all, and the shell contributes N duplicate
#: copies of the peak-anchored centre density — a systematic upward bias against phenix.
QSCORE_SHELLS = tuple(float(r) for r in np.linspace(0.1, 2.0, 20))
QSCORE_N_PROBES = 32
QSCORE_RTOL = 0.9


def _residue_key(chain, resseq, icode):
    try:
        return ResidueKey(str(chain).strip(), int(str(resseq).strip()),
                          str(icode or "").strip())
    except (TypeError, ValueError):
        return None


def extract_map_model_cc(mmm, d_min, *, radius=None, index=None,
                         components=("cc_mapmodel", "cc_half", "cc_star", "cc_gap"),
                         scattering="electron") -> List[ValidationEvent]:
    """Per-residue local map-model CC, the half-map CC* ceiling, and the gap
    (``mmtbx.maps.local_cc_star``). One event per residue per requested component,
    so each is its own colorable field. ``mmm`` boxed with map_manager +
    map_manager_1/2 + model; radius default ``max(2.5, d_min)``.
    """
    try:
        from mmtbx.maps import local_cc_star as lcs
    except ImportError as exc:   # pragma: no cover - depends on the cctbx build
        raise ImportError(
            "the cc_mapmodel / cc_half / cc_star / cc_gap channels need "
            "mmtbx.maps.local_cc_star, which is not in every cctbx build (it is absent "
            "from cctbx-base as installed here). Drop 'cc_gap' and the other cc_* entries "
            "from extract_fit(metrics=...) to use the channels that are available, or "
            "install a cctbx carrying that module.") from exc
    radius = max(2.5, float(d_min)) if radius is None else float(radius)
    model = mmm.model()
    model.setup_scattering_dictionaries(scattering_table=scattering)
    # scattering_table has to be given to generate_map too: it resolves its own from the
    # manager, not from the model's dictionaries, so without this a cryo-EM model map can be
    # computed with X-ray form factors while the call reads as if 'electron' were in force.
    mmm.generate_map(model=model, d_min=float(d_min), map_id="model_map",
                     scattering_table=scattering)
    hierarchy = model.get_hierarchy()
    index = residue_atom_index(hierarchy) if index is None else index
    xyz = _xyz(hierarchy)
    out = lcs.per_residue_local_cc_star(
        map_data_full=mmm.map_manager().map_data(),
        model_map_data=mmm.get_map_manager_by_id("model_map").map_data(),
        unit_cell=mmm.map_manager().crystal_symmetry().unit_cell(), model=model,
        map_data_half1=mmm.get_map_manager_by_id("map_manager_1").map_data(),
        map_data_half2=mmm.get_map_manager_by_id("map_manager_2").map_data(),
        radius=radius)
    events: List[ValidationEvent] = []
    for r in out.residues:
        key = _residue_key(r.chain_id, r.resseq, r.icode)
        picked = tuple(atoms_of(index, key)) if key is not None else ()
        axyz = tuple(tuple(float(c) for c in xyz[i]) for i in picked)
        vals = {"cc_mapmodel": r.cc_mapmodel, "cc_half": r.cc_half,
                "cc_star": r.cc_star, "cc_gap": r.cc_gap}
        overfit = (r.cc_star is not None and r.cc_mapmodel is not None
                   and r.cc_mapmodel > r.cc_star)
        for comp in components:
            v = vals.get(comp)
            if v is None:
                continue
            events.append(ValidationEvent(
                metric=comp, value=float(v), units="correlation",
                outlier=bool(comp == "cc_gap" and overfit),
                residue=key, atom_indices=picked, atoms_xyz=axyz,
                detail={"radius": radius}))
    return events


def extract_qscore(mmm, *, shells=None, n_probes=QSCORE_N_PROBES, rtol=QSCORE_RTOL,
                   index=None) -> List[ValidationEvent]:
    """Q-score per ATOM (``cctbx.maptbx.qscore``). Natively per-atom, one event per atom.

    Three things here are not optional, and each was got wrong once:

    * **Run against a copy.** ``calc_qscore`` strips hydrogens from the model it is handed and
      calls ``set_model`` back onto the manager. Handed a live one, it swaps the model its
      owner is using out from under it — in pxviewer, mid-session, while the viewer streams
      from it.
    * **Realign the result.** ``calc_qscore`` returns one value per **non-hydrogen** atom, not
      one per atom. Zipping it against the full atom list assigns every score to the wrong
      atom as soon as the model has hydrogens, silently — no exception, just wrong numbers on
      the wrong atoms. Hydrogens get ``nan`` (unmeasured), and the count is checked.
    * **Match the reference convention.** The shells and probe count *are* the definition; a
      different set is a different number wearing the same name. The defaults here are
      cctbx's own (see ``cctbx.programs.qscore``'s phil), so this is the value
      ``phenix.qscore`` reports rather than a lookalike.
    """
    from cctbx.maptbx.qscore import calc_qscore

    model = mmm.model()
    hierarchy = model.get_hierarchy()
    elements = [e.strip().upper() for e in hierarchy.atoms().extract_element()]
    scored_atoms = np.array([e not in _HYDROGEN for e in elements], dtype=bool)

    result = calc_qscore(mmm.deep_copy(),
                         shells=list(QSCORE_SHELLS if shells is None else shells),
                         n_probes=n_probes, rtol=rtol, nproc=1)
    scored = np.asarray(result["qscore_per_atom"], dtype=float).reshape(-1)
    expected = int(scored_atoms.sum())
    if scored.size != expected:
        raise ValueError(
            f"q-score returned {scored.size} values for {expected} non-hydrogen atoms")
    q = np.full(len(elements), np.nan, dtype=float)
    q[scored_atoms] = scored

    xyz = _xyz(hierarchy)
    atoms = list(hierarchy.atoms_with_labels())
    events: List[ValidationEvent] = []
    for i in range(q.size):
        if q[i] != q[i]:   # hydrogens: never scored, so absent rather than a bad fit
            continue
        a = atoms[i]
        events.append(ValidationEvent(
            metric="qscore", value=float(q[i]), units="correlation", outlier=False,
            residue=ResidueKey(a.chain_id.strip(), int(a.resseq_as_int()), a.icode.strip()),
            atom_indices=(int(i),), atoms_xyz=(tuple(float(c) for c in xyz[i]),), detail={}))
    return events


#: FSC threshold defining "resolved" for the local-resolution map. 0.143 is the convention
#: cctbx defaults to and the one ``pxviewer.volume_io.local_resolution_from_half_maps``
#: already uses. It matters that these agree: the value travels as plain angstroms under one
#: metric name, so a field built at 0.5 is systematically worse-looking than one built at
#: 0.143 and any palette or threshold calibrated on one misreads the other.
LOCAL_RESOLUTION_FSC_CUTOFF = 0.143


def extract_local_resolution(mmm, *, fsc_cutoff=LOCAL_RESOLUTION_FSC_CUTOFF, n_bins=20,
                             smoothing_radius_ratio=1.0, index=None
                             ) -> List[ValidationEvent]:
    """Local resolution (angstrom) per ATOM, half-map FSC
    (``map_model_manager.local_resolution_map``). Needs both half maps.

    **Lower is better here** — the only channel in this file with that polarity. Carried
    native so a colorer inverts as it likes, but note that :func:`per_atom` combines with a
    max, which on an inverted field selects the *best*-resolved value rather than the worst.
    Pass a negating ``transform=`` if you want worst-case behaviour from it.
    """
    from cctbx import maptbx
    lr = mmm.local_resolution_map(
        map_id_1="map_manager_1", map_id_2="map_manager_2",
        fsc_cutoff=float(fsc_cutoff), n_bins=n_bins,
        smoothing_radius_ratio=smoothing_radius_ratio).map_data()
    model = mmm.model()
    xrs = model.get_xray_structure()
    sf = xrs.unit_cell().fractionalize(xrs.sites_cart())
    hierarchy = model.get_hierarchy()
    xyz = _xyz(hierarchy)
    atoms = list(hierarchy.atoms_with_labels())
    events: List[ValidationEvent] = []
    for i in range(sf.size()):
        v = maptbx.eight_point_interpolation(lr, sf[i])
        if v != v:
            continue
        a = atoms[i]
        events.append(ValidationEvent(
            metric="local_resolution", value=float(v), units="angstrom", outlier=False,
            # Carry the residue like every other channel: a roll-up or join keyed on
            # .residue would otherwise drop this channel entirely and silently.
            residue=ResidueKey(a.chain_id.strip(), int(a.resseq_as_int()), a.icode.strip()),
            atom_indices=(int(i),),
            atoms_xyz=(tuple(float(c) for c in xyz[i]),), detail={}))
    return events


def extract_rsr(mmm, d_min, *, radius=None, scattering="electron", index=None
                ) -> List[ValidationEvent]:
    """Real-space R-value per residue: R = sum|obs - calc| / sum|obs + calc| over a
    radius-angstrom window, with calc LINEARLY SCALED to obs over the molecular
    envelope (least squares) so the R-value is not dominated by a global scale
    mismatch. obs = full map, calc = model map at ``d_min``.

    NOTE: this is the pipeline's real-space R-value; it still owes a calibration vs
    ``phenix.real_space_correlation`` / EDSTATS before its numbers are quoted
    (map-model V2 Gate 0.3). The extractor is stable; the *validation* is the open item.
    """
    from cctbx import maptbx
    from scitbx.array_family import flex
    radius = max(2.5, float(d_min)) if radius is None else float(radius)
    model = mmm.model()
    model.setup_scattering_dictionaries(scattering_table=scattering)
    mmm.generate_map(model=model, d_min=float(d_min), map_id="rsr_calc")
    uc = mmm.map_manager().crystal_symmetry().unit_cell()
    obs = mmm.map_manager().map_data()
    calc = mmm.get_map_manager_by_id("rsr_calc").map_data()
    n_real = obs.all()
    xrs = model.get_xray_structure()
    sites = xrs.sites_cart()
    obs_np = obs.as_1d().as_numpy_array().astype(float)
    calc_np = calc.as_1d().as_numpy_array().astype(float)
    env = maptbx.grid_indices_around_sites(
        uc, n_real, n_real, sites, flex.double(sites.size(), radius))
    e = np.array(list(env), dtype=np.int64)
    A = np.vstack([calc_np[e], np.ones(e.size)]).T
    a, b = np.linalg.lstsq(A, obs_np[e], rcond=None)[0]
    calc_s = a * calc_np + b
    hierarchy = model.get_hierarchy()
    index = residue_atom_index(hierarchy) if index is None else index
    xyz = _xyz(hierarchy)
    events: List[ValidationEvent] = []
    for ch in hierarchy.only_model().chains():
        for rg in ch.residue_groups():
            atoms = rg.atoms()
            sel = maptbx.grid_indices_around_sites(
                uc, n_real, n_real, atoms.extract_xyz(), flex.double(atoms.size(), radius))
            idx = np.array(list(sel), dtype=np.int64)
            if idx.size < 10:
                continue
            oo, cc = obs_np[idx], calc_s[idx]
            denom = float(np.abs(oo + cc).sum())
            if denom <= 0:
                continue
            key = ResidueKey(ch.id.strip(), rg.resseq_as_int(), rg.icode.strip())
            picked = tuple(atoms_of(index, key))
            events.append(ValidationEvent(
                metric="rsr", value=float(np.abs(oo - cc).sum() / denom), units="rvalue",
                outlier=False, residue=key, atom_indices=picked,
                atoms_xyz=tuple(tuple(float(c) for c in xyz[i]) for i in picked),
                detail={"radius": radius, "scale_a": float(a), "scale_b": float(b)}))
    return events


def extract_fit(mmm, d_min, *,
                metrics: Sequence[str] = ("cc_gap", "qscore", "local_resolution", "rsr"),
                radius=None, index=None) -> List[ValidationEvent]:
    """Map-fit events for ``metrics`` on a boxed ``map_model_manager`` -- the map-fit
    counterpart to :func:`extract_all`. ``cc_mapmodel``/``cc_half``/``cc_star``/``cc_gap``
    all come from one ``extract_map_model_cc`` call. A consumer wanting every colorable
    field runs ``extract_all(mmm.model()) + extract_fit(mmm, d_min)``.
    """
    hierarchy = mmm.model().get_hierarchy()
    index = residue_atom_index(hierarchy) if index is None else index
    cc = tuple(m for m in metrics if m in ("cc_mapmodel", "cc_half", "cc_star", "cc_gap"))
    events: List[ValidationEvent] = []
    if cc:
        events += extract_map_model_cc(mmm, d_min, radius=radius, index=index, components=cc)
    if "qscore" in metrics:
        events += extract_qscore(mmm, index=index)
    if "local_resolution" in metrics:
        events += extract_local_resolution(mmm, index=index)
    if "rsr" in metrics:
        events += extract_rsr(mmm, d_min, radius=radius, index=index)
    return events


# -- rolling up ---------------------------------------------------------------


_UNSET = object()


def per_atom(events: Iterable[ValidationEvent], n_atoms: int, *, metric: Optional[str] = None,
             transform=None, outliers_only: bool = False, skip_nonpositive=_UNSET,
             fill=_UNSET) -> np.ndarray:
    """Roll events onto atoms with a **max**, never a sum — for a SEVERITY field.

    Max because one physical mistake is routinely seen several times — a badly placed atom
    clashes with three neighbours, and a phi/psi is assigned to four backbone atoms — and
    summing would count it once per sighting and rank a large residue above a small one for
    no reason but size.

    ``transform(event) -> float`` converts a native value to whatever scale the caller uses;
    the default records the native value itself.

    The defaults (``skip_nonpositive=True``, ``fill=0.0``) encode badness-from-zero: a value
    at or below zero is nothing to report, and an atom no event touched is clean. That is
    true of the geometry channels and false of every map-fit one, so handing this function
    map-fit events without saying how to treat them **raises** rather than quietly returning
    a corrupted field — use :func:`per_atom_field`. Passing ``transform`` counts as saying:
    a caller mapping a correlation onto a badness scale has already thought about it.
    """
    events = list(events)
    if skip_nonpositive is _UNSET and fill is _UNSET and transform is None:
        offenders = sorted({e.metric for e in events
                            if e.metric in MAP_FIT_METRICS
                            and (metric is None or e.metric == metric)})
        if offenders:
            raise ValueError(
                f"per_atom() got continuous map-fit events ({', '.join(offenders)}) with the "
                "severity defaults, which would drop every negative value and read unmeasured "
                "atoms back as 0.0 — for a cc gap, normally negative, that is a field of "
                "zeros; for local resolution, 0.0 A is the best possible value. Use "
                "per_atom_field() for a continuous field, or pass an explicit transform= "
                "(and skip_nonpositive=/fill=) to map these onto a severity scale on purpose.")
    skip_nonpositive = True if skip_nonpositive is _UNSET else skip_nonpositive
    fill = 0.0 if fill is _UNSET else fill

    out = np.full(int(n_atoms), float(fill), dtype=float)
    seen = np.zeros(int(n_atoms), dtype=bool)
    for event in events:
        if metric is not None and event.metric != metric:
            continue
        if outliers_only and not event.outlier:
            continue
        value = float(event.value) if transform is None else float(transform(event))
        if skip_nonpositive and value <= 0:
            continue
        for i in event.atom_indices:
            if 0 <= i < n_atoms and (not seen[i] or value > out[i]):
                out[i] = value
                seen[i] = True
    return out


def per_atom_field(events: Iterable[ValidationEvent], n_atoms: int, *,
                   metric: Optional[str] = None, transform=None) -> np.ndarray:
    """Roll a **continuous** field onto atoms — the map-fit counterpart of :func:`per_atom`.

    Same max combination, but nothing is filtered and nothing is invented: negative values
    survive, and an atom no event touched reads ``nan``. Those two differences are the whole
    point.

    * Keeping negatives matters because they are the interesting atoms. A negative
      correlation is the worst possible fit, not an absence of one; dropping it and filling
      ``0.0`` turns the worst atom in the structure into an average-looking one.
    * ``nan`` rather than ``0.0`` matters because zero is a real value on these scales. For a
      resolution in angstroms it is the *best* possible reading, so an unmeasured atom would
      display as perfectly resolved.

    ``nan`` is also what a viewer's attribute theme draws in its "missing" colour, so
    unmeasured atoms read as unmeasured rather than as a score.

    Note the polarity is the caller's to handle: local resolution is lower-is-better while
    every other field here is higher-is-better, and the max combination does not know that.
    It only bites when several events touch one atom (a per-residue field fanned out over its
    atoms); the per-atom metrics emit exactly one event per atom, so there is nothing to
    combine. Pass a negating ``transform`` if you need worst-case semantics.
    """
    return per_atom(events, n_atoms, metric=metric, transform=transform,
                    skip_nonpositive=False, fill=np.nan)


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


# -- region roll-up -----------------------------------------------------------


def restrict(events: Iterable[ValidationEvent], atom_indices: Iterable[int]
             ) -> List[ValidationEvent]:
    """Events that TOUCH a region, given as a set of atom indices.

    Membership is by atom, not residue, because a clash / bond / angle spans
    several residues and the event carries all of them: a clash reaching into the
    region is the region's problem even if its "first" atom sits outside. Every
    event with at least one implicated atom in ``atom_indices`` is kept. (rama/rota
    are single-residue, so this reduces to residue membership for them.)
    """
    region = set(int(i) for i in atom_indices)
    return [e for e in events if region.intersection(e.atom_indices)]


def summarize(events: Iterable[ValidationEvent], *, n_atoms: Optional[int] = None
              ) -> Dict[str, float]:
    """Standard MolProbity-style aggregates over a set of events -- whole model, or a
    region from :func:`restrict`. Objective counts and the conventional derived
    numbers only; no project-specific calibration (that stays with the caller).

    Returns, for each metric present: ``n_<metric>`` and ``n_<metric>_outliers``, plus
      ``rota_outlier_pct`` / ``rama_outlier_pct`` = 100 * outliers / evaluated residues,
      ``clashscore``       = 1000 * clash outliers / ``n_atoms``  (needs ``n_atoms``;
                             for an H-less model this normalises on heavy atoms -- state it),
      ``bond_rmsd`` (A) / ``angle_rmsd`` (deg) = RMS of the native deviations.
    """
    by: Dict[str, List[ValidationEvent]] = {}
    for e in events:
        by.setdefault(e.metric, []).append(e)

    def rms(vals):
        vals = [v for v in vals if v == v]
        return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("nan")

    out: Dict[str, float] = {}
    for metric, evs in by.items():
        out["n_%s" % metric] = len(evs)
        out["n_%s_outliers" % metric] = sum(1 for e in evs if e.outlier)
    if "rota" in by:
        out["rota_outlier_pct"] = 100.0 * out["n_rota_outliers"] / max(1, out["n_rota"])
    if "rama" in by:
        out["rama_outlier_pct"] = 100.0 * out["n_rama_outliers"] / max(1, out["n_rama"])
    if "clash" in by and n_atoms:
        out["clashscore"] = 1000.0 * out["n_clash_outliers"] / float(n_atoms)
    if "bond" in by:
        out["bond_rmsd"] = rms([e.value for e in by["bond"]])
    if "angle" in by:
        out["angle_rmsd"] = rms([e.value for e in by["angle"]])
    return out


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
    events = extract_all(model, metrics=("rama", "rota", "clash", "bond", "angle"))
    n = model.get_number_of_atoms()
    print(f"{path}: {n} atoms, {len(events)} events")
    for metric in ("rama", "rota", "clash", "bond", "angle"):
        subset = [e for e in events if e.metric == metric]
        flagged = outlier_atoms(subset, metric=metric)
        print(f"  {metric:6s} {len(subset):5d} events, "
              f"{sum(1 for e in subset if e.outlier):4d} outliers, "
              f"{len(flagged):5d} atoms implicated")
    print("  whole-model summary:", summarize(events, n_atoms=n))
    # region roll-up demo: first ~15 residues by atom
    index = residue_atom_index(model.get_hierarchy())
    region_keys = sorted(index, key=lambda k: (k.chain, k.resseq, k.icode))[:15]
    region_atoms = [i for k in region_keys for i in atoms_of(index, k)]
    print(f"  region ({len(region_keys)} residues, {len(region_atoms)} atoms) summary:",
          summarize(restrict(events, region_atoms), n_atoms=len(region_atoms)))
