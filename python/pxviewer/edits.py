"""Read, write and apply cctbx/phenix geometry-restraints *edits* — custom bond, angle and
dihedral restraints a user adds on top of the library defaults.

When cctbx builds restraints it only knows a monomer's *internal* geometry and the standard
links between adjacent residues; it is blind to anything the library does not already
enumerate. Two common cases where that bites:

  * a **covalent ligand** — the bond from a ligand's warhead to a catalytic Cys is not in
    any link definition, so minimization/refinement lets the two drift apart;
  * a **metal centre** — there are no Zn–N/Zn–S bonds in the library, so the site collapses.

An *edits* file is the escape hatch: a small PHIL scope
(``geometry_restraints.edits``) adding those bonds/angles/dihedrals by hand. This module
parses such a file into a simple list of dicts, serialises the list back out, and applies it
when a model's restraints are built — so pxviewer's own minimize/drag honour the same
restraints a downstream phenix.refine would. See :mod:`pxviewer.desktop` for the wiring and
the authoring UI (which turns two/three/four picked atoms into an edit).

Each edit is a dict: ``{"kind", "action", "selections", "ideal", "sigma", ...}`` where
``kind`` is ``bond`` | ``angle`` | ``dihedral``, ``selections`` is the matching list of cctbx
atom-selection strings, and ``ideal`` is the target distance (Å) or angle (deg).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

# The structured edits are carried on a cctbx model here, so restraint builds pick them up.
_ATTR = "_pxviewer_edits"

KIND_ARITY = {"bond": 2, "angle": 3, "dihedral": 4}
_DEFAULT_SIGMA = {"bond": 0.02, "angle": 3.0, "dihedral": 20.0}


def _master() -> Any:
    """The phil master for just the edits scope, so external files parse against the real
    cctbx definition. Carries the ``.alias = refinement.geometry_restraints`` that the full
    master has, so files phenix.refine writes (``refinement.geometry_restraints.edits``)
    resolve as well as the bare ``geometry_restraints.edits``."""
    import iotbx.phil
    from mmtbx.monomer_library.pdb_interpretation import geometry_restraints_edits_str

    return iotbx.phil.parse(
        "geometry_restraints\n  .alias = refinement.geometry_restraints\n{\n"
        "  edits {\n%s\n  }\n}" % geometry_restraints_edits_str)


def parse_edits(text: str) -> Tuple[List[dict], int]:
    """Parse a ``geometry_restraints.edits`` PHIL string into ``(edits, n_unsupported)``.

    Bond/angle/dihedral edits become dicts; planarity/parallelity are counted but not
    returned (not yet supported here). Tolerates the ``refinement.`` prefix phenix writes.
    """
    import iotbx.phil

    scope = _master().fetch(source=iotbx.phil.parse(text)).extract().geometry_restraints.edits
    edits: List[dict] = []
    for b in scope.bond:
        if b.atom_selection_1 and b.atom_selection_2 and b.distance_ideal is not None:
            edits.append({
                "kind": "bond", "action": b.action or "add",
                "selections": [b.atom_selection_1, b.atom_selection_2],
                "ideal": float(b.distance_ideal),
                "sigma": float(b.sigma) if b.sigma else _DEFAULT_SIGMA["bond"]})
    for a in scope.angle:
        if a.atom_selection_1 and a.atom_selection_2 and a.atom_selection_3 \
                and a.angle_ideal is not None:
            edits.append({
                "kind": "angle", "action": a.action or "add",
                "selections": [a.atom_selection_1, a.atom_selection_2, a.atom_selection_3],
                "ideal": float(a.angle_ideal),
                "sigma": float(a.sigma) if a.sigma else _DEFAULT_SIGMA["angle"]})
    for d in scope.dihedral:
        if all([d.atom_selection_1, d.atom_selection_2, d.atom_selection_3,
                d.atom_selection_4]) and d.angle_ideal is not None:
            edits.append({
                "kind": "dihedral", "action": d.action or "add",
                "selections": [d.atom_selection_1, d.atom_selection_2,
                               d.atom_selection_3, d.atom_selection_4],
                "ideal": float(d.angle_ideal),
                "sigma": float(d.sigma) if d.sigma else _DEFAULT_SIGMA["dihedral"],
                "periodicity": int(d.periodicity) if d.periodicity else 1})
    n_unsupported = len(scope.planarity) + len(scope.parallelity)
    return edits, n_unsupported


def _q(selection: str) -> str:
    """Quote an atom-selection for PHIL — they contain spaces, so always quote."""
    return '"' + str(selection).replace('"', "'") + '"'


def edits_to_phil(edits: List[dict]) -> str:
    """Serialise structured edits to a ``geometry_restraints.edits`` PHIL file (one
    definition per line, as PHIL requires)."""
    out = [
        "# Custom geometry-restraints edits, written by pxviewer.",
        "# Bond/angle/dihedral restraints added on top of the monomer-library defaults —",
        "# e.g. a covalent-ligand link or metal coordination the library does not know.",
        "# Read by cctbx/phenix (refinement.geometry_restraints.edits) and by pxviewer.",
        "geometry_restraints.edits {",
    ]
    for e in edits:
        sels = e["selections"]
        if e["kind"] == "bond":
            out += ["  bond {",
                    f"    action = {e.get('action', 'add')}",
                    f"    atom_selection_1 = {_q(sels[0])}",
                    f"    atom_selection_2 = {_q(sels[1])}",
                    f"    distance_ideal = {e['ideal']:.4f}",
                    f"    sigma = {e['sigma']:.4f}",
                    "  }"]
        elif e["kind"] == "angle":
            out += ["  angle {",
                    f"    action = {e.get('action', 'add')}",
                    f"    atom_selection_1 = {_q(sels[0])}",
                    f"    atom_selection_2 = {_q(sels[1])}",
                    f"    atom_selection_3 = {_q(sels[2])}",
                    f"    angle_ideal = {e['ideal']:.4f}",
                    f"    sigma = {e['sigma']:.4f}",
                    "  }"]
        elif e["kind"] == "dihedral":
            out += ["  dihedral {",
                    f"    action = {e.get('action', 'add')}",
                    f"    atom_selection_1 = {_q(sels[0])}",
                    f"    atom_selection_2 = {_q(sels[1])}",
                    f"    atom_selection_3 = {_q(sels[2])}",
                    f"    atom_selection_4 = {_q(sels[3])}",
                    f"    angle_ideal = {e['ideal']:.4f}",
                    f"    sigma = {e['sigma']:.4f}",
                    f"    periodicity = {int(e.get('periodicity', 1))}",
                    "  }"]
    out.append("}")
    return "\n".join(out) + "\n"


def summarize(edit: dict) -> str:
    """A one-line human label for the edits list in the UI."""
    tags = [_short(s) for s in edit["selections"]]
    if edit["kind"] == "bond":
        return f"bond  {tags[0]} – {tags[1]}   {edit['ideal']:.2f} Å  (σ {edit['sigma']:g})"
    if edit["kind"] == "angle":
        return f"angle  {tags[0]} – {tags[1]} – {tags[2]}   {edit['ideal']:.1f}°  (σ {edit['sigma']:g})"
    return (f"dihedral  {tags[0]} – {tags[1]} – {tags[2]} – {tags[3]}   "
            f"{edit['ideal']:.1f}°  (σ {edit['sigma']:g})")


def _short(selection: str) -> str:
    """Compact an atom selection to 'A/145 SG' for the list, keeping chain/resseq/name."""
    parts = {}
    toks = str(selection).split()
    for key in ("chain", "resseq", "resid", "name", "resname"):
        if key in toks:
            i = toks.index(key)
            if i + 1 < len(toks):
                parts[key] = toks[i + 1].strip("'\"")
    chain = parts.get("chain", "")
    res = parts.get("resseq", parts.get("resid", ""))
    name = parts.get("name", "")
    label = "/".join(x for x in (chain, res) if x)
    return f"{label} {name}".strip() or str(selection)


def selection_for_atom(model: Any, atom_index: int) -> str:
    """A cctbx atom-selection string uniquely naming one atom of ``model`` — chain, residue
    and atom name (plus altloc if any) — for authoring an edit from a picked atom."""
    atom = model.get_hierarchy().atoms()[atom_index]
    ag = atom.parent()             # atom_group: resname, altloc
    rg = ag.parent()               # residue_group: resseq, icode
    chain = rg.parent()            # chain: id
    terms = [f"chain {chain.id.strip() or 'A'}",
             f"resseq {rg.resseq.strip()}",
             f"name {atom.name.strip()}"]
    if ag.altloc.strip():
        terms.append(f"altloc {ag.altloc.strip()}")
    icode = rg.icode.strip()
    if icode:
        terms.append(f"icode {icode}")
    return " and ".join(terms)


def geometry_value(kind: str, points: List[Any]) -> float:
    """The current distance (Å) / angle / dihedral (deg) of ``points`` (each an (x,y,z)),
    so an edit authored from picked atoms defaults its target to what is already there."""
    import numpy as np

    p = [np.asarray(x, dtype=float) for x in points]
    if kind == "bond":
        return float(np.linalg.norm(p[0] - p[1]))
    if kind == "angle":
        u, v = p[0] - p[1], p[2] - p[1]
        c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))
    # dihedral p0-p1-p2-p3
    b1, b2, b3 = p[1] - p[0], p[2] - p[1], p[3] - p[2]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    x, y = float(np.dot(n1, n2)), float(np.dot(m, n2))
    return float(np.degrees(np.arctan2(y, x)))


def get_edits(model: Any) -> List[dict]:
    """The structured edits carried on ``model`` (empty list if none)."""
    return list(getattr(model, _ATTR, None) or [])


def set_edits(model: Any, edits: List[dict]) -> None:
    """Carry ``edits`` on ``model`` so the next restraint build applies them."""
    setattr(model, _ATTR, list(edits or []))


def build_restraints(model: Any, *, make_restraints: bool = True, force: bool = False) -> None:
    """Process ``model``'s restraints, folding in any edits carried on it — the one call
    every pxviewer restraint build (minimize, drag) goes through, so custom bonds/angles/
    dihedrals are honoured everywhere.

    ``force=False`` (minimize/drag): reuse an existing restraints manager if there is one —
    matching the old ``process()`` behaviour and avoiding a costly rebuild every minimize
    cycle. The manager it reuses already reflects the current edits, because changing them
    goes through ``force=True``, which unsets and rebuilds. ``force=True`` (after edits
    change): always rebuild, so an edit added or removed takes effect (and a cleared edit
    is really gone — ``process()`` only drops the old manager when given explicit params).
    """
    if not force and model.restraints_manager_available():
        return
    params = model.get_default_pdb_interpretation_params()
    edits = get_edits(model)
    if edits:
        _fill_params(params.geometry_restraints.edits, edits)
    model.process(pdb_interpretation_params=params, make_restraints=make_restraints)


def _fill_params(scope: Any, edits: List[dict]) -> None:
    """Replace a params ``geometry_restraints.edits`` scope's bond/angle/dihedral lists with
    objects built from ``edits`` (deep-copying the scope's own template so every field the
    cctbx scope expects is present)."""
    import copy

    bond_t, angle_t, dih_t = scope.bond[0], scope.angle[0], scope.dihedral[0]
    bonds, angles, dihedrals = [], [], []
    for e in edits:
        sels = e["selections"]
        if e["kind"] == "bond":
            o = copy.deepcopy(bond_t)
            o.action = e.get("action", "add")
            o.atom_selection_1, o.atom_selection_2 = sels
            o.distance_ideal, o.sigma = e["ideal"], e["sigma"]
            bonds.append(o)
        elif e["kind"] == "angle":
            o = copy.deepcopy(angle_t)
            o.action = e.get("action", "add")
            o.atom_selection_1, o.atom_selection_2, o.atom_selection_3 = sels
            o.angle_ideal, o.sigma = e["ideal"], e["sigma"]
            angles.append(o)
        elif e["kind"] == "dihedral":
            o = copy.deepcopy(dih_t)
            o.action = e.get("action", "add")
            (o.atom_selection_1, o.atom_selection_2,
             o.atom_selection_3, o.atom_selection_4) = sels
            o.angle_ideal, o.sigma = e["ideal"], e["sigma"]
            o.periodicity = int(e.get("periodicity", 1))
            dihedrals.append(o)
    scope.bond, scope.angle, scope.dihedral = bonds, angles, dihedrals
