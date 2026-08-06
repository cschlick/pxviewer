"""Read, write and apply cctbx/phenix geometry-restraints *edits* — custom bond, angle and
dihedral restraints a user adds on top of the library defaults.

When cctbx builds restraints it only knows a monomer's *internal* geometry and the standard
links between adjacent residues; it is blind to anything the library does not already
enumerate. Two common cases where that bites:

  * a **covalent ligand** — the bond from a ligand's warhead to a catalytic Cys is not in
    any link definition, so minimization/refinement lets the two drift apart;
  * a **metal center** — there are no Zn–N/Zn–S bonds in the library, so the site collapses.

An *edits* file is the escape hatch: a small PHIL scope
(``geometry_restraints.edits``) adding those bonds/angles/dihedrals by hand.

**The file is cctbx's, and it is kept that way.** A user's PHIL is fetched against cctbx's
own master scope and the resulting scope is handed to ``pdb_interpretation`` whole, so the
proxies fall out of it exactly as they would in phenix. This module holds no representation
of an edit of its own — it stores the scope, projects it for display, and appends to it when
the GUI authors one.

That is a correction, not a preference. An earlier version parsed each edit into a dict and
rebuilt the params from those dicts at build time. Every field the dict had no key for was
silently lost: ``symmetry_operation`` (so a bond to a symmetry mate restrained the wrong
pair of atoms), ``slack``, ``limit``, ``top_out``, and planarity and parallelity restraints
entirely — which were counted as "unsupported" and discarded. Nothing about them needed
supporting; they only needed not to be thrown away.

See :mod:`pxviewer.desktop` for the wiring and the authoring UI (which turns two/three/four
picked atoms into an edit).
"""

from __future__ import annotations

import threading

from typing import Any, List, Optional, Tuple

# The structured edits are carried on a cctbx model here, so restraint builds pick them up.
_ATTR = "_pxviewer_edits"

KIND_ARITY = {"bond": 2, "angle": 3, "dihedral": 4}

#: Sigmas used when the *user authors* an edit in the GUI, where there is no file to read
#: one from and clicking "add a bond restraint" has to mean something. Deliberately **not**
#: applied to a PHIL: a file that leaves sigma out is refused by :func:`validate`, cctbx's
#: own check, rather than quietly restrained at a tightness nobody chose.
AUTHORING_SIGMA = {"bond": 0.02, "angle": 3.0, "dihedral": 20.0}


def _master_scope(model: Any) -> Any:
    """cctbx's own pdb_interpretation phil scope, taken from the model.

    The master, not a hand-rolled subset: everything a user may legitimately write is
    defined in it, and fetching against it is what makes their file mean the same thing
    here as it does to phenix.
    """
    return model.get_default_pdb_interpretation_scope()


def _standalone_master() -> Any:
    """The edits scope alone, for reading a file with no model in hand.

    Carries the ``.alias = refinement.geometry_restraints`` the full master has, so a file
    phenix.refine wrote (``refinement.geometry_restraints.edits``) resolves as well as a
    bare ``geometry_restraints.edits``.
    """
    import iotbx.phil
    from mmtbx.monomer_library.pdb_interpretation import geometry_restraints_edits_str

    return iotbx.phil.parse(
        "geometry_restraints\n  .alias = refinement.geometry_restraints\n{\n"
        "  edits {\n%s\n  }\n}" % geometry_restraints_edits_str)


def edits_from_phil(text: str, model: Any = None) -> Any:
    """Fetch a ``geometry_restraints.edits`` PHIL string against cctbx's master.

    Returns the extracted ``edits`` scope -- cctbx's own object, with every field it
    defines, including the ones pxviewer has no opinion about (``symmetry_operation``,
    ``slack``, ``limit``, ``top_out``, planarity and parallelity restraints).

    This deliberately does **not** convert to any structure of pxviewer's own. An earlier
    version parsed each edit into a dict and rebuilt the params from those dicts at build
    time, which silently dropped every field the dict had no key for: a symmetry-related
    metal bond lost its ``symmetry_operation`` and was restrained to the wrong atom, and
    planarity restraints were counted and discarded. The file the user wrote is now handed
    to cctbx as-is and the proxies fall out of it normally.
    """
    import iotbx.phil

    master = _master_scope(model) if model is not None else _standalone_master()
    extracted = master.fetch(sources=[iotbx.phil.parse(text)]).extract()
    return extracted.geometry_restraints.edits


def empty_edits(model: Any = None) -> Any:
    """An edits scope with nothing in it."""
    return edits_from_phil("", model)


def validate(scope: Any) -> None:
    """Check an edits scope with cctbx's own validator, raising on anything incomplete.

    ``pdb_interpretation.validate_geometry_edits_params`` is the check phenix runs over
    this scope, and it has to be called deliberately: nothing inside ``model.process``
    invokes it, so an edit missing its ``sigma`` or its ideal value is not refused there
    -- it is **silently skipped**, and the user is left with a restraint they believe is
    holding two atoms together and is not.

    Calling cctbx's function rather than writing the same checks here is the point: the
    rules for a well-formed edit belong to cctbx, and a copy of them would drift.
    """
    from mmtbx.monomer_library.pdb_interpretation import validate_geometry_edits_params

    validate_geometry_edits_params(scope)


#: The edit kinds cctbx defines, and the atom-selection fields each one carries. Order is
#: the order they are listed to the user.
EDIT_FIELDS = {
    "bond": ("atom_selection_1", "atom_selection_2"),
    "angle": ("atom_selection_1", "atom_selection_2", "atom_selection_3"),
    "dihedral": ("atom_selection_1", "atom_selection_2", "atom_selection_3",
                 "atom_selection_4"),
    "planarity": ("atom_selection",),
    "parallelity": ("atom_selection_1", "atom_selection_2"),
}

#: What each kind calls its target value, and how to render it.
_IDEAL_FIELD = {"bond": "distance_ideal", "angle": "angle_ideal",
                "dihedral": "angle_ideal", "parallelity": "target_angle_deg"}


def entries(scope: Any):
    """``(kind, object)`` for every populated edit in ``scope``, in cctbx's own order.

    A *view* of the scope for display and removal -- never a copy that gets applied. An
    entry counts as populated when its selections are filled in; cctbx's extract yields no
    empty templates, but a partially written one should not be listed as a restraint.
    """
    out = []
    for kind, fields in EDIT_FIELDS.items():
        for obj in getattr(scope, kind, None) or []:
            if all(getattr(obj, f, None) for f in fields):
                out.append((kind, obj))
    return out


def selections_of(kind: str, obj: Any) -> List[str]:
    return [getattr(obj, f) for f in EDIT_FIELDS[kind]]


def summarize(kind: str, obj: Any) -> str:
    """A one-line human label for the edits list in the UI."""
    tags = [_short(s) for s in selections_of(kind, obj)]
    sigma = getattr(obj, "sigma", None)
    suffix = f"  (\u03c3 {sigma:g})" if sigma is not None else ""
    if kind == "bond":
        value = obj.distance_ideal
        symop = getattr(obj, "symmetry_operation", None)
        via = f"  [{symop}]" if symop else ""
        return f"bond  {tags[0]} \u2013 {tags[1]}   {value:.2f} \u00c5{via}{suffix}"
    if kind == "angle":
        return (f"angle  {tags[0]} \u2013 {tags[1]} \u2013 {tags[2]}   "
                f"{obj.angle_ideal:.1f}\u00b0{suffix}")
    if kind == "dihedral":
        return (f"dihedral  {tags[0]} \u2013 {tags[1]} \u2013 {tags[2]} \u2013 {tags[3]}   "
                f"{obj.angle_ideal:.1f}\u00b0{suffix}")
    if kind == "planarity":
        return f"planarity  {tags[0]}{suffix}"
    return (f"parallelity  {tags[0]} \u2013 {tags[1]}   "
            f"{getattr(obj, 'target_angle_deg', 0) or 0:.1f}\u00b0{suffix}")


def count(scope: Any) -> int:
    return len(entries(scope))


def merge(scope: Any, other: Any) -> int:
    """Append ``other``'s edits onto ``scope``. Returns how many were added."""
    added = 0
    for kind in EDIT_FIELDS:
        incoming = [obj for k, obj in entries(other) if k == kind]
        if incoming:
            setattr(scope, kind, list(getattr(scope, kind, None) or []) + incoming)
            added += len(incoming)
    return added


def remove(scope: Any, index: int) -> None:
    """Drop the ``index``-th populated edit, numbered as :func:`entries` lists them."""
    listing = entries(scope)
    if not 0 <= index < len(listing):
        raise IndexError("no edit at position %d" % index)
    kind, target = listing[index]
    setattr(scope, kind,
            [obj for obj in getattr(scope, kind) if obj is not target])


def new_entry(model: Any, kind: str, selections: List[str], *, ideal: float,
              sigma: float, periodicity: int = 1) -> Any:
    """A populated cctbx edit object of ``kind``, for the GUI's authoring path.

    Written as PHIL and fetched back, rather than assembled field by field. It is the same
    round trip a file takes, so an authored edit and a loaded one are the same kind of
    object with the same defaults filled in, and there is still no place where pxviewer
    decides what an edit is made of.
    """
    if kind not in EDIT_FIELDS:
        raise ValueError("unknown edit kind %r" % kind)
    fields = EDIT_FIELDS[kind]
    if len(selections) != len(fields):
        raise ValueError("a %s needs %d selections (got %d)"
                         % (kind, len(fields), len(selections)))

    lines = ["    action = add"]
    for field, selection in zip(fields, selections):
        lines.append('    %s = "%s"' % (field, str(selection).replace('"', "'")))
    if kind in _IDEAL_FIELD:
        lines.append("    %s = %.6f" % (_IDEAL_FIELD[kind], float(ideal)))
    lines.append("    sigma = %.6f" % float(sigma))
    if kind == "dihedral":
        lines.append("    periodicity = %d" % int(periodicity))

    text = "geometry_restraints.edits {\n  %s {\n%s\n  }\n}" % (
        kind, "\n".join(lines))
    return getattr(edits_from_phil(text, model), kind)[0]


def add_entry(scope: Any, obj: Any, kind: str) -> None:
    setattr(scope, kind, list(getattr(scope, kind, None) or []) + [obj])


def edits_as_phil(scope: Any, model: Any = None) -> str:
    """Serialise an edits scope back to PHIL text, via cctbx's own formatter.

    Formatted by cctbx rather than written by hand, so a field pxviewer never looks at is
    still written out and still means the same thing when the file is read again -- by
    this, by phenix, or by a person.
    """
    master = _master_scope(model) if model is not None else _standalone_master()
    params = master.fetch(sources=[]).extract()
    params.geometry_restraints.edits = scope
    formatted = master.format(python_object=params).get_without_substitution(
        "geometry_restraints.edits")
    block = formatted[0] if isinstance(formatted, list) else formatted
    return "\n".join([
        "# Custom geometry-restraints edits, written by pxviewer.",
        "# Bond/angle/dihedral restraints added on top of the monomer-library defaults \u2014",
        "# e.g. a covalent-ligand link or metal coordination the library does not know.",
        "# Read by cctbx/phenix (refinement.geometry_restraints.edits) and by pxviewer.",
        "geometry_restraints {",
        block.as_str().rstrip(),
        "}",
        "",
    ])


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


def get_edits(model: Any) -> Any:
    """The edits scope carried on ``model`` -- cctbx's own object, never a copy of it."""
    scope = getattr(model, _ATTR, None)
    if scope is None:
        scope = empty_edits(model)
        setattr(model, _ATTR, scope)
    return scope


#: Serialises restraint builds; see build_restraints.
_BUILD_LOCK = threading.Lock()


def set_edits(model: Any, scope: Any) -> None:
    """Carry an edits scope on ``model`` so the next restraint build applies it."""
    setattr(model, _ATTR, scope if scope is not None else empty_edits(model))


def build_restraints(model: Any, *, make_restraints: bool = True, force: bool = False) -> None:
    """Process ``model``'s restraints, folding in any edits carried on it — the one call
    every pxviewer restraint build (minimize, drag) goes through, so custom bonds/angles/
    dihedrals are honored everywhere.

    ``force=False`` (minimize/drag): reuse an existing restraints manager if there is one —
    matching the old ``process()`` behavior and avoiding a costly rebuild every minimize
    cycle. The manager it reuses already reflects the current edits, because changing them
    goes through ``force=True``, which unsets and rebuilds. ``force=True`` (after edits
    change): always rebuild, so an edit added or removed takes effect (and a cleared edit
    is really gone — ``process()`` only drops the old manager when given explicit params).

    Serialised across threads. Building replaces the model's restraints manager in place, so
    two builds of the same model at once leave it in a state neither asked for. There is a
    real chance of that: the drag pre-warm runs on its own thread precisely so the user is
    not waiting on it, while minimize and the drag itself build from theirs. The lock is
    global rather than per-model because a build is rare, seconds long, and CPU-bound —
    letting two run at once would not help even on different models.
    """
    with _BUILD_LOCK:
        if not force and model.restraints_manager_available():
            return
        params = model.get_default_pdb_interpretation_params()
        scope = getattr(model, _ATTR, None)
        if scope is not None and count(scope):
            # cctbx skips an incomplete edit rather than refusing it, so check first --
            # see validate().
            validate(scope)
            # The user's scope, handed over whole. Nothing is rebuilt from an
            # intermediate of pxviewer's own, so nothing cctbx understands is lost on
            # the way -- see edits_from_phil.
            params.geometry_restraints.edits = scope
        model.process(pdb_interpretation_params=params, make_restraints=make_restraints)
