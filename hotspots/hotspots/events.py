"""Validation events: the physics layer of the hotspot field.

Each validation problem (a Ramachandran outlier, a bad rotamer, a clash) becomes
an *event* carrying:

  - the atom positions it implicates in space (its native spatial footprint), and
  - a calibrated *severity* mark, anchored so 1.0 == the community outlier cut.

This layer does NOT smooth anything into a grid. It just extracts the marked
events. The field is a cheap convolution built on top of these (separate module),
so the expensive mmtbx work is decoupled from the spatial smoothing and its knob.

**Extraction now comes from ``validation_events``**, the module shared verbatim with
pxviewer. That file owns the one definition of *which residue a result belongs to and
which atoms it implicates*; this file owns only the calibration on top. Keeping those
separate is the point: pxviewer maps the same events to unbounded surprisal severity
while ``concern.py`` here maps them to bounded [0, 1] concern, and neither is a
rescaling of the other -- but they must never disagree about *where* a problem is.

Severity marks below are the legacy surprisal scale, retained because they are what
``is_outlier`` and ``meta['native_severity']`` mean. The bounded concern calibration
that actually drives the maps lives in ``concern.py``.

  - Ramachandran / rotamer: surprisal  -log10(p) / -log10(p_cut), p from the
    percentage the analyzer already returns.  (p_cut = 0.05% rama, 0.30% rota.)
  - Clash: |overlap| / 0.40 A.  No reference tail exists for clash overlap, so
    this is a linear anchor at the 0.4 A MolProbity cut (an assumed shape).

Run under the phenix python (libtbx.python) so the reference data (rotarama_data,
geostd, reduce2/probe2) is wired up automatically.
"""
from __future__ import annotations

import gzip
import importlib.util
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import iotbx.pdb


def _load_shared(name: str = "validation_events"):
    """Import the shared extractor from the one place it lives.

    There is exactly one copy of that file, never two: a second copy that drifts looks like
    one definition while being two, which is the failure the file exists to prevent. While
    this directory sits inside pxviewer, the copy is ``python/pxviewer/validation_events.py``.

    Loaded by explicit path rather than through ``sys.path`` on purpose — pxviewer's package
    directory holds its own ``concern.py`` and ``field.py``, whose names collide with this
    package's, so putting it on the path would risk shadowing them.

    **Splitting this directory back out:** drop a copy of the file beside this one. The
    sibling location is tried first, so it takes over with no code change here.
    """
    here = Path(__file__).resolve().parent
    candidates = (here / f"{name}.py",                                    # standalone
                  here.parent.parent / "python" / "pxviewer" / f"{name}.py")   # in pxviewer
    for path in candidates:
        if not path.exists():
            continue
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError(
        f"could not find {name}.py — expected beside this file or at "
        f"{candidates[-1]}. It is shared verbatim with pxviewer; copy it, do not rewrite it.")


ve = _load_shared()

# --- community outlier thresholds (the calibration we inherit) ------------------
RAMA_CUT_PCT = 0.05   # Ramachandran outlier: score < 0.05 %
ROTA_CUT_PCT = 0.30   # rotamer outlier:      score < 0.30 %
CLASH_CUT_A = ve.CLASH_OUTLIER_A   # MolProbity clash: overlap >= 0.40 A

# floor the probability so score==0 gives a finite (large) surprisal rather than inf
_SCORE_FLOOR_PCT = 1.0e-3

#: Kept for reference; the shared module owns the localization rules now.
_BACKBONE = set(ve.MAINCHAIN)

Xyz = Tuple[float, float, float]


@dataclass
class Event:
    """One validation problem, localized in space with a calibrated severity."""
    metric: str                 # 'rama' | 'rota' | 'clash'
    severity: float             # dimensionless mark, 1.0 == outlier cut
    atoms_xyz: List[Xyz]        # positions this event deposits severity at
    meta: Dict = field(default_factory=dict)

    @property
    def is_outlier(self) -> bool:
        return self.severity >= 1.0


# --- severity marks -------------------------------------------------------------
def _surprisal_mark(score_pct: float, cut_pct: float) -> float:
    """-log10(p) / -log10(p_cut), with p = score_pct/100. ==1.0 at the cut."""
    p = max(float(score_pct), _SCORE_FLOOR_PCT) / 100.0
    pc = cut_pct / 100.0
    return math.log10(p) / math.log10(pc)   # both logs negative -> positive ratio


def _clash_mark(overlap_a: float) -> float:
    """|overlap|/0.4. overlap is negative for a real clash; >=0 -> no severity."""
    return max(0.0, -float(overlap_a)) / CLASH_CUT_A


# --- model loading --------------------------------------------------------------
def load_hierarchy(path: str):
    """Load a .pdb/.cif (optionally .gz) into a pdb hierarchy."""
    return load_model(path).get_hierarchy()


def load_model(path: str):
    """Load a .pdb/.cif (optionally .gz) into an mmtbx model manager.

    A model, not just a hierarchy: clash extraction runs probe2, which needs one.
    """
    from mmtbx.model import manager as model_manager

    if path.endswith(".gz"):
        suffix = ".cif" if ".cif" in path else ".pdb"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(gzip.open(path, "rb").read())
        tmp.close()
        path = tmp.name
    return model_manager(model_input=iotbx.pdb.input(file_name=path), log=None)


# --- extractors -----------------------------------------------------------------
#
# Thin adapters over validation_events: it decides which atoms a result implicates,
# this decides what number rides along. Keep it that way -- a localization rule that
# lives here and not there is exactly the drift the shared file exists to stop.


def _rama_event(e) -> Event:
    return Event("rama", _surprisal_mark(e.value, RAMA_CUT_PCT), list(e.atoms_xyz),
                 meta=dict(id=e.detail.get("id", ""), score=e.value,
                           res_type=e.detail.get("res_type"),
                           rama_type=e.detail.get("rama_type"),
                           outlier=bool(e.outlier)))


def _rota_event(e) -> Event:
    return Event("rota", _surprisal_mark(e.value, ROTA_CUT_PCT), list(e.atoms_xyz),
                 meta=dict(id=e.detail.get("id", ""), score=e.value,
                           rotamer=e.detail.get("rotamer"),
                           outlier=bool(e.outlier)))


def _clash_event(e) -> Event:
    # Deposit at the contact point: the true interface, always populated, and robust to
    # H atoms that live only in the H-added model. The shared extractor implicates both
    # atoms (and each one's parent heavy atom) and carries the contact point alongside,
    # so this choice stays here rather than being made for us. Falls back to the atoms if
    # a dot arrived without a location.
    contact = e.detail.get("contact_xyz")
    xyz = [tuple(contact)] if contact else list(e.atoms_xyz)
    # concern.py reads meta['overlap'] in probe's signed convention (negative == clash);
    # the shared event carries it as a positive magnitude.
    return Event("clash", _clash_mark(-e.value), xyz,
                 meta=dict(id="", overlap=-float(e.value), outlier=bool(e.outlier)))


def extract_ramachandran(hierarchy, amap=None) -> List[Event]:
    """Rama result -> deposit on residue i's backbone atoms (N, CA, C, O)."""
    return [_rama_event(e) for e in ve.extract_ramachandran(hierarchy)]


def extract_rotamer(hierarchy, amap=None) -> List[Event]:
    """Rotamer result -> deposit on the sidechain atoms only (not N/CA/C/O)."""
    return [_rota_event(e) for e in ve.extract_rotamer(hierarchy)]


def model_has_hydrogens(model) -> bool:
    elements = {e.strip().upper() for e in model.get_hierarchy().atoms().extract_element()}
    return bool(elements & {"H", "D"})


def add_hydrogens(model):
    """Return a new model with explicit H placed and optimized by reduce2.

    Clashes are overwhelmingly hydrogen-mediated and MolProbity's clashscore is *defined*
    with explicit H, so this is the calibrated path; a heavy-atom-only pass finds a small
    fraction of the contacts. ``approach=add`` with ``n_terminal_charge=no_charge`` avoids
    the N-terminal propeller placement that crashes on these inputs, and
    ``ignore_missing_restraints`` keeps ions from stopping the run. cctbx exposes no
    in-memory entry point, so this round-trips through a temp file.
    """
    from iotbx.cli_parser import run_program
    from iotbx.data_manager import DataManager
    from mmtbx.programs import reduce2

    workdir = tempfile.mkdtemp(prefix="hotspots-reduce2-")
    in_path = os.path.join(workdir, "in.pdb")
    out_path = os.path.join(workdir, "in_H.pdb")
    with open(in_path, "w") as fh:
        fh.write(model.model_as_pdb())
    args = [in_path, "approach=add", "n_terminal_charge=no_charge",
            "ignore_missing_restraints=True", "add_flip_movers=True",
            f"output.filename={out_path}", "output.overwrite=True"]
    with open(os.devnull, "w") as devnull:
        run_program(program_class=reduce2.Program, args=args, logger=devnull)

    dm = DataManager()
    dm.set_overwrite(True)
    dm.process_model_file(out_path)
    return dm.get_model(out_path)


def extract_clash(model, keep_hydrogens=False, dots=None):
    """Clash -> deposit at the contact point. Returns ``(events, clashscore)``.

    Uses **probe2** (ships with cctbx) rather than ``mmtbx.validation.clashscore``, which
    shells out to the classic Duke ``probe`` binary that is not installed in every
    environment -- that absence is what used to make this channel impossible to generate
    here.

    ``model`` is used as given: add hydrogens first (see :func:`add_hydrogens`) for the
    calibrated MolProbity path. The returned clashscore normalises on this model's atom
    count, so it is comparable to MolProbity's only when H are present.
    """
    shared = ve.extract_clashes(model, dots=dots)
    events = [_clash_event(e) for e in shared]
    n_atoms = model.get_hierarchy().atoms_size()
    n_outliers = sum(1 for e in shared if e.outlier)
    clashscore = (1000.0 * n_outliers / n_atoms) if n_atoms else float("nan")
    return events, clashscore


def extract_all(model, use_hydrogens=True, dots=None) -> Dict:
    """Extract the core event set (rama + rota + clash). Returns events plus a
    manifest declaring what went into them (H mode, clashscore).

    ``model`` is an mmtbx model manager (see :func:`load_model`). With ``use_hydrogens``
    (the default) hydrogens are placed by reduce2 first, which is the calibrated
    MolProbity clash path; without, only heavy-atom contacts are found and the manifest
    says so. Rama and rota read the *original* model either way, so the residue numbering
    in the manifest matches the input.
    """
    hierarchy = model.get_hierarchy()
    index = ve.residue_atom_index(hierarchy)
    events: List[Event] = []
    events += [_rama_event(e) for e in ve.extract_ramachandran(hierarchy, index=index)]
    events += [_rota_event(e) for e in ve.extract_rotamer(hierarchy, index=index)]

    # Hydrogens are added to a *separate* model used only for probe2. The events carry
    # Cartesian footprints, so they need no atom-index agreement with the original -- and
    # keeping the original untouched means rama/rota above are unaffected by it.
    clash_model = model
    hydrogens_used = bool(use_hydrogens)
    if use_hydrogens and not model_has_hydrogens(model):
        try:
            clash_model = add_hydrogens(model)
        except Exception as exc:   # reduce2 unavailable or refused this model
            hydrogens_used = False
            print(f"warning: reduce2 could not add hydrogens ({exc}); "
                  f"falling back to the heavy-atom clash pass", flush=True)
    clash_events, clashscore_val = extract_clash(clash_model, dots=dots)
    events += clash_events
    return dict(
        events=events,
        manifest=dict(
            core_metrics=["rama", "rota", "clash"],
            hydrogens=hydrogens_used,
            clashscore=clashscore_val,
            n_atoms=hierarchy.atoms_size(),
            clash_n_atoms=clash_model.get_hierarchy().atoms_size(),
        ),
    )


if __name__ == "__main__":
    import sys
    from collections import Counter

    path = sys.argv[1]
    model = load_model(path)
    out = extract_all(model, use_hydrogens=True)
    evs, man = out["events"], out["manifest"]

    print(f"model: {path}")
    print(f"manifest: {man}")
    by = Counter(e.metric for e in evs)
    print(f"events: {len(evs)}  by metric: {dict(by)}")
    outliers = [e for e in evs if e.is_outlier]
    outc = Counter(e.metric for e in outliers)
    print(f"outliers (severity>=1.0): {len(outliers)}  by metric: {dict(outc)}")
    for m in ("rama", "rota", "clash"):
        top = sorted((e for e in evs if e.metric == m),
                     key=lambda e: e.severity, reverse=True)[:3]
        print(f"\n  top {m}:")
        for e in top:
            print(f"    sev={e.severity:5.2f}  natoms={len(e.atoms_xyz)}  {e.meta}")
