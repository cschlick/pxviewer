"""Corpus driver for the hotspots figures. Computes the dump once; figures come after.

This is the harness for figures A, B and C in ``../../FIGURES.md``. It is deliberately its
own thing rather than a general corpus runner: it generates fields **in memory** and never
writes a CCP4, because the figures need sampled numbers and a corpus of maps nobody opens
costs ~9 MB per small structure at the 1.0 A the figures require.

What it measures, and the two decisions that shape it:

* **The evaluation universe is heavy atoms only.** Hydrogens sit ~1 A from their parent
  heavy atom, so at sigma = 2 A they carry no independent spatial signal, and including them
  makes recall mean different things for depositions that ship hydrogens and those that do
  not -- a corpus-level confound. Clash survives the restriction because
  ``extract_clashes(inherit_to_heavy=True)`` already hands each hydrogen's clash to its
  parent heavy atom (verified: 0 of 96 clash outlier events on 1TEC lose their heavy atom).
  The restriction applies to the whole universe -- flagged set, hot set, and sampled values
  alike -- never to one side of a ratio.
* **Figure B's target set is *concerning* atoms, not flagged outliers.** The concern curve
  rises from the 2.0% favored boundary, well before the outlier cut, so the field
  legitimately marks residues MolProbity never flags. Judged against outliers alone the
  figure grows a 23 A tail and reports correct behaviour as failure.

Usage -- sharded, resumable, one JSON line per model appended as it goes:

    libtbx.python corpus/figure_data.py SAMPLE.txt OUT_DIR --shard 0/8 --resume
    libtbx.python corpus/figure_data.py SAMPLE.txt OUT_DIR --report

Sharding rather than a process pool: forked cctbx state is a poor bet, and a shard that dies
should take nothing else with it.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hotspots"))

from concern import build_concern_fields, molprobity_concern_events  # noqa: E402
from density import build_density_fields  # noqa: E402
from events import _load_shared, extract_all, load_model  # noqa: E402

ve = _load_shared()

MIRROR = "/root/data/pdb_mmcif"

#: The display threshold. Concern's yellow anchor -- the point the viewer shows anything.
HOT = 0.5
SIGMA = 2.0
#: Figures are generated at 1.0 A, not the 2.0 A fast-viewing default: at 2.5 A the field
#: already loses one flagged outlier in five. See ../README.md.
SPACING = 1.0
#: Figure B's target set: "worse than 2.0%" is where the concern curve starts rising.
CONCERNING_PCT = 2.0
#: The same idea for clash, in angstroms of overlap: its concern curve starts rising here.
#: Imported rather than hardcoded so the figure cannot drift from the calibration it judges.
from concern import CLASH_ZERO_OVERLAP_A as CLASH_CONCERNING_OVERLAP_A  # noqa: E402
#: Skip anything larger. Both guards are recorded when they fire, never silent.
#:
#: Measured over this 2000-structure sample: median 4,197 atoms, p90 16,096, p99 73,475,
#: max 129,875. A cap here excludes 1.6% of the sample. It is set on the atom count because
#: the blow-up happens inside reduce2/probe2 -- *before* the grid exists -- so the voxel
#: guard below never gets a chance on a huge model: 1k8a (98,560 atoms) reached 11.6 GB RSS
#: and took a shard with it.
MAX_ATOMS = 50_000

#: Aggregate memory headroom, in GB, required before starting a model, so N shards cannot
#: all allocate at the same moment and have the kernel shoot the largest.
#:
#: **The threshold must sit above what the shards consume at rest, or it never clears.** Set
#: at 5.0 GB with 8 shards it was self-defeating: steady state was ~20 of 23 GB, the guard
#: never cleared, and 82 models each burned the full timeout -- 13.1 h of wall clock spent
#: sleeping. The concurrency has to leave the headroom the guard asks for; at 6 shards
#: (~15 GB resident) there is ~8 GB free and this clears immediately in the normal case.
#: The timeout is short for the same reason: waiting is a last resort, not a scheduler.
MIN_AVAILABLE_GB = 3.0
MEM_WAIT_TIMEOUT_S = 180

#: Peak memory per atom, GB. A flat headroom threshold is not enough: what actually blows up
#: is probe2's dot list, which is a Python dict per dot and scales with *contacts*, so it
#: grows faster than the atom count and is invisible to both caps above. 7cdl -- 45,133 atoms,
#: a 4.4e6-voxel grid, comfortably inside both -- still reached 8.6 GB and took a shard with
#: it. That measurement sets this constant: 8.6 GB / 45,133 atoms.
GB_PER_ATOM = 1.9e-4
#: How long to wait for a large model's headroom before deferring it. Deferral is recorded as
#: its own status so a later low-concurrency pass can pick these up; it is NOT a size skip,
#: and conflating the two would quietly shrink the corpus by whatever the machine was doing
#: at the time.
BIG_MODEL_WAIT_S = 420

#: The guard that actually matters. Memory scales with the *bounding-box volume*, not the
#: atom count -- an elongated complex of modest atom count can want a far bigger grid than a
#: compact one twice its mass. build_concern_fields holds ~8-10 grids at peak (a target, a
#: raw and a clipped copy per metric, the combined, then figure C's held-out set), so at 8
#: bytes a voxel this cap keeps a shard under ~1 GB. Measured the hard way: without it the
#: kernel OOM-killed two shards at 6-10 GB RSS. 1.2e7 voxels is a ~228 A cube, which no
#: structure in the 30-model pilot came close to (max 5.8e6).
MAX_VOXELS = 12_000_000

#: Figure B histogram: 0-30 A at 0.05 A, plus one overflow bin. Fine enough that a pooled
#: median from summed counts is exact to within a bin.
B_EDGES = np.arange(0.0, 30.0 + 1e-9, 0.05)

#: Figure C's spatially matched null.
NULL_TRIALS = 50
NULL_HOT_CAP = 20_000     # subsample the hot region past this, for tractable KD queries
NULL_ENV_R = 5.0          # a point is "inside the molecular envelope" within this of an atom
NULL_MIN_INSIDE = 0.90    # a placement is valid if this fraction of it lands in the envelope
NULL_PLACEMENT_TRIES = 40
#: Total placement searches allowed per structure. The null costs
#: NULL_TRIALS x n_components placements, and n_components grows with the number of
#: concerning residues -- so a badly-strained structure can want hundreds of thousands of KD
#: queries and grind for hours. Trials are reduced (never below a floor) to stay inside this,
#: and ``n_null`` is recorded so a noisier null is visible rather than assumed away.
NULL_PLACEMENT_BUDGET = 4_000
NULL_MIN_TRIALS = 8

#: Hard per-model wall clock. Some structures in the tail of this corpus ran for hours in a
#: single stage; without a bound one model can hold a whole shard indefinitely. A model that
#: hits this is recorded as ``timeout`` -- a stated exclusion, not a silent hole.
MODEL_TIMEOUT_S = 1200


#: Overridable at the command line (``--null-budget``) so a structure the budget throttled
#: can be recomputed under the flat policy. A corpus figure whose configuration varies between
#: structures is a corpus figure nobody can restate, so the override exists to *remove* that
#: variation, not to add a knob.
_NULL_BUDGET_OVERRIDE = None


def _null_budget() -> int:
    return NULL_PLACEMENT_BUDGET if _NULL_BUDGET_OVERRIDE is None else _NULL_BUDGET_OVERRIDE


#: Overridable at the command line (``--hot-threshold``). Exists so a field can be measured at
#: its *own* operating point rather than at another field's: the density construction's natural
#: knee is 1.0 while concern's is 0.5, and comparing them at one shared number prices only one
#: point on each curve.
_HOT_OVERRIDE = None


def _hot() -> float:
    return HOT if _HOT_OVERRIDE is None else _HOT_OVERRIDE


class _Timeout(Exception):
    """A model exceeded MODEL_TIMEOUT_S."""


def _arm_timeout(seconds: int):
    """Arm (or with 0, disarm) the per-model wall clock.

    SIGALRM is delivered between Python bytecodes, so this bounds Python-level work -- which
    is where the pathological cost sat (the figure C null's placement loop). It cannot
    interrupt a single long call inside numpy or scipy C code; such a call would overrun and
    the alarm would fire when it returns. A hard bound would need a subprocess per model,
    which is not worth its cost for the tail this catches.
    """
    import signal

    if not hasattr(signal, "SIGALRM"):
        return
    if seconds:
        signal.signal(signal.SIGALRM, lambda *_a: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(int(seconds))


def model_path(pdb_id: str) -> str:
    return os.path.join(MIRROR, pdb_id[1:3], pdb_id + ".cif.gz")


def read_ids(path: str):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def available_gb() -> float:
    """System memory actually available, from /proc/meminfo — not 'free', which excludes
    reclaimable cache and so reads far too pessimistically on a machine doing heavy I/O."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1_048_576.0
    except Exception:
        pass
    return float("inf")     # unknown: never block on a guard we cannot evaluate


def wait_for_memory(min_gb=MIN_AVAILABLE_GB, timeout_s=MEM_WAIT_TIMEOUT_S):
    """Block until the machine has headroom. Returns (waited_seconds, available_gb).

    Proceeds anyway on timeout rather than skipping the model: a shard that quietly declined
    work because the machine was briefly busy would leave a hole in the corpus that looks
    like a result. If it then dies, ``--resume`` picks the model up next time.
    """
    started = time.time()
    while time.time() - started < timeout_s:
        avail = available_gb()
        if avail >= min_gb:
            return round(time.time() - started, 1), avail
        time.sleep(10)
    return round(time.time() - started, 1), available_gb()


def sample_field(field, xyz) -> np.ndarray:
    """Read a Field at Cartesian points, trilinearly — the same thing the viewer does."""
    from scipy.ndimage import map_coordinates

    sites = np.asarray(xyz, dtype=float).reshape(-1, 3)
    ijk = np.empty((3, sites.shape[0]), dtype=float)
    for axis in range(3):
        ijk[axis] = (sites[:, axis] - field.origin[axis]) / field.spacing
    return map_coordinates(field.data, ijk, order=1, mode="constant", cval=0.0)


def heavy_mask(hierarchy) -> np.ndarray:
    """True for every non-hydrogen atom. See the module docstring for why this exists."""
    elements = [e.strip().upper() for e in hierarchy.atoms().extract_element()]
    return np.array([e not in ("H", "D") for e in elements], dtype=bool)


def hot_voxel_xyz(field, threshold=None) -> np.ndarray:
    """Cartesian centres of every voxel at or above the display threshold."""
    idx = np.argwhere(field.data >= (_hot() if threshold is None else threshold))
    if not idx.size:
        return np.empty((0, 3))
    return idx.astype(float) * field.spacing + np.asarray(field.origin, dtype=float)


def _scored_events(shared, metric):
    """The events recall should be scored against, and how many were set aside.

    For every channel but one this is just the validator's outlier set. omega is the
    exception, and not because of anything the field does: omegalyze flags *every* non-trans
    peptide, so an ordinary cis-proline arrives flagged, and ``concern._omega_concern``
    deliberately scores cis-proline 0.0 on the grounds that it is common, legitimate, and
    would otherwise light up a hotspot on most structures in the PDB.

    Scoring recall against the raw omegalyze flag therefore measures a disagreement the
    project chose to have, and reports it as a field failure -- it comes out at exactly 0.000
    on structures whose only flagged peptides are cis-prolines. The set aside count is
    returned so the exclusion is visible in the data rather than buried here; figure B made
    the same move for the same reason, scoring against *concerning* atoms rather than
    outliers.
    """
    if metric != "omega":
        return shared, 0
    kept, dropped = [], 0
    for e in shared:
        if (e.metric == "omega" and e.detail.get("kind") == "cis"
                and e.detail.get("is_proline") and e.outlier):
            dropped += 1
            continue
        kept.append(e)
    return kept, dropped


def figure_a(shared, sampled, heavy, metric) -> dict:
    """Operating point: recall and precision at the display threshold, over heavy atoms.

    Recall is the number that matters -- a field that loses a real outlier is wrong.
    Precision is reported for completeness, but see figure B before drawing any conclusion
    from it: the sigma ~ 2 A splat is wider than the ~3.8 A between adjacent CA atoms, so
    most "false positives" are neighbours of concerning residues.
    """
    scored, n_excluded = _scored_events(shared, metric)
    flagged = np.zeros(heavy.size, dtype=bool)
    for i in ve.outlier_atoms(scored, metric=metric):
        flagged[i] = True
    flagged &= heavy
    marked = (sampled >= _hot()) & heavy
    n_flagged, n_marked = int(flagged.sum()), int(marked.sum())
    hit = int((flagged & marked).sum())
    return {
        "n_heavy": int(heavy.sum()),
        "n_flagged": n_flagged,
        "n_marked": n_marked,
        "n_hit": hit,
        "recall": (hit / n_flagged) if n_flagged else None,
        "precision": (hit / n_marked) if n_marked else None,
        "prevalence": (n_flagged / int(heavy.sum())) if heavy.any() else None,
        # Non-zero for omega only: legitimate cis-prolines the calibration scores 0.0 on
        # purpose. Recorded so the corpus can say how large that population is rather than
        # leaving the exclusion invisible.
        "n_excluded": n_excluded,
    }


def figure_b(field, shared, sites, heavy, metric) -> dict:
    """Spatial error: distance from every hot voxel to the nearest *concerning* atom.

    ``concerning`` widens past the outlier boolean on purpose for rama/rota. Clash has no
    percentage tail and its concern is gated at the 0.40 A cut, so for clash the atoms that
    deposit are exactly the flagged ones and the default predicate is already right.
    """
    from scipy.spatial import cKDTree

    # Judge each channel against the set it was actually built from, never against flagged
    # outliers alone -- that is the whole point of this figure. For rama/rota the concern
    # curve rises from the 2% favored boundary. Clash used to have no sub-threshold tail, so
    # its flagged set *was* its deposited set and the default outlier predicate was correct;
    # since the re-anchoring it deposits from CLASH_ZERO_OVERLAP_A upward, and judging it
    # against outliers alone grew exactly the artifact this figure exists to avoid (1TEC max
    # 3.79 -> 10.10 A, from hot voxels legitimately placed on sub-threshold contacts).
    if metric == "clash":
        predicate = (lambda e: e.units == "angstrom"
                     and abs(float(e.value)) >= CLASH_CONCERNING_OVERLAP_A)
    else:
        predicate = ve.worse_than_percent(CONCERNING_PCT)
    idx = sorted(i for i in ve.outlier_atoms(shared, metric=metric, predicate=predicate)
                 if heavy[i])
    hot = hot_voxel_xyz(field)
    if not idx or not len(hot):
        return {"n_hot_voxels": int(len(hot)), "n_target_atoms": len(idx), "hist": None}
    # workers=-1: this query runs over every hot voxel (millions on a large
    # structure) and is otherwise single-threaded for no reason.
    d, _ = cKDTree(sites[idx]).query(hot, k=1, workers=-1)
    counts, _ = np.histogram(d, bins=B_EDGES)
    return {
        "n_hot_voxels": int(len(hot)),
        "n_target_atoms": len(idx),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
        "max": float(d.max()),
        "n_over_30a": int((d > B_EDGES[-1]).sum()),
        "hist": counts.astype(int).tolist(),
    }


def _random_rotation(rng) -> np.ndarray:
    """Uniform random rotation, via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def figure_c(fields_heldout, shared_clash, clash_sites, clash_heavy, seed) -> dict:
    """Is a flagged region worth visiting for reasons beyond the one that flagged it?

    Build the field from Ramachandran + rotamer only, then ask what fraction of the atoms in
    the resulting hot region carry a *clash* outlier, against the base rate.

    **The null is the whole point.** Both signals live on atoms in a protein-shaped region,
    so some enrichment comes free from co-localization rather than from any relationship. A
    label shuffle would not capture that. Instead the hot region is taken as a rigid shape
    and re-placed at random rotations and positions inside the same molecular envelope,
    preserving its volume and shape, and enrichment is measured against *that*.

    Region membership is defined identically for the observed and null cases -- an atom is
    in the region if it lies within one voxel of a hot voxel -- so the two are exactly
    comparable. The sampled-value count is recorded alongside for reference only.
    """
    from scipy.spatial import cKDTree

    field = fields_heldout.get("combined")
    if field is None or not len(clash_sites):
        return None
    hot = hot_voxel_xyz(field)
    if not len(hot):
        return {"n_hot_voxels": 0, "enrichment": None, "reason": "no hot voxels"}

    clash_flag = np.zeros(clash_heavy.size, dtype=bool)
    for i in ve.outlier_atoms(shared_clash, metric="clash"):
        clash_flag[i] = True
    clash_flag &= clash_heavy
    heavy_idx = np.flatnonzero(clash_heavy)
    if not heavy_idx.size:
        return None
    base = float(clash_flag[heavy_idx].mean())
    sites = clash_sites[heavy_idx]
    flags = clash_flag[heavy_idx]

    rng = np.random.default_rng(seed)
    member_r = float(field.spacing)

    # The hot region is NOT one compact blob: it is one blob per concerning residue,
    # scattered across the whole molecule (measured on 6cg7: the hot set spans 49x57x82 A
    # inside a 72x65x115 A protein). Rigidly rotating a molecule-spanning scatter can never
    # land back inside the envelope -- every placement fails and the null is empty. So the
    # region is decomposed into connected components and each is re-placed independently.
    # That preserves total volume and each blob's shape, and randomizes only *where* the
    # blobs sit, which is precisely the co-localization the null has to control for.
    from scipy.ndimage import label

    labels, n_comp = label(field.data >= _hot())
    origin = np.asarray(field.origin, dtype=float)
    # Group the voxels by label in ONE pass. The obvious loop -- argwhere(labels == k) for
    # each k -- rescans the whole grid per component, i.e. O(n_components x n_voxels); on a
    # structure with hundreds of blobs on a 12M-voxel grid that is billions of comparisons
    # and it burned over two CPU-hours on a single model before this was caught.
    vox = np.argwhere(labels > 0)
    comps = []
    if vox.size:
        lab = labels[vox[:, 0], vox[:, 1], vox[:, 2]]
        order = np.argsort(lab, kind="stable")
        vox, lab = vox[order], lab[order]
        starts = np.searchsorted(lab, np.arange(1, n_comp + 1), side="left")
        ends = np.searchsorted(lab, np.arange(1, n_comp + 1), side="right")
        comps = [vox[s:e].astype(float) * field.spacing + origin
                 for s, e in zip(starts, ends) if e > s]
    if not comps:
        return {"n_hot_voxels": int(len(hot)), "enrichment": None, "reason": "no components"}
    total = sum(len(c) for c in comps)
    if total > NULL_HOT_CAP:                       # subsample each blob proportionally
        keep = NULL_HOT_CAP / total
        comps = [c[rng.choice(len(c), max(1, int(len(c) * keep)), replace=False)]
                 for c in comps]

    def region_rate(points):
        """Clash-outlier rate among atoms within one voxel of the given point cloud."""
        tree = cKDTree(points)
        near = tree.query_ball_point(sites, r=member_r, return_length=True) > 0
        n = int(near.sum())
        return (float(flags[near].mean()) if n else None), n

    pts = np.vstack(comps)
    observed_rate, n_in = region_rate(pts)
    sampled = sample_field(field, sites)
    env = cKDTree(sites)
    centred = [c - c.mean(axis=0) for c in comps]

    trials = max(NULL_MIN_TRIALS,
                 min(NULL_TRIALS, _null_budget() // max(1, len(comps))))
    null_rates, inside_fracs = [], []
    for _ in range(trials):
        placed = []
        for shape in centred:
            best, best_frac = None, -1.0
            for _try in range(NULL_PLACEMENT_TRIES):
                moved = shape @ _random_rotation(rng).T + sites[rng.integers(len(sites))]
                frac = float((env.query_ball_point(
                    moved, r=NULL_ENV_R, return_length=True) > 0).mean())
                if frac > best_frac:
                    best, best_frac = moved, frac
                if frac >= NULL_MIN_INSIDE:
                    break
            placed.append(best)
            inside_fracs.append(best_frac)
        rate, _n = region_rate(np.vstack(placed))
        if rate is not None:
            null_rates.append(rate)

    out = {
        "n_hot_voxels": int(len(hot)),
        "n_components": len(comps),
        "n_points_used": int(len(pts)),
        "null_inside_fraction_mean": (float(np.mean(inside_fracs))
                                      if inside_fracs else None),
        "n_atoms_in_region": n_in,
        "n_atoms_hot_by_sample": int((sampled >= _hot()).sum()),
        "n_heavy": int(heavy_idx.size),
        "base_rate": base,
        "region_rate": observed_rate,
        "enrichment": (observed_rate / base) if (base > 0 and observed_rate is not None)
                      else None,
        "n_null": len(null_rates),
        "null_trials_requested": trials,
    }
    if null_rates and base > 0:
        nulls = np.array(null_rates) / base
        out["null_enrichment_mean"] = float(nulls.mean())
        out["null_enrichment_sd"] = float(nulls.std())
        out["null_enrichment_p95"] = float(np.percentile(nulls, 95))
        if out["enrichment"] is not None:
            # +1 in both places: the observed value is one of the possible arrangements.
            out["p_value"] = float((int((nulls >= out["enrichment"]).sum()) + 1)
                                   / (len(nulls) + 1))
    return out


def run_one(pdb_id, *, spacing=SPACING, sigma=SIGMA, seed=0, want_dump=True,
            metrics=("rama", "rota", "clash"), combine="max", norm_p=1.0,
            field="concern", radius=6.0, want_c=True):
    """Returns ``(record, observation_lines)``.

    The lines are *returned* rather than streamed so the caller can append them only after
    the model is recorded ok. A shard that is OOM-killed mid-model then leaves no partial
    observations behind, so ``--resume`` re-running that model cannot double-count it.
    """
    path = model_path(pdb_id)
    started = time.time()
    obs: list = []
    rec = {"id": pdb_id, "model": path}
    _arm_timeout(MODEL_TIMEOUT_S)
    try:
        waited, avail = wait_for_memory()
        if waited:
            rec["mem_wait_s"], rec["mem_available_gb"] = waited, round(avail, 1)
        model = load_model(path)
        hierarchy = model.get_hierarchy()
        n_atoms = hierarchy.atoms_size()
        rec["n_atoms"] = int(n_atoms)
        if n_atoms > MAX_ATOMS:
            rec.update(status="skipped", reason=f"n_atoms {n_atoms} > {MAX_ATOMS}")
            rec["seconds"] = round(time.time() - started, 1)
            return rec, obs

        # Now that the size is known, demand headroom proportional to it. Defer rather than
        # proceed if the machine cannot supply it: proceeding is what kills the shard, and a
        # dead shard costs far more than a deferred model.
        need = max(MIN_AVAILABLE_GB, n_atoms * GB_PER_ATOM)
        if need > MIN_AVAILABLE_GB:
            waited, avail = wait_for_memory(min_gb=need, timeout_s=BIG_MODEL_WAIT_S)
            rec["mem_wait_s"], rec["mem_available_gb"] = waited, round(avail, 1)
            if avail < need:
                rec.update(status="deferred",
                           reason=f"needs ~{need:.1f} GB, {avail:.1f} GB available")
                rec["seconds"] = round(time.time() - started, 1)
                return rec, obs

        extras = {}
        extracted = extract_all(model, use_hydrogens=True,
                                metrics=tuple(metrics), extras_out=extras)
        shared = extras["shared_events"]
        shared_clash = extras.get("shared_clash") or []
        clash_model = extras.get("clash_model", model)
        rec["hydrogens"] = extracted["manifest"]["hydrogens"]
        rec["clashscore"] = extracted["manifest"]["clashscore"]

        concern_events = molprobity_concern_events(extracted["events"])
        by_metric = {}
        for m in metrics:
            picked = [e for e in concern_events if e.metric == m]
            if picked:
                by_metric[m] = picked
        if not by_metric:
            rec.update(status="empty", reason="no concern events")
            rec["seconds"] = round(time.time() - started, 1)
            return rec, obs

        # Same grid compute_field will derive, computed before anything is allocated.
        pts = np.vstack([np.asarray(e.atoms_xyz, dtype=float).reshape(-1, 3)
                         for evs in by_metric.values() for e in evs if e.atoms_xyz])
        pad = 3.0 * sigma
        lo = np.floor((pts.min(axis=0) - pad) / spacing) * spacing
        shape = np.ceil((pts.max(axis=0) + pad - lo) / spacing).astype(int) + 1
        n_vox = int(np.prod(shape.astype(np.int64)))
        rec["n_voxels"] = n_vox
        if n_vox > MAX_VOXELS:
            rec.update(status="skipped",
                       reason=f"grid {list(shape)} = {n_vox} voxels > {MAX_VOXELS}")
            rec["seconds"] = round(time.time() - started, 1)
            return rec, obs

        # Both field constructions go through identical A/B/C code below, so the comparison
        # is of the fields and not of two analyses. The threshold is 0.5 for both: in each, a
        # flagged outlier peaks at 1.0, so "half the community cut" means the same thing. It
        # also avoids the failure the clash calibration just taught us -- a threshold sitting
        # exactly on an event's peak makes recall a coin flip.
        fields = (build_concern_fields(by_metric, spacing=spacing, sigma=sigma,
                                       combine=combine, p=norm_p) if field == "concern"
                  else build_density_fields(by_metric, spacing=spacing, radius=radius))
        rec["field"] = field
        rec["grid"] = list(fields["combined"].data.shape)

        sites = np.asarray(hierarchy.atoms().extract_xyz()).reshape(-1, 3)
        heavy = heavy_mask(hierarchy)
        ch = clash_model.get_hierarchy()
        clash_sites = np.asarray(ch.atoms().extract_xyz()).reshape(-1, 3)
        clash_heavy = heavy_mask(ch)

        rec["A"], rec["B"] = {}, {}
        # The density construction keys its outputs by family, so a per-channel figure needs
        # its own single-channel field rather than the family one it happens to sit in.
        def _channel_field(metric):
            if field == "concern":
                return fields.get(metric)
            single = build_density_fields({metric: by_metric[metric]}, spacing=spacing,
                                          radius=radius, grid_events=all_ev)
            return single.get("combined")
        all_ev = [e for evs in by_metric.values() for e in evs]
        for m in metrics:
            if m not in by_metric:
                continue
            fld = _channel_field(m)
            if fld is None:
                continue
            if m == "clash":
                sv = sample_field(fld, clash_sites)
                rec["A"][m] = figure_a(shared_clash, sv, clash_heavy, m)
                rec["B"][m] = figure_b(fld, shared_clash, clash_sites, clash_heavy, m)
            else:
                sv = sample_field(fld, sites)
                rec["A"][m] = figure_a(shared, sv, heavy, m)
                rec["B"][m] = figure_b(fld, shared, sites, heavy, m)

        # Figure C holds the clash channel out and asks what the geometry-only field finds
        # -- a *predictive* claim. It costs a second field build per structure and the
        # project no longer makes that claim: held-out enrichment was 1.92x against a 0.98x
        # null, and the 6.1x quoted in FIGURES.md came from the uncalibrated heavy-atom path.
        # Kept behind a switch rather than deleted, so the number can be regenerated.
        if want_c:
            geom = {m: v for m, v in by_metric.items() if m in ("rama", "rota")}
            rec["C"] = (figure_c((build_concern_fields(geom, spacing=spacing, sigma=sigma,
                                                      combine=combine, p=norm_p)
                                  if field == "concern" else
                                  build_density_fields(geom, spacing=spacing, radius=radius)),
                                 shared_clash, clash_sites, clash_heavy, seed)
                        if geom and shared_clash else None)
        else:
            rec["C"] = None

        if want_dump:
            obs = _observation_lines(pdb_id, shared, shared_clash, fields,
                                     sites, heavy, clash_sites, clash_heavy)
        rec["status"] = "ok"
    except _Timeout:
        rec.update(status="timeout", reason=f"exceeded {MODEL_TIMEOUT_S}s")
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-1500:])
    finally:
        _arm_timeout(0)
    rec["seconds"] = round(time.time() - started, 1)
    return rec, obs


def _observation_lines(pdb_id, shared, shared_clash, fields,
                       sites, heavy, clash_sites, clash_heavy):
    """One line per validation observation — the dump every figure is remade from.

    Concern is not stored: it is a pure function of the native value under the calibration in
    ``hotspots/concern.py``, so it is recomputable offline and storing it would only create a
    second copy that can disagree with the first.
    """
    lines: list = []
    for evs, xyz, mask, tag in ((shared, sites, heavy, "geom"),
                                (shared_clash, clash_sites, clash_heavy, "clash")):
        for e in evs:
            if e.metric not in fields:
                continue
            idx = [i for i in e.atom_indices if mask[i]]
            if not idx:
                continue
            s = sample_field(fields[e.metric], xyz[idx])
            lines.append(json.dumps({
                "id": pdb_id, "metric": e.metric, "chain": e.residue.chain,
                "resseq": e.residue.resseq, "icode": e.residue.icode,
                "value": float(e.value), "units": e.units, "outlier": bool(e.outlier),
                "n_heavy": len(idx), "sampled_max": float(s.max()),
                "sampled_min": float(s.min()),
            }) + "\n")
    return lines


def _paths(out_dir, shard):
    """Results are appended across restarts; observations are NOT.

    A gzip stream that is appended to across process restarts is a trap: SIGKILL (OOM, or a
    deliberate stop) loses the compressor's buffered tail, leaving a truncated member, and
    the next process appends a fresh member after it. The result decompresses up to the
    break and then raises ``Error -3 while decompressing data`` -- every byte after the first
    kill is unreadable. That cost the observations for 366 structures here. Writing per-model
    was not enough: the loss is in zlib's buffer, not in line boundaries.

    So each *process* owns its own observations file, named with its pid and start time, and
    never appends to a stream another process opened. A killed process can still truncate its
    own tail -- losing at most the last model -- but it can no longer corrupt anyone else's,
    and every other file stays independently readable. Reduction globs them all.
    """
    tag = "" if shard is None else f".shard{shard[0]}of{shard[1]}"
    stamp = f"{os.getpid()}_{int(time.time())}"
    return (os.path.join(out_dir, f"results{tag}.jsonl"),
            os.path.join(out_dir, f"observations{tag}.{stamp}.jsonl.gz"))


def report(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "results*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    if not recs:
        print("no results in", out_dir)
        return recs
    ok = [r for r in recs if r["status"] == "ok"]
    print(f"{len(recs)} models: {len(ok)} ok, "
          f"{sum(1 for r in recs if r['status'] == 'failed')} failed, "
          f"{sum(1 for r in recs if r['status'] == 'skipped')} skipped")
    if ok:
        secs = [r["seconds"] for r in ok]
        print(f"  seconds/model: median {np.median(secs):.1f}  mean {np.mean(secs):.1f}  "
              f"max {max(secs):.1f}  total {sum(r['seconds'] for r in recs)/3600:.2f} h")
    for m in ("rama", "rota", "clash"):
        rc = [r["A"][m]["recall"] for r in ok
              if r.get("A", {}).get(m, {}).get("recall") is not None]
        if rc:
            rc = np.array(rc)
            print(f"  {m:5s} recall: min {rc.min():.3f}  median {np.median(rc):.3f}  "
                  f"mean {rc.mean():.4f}  <1.0 in {(rc < 1.0).sum()}/{len(rc)}")
    en = [r["C"]["enrichment"] for r in ok
          if r.get("C") and r["C"].get("enrichment") is not None]
    if en:
        print(f"  figure C enrichment: median {np.median(en):.2f}x over {len(en)} models")
    bad = [r for r in recs if r["status"] == "failed"]
    if bad:
        print("\n  failures:")
        for r in bad[:15]:
            print(f"    {r['id']:6s} {r['error'][:110]}")
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sample", help="file listing PDB ids, one per line")
    ap.add_argument("out_dir")
    ap.add_argument("--output-pixel-size", dest="spacing", type=float, default=SPACING)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--shard", metavar="K/N")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-dump", action="store_true",
                    help="skip the per-observation dump (summaries only)")
    ap.add_argument("--null-budget", type=int, metavar="N",
                    help="override the figure C null placement budget; a large value "
                         "forces the flat NULL_TRIALS policy")
    ap.add_argument("--metrics", default="rama,rota,clash",
                    help="comma-separated channels, or 'all'")
    ap.add_argument("--combine", choices=("max", "family"), default="max",
                    help="cross-metric combination (see concern.combine_arrays)")
    ap.add_argument("--norm-p", type=float, default=1.0)
    ap.add_argument("--hot-threshold", type=float, default=None,
                    help="operating point; defaults to 0.5 (half the community cut)")
    ap.add_argument("--field", choices=("concern", "density"), default="concern",
                    help="which field construction to measure")
    ap.add_argument("--radius", type=float, default=6.0,
                    help="neighbourhood radius for --field density")
    ap.add_argument("--no-figure-c", dest="want_c", action="store_false",
                    help="skip the held-out-clash prediction figure (a second field build "
                         "per structure); the project no longer makes that claim")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.null_budget:
        global _NULL_BUDGET_OVERRIDE
        _NULL_BUDGET_OVERRIDE = int(args.null_budget)
    if args.hot_threshold is not None:
        global _HOT_OVERRIDE
        _HOT_OVERRIDE = float(args.hot_threshold)

    from events import ALL_METRICS
    metrics = (tuple(ALL_METRICS) if args.metrics.strip() == "all"
               else tuple(m.strip() for m in args.metrics.split(",") if m.strip()))
    unknown = [m for m in metrics if m not in ALL_METRICS]
    if unknown:
        ap.error("unknown metric(s) %s; known: %s" % (unknown, list(ALL_METRICS)))

    if args.report:
        report(args.out_dir)
        return

    shard = None
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not (0 <= k < n):
            ap.error("--shard K/N needs 0 <= K < N")
        shard = (k, n)

    ids = read_ids(args.sample)
    if shard:
        ids = [x for i, x in enumerate(ids) if i % shard[1] == shard[0]]
    if args.limit:
        ids = ids[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    results_path, obs_path = _paths(args.out_dir, shard)

    # Resume against *every* results file in the directory, not just this shard's. Keying on
    # one shard's file silently breaks the moment the shard count changes -- a re-shard makes
    # each worker see an empty done-set, re-run everything, and duplicate it into the merged
    # results. Reading them all makes the shard count a free parameter between restarts.
    done, deferred = set(), set()
    if args.resume:
        for p in glob.glob(os.path.join(args.out_dir, "results*.jsonl")):
            with open(p) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    # A deferral means "the machine was busy", not "this model is done".
                    # Leaving it in the done-set would turn a transient condition into a
                    # permanent hole in the corpus.
                    (deferred if r.get("status") == "deferred" else done).add(r["id"])
    done -= deferred
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} models{f' (shard {shard[0]}/{shard[1]})' if shard else ''}"
          f"{f', {len(done)} done' if done else ''} -> {results_path}", flush=True)

    dump = None if args.no_dump else gzip.open(obs_path, "at")
    try:
        for n, pdb_id in enumerate(todo, 1):
            # Deterministic per-model seed: builtin hash() is salted per process, so the
            # null would be unreproducible across runs and differ between shards.
            seed = int(hashlib.sha256(pdb_id.encode()).hexdigest()[:8], 16)
            rec, obs = run_one(pdb_id, spacing=args.spacing, sigma=args.sigma,
                               seed=seed, want_dump=dump is not None,
                               metrics=metrics, combine=args.combine,
                               norm_p=args.norm_p, field=args.field, radius=args.radius,
                               want_c=args.want_c)
            # Results first, then observations: both are appended only once the model is
            # complete, so the two files always agree about which models are in the run.
            with open(results_path, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                fh.flush()
            if dump is not None and obs:
                dump.write("".join(obs))
                dump.flush()
            note = rec.get("error", "") or rec.get("reason", "")
            if rec["status"] == "ok":
                note = " ".join(
                    f"{m}:{rec['A'][m]['recall']:.2f}" for m in ("rama", "rota", "clash")
                    if rec["A"].get(m, {}).get("recall") is not None)
            print(f"[{n}/{len(todo)}] {pdb_id:6s} {rec['status']:8s} "
                  f"{rec['seconds']:6.1f}s  {note[:80]}", flush=True)
    finally:
        if dump is not None:
            dump.close()
    print()
    report(args.out_dir)


if __name__ == "__main__":
    main()
