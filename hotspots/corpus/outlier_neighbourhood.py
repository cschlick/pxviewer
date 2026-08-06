"""Are outliers islands? Profile the quality of a flagged residue's surroundings.

The hypothesis this project should have been testing all along: **an outlier is not an island.
The residues around it carry more imperfect geometry than the noise floor** — and a field's job
is to *show* that continuous structure, exactly as a Ramachandran plot shows contours instead
of a boolean. Concern 0.5 is then a literal statement, "half a Ramachandran outlier", and it
means the same thing in every channel because every community cut is anchored at 1.0.

Note what this is *not*. Every earlier test here asked whether the field **predicts** something
it was not told — held-out clash, cross-metric coincidence — and all were negative. This is a
**within-metric** question about **extent**: near a flagged rama outlier, are the neighbours
worse on *rama*? That is what decides whether a field showing gradient is showing anything.

Measured two ways, per metric:

* **sequence offset** ±1…±8 along the chain;
* **3D distance** from the outlier's CA, in bins.

against two baselines, because one is not enough:

* the structure's own mean concern for that metric (the noise floor);
* the same profile around **randomly chosen non-outlier residues**, which controls for the
  fact that any residue's neighbourhood is not a random sample of the structure.

**Outlier neighbours are excluded from the profile.** Otherwise the curve measures the known
clustering of outliers with each other (Clark-Evans R = 0.825) rather than the hypothesis,
which is about sub-outlier elevation in the surroundings.

**Confound stated up front:** phi/psi of residue *i* involve atoms of *i±1*, so Ramachandran
scores of immediate sequence neighbours are structurally coupled whatever the quality.
Elevation at ±1 proves little. Elevation at ±3 or beyond is the real signal.

    libtbx.python corpus/outlier_neighbourhood.py IDS.txt OUT_DIR --shard 0/4
    libtbx.python corpus/outlier_neighbourhood.py IDS.txt OUT_DIR --report
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hotspots"))
sys.path.insert(0, HERE)

from concern import FAMILIES, molprobity_concern_events  # noqa: E402
from events import _ADAPTERS, _load_shared, add_hydrogens, load_model  # noqa: E402
from figure_data import MAX_ATOMS, model_path  # noqa: E402

ve = _load_shared()

#: Residue-level channels only, and the ones whose adapters are 1:1 with the shared events so
#: a concern can be paired back to its residue. bond/angle are per-restraint and get rolled up
#: by residue inside the concern layer, which breaks the pairing.
RESIDUE_METRICS = ("rama", "rota", "cablam", "ca_geom", "cbeta", "omega")

#: Clash is per-*contact*, so it is not in the list above; :func:`_clash_by_residue` rolls it
#: up explicitly. It is opt-in (``--clash``) because it is the only channel that needs
#: hydrogens, and reduce2 + probe2 cost roughly ten times the other six put together.
METRICS = RESIDUE_METRICS
FLAGGED = 1.0
OFFSETS = list(range(-8, 9))
DIST_EDGES = np.arange(0.0, 20.001, 2.0)
CONTROL_TRIALS = 20
MIN_OUTLIERS = 3

#: Sequence separation a residue must exceed to count as a *through-space* neighbour. The
#: plain distance profile cannot separate the two effects: a residue 4 A from an outlier is
#: usually the one next to it in sequence, so the spatial curve could be the sequence curve
#: wearing a costume. Requiring >5 residues of chain separation (or a different chain
#: entirely) leaves only neighbours that are close in space for a reason other than being
#: adjacent in the chain.
MIN_SEQ_SEP = 6

#: Which family each channel belongs to, so "a different kind of problem" is well defined.
FAMILY_OF = {m: f for f, ms in FAMILIES.items() for m in ms}


def _residue_ca(hierarchy):
    """(chain, resseq, icode) -> CA xyz, falling back to the residue centroid."""
    out = {}
    for ch in hierarchy.only_model().chains():
        for rg in ch.residue_groups():
            key = (ch.id.strip(), rg.resseq_as_int(), rg.icode.strip())
            xyz = None
            for at in rg.atoms():
                if at.name.strip() == "CA":
                    xyz = tuple(float(c) for c in at.xyz)
                    break
            if xyz is None:
                pts = np.asarray(rg.atoms().extract_xyz()).reshape(-1, 3)
                if not pts.size:
                    continue
                xyz = tuple(pts.mean(axis=0))
            out[key] = xyz
    return out


def _dist_profile_seqfar(conc, ca, centres, exclude, min_sep=MIN_SEQ_SEP):
    """Distance profile counting only neighbours far away in sequence (or on another chain).

    Each residue is binned by its distance to the nearest *sequence-far* outlier, so it is
    counted once and the statistic stays comparable to the plain distance profile.
    """
    keys = [k for k in ca if k not in exclude]
    centres = [c for c in centres if c in ca]
    if not keys or not centres:
        return {i: (None, 0) for i in range(len(DIST_EDGES) - 1)}
    chain_id = {}
    for k in keys:
        chain_id.setdefault(k[0], len(chain_id))
    for c in centres:
        chain_id.setdefault(c[0], len(chain_id))
    axyz = np.array([ca[k] for k in keys])
    kchain = np.array([chain_id[k[0]] for k in keys])
    kres = np.array([k[1] for k in keys], dtype=float)
    best = np.full(len(keys), np.inf)
    for c in centres:
        d = np.linalg.norm(axyz - np.asarray(ca[c], float), axis=1)
        near_in_seq = (kchain == chain_id[c[0]]) & (np.abs(kres - c[1]) < min_sep)
        best = np.minimum(best, np.where(near_in_seq, np.inf, d))
    out = {i: [] for i in range(len(DIST_EDGES) - 1)}
    finite = np.isfinite(best)
    b = np.digitize(best[finite], DIST_EDGES) - 1
    kf = [k for k, ok in zip(keys, finite) if ok]
    for i, k in enumerate(kf):
        if 0 <= b[i] < len(DIST_EDGES) - 1:
            out[b[i]].append(conc.get(k, 0.0))
    return {i: (float(np.mean(v)) if v else None, len(v)) for i, v in out.items()}


def _cross_profile(by_metric, ca, centres, exclude, min_sep=None):
    """Mean concern of *every* metric, binned by distance to the nearest centre.

    One binning serves all metrics, so the whole metric-by-metric matrix costs about what the
    single-metric profile cost: six binnings rather than thirty-six.
    """
    keys = [k for k in ca if k not in exclude]
    centres = [c for c in centres if c in ca]
    nb = len(DIST_EDGES) - 1
    if not keys or not centres:
        return {m: [None] * nb for m in by_metric}
    axyz = np.array([ca[k] for k in keys])
    if min_sep is None:
        from scipy.spatial import cKDTree
        d = cKDTree(np.array([ca[c] for c in centres])).query(axyz, k=1)[0]
    else:
        # Distance to the nearest centre that is NOT a chain neighbour.
        chain_id = {}
        for k in list(keys) + list(centres):
            chain_id.setdefault(k[0], len(chain_id))
        kchain = np.array([chain_id[k[0]] for k in keys])
        kres = np.array([k[1] for k in keys], dtype=float)
        d = np.full(len(keys), np.inf)
        for c in centres:
            dc = np.linalg.norm(axyz - np.asarray(ca[c], float), axis=1)
            near_seq = (kchain == chain_id[c[0]]) & (np.abs(kres - c[1]) < min_sep)
            d = np.minimum(d, np.where(near_seq, np.inf, dc))
    b = np.digitize(d, DIST_EDGES) - 1
    b = np.where(np.isfinite(d), b, -1)
    out = {}
    for m, conc in by_metric.items():
        vals = np.array([conc.get(k, 0.0) for k in keys])
        prof = []
        for i in range(nb):
            sel = b == i
            prof.append(float(vals[sel].mean()) if sel.any() else None)
        out[m] = prof
    return out


def _profiles(conc, keys_by_chain, ca, centres, exclude, offsets=OFFSETS):
    """Mean concern at each sequence offset and distance bin around ``centres``."""
    seq = {o: [] for o in offsets}
    for (chain, resseq, icode) in centres:
        for o in offsets:
            k = (chain, resseq + o, icode)
            if k in ca and k not in exclude:
                seq[o].append(conc.get(k, 0.0))
    dist = {i: [] for i in range(len(DIST_EDGES) - 1)}
    if centres:
        cxyz = np.array([ca[c] for c in centres if c in ca])
        if cxyz.size:
            allk = [k for k in ca if k not in exclude]
            axyz = np.array([ca[k] for k in allk])
            from scipy.spatial import cKDTree
            d = cKDTree(cxyz).query(axyz, k=1)[0]
            b = np.digitize(d, DIST_EDGES) - 1
            for i, k in enumerate(allk):
                if 0 <= b[i] < len(DIST_EDGES) - 1:
                    dist[b[i]].append(conc.get(k, 0.0))
    return ({o: (float(np.mean(v)) if v else None, len(v)) for o, v in seq.items()},
            {i: (float(np.mean(v)) if v else None, len(v)) for i, v in dist.items()})


def _clash_by_residue(model, ca):
    """Worst clash concern per residue, over both sides of every contact.

    Returns ``(concern_by_residue, n_contacts)``. Hydrogens are placed first: MolProbity's
    clash channel is *defined* on a hydrogenated model, and the calibration anchors
    (0.30 A zero, 0.40 A cut) were set against that path.

    ``e.residue`` is deliberately not used. ``extract_clashes`` sets it to the residue of the
    lowest-indexed atom in the contact, which is one arbitrary side of a two-sided
    relationship; ``atom_indices`` carries both sides plus hydrogen parents.
    """
    hmodel = add_hydrogens(model)
    dots = ve.probe2_dots(hmodel)
    shared = ve.extract_clashes(hmodel, dots=dots)
    if not shared:
        return {}, 0
    concern = molprobity_concern_events([_ADAPTERS["clash"](e) for e in shared])
    hatoms = list(hmodel.get_hierarchy().atoms_with_labels())
    res_of = [(a.chain_id.strip(), int(a.resseq_as_int()), a.icode.strip()) for a in hatoms]
    out = {}
    for e, c in zip(shared, concern):
        v = float(c.severity)
        if v <= 0.0:
            continue
        for i in e.atom_indices:
            if not (0 <= i < len(res_of)):
                continue
            k = res_of[i]
            if k in ca and v > out.get(k, 0.0):
                out[k] = v
    return out, len(shared)


def run_one(pdb_id, seed=0, with_clash=False):
    started = time.time()
    rec = {"id": pdb_id}
    try:
        model = load_model(model_path(pdb_id))
        hierarchy = model.get_hierarchy()
        if hierarchy.atoms_size() > MAX_ATOMS:
            rec.update(status="skipped", reason="too many atoms")
            return _done(rec, started)

        shared = ve.extract_all(model, metrics=RESIDUE_METRICS)
        adapted = [_ADAPTERS[s.metric](s) for s in shared]
        calibrated = molprobity_concern_events(adapted)
        if len(calibrated) != len(shared):
            rec.update(status="failed", error="calibration changed event count")
            return _done(rec, started)

        ca = _residue_ca(hierarchy)
        by_metric = defaultdict(dict)
        for s, c in zip(shared, calibrated):
            if s.residue is None:
                continue
            k = (s.residue.chain, s.residue.resseq, s.residue.icode)
            if k not in ca:
                continue
            cur = by_metric[s.metric].get(k, 0.0)
            by_metric[s.metric][k] = max(cur, float(c.severity))

        if with_clash:
            # Failure is recorded, not absorbed: a heavy-atom fallback would answer a
            # different question under the same name.
            clash_conc, n_contacts = _clash_by_residue(model, ca)
            by_metric["clash"] = clash_conc
            rec["n_contacts"] = n_contacts

        keys_by_chain = defaultdict(list)
        for (chain, rs, ic) in ca:
            keys_by_chain[chain].append(rs)

        rng = np.random.default_rng(seed)
        out = {}
        for metric in (RESIDUE_METRICS + ("clash",) if with_clash else RESIDUE_METRICS):
            conc = by_metric.get(metric)
            if not conc:
                continue
            outliers = [k for k, v in conc.items() if v >= FLAGGED]
            if len(outliers) < MIN_OUTLIERS:
                continue
            excl = set(outliers)          # profile the SURROUNDINGS, not other outliers
            base = float(np.mean([conc.get(k, 0.0) for k in ca if k not in excl]))
            seq, dist = _profiles(conc, keys_by_chain, ca, outliers, excl)
            far = _dist_profile_seqfar(conc, ca, outliers, excl)
            # The same question asked of EVERY channel on the same axis: near an outlier of
            # this metric, how elevated is each other metric? The diagonal of the resulting
            # matrix is "clusters with its own kind"; the off-diagonal is "kinds co-locate".
            cross = _cross_profile(by_metric, ca, outliers, excl,
                                   min_sep=MIN_SEQ_SEP)
            # The same question asked about a DIFFERENT kind of problem: at each residue, the
            # worst concern from any channel outside this metric's family. Put on the same
            # axis and in the same units as the curve above, so "clusters with its own kind
            # but not across kinds" is one comparison rather than two methods.
            other = {}
            for k in ca:
                best = 0.0
                for m2, c2 in by_metric.items():
                    if FAMILY_OF.get(m2) == FAMILY_OF.get(metric):
                        continue
                    v2 = c2.get(k, 0.0)
                    if v2 > best:
                        best = v2
                other[k] = best
            _oseq, odist = _profiles(other, keys_by_chain, ca, outliers, excl)

            # Control: the same profile around random non-outlier residues.
            pool = [k for k in ca if k not in excl]
            cseq = defaultdict(list)
            cdist = defaultdict(list)
            cfar = defaultdict(list)
            ccross = defaultdict(list)
            codist = defaultdict(list)
            for _ in range(CONTROL_TRIALS):
                pick = [pool[i] for i in rng.integers(0, len(pool), size=len(outliers))]
                s2, d2 = _profiles(conc, keys_by_chain, ca, pick, excl)
                f2 = _dist_profile_seqfar(conc, ca, pick, excl)
                x2 = _cross_profile(by_metric, ca, pick, excl,
                                    min_sep=MIN_SEQ_SEP)
                _o2s, o2 = _profiles(other, keys_by_chain, ca, pick, excl)
                for o, (m, _n) in s2.items():
                    if m is not None:
                        cseq[o].append(m)
                for i, (m, _n) in d2.items():
                    if m is not None:
                        cdist[i].append(m)
                for i, (m, _n) in f2.items():
                    if m is not None:
                        cfar[i].append(m)
                for m2, prof in x2.items():
                    for i, val in enumerate(prof):
                        if val is not None:
                            ccross[(m2, i)].append(val)
                for i, (m, _n) in o2.items():
                    if m is not None:
                        codist[i].append(m)
            out[metric] = {
                "n_outliers": len(outliers),
                "baseline": base,
                "seq": {str(o): seq[o][0] for o in seq},
                "seq_n": {str(o): seq[o][1] for o in seq},
                "seq_control": {str(o): (float(np.mean(v)) if v else None)
                                for o, v in cseq.items()},
                "dist": {str(i): dist[i][0] for i in dist},
                "dist_control": {str(i): (float(np.mean(v)) if v else None)
                                 for i, v in cdist.items()},
                "far": {str(i): far[i][0] for i in far},
                "far_n": {str(i): far[i][1] for i in far},
                "far_control": {str(i): (float(np.mean(v)) if v else None)
                                for i, v in cfar.items()},
                "cross": dict(cross),
                "cross_control": {m2: [
                    (float(np.mean(ccross[(m2, i)])) if ccross[(m2, i)] else None)
                    for i in range(len(DIST_EDGES) - 1)] for m2 in cross},
                "other": {str(i): odist[i][0] for i in odist},
                "other_control": {str(i): (float(np.mean(v)) if v else None)
                                  for i, v in codist.items()},
            }
        if not out:
            rec.update(status="empty", reason="no metric with enough outliers")
            return _done(rec, started)
        rec.update(status="ok", metrics=out)
    except Exception as exc:
        rec.update(status="failed", error="%s: %s" % (type(exc).__name__, exc),
                   traceback=traceback.format_exc()[-1000:])
    return _done(rec, started)


def _done(rec, started):
    rec["seconds"] = round(time.time() - started, 1)
    return rec


#: Distance band the matrix is quoted at. DIST_EDGES starts at 0, so band 2 is 4-6 A -- the
#: bin where the through-space effect peaks once chain neighbours are excluded. Band 1 (2-4 A)
#: was used first and is unusable: it is where sequence neighbours live, and adjacent residues
#: share atoms across channels as well as within them, so it measures validator coupling.
MATRIX_BAND = 2
MATRIX_ROWS = []


def _matrix_row(rows, channels, band=MATRIX_BAND):
    """obs/ctl for every channel measured near this channel's outliers."""
    out = {}
    for m2 in channels:
        v = [x["cross"][m2][band] for x in rows
             if x.get("cross", {}).get(m2) and x["cross"][m2][band] is not None]
        c = [x["cross_control"][m2][band] for x in rows
             if x.get("cross_control", {}).get(m2) and x["cross_control"][m2][band] is not None]
        out[m2] = (float(np.mean(v) / np.mean(c))
                   if v and c and np.mean(c) > 0 else None)
    return out


def report(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "nbhd*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no results in", out_dir)
        return
    # Which channels are present is a property of the run (--clash or not), so read it
    # off the data rather than assuming the module constant matches what was measured.
    present = {m for r in ok for m in (r.get("metrics") or {})}
    channels = [m for m in RESIDUE_METRICS + ("clash",) if m in present]
    print("%d structures: %d ok\n" % (len(recs), len(ok)))
    print("Mean concern of NON-OUTLIER residues near a flagged outlier of the same metric,")
    print("as a ratio to the same profile around random non-outlier residues.")
    print("Ratio > 1 means the surroundings of an outlier are worse than the noise floor.\n")

    for metric in channels:
        rows = [r["metrics"][metric] for r in ok if r.get("metrics", {}).get(metric)]
        if len(rows) < 5:
            continue
        print("%s  (%d structures, %d outliers total)" % (
            metric, len(rows), sum(x["n_outliers"] for x in rows)))
        print("   offset :  %s" % "  ".join("%5d" % o for o in range(0, 6)))
        obs, ctl = [], []
        for o in range(0, 6):
            v = [x["seq"].get(str(o)) for x in rows if x["seq"].get(str(o)) is not None]
            c = [x["seq_control"].get(str(o)) for x in rows
                 if x["seq_control"].get(str(o)) is not None]
            obs.append(np.mean(v) if v else np.nan)
            ctl.append(np.mean(c) if c else np.nan)
        print("   obs/ctl:  %s" % "  ".join(
            "%5.2f" % (o / c) if c and c > 0 else "    -" for o, c in zip(obs, ctl)))
        d_obs, d_ctl = [], []
        for i in range(len(DIST_EDGES) - 1):
            v = [x["dist"].get(str(i)) for x in rows if x["dist"].get(str(i)) is not None]
            c = [x["dist_control"].get(str(i)) for x in rows
                 if x["dist_control"].get(str(i)) is not None]
            d_obs.append(np.mean(v) if v else np.nan)
            d_ctl.append(np.mean(c) if c else np.nan)
        print("   dist(A):  %s" % "  ".join(
            "%5.0f" % DIST_EDGES[i + 1] for i in range(6)))
        print("   obs/ctl:  %s" % "  ".join(
            "%5.2f" % (o / c) if c and c > 0 else "    -"
            for o, c in list(zip(d_obs, d_ctl))[:6]))
        # where does elevation fall to within 10% of the control?
        decay = None
        for i, (o, c) in enumerate(zip(d_obs, d_ctl)):
            if c and c > 0 and o / c < 1.10:
                decay = DIST_EDGES[i + 1]
                break
        f_obs, f_ctl, f_n = [], [], []
        for i in range(len(DIST_EDGES) - 1):
            v = [x["far"].get(str(i)) for x in rows if x.get("far", {}).get(str(i)) is not None]
            c = [x["far_control"].get(str(i)) for x in rows
                 if x.get("far_control", {}).get(str(i)) is not None]
            nn = [x["far_n"].get(str(i), 0) for x in rows if x.get("far_n")]
            f_obs.append(np.mean(v) if v else np.nan)
            f_ctl.append(np.mean(c) if c else np.nan)
            f_n.append(int(np.sum(nn)) if nn else 0)
        o_obs, o_ctl = [], []
        for i in range(len(DIST_EDGES) - 1):
            v = [x["other"].get(str(i)) for x in rows if x.get("other", {}).get(str(i)) is not None]
            c = [x["other_control"].get(str(i)) for x in rows
                 if x.get("other_control", {}).get(str(i)) is not None]
            o_obs.append(np.mean(v) if v else np.nan)
            o_ctl.append(np.mean(c) if c else np.nan)
        print("   A DIFFERENT KIND of problem, same distance axis")
        print("   obs/ctl:  %s" % "  ".join(
            "%5.2f" % (o / c) if c and c > 0 else "    -"
            for o, c in list(zip(o_obs, o_ctl))[:6]))
        print("   THROUGH SPACE ONLY (>5 residues apart in sequence, or another chain)")
        print("   dist(A):  %s" % "  ".join("%5.0f" % DIST_EDGES[i + 1] for i in range(6)))
        print("   obs/ctl:  %s" % "  ".join(
            "%5.2f" % (o / c) if c and c > 0 else "    -"
            for o, c in list(zip(f_obs, f_ctl))[:6]))
        print("   n resid:  %s" % "  ".join("%5d" % n for n in f_n[:6]))
        print("   elevation falls within 10%% of control by: %s\n" % (
            "%.0f A" % decay if decay else "> %.0f A" % DIST_EDGES[-1]))
        MATRIX_ROWS.append((metric, len(rows), _matrix_row(rows, channels)))

    if MATRIX_ROWS:
        print("\nNEIGHBOURHOOD MATRIX at %.0f-%.0f A (obs/ctl)" % (
            DIST_EDGES[MATRIX_BAND], DIST_EDGES[MATRIX_BAND + 1]))
        print("rows = the outlier's channel; columns = the channel measured nearby\n")
        print("%-9s %s" % ("near \u2193", "  ".join("%8s" % m[:8] for m in channels)))
        for metric, _n, row in MATRIX_ROWS:
            print("%-9s %s" % (metric, "  ".join(
                ("%8.2f" % row[m]) if row.get(m) is not None else "%8s" % "-"
                for m in channels)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids")
    ap.add_argument("out_dir")
    ap.add_argument("--shard", metavar="K/N")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--clash", action="store_true",
                    help="also measure the clash channel (needs reduce2 + probe2; ~10x slower)")
    args = ap.parse_args()

    if args.report:
        report(args.out_dir)
        return

    ids = [l.strip() for l in open(args.ids) if l.strip()]
    shard = None
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        shard = (k, n)
        ids = [x for i, x in enumerate(ids) if i % n == k]
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "" if shard is None else ".shard%dof%d" % shard
    path = os.path.join(args.out_dir, "nbhd%s.jsonl" % tag)
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [i for i in ids if i not in done]
    print("%d structures%s -> %s" % (
        len(ids), " (shard %d/%d)" % shard if shard else "", path), flush=True)

    for i, pid in enumerate(todo, 1):
        seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
        rec = run_one(pid, seed=seed, with_clash=args.clash)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        note = (",".join("%s:%d" % (m, v["n_outliers"])
                         for m, v in (rec.get("metrics") or {}).items())
                if rec["status"] == "ok" else rec.get("error", rec.get("reason", ""))[:45])
        print("[%d/%d] %-5s %-8s %6.1fs  %s" % (
            i, len(todo), pid, rec["status"], rec["seconds"], note[:60]), flush=True)


if __name__ == "__main__":
    main()
