"""Do validation problems cluster in space — and does that depend on severity?

Three side-observations in this project pointed the same way: cross-family residue-level
Jaccard is 0.000, sub-threshold regions are *depleted* for held-out clash (0.86x), and marked
regions are enriched (2.07x). Together they suggest **serious problems cluster and mild ones
do not**. That was assembled from three different measurements and one back-of-envelope
Poisson estimate; this measures it directly.

**The statistic** is the nearest-neighbour distance between events, and the comparison is the
Clark-Evans ratio ``R = observed mean NN / null mean NN``:

    R < 1   clustered      events sit closer together than chance
    R ~ 1   random
    R > 1   dispersed      events avoid each other

**The null re-places every event at a randomly chosen heavy atom of the same structure.** That
is the control that matters: events can only occur where atoms are, and atoms are far from
uniformly distributed — a buried core is dense, a surface loop is not. Comparing against a
uniform Poisson process in a box would report "clustering" that is nothing but the shape of
the protein. Count and severity composition are preserved; only position is randomized.

Split by severity, because that is the hypothesis:

* **flagged** (concern >= 1.0) — a validator called these outliers;
* **sub-threshold** (0 < concern < 1.0) — the events the accumulation idea depended on.

and by family pairing, because accumulation needs *different kinds* of problem to coincide:

* **any** — nearest event of the same severity class, whatever family;
* **cross-family** — nearest event of a *different* family, which is the arrangement a
  cross-metric field would need in order to have anything to add up.

    libtbx.python corpus/clustering.py IDS.txt OUT_DIR --shard 0/4
    libtbx.python corpus/clustering.py IDS.txt OUT_DIR --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hotspots"))
sys.path.insert(0, HERE)

from concern import FAMILIES, molprobity_concern_events  # noqa: E402
from events import ALL_METRICS, extract_all, load_model  # noqa: E402
from figure_data import MAX_ATOMS, heavy_mask, model_path  # noqa: E402

FAMILY_OF = {m: f for f, ms in FAMILIES.items() for m in ms}
FLAGGED = 1.0
NULL_TRIALS = 50
MIN_EVENTS = 20          # below this a nearest-neighbour distribution is not worth reporting


def _nn_mean(points, groups=None, cross_family=False):
    """Mean nearest-neighbour distance. With ``cross_family``, the neighbour must differ."""
    n = len(points)
    if n < 2:
        return None
    tree = cKDTree(points)
    if not cross_family:
        d, _ = tree.query(points, k=2)
        return float(d[:, 1].mean())
    k = min(n, 40)
    d, idx = tree.query(points, k=k)
    out = []
    for i in range(n):
        for j in range(1, k):
            if groups[idx[i, j]] != groups[i]:
                out.append(d[i, j])
                break
    return float(np.mean(out)) if out else None


def _clark_evans(points, groups, atoms, rng, cross_family=False):
    """Observed mean NN against the same events re-placed on random atoms."""
    obs = _nn_mean(points, groups, cross_family)
    if obs is None:
        return None
    nulls = []
    for _ in range(NULL_TRIALS):
        pick = atoms[rng.integers(0, len(atoms), size=len(points))]
        v = _nn_mean(pick, groups, cross_family)
        if v is not None:
            nulls.append(v)
    if not nulls:
        return None
    null = float(np.mean(nulls))
    return {"observed": obs, "null": null, "R": obs / null if null > 0 else None,
            "n": len(points), "n_null": len(nulls)}


def run_one(pdb_id, seed=0):
    started = time.time()
    rec = {"id": pdb_id}
    try:
        model = load_model(model_path(pdb_id))
        hierarchy = model.get_hierarchy()
        n_atoms = hierarchy.atoms_size()
        rec["n_atoms"] = int(n_atoms)
        if n_atoms > MAX_ATOMS:
            rec.update(status="skipped", reason="n_atoms %d > %d" % (n_atoms, MAX_ATOMS))
            return _done(rec, started)

        events = molprobity_concern_events(
            extract_all(model, use_hydrogens=True, metrics=ALL_METRICS)["events"])
        sites = np.asarray(hierarchy.atoms().extract_xyz()).reshape(-1, 3)
        atoms = sites[heavy_mask(hierarchy)]
        if len(atoms) < 50:
            rec.update(status="empty", reason="too few heavy atoms")
            return _done(rec, started)

        rows = []
        for e in events:
            if e.severity <= 0 or not e.atoms_xyz:
                continue
            xyz = np.asarray(e.atoms_xyz, float).reshape(-1, 3).mean(axis=0)
            rows.append((xyz, float(e.severity), FAMILY_OF.get(e.metric, e.metric)))
        if len(rows) < MIN_EVENTS:
            rec.update(status="empty", reason="only %d events" % len(rows))
            return _done(rec, started)

        rng = np.random.default_rng(seed)
        classes = {
            "flagged": [r for r in rows if r[1] >= FLAGGED],
            "sub_threshold": [r for r in rows if r[1] < FLAGGED],
            "all": rows,
        }
        out = {}
        for name, sel in classes.items():
            if len(sel) < MIN_EVENTS:
                continue
            pts = np.array([r[0] for r in sel])
            fam = [r[2] for r in sel]
            out[name] = {
                "any": _clark_evans(pts, fam, atoms, rng, cross_family=False),
                "cross_family": _clark_evans(pts, fam, atoms, rng, cross_family=True),
            }
        rec.update(status="ok", classes=out)
    except Exception as exc:
        rec.update(status="failed", error="%s: %s" % (type(exc).__name__, exc),
                   traceback=traceback.format_exc()[-1000:])
    return _done(rec, started)


def _done(rec, started):
    rec["seconds"] = round(time.time() - started, 1)
    return rec


def report(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "clust*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no results in", out_dir)
        return
    print("%d structures: %d ok, %d failed, %d other\n" % (
        len(recs), len(ok), sum(1 for r in recs if r["status"] == "failed"),
        sum(1 for r in recs if r["status"] not in ("ok", "failed"))))
    print("Clark-Evans R = observed mean NN / null mean NN, null = events re-placed on")
    print("random heavy atoms of the same structure.  R<1 clustered, R=1 random.\n")
    print("  %-15s %-14s %6s %10s %8s %9s %10s" % (
        "severity", "neighbour", "n", "observed", "null", "R", "clustered"))
    summary = {}
    for cls in ("flagged", "sub_threshold", "all"):
        for pair in ("any", "cross_family"):
            vals = [(r["classes"].get(cls) or {}).get(pair) for r in ok
                    if r.get("classes", {}).get(cls)]
            vals = [v for v in vals if v and v.get("R")]
            if not vals:
                continue
            R = np.array([v["R"] for v in vals])
            obs = np.array([v["observed"] for v in vals])
            nul = np.array([v["null"] for v in vals])
            summary[(cls, pair)] = float(np.median(R))
            frac = float((R < 1.0).mean())
            print("  %-15s %-14s %6d %9.2fA %7.2fA %9.3f %8.0f%%" % (
                cls, pair, len(vals), np.median(obs), np.median(nul),
                np.median(R), 100 * frac))

    f = summary.get(("flagged", "any"))
    s = summary.get(("sub_threshold", "any"))
    fx = summary.get(("flagged", "cross_family"))
    sx = summary.get(("sub_threshold", "cross_family"))
    print("\nVERDICT")
    if f is not None and s is not None:
        print("  flagged        R = %.3f  (%s)" % (
            f, "clustered" if f < 0.95 else "random" if f < 1.05 else "dispersed"))
        print("  sub-threshold  R = %.3f  (%s)" % (
            s, "clustered" if s < 0.95 else "random" if s < 1.05 else "dispersed"))
        if f < s - 0.02:
            print("  -> severity-dependent clustering: serious problems sit closer together")
            print("     than chance, mild ones %s." % (
                "less so" if s < 0.95 else "do not"))
        else:
            print("  -> no severity dependence in clustering.")
    if fx is not None and sx is not None:
        print("\n  cross-family (what a cross-metric field would need to accumulate):")
        print("    flagged       R = %.3f" % fx)
        print("    sub-threshold R = %.3f  %s" % (
            sx, "<- accumulation had nothing to work with" if sx >= 0.95 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids")
    ap.add_argument("out_dir")
    ap.add_argument("--shard", metavar="K/N")
    ap.add_argument("--report", action="store_true")
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
    path = os.path.join(args.out_dir, "clust%s.jsonl" % tag)
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [i for i in ids if i not in done]
    print("%d structures%s -> %s" % (
        len(ids), " (shard %d/%d)" % shard if shard else "", path), flush=True)

    import hashlib
    for i, pid in enumerate(todo, 1):
        seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
        rec = run_one(pid, seed=seed)
        note = ""
        if rec["status"] == "ok":
            fl = (rec["classes"].get("flagged") or {}).get("any")
            sb = (rec["classes"].get("sub_threshold") or {}).get("any")
            note = "R flagged %s  sub %s" % (
                "%.2f" % fl["R"] if fl and fl.get("R") else "-",
                "%.2f" % sb["R"] if sb and sb.get("R") else "-")
        else:
            note = rec.get("error", rec.get("reason", ""))[:50]
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        print("[%d/%d] %-5s %-8s %6.1fs  %s" % (
            i, len(todo), pid, rec["status"], rec["seconds"], note), flush=True)


if __name__ == "__main__":
    main()
