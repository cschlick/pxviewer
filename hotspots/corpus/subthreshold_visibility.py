"""Near an outlier, how many neighbours are *visibly* bad without being flagged?

The neighbourhood measurement says the mean concern around an outlier is elevated 1.35-2.51x.
That is a real ratio and a misleading basis for a claim about a picture: the absolute means
are 0.03-0.11 on a scale where 1.0 is the community cut, and a field value of 0.03 renders as
nothing. A mean of 0.037 is equally consistent with

  * every neighbour faintly warm  -> the field shows a uniform haze, nothing to look at;
  * 97% at zero and 3% at 0.6     -> a handful of clearly-drawn residues the outlier markup
                                     cannot show at all.

Those are opposite claims about what a hotspot field is *for*, and a mean cannot separate
them. So count residues by concern band instead of averaging them: near a flagged outlier,
what fraction of the NON-outlier residues carry at least a quarter, or at least half, of an
outlier's worth of concern -- against the same fraction around random non-outlier residues.

Bands, in units where 1.0 is the community cut:
  0.25 -- a quarter of an outlier. Below anything a markup shows; renders faintly.
  0.50 -- half an outlier. Clearly drawn in an absolute-domain field, invisible in a markup.

    libtbx.python corpus/subthreshold_visibility.py IDS.txt OUT_DIR --shard 0/6
    libtbx.python corpus/subthreshold_visibility.py IDS.txt OUT_DIR --report
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

from concern import molprobity_concern_events  # noqa: E402
from events import _ADAPTERS, _load_shared, load_model  # noqa: E402
from figure_data import MAX_ATOMS, model_path  # noqa: E402
from outlier_neighbourhood import (  # noqa: E402
    CONTROL_TRIALS, DIST_EDGES, FLAGGED, MIN_OUTLIERS, MIN_SEQ_SEP,
    RESIDUE_METRICS, _residue_ca,
)

ve = _load_shared()

BANDS = (0.25, 0.50)
#: Sequence window counted as "near", chosen to exclude the +-1 shared-atom coupling that
#: the neighbourhood measurement already showed proves nothing on its own.
NEAR_OFFSETS = (-3, -2, 2, 3)
#: Through-space band, the 4-6 A bin where the only surviving spatial effect lives.
FAR_LO, FAR_HI = 4.0, 6.0


def _frac_above(conc, keys, band):
    if not keys:
        return None, 0
    hits = sum(1 for k in keys if conc.get(k, 0.0) >= band)
    return hits / float(len(keys)), len(keys)


def _near_seq(ca, centres, exclude):
    out = set()
    for (chain, rs, ic) in centres:
        for o in NEAR_OFFSETS:
            k = (chain, rs + o, ic)
            if k in ca and k not in exclude:
                out.add(k)
    return sorted(out)


def _near_space(ca, centres, exclude, min_sep=MIN_SEQ_SEP):
    """Residues in the 4-6 A shell of a *sequence-far* centre."""
    keys = [k for k in ca if k not in exclude]
    centres = [c for c in centres if c in ca]
    if not keys or not centres:
        return []
    axyz = np.array([ca[k] for k in keys])
    chain_id = {}
    for k in list(keys) + list(centres):
        chain_id.setdefault(k[0], len(chain_id))
    kchain = np.array([chain_id[k[0]] for k in keys])
    kres = np.array([k[1] for k in keys], dtype=float)
    best = np.full(len(keys), np.inf)
    for c in centres:
        d = np.linalg.norm(axyz - np.asarray(ca[c], float), axis=1)
        near_seq = (kchain == chain_id[c[0]]) & (np.abs(kres - c[1]) < min_sep)
        best = np.minimum(best, np.where(near_seq, np.inf, d))
    return [k for k, d in zip(keys, best) if FAR_LO <= d < FAR_HI]


def run_one(pdb_id, seed=0):
    started = time.time()
    rec = {"id": pdb_id}
    try:
        model = load_model(model_path(pdb_id))
        hierarchy = model.get_hierarchy()
        if hierarchy.atoms_size() > MAX_ATOMS:
            rec.update(status="skipped", reason="too many atoms")
            rec["seconds"] = round(time.time() - started, 1)
            return rec

        shared = ve.extract_all(model, metrics=RESIDUE_METRICS)
        calibrated = molprobity_concern_events([_ADAPTERS[s.metric](s) for s in shared])
        ca = _residue_ca(hierarchy)
        by_metric = defaultdict(dict)
        for s, c in zip(shared, calibrated):
            if s.residue is None:
                continue
            k = (s.residue.chain, s.residue.resseq, s.residue.icode)
            if k in ca:
                by_metric[s.metric][k] = max(by_metric[s.metric].get(k, 0.0),
                                             float(c.severity))

        rng = np.random.default_rng(seed)
        out = {}
        for metric in RESIDUE_METRICS:
            conc = by_metric.get(metric)
            if not conc:
                continue
            outliers = [k for k, v in conc.items() if v >= FLAGGED]
            if len(outliers) < MIN_OUTLIERS:
                continue
            excl = set(outliers)
            pool = [k for k in ca if k not in excl]

            res = {"n_outliers": len(outliers), "n_residues": len(ca)}
            for label, picker in (("seq", _near_seq), ("space", _near_space)):
                keys = picker(ca, outliers, excl)
                for band in BANDS:
                    f, n = _frac_above(conc, keys, band)
                    res["%s_%.2f" % (label, band)] = f
                    res["%s_n" % label] = n
                # control: the same shell around random non-outlier centres
                ctl = defaultdict(list)
                for _ in range(CONTROL_TRIALS):
                    pick = [pool[i] for i in rng.integers(0, len(pool), size=len(outliers))]
                    ckeys = picker(ca, pick, excl)
                    for band in BANDS:
                        f, _n = _frac_above(conc, ckeys, band)
                        if f is not None:
                            ctl[band].append(f)
                for band in BANDS:
                    res["%s_ctl_%.2f" % (label, band)] = (
                        float(np.mean(ctl[band])) if ctl[band] else None)
            # How much of the sub-threshold mass is near an outlier at all? A field only
            # earns the claim if the visible sub-threshold residues are mostly near
            # something, rather than scattered everywhere.
            vis = [k for k in pool if conc.get(k, 0.0) >= 0.50]
            near = set(_near_seq(ca, outliers, excl)) | set(_near_space(ca, outliers, excl))
            res["n_visible"] = len(vis)
            res["n_visible_near"] = sum(1 for k in vis if k in near)
            res["frac_residues_near"] = len(near) / float(max(len(pool), 1))
            out[metric] = res

        if not out:
            rec.update(status="empty")
        else:
            rec.update(status="ok", metrics=out)
    except Exception as exc:
        rec.update(status="failed", error="%s: %s" % (type(exc).__name__, exc),
                   traceback=traceback.format_exc()[-800:])
    rec["seconds"] = round(time.time() - started, 1)
    return rec


def report(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "subvis*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no results in", out_dir)
        return
    print("%d structures: %d ok\n" % (len(recs), len(ok)))
    print("Fraction of NON-OUTLIER residues carrying at least a given share of an outlier's")
    print("concern, near a flagged outlier vs around random non-outlier residues.")
    print("Concern 1.0 = the community cut, so 0.50 means 'half an outlier'.\n")

    for label, title in (("seq", "SEQUENCE: +-2 and +-3 residues along the chain"),
                         ("space", "THROUGH SPACE: 4-6 A shell, sequence-far centres only")):
        print(title)
        print("%-9s %9s %9s %8s %9s %9s %8s" % (
            "channel", ">=0.25", "ctl", "x", ">=0.50", "ctl", "x"))
        for metric in RESIDUE_METRICS:
            rows = [r["metrics"][metric] for r in ok if r.get("metrics", {}).get(metric)]
            if len(rows) < 5:
                continue
            cells = []
            for band in BANDS:
                o = [x["%s_%.2f" % (label, band)] for x in rows
                     if x.get("%s_%.2f" % (label, band)) is not None]
                c = [x["%s_ctl_%.2f" % (label, band)] for x in rows
                     if x.get("%s_ctl_%.2f" % (label, band)) is not None]
                mo = float(np.mean(o)) if o else float("nan")
                mc = float(np.mean(c)) if c else float("nan")
                cells += [mo, mc, (mo / mc if mc else float("nan"))]
            print("%-9s %8.2f%% %8.2f%% %8.2f %8.2f%% %8.2f%% %8.2f" % (
                metric, 100 * cells[0], 100 * cells[1], cells[2],
                100 * cells[3], 100 * cells[4], cells[5]))
        print()

    print("CONCENTRATION: of the non-outlier residues drawn at >=0.50, how many sit in the")
    print("near zone -- and how big is that zone as a share of the structure?\n")
    print("%-9s %14s %14s %10s" % ("channel", "visible >=0.50", "of those, near", "zone size"))
    for metric in RESIDUE_METRICS:
        rows = [r["metrics"][metric] for r in ok if r.get("metrics", {}).get(metric)]
        if len(rows) < 5:
            continue
        vis = sum(x["n_visible"] for x in rows)
        near = sum(x["n_visible_near"] for x in rows)
        zone = float(np.mean([x["frac_residues_near"] for x in rows]))
        print("%-9s %14d %13.1f%% %9.1f%%" % (
            metric, vis, (100.0 * near / vis if vis else float("nan")), 100 * zone))


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
    path = os.path.join(args.out_dir, "subvis%s.jsonl" % tag)
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [i for i in ids if i not in done]
    print("%d structures -> %s" % (len(todo), path), flush=True)
    for i, pid in enumerate(todo, 1):
        seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
        rec = run_one(pid, seed=seed)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        print("[%d/%d] %-5s %-7s %5.1fs" % (i, len(todo), pid, rec["status"],
                                            rec["seconds"]), flush=True)


if __name__ == "__main__":
    main()
