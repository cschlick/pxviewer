"""Are the regions only the field can show actually worth showing?

`alpha_accumulation.py` established that faint concern composites into a visible contrast on
~18% of the structure, where a marker representation shows nothing. That is a capability. It
is not yet a benefit: **visible is not the same as worth seeing**, and a field that reliably
renders noise would score exactly the same on that test.

So this asks the other half. Regions that never cross the display threshold — invisible to any
outlier list, and to the threshold itself — are they enriched for problems the field was never
told about?

Method mirrors figure C, with clash held out of the field entirely:

* build the concern field from the eight non-clash channels;
* **sub_threshold** = voxels at ``c >= FAINT`` that lie more than ``AWAY`` from any marked
  voxel. Not merely below the threshold, but far enough from a marked region to be a separate
  *place* rather than its halo -- otherwise the set is just the skirt of problems the markers
  already show, and the test answers nothing.

  A connected-component definition was tried first and does not work: at FAINT the faint
  regions merge into one component touching a marked voxel, so on a strained structure
  (1TEC) the set came out empty. Topological isolation is not what the claim needs.
* **marked** = the ordinary hot set, as the control that should reproduce figure C;
* held-out clash enrichment in each, against the same spatially matched null.

Each region is handed to ``figure_c`` as a binary mask, so the null construction, inclusion
rule and p-value mean exactly what they mean in figure C, and the two partitions are measured
on their own terms rather than one diluting the other.

    libtbx.python corpus/subthreshold_value.py IDS.txt OUT_DIR --shard 0/4
    libtbx.python corpus/subthreshold_value.py IDS.txt OUT_DIR --report
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hotspots"))
sys.path.insert(0, HERE)

from concern import build_concern_fields, molprobity_concern_events  # noqa: E402
from events import ALL_METRICS, extract_all, load_model  # noqa: E402
from figure_data import (MAX_ATOMS, MAX_VOXELS, MODEL_TIMEOUT_S, SIGMA,  # noqa: E402
                         SPACING, _arm_timeout, _Timeout, figure_c, heavy_mask,
                         model_path, wait_for_memory)

BUILD_METRICS = ("rama", "rota", "cablam", "ca_geom", "omega", "bond", "angle", "cbeta")
HOT = 0.5

#: Floor for "the field shows something here at all". Below this a voxel contributes nothing a
#: viewer could see even after compositing, so including it would pad the sub-threshold set
#: with empty space and drag its enrichment toward the base rate for free.
FAINT = 0.05

#: How far a faint voxel must be from any marked voxel to count as its own place. Set at the
#: kernel's reach (3 sigma) so the set excludes the skirt of an already-marked problem: within
#: that distance a viewer following the marked region would arrive here anyway, and crediting
#: the field for it would be double-counting what the threshold already shows.
AWAY_A = 6.0


def _binary(field, mask):
    """Hand figure_c a region as a 0/1 field, so its own 0.5 threshold selects exactly it.

    Cleaner than lowering the threshold for one call: the null construction, the connected
    components it re-places, and the inclusion rule then all mean what they mean in figure C.
    """
    return type(field)(np.where(mask, 1.0, 0.0), field.origin.copy(),
                       field.spacing, field.sigma, field.reference_level)


def run_one(pdb_id, *, spacing=SPACING, sigma=SIGMA, seed=0):
    started = time.time()
    rec = {"id": pdb_id}
    _arm_timeout(MODEL_TIMEOUT_S)
    try:
        model = load_model(model_path(pdb_id))
        n_atoms = model.get_hierarchy().atoms_size()
        rec["n_atoms"] = int(n_atoms)
        if n_atoms > MAX_ATOMS:
            rec.update(status="skipped", reason="n_atoms %d > %d" % (n_atoms, MAX_ATOMS))
            return _done(rec, started)
        need = max(3.0, n_atoms * 1.9e-4)
        _w, avail = wait_for_memory(min_gb=need, timeout_s=420)
        if avail < need:
            rec.update(status="deferred", reason="needs ~%.1f GB, %.1f free" % (need, avail))
            return _done(rec, started)

        extras = {}
        extract = extract_all(model, use_hydrogens=True, metrics=ALL_METRICS,
                              extras_out=extras)
        events = molprobity_concern_events(extract["events"])
        by_metric = {}
        for e in events:
            if e.severity > 0 and e.atoms_xyz and e.metric in BUILD_METRICS:
                by_metric.setdefault(e.metric, []).append(e)
        if not by_metric:
            rec.update(status="empty", reason="no depositing events")
            return _done(rec, started)

        fields = build_concern_fields(by_metric, spacing=spacing, sigma=sigma)
        f = fields["combined"]
        if f.data.size > MAX_VOXELS:
            rec.update(status="skipped", reason="grid %d voxels" % f.data.size)
            return _done(rec, started)

        from scipy.ndimage import distance_transform_edt
        marked_mask = f.data >= HOT
        # Distance to the nearest marked voxel, in angstrom.
        far = (distance_transform_edt(~marked_mask) * spacing > AWAY_A
               if marked_mask.any() else np.ones_like(marked_mask))
        sub_mask = (f.data >= FAINT) & far
        rec["counts"] = {"sub_threshold": int(sub_mask.sum()),
                         "marked": int(marked_mask.sum()),
                         "faint_total": int((f.data >= FAINT).sum())}

        shared_clash = extras.get("shared_clash") or []
        clash_model = extras.get("clash_model")
        if shared_clash and clash_model is not None:
            ch = clash_model.get_hierarchy()
            sites = np.asarray(ch.atoms().extract_xyz()).reshape(-1, 3)
            heavy = heavy_mask(ch)
            rec["enrichment"] = {}
            for name, mask in (("sub_threshold", sub_mask), ("marked", marked_mask)):
                rec["enrichment"][name] = (
                    figure_c({"combined": _binary(f, mask)}, shared_clash, sites, heavy, seed)
                    if mask.any() else None)
        rec["status"] = "ok"
    except _Timeout:
        rec.update(status="timeout", reason="exceeded %ds" % MODEL_TIMEOUT_S)
    except Exception as exc:
        rec.update(status="failed", error="%s: %s" % (type(exc).__name__, exc),
                   traceback=traceback.format_exc()[-1200:])
    finally:
        _arm_timeout(0)
    return _done(rec, started)


def _done(rec, started):
    rec["seconds"] = round(time.time() - started, 1)
    return rec


def report(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "subval*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no results in", out_dir)
        return
    print("%d structures: %d ok, %d failed, %d other\n" % (
        len(recs), len(ok), sum(1 for r in recs if r["status"] == "failed"),
        sum(1 for r in recs if r["status"] not in ("ok", "failed"))))

    sub = np.array([r["counts"]["sub_threshold"] for r in ok], float)
    mk = np.array([r["counts"]["marked"] for r in ok], float)
    print("VOLUME")
    print("  sub-threshold regions : %12d voxels  (%.1f%% of the two sets)" % (
        sub.sum(), 100 * sub.sum() / max(1.0, sub.sum() + mk.sum())))
    print("  marked regions        : %12d voxels" % mk.sum())

    print("\nHELD-OUT CLASH ENRICHMENT (clash held out of the field entirely)")
    print("  %-15s %5s %10s %8s %10s %8s" % (
        "region", "n", "observed", "null", "obs/null", "p<0.05"))
    out = {}
    for key in ("sub_threshold", "marked"):
        rows = [(r.get("enrichment") or {}).get(key) for r in ok]
        rows = [x for x in rows if x and x.get("enrichment") is not None
                and x.get("n_atoms_in_region", 0) >= 50 and x.get("n_null", 0) >= 10]
        if not rows:
            print("  %-15s %5s" % (key, "none"))
            continue
        obs = np.array([x["enrichment"] for x in rows])
        nul = np.array([x.get("null_enrichment_mean", np.nan) for x in rows])
        pv = np.array([x.get("p_value", np.nan) for x in rows])
        ratio = float(np.nanmedian(obs / np.where(nul > 0, nul, np.nan)))
        out[key] = ratio
        print("  %-15s %5d %10.2f %8.2f %10.2f %7.0f%%" % (
            key, len(rows), np.median(obs), np.nanmedian(nul), ratio,
            100 * np.nanmean(pv < 0.05)))

    if "sub_threshold" in out:
        r = out["sub_threshold"]
        print("\nVERDICT — are the regions only the field can show worth showing?")
        if r < 1.15:
            print("  %.2fx, at the null. The field visibly renders NOISE in those regions:" % r)
            print("  a capability that shows nothing worth finding is not a benefit.")
        elif r < 1.6:
            print("  %.2fx. Real but weak signal -- above the null, well below the %.2fx" % (
                r, out.get("marked", float("nan"))))
            print("  the marked regions carry. Worth showing, worth not overselling.")
        else:
            print("  %.2fx. The regions only the field can show carry real signal," % r)
            print("  comparable to the marked ones. That is the strongest form of the claim.")


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
    path = os.path.join(args.out_dir, "subval%s.jsonl" % tag)
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [i for i in ids if i not in done]
    print("%d structures%s -> %s" % (
        len(ids), " (shard %d/%d)" % shard if shard else "", path), flush=True)

    for i, pid in enumerate(todo, 1):
        seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
        rec = run_one(pid, seed=seed)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        note = ("sub=%d marked=%d" % (rec["counts"]["sub_threshold"], rec["counts"]["marked"])
                if rec["status"] == "ok" else rec.get("error", rec.get("reason", ""))[:50])
        print("[%d/%d] %-5s %-8s %6.1fs  %s" % (
            i, len(todo), pid, rec["status"], rec["seconds"], note), flush=True)


if __name__ == "__main__":
    main()
