"""Does the density field find anything the concern field misses — and is it worth finding?

Figures A, B and C do not answer this. They ask whether a field loses flagged problems, how
blurry it is, and whether its hot regions are informative. None of them asks the question the
density construction exists for: **does it mark places the concern field structurally cannot,
and do those places carry real signal?**

The head-to-head compared the two fields on those three figures and the density field lost —
but on measures that do not test its purpose. Precision is prevalence-confounded, figure B
essentially measures kernel width (a 6 Å kernel is blurrier than a 2 Å one by construction,
which is not a discovery), and whole-field enrichment dilutes with region size, which penalises
the field with larger regions regardless of quality. So the loss was not yet decisive.

This measures the thing directly. Both fields are built from the same events on the same grid,
with **clash held out of both**, and the voxels are partitioned:

    novel        density-hot, concern-cold   <- what density adds
    shared       hot in both
    concern_only concern-hot, density-cold   <- what density loses

Then held-out clash enrichment in each partition, against the same spatially matched null
figure C uses. The partition removes the volume confound: each is measured on its own set.

* novel enrichment near the null (~1.0)  -> density adds volume and nothing else.
* novel enrichment well above the null   -> density finds something concern cannot, and the
  precision/blur trade becomes a real choice rather than a loss.
* concern_only enrichment high           -> density is *losing* real signal, which would be a
  cost the head-to-head never priced.

    libtbx.python corpus/novel_regions.py IDS.txt OUT_DIR --shard 0/4
    libtbx.python corpus/novel_regions.py IDS.txt OUT_DIR --report
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
from density import build_density_fields  # noqa: E402
from events import ALL_METRICS, extract_all, load_model  # noqa: E402
from figure_data import (MAX_ATOMS, MAX_VOXELS, SIGMA, SPACING,  # noqa: E402
                         MODEL_TIMEOUT_S, _arm_timeout, _Timeout, figure_c,
                         heavy_mask, model_path, wait_for_memory)

#: Clash is held out of BOTH fields so it can serve as the signal neither was told about.
BUILD_METRICS = ("rama", "rota", "cablam", "ca_geom", "omega", "bond", "angle", "cbeta")

CONCERN_HOT = 0.5
#: The density field is measured at its own knee, not at concern's. Comparing two fields at
#: one shared number prices a single point on each curve, which is what made the head-to-head
#: objectionable.
DENSITY_HOT = 1.0


def _masked(field, mask):
    """A copy of ``field`` zeroed outside ``mask`` — figure_c reads its hot set from this."""
    return type(field)(np.where(mask, field.data, 0.0), field.origin.copy(),
                       field.spacing, field.sigma, field.reference_level)


def run_one(pdb_id, *, spacing=SPACING, sigma=SIGMA, radius=6.0, seed=0,
            concern_hot=CONCERN_HOT, density_hot=DENSITY_HOT):
    started = time.time()
    rec = {"id": pdb_id, "concern_hot": concern_hot, "density_hot": density_hot}
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
        extracted = extract_all(model, use_hydrogens=True, metrics=ALL_METRICS,
                                extras_out=extras)
        events = molprobity_concern_events(extracted["events"])
        by_metric = {}
        for e in events:
            if e.severity > 0 and e.atoms_xyz and e.metric in BUILD_METRICS:
                by_metric.setdefault(e.metric, []).append(e)
        if not by_metric:
            rec.update(status="empty", reason="no depositing events")
            return _done(rec, started)

        all_ev = [e for evs in by_metric.values() for e in evs]
        cf = build_concern_fields(by_metric, spacing=spacing, sigma=sigma, grid_events=all_ev)
        df = build_density_fields(by_metric, spacing=spacing, radius=radius,
                                  grid_events=all_ev)
        if "combined" not in cf or "combined" not in df:
            rec.update(status="empty", reason="missing a combined field")
            return _done(rec, started)
        c, d = cf["combined"], df["combined"]
        # compute_field pads 3*sigma and compute_density pads radius; both are 6 A here, so
        # the boxes coincide. Assert rather than assume -- a silent mismatch would make every
        # partition below empty, which is exactly the bug that hid in the accumulation run.
        if c.data.shape != d.data.shape:
            rec.update(status="failed",
                       error="grid mismatch %s vs %s" % (c.data.shape, d.data.shape))
            return _done(rec, started)
        if c.data.size > MAX_VOXELS:
            rec.update(status="skipped", reason="grid %d voxels" % c.data.size)
            return _done(rec, started)

        c_hot = c.data >= concern_hot
        d_hot = d.data >= density_hot
        parts = {
            "novel": d_hot & ~c_hot,
            "shared": d_hot & c_hot,
            "concern_only": c_hot & ~d_hot,
        }
        rec["n_voxels"] = int(c.data.size)
        rec["counts"] = {k: int(v.sum()) for k, v in parts.items()}

        shared_clash = extras.get("shared_clash") or []
        clash_model = extras.get("clash_model")
        if shared_clash and clash_model is not None:
            ch = clash_model.get_hierarchy()
            sites = np.asarray(ch.atoms().extract_xyz()).reshape(-1, 3)
            heavy = heavy_mask(ch)
            rec["enrichment"] = {}
            for name, mask in parts.items():
                if not mask.any():
                    rec["enrichment"][name] = None
                    continue
                src = d if name != "concern_only" else c
                rec["enrichment"][name] = figure_c({"combined": _masked(src, mask)},
                                                   shared_clash, sites, heavy, seed)
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
    for p in sorted(glob.glob(os.path.join(out_dir, "novel*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no results in", out_dir)
        return
    print("%d structures: %d ok, %d failed, %d other\n" % (
        len(recs), len(ok), sum(1 for r in recs if r["status"] == "failed"),
        sum(1 for r in recs if r["status"] not in ("ok", "failed"))))

    print("VOLUME PARTITION (share of the union of the two hot sets)")
    tot = {k: 0 for k in ("novel", "shared", "concern_only")}
    for r in ok:
        for k in tot:
            tot[k] += r["counts"].get(k, 0)
    union = sum(tot.values()) or 1
    for k in ("novel", "shared", "concern_only"):
        print("  %-13s %12d voxels  %5.1f%%" % (k, tot[k], 100 * tot[k] / union))

    print("\nHELD-OUT CLASH ENRICHMENT, by partition (vs spatially matched null)")
    print("  %-13s %6s %9s %9s %9s %8s" % (
        "partition", "n", "observed", "null", "obs/null", "p<0.05"))
    for k in ("novel", "shared", "concern_only"):
        rows = [(r["enrichment"] or {}).get(k) for r in ok if (r.get("enrichment") or {}).get(k)]
        rows = [x for x in rows if x and x.get("enrichment") is not None
                and x.get("n_atoms_in_region", 0) >= 50 and x.get("n_null", 0) >= 10]
        if not rows:
            print("  %-13s %6s" % (k, "none"))
            continue
        obs = np.array([x["enrichment"] for x in rows])
        nul = np.array([x.get("null_enrichment_mean", np.nan) for x in rows])
        pv = np.array([x.get("p_value", np.nan) for x in rows])
        ratio = obs / np.where(nul > 0, nul, np.nan)
        print("  %-13s %6d %9.2f %9.2f %9.2f %7.0f%%" % (
            k, len(rows), np.median(obs), np.nanmedian(nul), np.nanmedian(ratio),
            100 * np.nanmean(pv < 0.05)))

    nov = [(r["enrichment"] or {}).get("novel") for r in ok]
    nov = [x for x in nov if x and x.get("enrichment") is not None
           and x.get("n_atoms_in_region", 0) >= 50 and x.get("n_null", 0) >= 10]
    if nov:
        obs = np.array([x["enrichment"] for x in nov])
        nul = np.array([x.get("null_enrichment_mean", np.nan) for x in nov])
        ratio = float(np.nanmedian(obs / np.where(nul > 0, nul, np.nan)))
        print("\n  VERDICT on what the density field ADDS:")
        if ratio < 1.15:
            print("    novel regions sit at the null (%.2fx). Density adds volume, not signal." % ratio)
        elif ratio < 1.6:
            print("    novel regions carry weak signal (%.2fx). Real but modest." % ratio)
        else:
            print("    novel regions carry real signal (%.2fx). Density finds something" % ratio)
            print("    the concern field structurally cannot, and the trade is a choice.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids")
    ap.add_argument("out_dir")
    ap.add_argument("--shard", metavar="K/N")
    ap.add_argument("--density-hot", type=float, default=DENSITY_HOT)
    ap.add_argument("--concern-hot", type=float, default=CONCERN_HOT)
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
    path = os.path.join(args.out_dir, "novel%s.jsonl" % tag)
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [i for i in ids if i not in done]
    print("%d structures%s -> %s" % (
        len(ids), " (shard %d/%d)" % shard if shard else "", path), flush=True)

    for n, pid in enumerate(todo, 1):
        seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
        rec = run_one(pid, seed=seed, concern_hot=args.concern_hot,
                      density_hot=args.density_hot)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        note = ("novel=%d shared=%d concern_only=%d" % (
            rec["counts"]["novel"], rec["counts"]["shared"], rec["counts"]["concern_only"])
            if rec["status"] == "ok" else rec.get("error", rec.get("reason", ""))[:50])
        print("[%d/%d] %-5s %-8s %6.1fs  %s" % (
            n, len(todo), pid, rec["status"], rec["seconds"], note), flush=True)


if __name__ == "__main__":
    main()
