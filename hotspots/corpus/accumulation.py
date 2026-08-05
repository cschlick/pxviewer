"""Does co-locality of *never-flagged* observations create regions worth visiting?

This is the claim a continuous field can make and an outlier list structurally cannot: a place
where nothing individually crosses its threshold, but several sub-threshold concerns coincide
into a region that deserves attention.

An earlier probe put "hot voxels no single event could have made hot" at ~45% of hot volume.
That number conflated the interesting case with a dull one -- two *flagged* outliers' Gaussian
tails overlapping in the space between them is accumulation too, and says nothing. This
measures the strict version by construction rather than by proximity argument:

    build a field from ONLY the sub-threshold events, and see what is hot in it.

Two strictness levels, because they support different sentences:

* ``sub_outlier``  -- every contributing observation is one MolProbity never flagged
  (concern < 1.0). Supports "a region no outlier list would send you to".
* ``sub_display``  -- every contributing observation is individually invisible in the viewer
  (concern < 0.5). Stricter, and the honest version of "nothing here is visible on its own".

And then the part that decides whether the claim is worth making: **are those regions
enriched for problems the field was not told about?** The accumulation field is built with
clash held out entirely, and clash-outlier enrichment inside the accumulated region is
measured against the same spatially matched null figure C uses. Without that, "we light up
places nothing flagged" is equally consistent with "we light up noise".

    libtbx.python corpus/accumulation.py IDS.txt OUT_DIR --shard 0/6
    libtbx.python corpus/accumulation.py IDS.txt OUT_DIR --report
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
from figure_data import (HOT, MAX_ATOMS, MAX_VOXELS, SIGMA, SPACING,  # noqa: E402
                         _arm_timeout, _Timeout, MODEL_TIMEOUT_S, figure_c,
                         heavy_mask, hot_voxel_xyz, model_path, wait_for_memory)

#: Channels the accumulation field is built from. Clash is held out so it can serve as the
#: signal the field was never told about -- the same structure as figure C.
BUILD_METRICS = ("rama", "rota", "cablam", "ca_geom", "omega", "bond", "angle", "cbeta")
HELD_OUT = "clash"

#: An observation is "flagged" at concern 1.0 by construction: every channel's community cut
#: is anchored there (see hotspots/calibration_cuts.py). That is what makes "sub-outlier"
#: mean the same thing in every channel, and it is why the re-anchoring had to come first.
FLAGGED = 1.0


def _fields_from(events, spacing, sigma, combine, p, grid_events):
    """Fields for a subset of events, on the grid ``grid_events`` defines.

    The shared grid is the whole point: these fields get compared voxel-to-voxel against the
    full field, and a subset left to pick its own bounding box lands on different voxels and
    silently compares as empty.
    """
    by_metric = {}
    for e in events:
        if e.severity > 0:
            by_metric.setdefault(e.metric, []).append(e)
    if not by_metric:
        return None
    return build_concern_fields(by_metric, spacing=spacing, sigma=sigma, combine=combine, p=p,
                                grid_events=grid_events)


def run_one(pdb_id, *, spacing=SPACING, sigma=SIGMA, combine="family", p=1.0, seed=0):
    started = time.time()
    rec = {"id": pdb_id, "combine": combine, "p": p}
    _arm_timeout(MODEL_TIMEOUT_S)
    try:
        model = load_model(model_path(pdb_id))
        n_atoms = model.get_hierarchy().atoms_size()
        rec["n_atoms"] = int(n_atoms)
        if n_atoms > MAX_ATOMS:
            rec.update(status="skipped", reason="n_atoms %d > %d" % (n_atoms, MAX_ATOMS))
            return _done(rec, started)
        need = max(3.0, n_atoms * 1.9e-4)
        waited, avail = wait_for_memory(min_gb=need, timeout_s=420)
        if avail < need:
            rec.update(status="deferred", reason="needs ~%.1f GB, %.1f free" % (need, avail))
            return _done(rec, started)

        extras = {}
        extracted = extract_all(model, use_hydrogens=True, metrics=ALL_METRICS,
                                extras_out=extras)
        events = molprobity_concern_events(extracted["events"])
        build = [e for e in events if e.metric in BUILD_METRICS]

        pts = np.vstack([np.asarray(e.atoms_xyz, float).reshape(-1, 3)
                         for e in build if e.atoms_xyz and e.severity > 0]) \
            if any(e.severity > 0 and e.atoms_xyz for e in build) else None
        if pts is None:
            rec.update(status="empty", reason="no depositing events")
            return _done(rec, started)
        pad = 3.0 * sigma
        lo = np.floor((pts.min(axis=0) - pad) / spacing) * spacing
        shape = np.ceil((pts.max(axis=0) + pad - lo) / spacing).astype(int) + 1
        n_vox = int(np.prod(shape.astype(np.int64)))
        rec["n_voxels"] = n_vox
        if n_vox > MAX_VOXELS:
            rec.update(status="skipped", reason="grid %d voxels > %d" % (n_vox, MAX_VOXELS))
            return _done(rec, started)

        full = _fields_from(build, spacing, sigma, combine, p, build)
        flagged_only = _fields_from([e for e in build if e.severity >= FLAGGED],
                                    spacing, sigma, combine, p, build)
        levels = {
            "sub_outlier": [e for e in build if 0 < e.severity < FLAGGED],
            "sub_display": [e for e in build if 0 < e.severity < HOT],
        }
        if full is None:
            rec.update(status="empty", reason="no field")
            return _done(rec, started)

        f_full = full["combined"].data
        rec["hot_full"] = float((f_full >= HOT).mean())
        rec["n_vox_total"] = int(f_full.size)

        # Voxels hot in the flagged-only field, resampled onto the full field's grid so the
        # masks are comparable voxel for voxel.
        flagged_hot = (np.zeros_like(f_full, dtype=bool) if flagged_only is None
                       else flagged_only["combined"].data >= HOT)

        out = {}
        for name, evs in levels.items():
            sub = _fields_from(evs, spacing, sigma, combine, p, build)
            if sub is None:
                out[name] = {"hot": 0.0, "hot_novel": 0.0, "n_events": len(evs),
                             "n_hot_voxels": 0, "n_novel_voxels": 0,
                             "share_of_full_hot": 0.0}
                continue
            assert sub["combined"].data.shape == f_full.shape, (
                "subset field left the shared grid: %s vs %s"
                % (sub["combined"].data.shape, f_full.shape))
            hot = sub["combined"].data >= HOT
            novel = hot & ~flagged_hot
            out[name] = {
                "n_events": len(evs),
                "hot": float(hot.mean()),
                "hot_novel": float(novel.mean()),
                "n_hot_voxels": int(hot.sum()),
                "n_novel_voxels": int(novel.sum()),
                # share of the whole field's hot volume that only these events explain
                "share_of_full_hot": float(novel.sum() / max(1, (f_full >= HOT).sum())),
            }
            # Falsification: is the novel region enriched for the held-out channel?
            if name == "sub_outlier" and novel.any():
                out[name]["heldout"] = _heldout_enrichment(
                    sub["combined"], novel, extras, seed)
        rec["levels"] = out
        rec["status"] = "ok"
    except _Timeout:
        rec.update(status="timeout", reason="exceeded %ds" % MODEL_TIMEOUT_S)
    except Exception as exc:
        rec.update(status="failed", error="%s: %s" % (type(exc).__name__, exc),
                   traceback=traceback.format_exc()[-1200:])
    finally:
        _arm_timeout(0)
    return _done(rec, started)


def _heldout_enrichment(field, novel_mask, extras, seed):
    """Clash-outlier enrichment inside the accumulated region, against the matched null.

    Reuses figure_c wholesale rather than reimplementing the null: the region is handed over
    as a field whose hot set is exactly the accumulated voxels, so the null construction,
    inclusion rule and p-value all mean the same thing they mean in figure C.
    """
    shared_clash = extras.get("shared_clash") or []
    clash_model = extras.get("clash_model")
    if not shared_clash or clash_model is None:
        return None
    ch = clash_model.get_hierarchy()
    sites = np.asarray(ch.atoms().extract_xyz()).reshape(-1, 3)
    heavy = heavy_mask(ch)
    masked = type(field)(np.where(novel_mask, field.data, 0.0), field.origin.copy(),
                         field.spacing, field.sigma, field.reference_level)
    return figure_c({"combined": masked}, shared_clash, sites, heavy, seed)


def _done(rec, started):
    rec["seconds"] = round(time.time() - started, 1)
    return rec


def report(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "accum*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no results in", out_dir)
        return
    print("%d structures: %d ok, %d failed, %d skipped/deferred\n" % (
        len(recs), len(ok), sum(1 for r in recs if r["status"] == "failed"),
        sum(1 for r in recs if r["status"] in ("skipped", "deferred", "timeout"))))

    print("accumulation-only hot volume (clash held out of the field entirely)")
    print("%-12s %8s %12s %14s %16s" % (
        "level", "n", "events/str", "hot % of box", "novel % of hot"))
    for name in ("sub_outlier", "sub_display"):
        rows = [r["levels"][name] for r in ok if r.get("levels", {}).get(name)]
        rows = [x for x in rows if x.get("n_hot_voxels") is not None]
        if not rows:
            continue
        ev = np.array([x["n_events"] for x in rows], float)
        hot = np.array([100 * x["hot"] for x in rows])
        share = np.array([100 * x.get("share_of_full_hot", 0.0) for x in rows])
        print("%-12s %8d %12.0f %8.2f (med) %12.1f (med)" % (
            name, len(rows), np.median(ev), np.median(hot), np.median(share)))

    en = [(r["id"], r["levels"]["sub_outlier"]["heldout"]) for r in ok
          if (r.get("levels", {}).get("sub_outlier") or {}).get("heldout")]
    usable = [(i, c) for i, c in en
              if c and c.get("enrichment") is not None
              and c.get("n_atoms_in_region", 0) >= 50 and c.get("n_null", 0) >= 10]
    print("\nfalsification: held-out clash enrichment INSIDE the accumulated region")
    if not usable:
        print("  no structure met the inclusion rule (>=50 atoms in region, >=10 null runs)")
        return
    obs = np.array([c["enrichment"] for _i, c in usable])
    null = np.array([c.get("null_enrichment_mean", np.nan) for _i, c in usable])
    pv = np.array([c.get("p_value", np.nan) for _i, c in usable])
    print("  usable %d of %d structures" % (len(usable), len(en)))
    print("  observed median %.2fx   null median %.2fx   observed/null %.2fx" % (
        np.median(obs), np.nanmedian(null), np.nanmedian(obs / np.where(null > 0, null, np.nan))))
    print("  enriched over 1.0 in %.1f%%; p<0.05 in %.1f%%" % (
        100 * (obs > 1).mean(), 100 * np.nanmean(pv < 0.05)))
    print("\n  (figure C over the whole field, for comparison: 1.92x observed, 0.98x null)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids")
    ap.add_argument("out_dir")
    ap.add_argument("--shard", metavar="K/N")
    ap.add_argument("--combine", choices=("max", "family"), default="family")
    ap.add_argument("--norm-p", type=float, default=1.0)
    ap.add_argument("--output-pixel-size", dest="spacing", type=float, default=SPACING)
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
    path = os.path.join(args.out_dir, "accum%s.jsonl" % tag)
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [i for i in ids if i not in done]
    print("%d structures%s, %d done -> %s" % (
        len(ids), " (shard %d/%d)" % shard if shard else "", len(done), path), flush=True)

    for n, pid in enumerate(todo, 1):
        seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
        rec = run_one(pid, spacing=args.spacing, combine=args.combine, p=args.norm_p,
                      seed=seed)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        lv = (rec.get("levels") or {}).get("sub_outlier") or {}
        note = "novel %.2f%% of box" % (100 * lv["hot_novel"]) if lv.get("hot_novel") else \
            rec.get("error", rec.get("reason", ""))[:50]
        print("[%d/%d] %-5s %-8s %6.1fs  %s" % (
            n, len(todo), pid, rec["status"], rec["seconds"], note), flush=True)


if __name__ == "__main__":
    main()
