"""Reduce a corpus run into the figure data for FIGURES.md figures A, B and C.

Reads the per-model records `figure_data.py` wrote and emits one JSON file holding exactly
what each figure plots, plus the sentences each one supports. Kept separate from the run on
purpose: the run is hours, the reduction is seconds, and a figure that gets restated must
not force a recompute.

    libtbx.python corpus/make_figures.py OUT_DIR [--out figures.json]

Three things this deliberately does *not* do:

* **No ROC.** Positives are ~0.7% of atoms for Ramachandran and ROC is insensitive to
  prevalence, so it reads ~0.999 for a field that is only mediocre. PR is reported instead.
* **No pooling of figure C where it is undefined.** A structure whose held-out hot region
  contains almost no atoms produces a meaningless ratio; the inclusion rule is stated and
  applied, and the number of structures it keeps is reported rather than hidden.
* **No silent truncation.** Every count that gets dropped is reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

B_EDGES = np.arange(0.0, 30.0 + 1e-9, 0.05)

#: Figure C inclusion rule, fixed in advance. A held-out region with almost nothing in it
#: yields a ratio dominated by whether a single atom happened to clash; and with no clash
#: outlier anywhere in the structure the question is not asked at all. Stated here so the
#: number of structures it keeps can be reported alongside the result.
C_MIN_REGION_ATOMS = 50
C_MIN_BASE_CLASHES = 1

#: Corpus size cap, applied here as well as in the runner. The runner's cap was tightened
#: mid-run after large models OOM-killed shards, so some models above it had already been
#: processed successfully. Re-applying it at reduction makes the corpus definition uniform --
#: "protein-containing entries of at most this many atoms" -- rather than "whichever large
#: models happened to run before the cap changed", which is not a definition at all.
MAX_ATOMS = 50_000


#: Preference when one structure has several records. A deferral is provisional -- the model
#: is retried later -- so a structure can legitimately appear as ``deferred`` and again with
#: its real outcome. Reducing without collapsing these would count such structures twice and
#: let a superseded deferral sit in the corpus totals as if it were a result.
_STATUS_RANK = {"ok": 0, "failed": 1, "timeout": 2, "skipped": 3, "deferred": 4}


def _preference(rec):
    """Sort key for choosing between several records of the same structure.

    Status first. Then, among equally-terminal records, **more figure C null placements
    wins** — a structure recomputed under the flat null policy supersedes one the placement
    budget throttled, so the corpus converges on a single configuration instead of keeping
    whichever record happened to be read first. Making this explicit matters: the alternative
    is depending on filename sort order, which is invisible and breaks the moment a file is
    renamed.
    """
    n_null = ((rec.get("C") or {}).get("n_null") or 0) if rec.get("status") == "ok" else 0
    return (_STATUS_RANK.get(rec["status"], 9), -n_null)


def load(out_dir):
    """Every record, collapsed to one per structure (see :func:`_preference`)."""
    best = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "results*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                prev = best.get(r["id"])
                if prev is None or _preference(r) < _preference(prev):
                    best[r["id"]] = r
    return list(best.values())


def _quantiles_from_hist(counts, edges):
    """Median/p90/max from pooled histogram counts — exact to within one bin."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return None
    cum = np.cumsum(counts) / total
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    for name, q in (("median", 0.5), ("p90", 0.9), ("p99", 0.99)):
        out[name] = float(centres[int(np.searchsorted(cum, q))])
    out["max_bin"] = float(centres[int(np.flatnonzero(counts)[-1])])
    return out


def figure_a(ok):
    """Corpus distribution of recall and precision per channel, at the display threshold."""
    out = {}
    for m in ("rama", "rota", "clash"):
        rec = [r["A"][m] for r in ok if r.get("A", {}).get(m, {}).get("recall") is not None]
        if not rec:
            continue
        recall = np.array([x["recall"] for x in rec])
        prec = np.array([x["precision"] for x in rec if x["precision"] is not None])
        # Pooled = one number for the whole corpus, not the mean of per-structure ratios:
        # a 3-outlier structure should not weigh as much as a 300-outlier one.
        hit = sum(x["n_hit"] for x in rec)
        out[m] = {
            "n_structures": len(rec),
            "pooled_recall": hit / max(1, sum(x["n_flagged"] for x in rec)),
            "pooled_precision": hit / max(1, sum(x["n_marked"] for x in rec)),
            "recall_mean": float(recall.mean()),
            "recall_median": float(np.median(recall)),
            "recall_min": float(recall.min()),
            "recall_ecdf": {str(t): float((recall >= t).mean())
                            for t in (1.0, 0.99, 0.95, 0.9, 0.75, 0.5)},
            "n_structures_recall_below_1": int((recall < 1.0).sum()),
            "worst": sorted(({"id": r["id"], "recall": r["A"][m]["recall"],
                              "n_flagged": r["A"][m]["n_flagged"]}
                             for r in ok
                             if r.get("A", {}).get(m, {}).get("recall") is not None),
                            key=lambda d: d["recall"])[:15],
            "precision_median": float(np.median(prec)) if prec.size else None,
            "total_flagged_atoms": int(sum(x["n_flagged"] for x in rec)),
        }
    return out


def figure_b(ok):
    """Pooled distance histogram: hot voxel -> nearest concerning atom, in angstroms."""
    out = {}
    for m in ("rama", "rota", "clash"):
        hists = [r["B"][m]["hist"] for r in ok if r.get("B", {}).get(m, {}).get("hist")]
        if not hists:
            continue
        pooled = np.sum(np.array(hists, dtype=np.int64), axis=0)
        over = sum(r["B"][m].get("n_over_30a", 0) for r in ok
                   if r.get("B", {}).get(m, {}).get("hist"))
        q = _quantiles_from_hist(pooled, B_EDGES)
        per_struct_max = [r["B"][m]["max"] for r in ok
                          if r.get("B", {}).get(m, {}).get("hist")]
        out[m] = {
            "n_structures": len(hists),
            "n_hot_voxels": int(pooled.sum()) + int(over),
            "n_beyond_30a": int(over),
            "bin_edges_a": [float(B_EDGES[0]), float(B_EDGES[-1]), float(B_EDGES[1])],
            "hist": pooled.astype(int).tolist(),
            **(q or {}),
            "worst_structure_max_a": float(np.max(per_struct_max)),
            "p90_of_structure_maxima_a": float(np.percentile(per_struct_max, 90)),
        }
    return out


def figure_c(ok):
    """Held-out-channel enrichment against a spatially matched null."""
    usable, skipped = [], 0
    for r in ok:
        c = r.get("C")
        if not c or c.get("enrichment") is None:
            skipped += 1
            continue
        if (c.get("n_atoms_in_region", 0) < C_MIN_REGION_ATOMS
                or c.get("base_rate", 0) <= 0
                or c.get("n_null", 0) < 10):
            skipped += 1
            continue
        usable.append((r["id"], c))
    if not usable:
        return {"n_usable": 0, "n_excluded": skipped}
    obs = np.array([c["enrichment"] for _i, c in usable])
    null = np.array([c.get("null_enrichment_mean", np.nan) for _i, c in usable])
    pv = np.array([c.get("p_value", np.nan) for _i, c in usable])
    ratio = obs / np.where(null > 0, null, np.nan)
    return {
        "inclusion_rule": (f"n_atoms_in_region >= {C_MIN_REGION_ATOMS}, "
                           f">= {C_MIN_BASE_CLASHES} clash outlier, >= 10 null placements"),
        "n_usable": len(usable),
        "n_excluded": skipped,
        "observed_enrichment_median": float(np.median(obs)),
        "observed_enrichment_mean": float(obs.mean()),
        "null_enrichment_median": float(np.nanmedian(null)),
        "observed_over_null_median": float(np.nanmedian(ratio)),
        "fraction_enriched_over_1": float((obs > 1.0).mean()),
        "fraction_p_below_0.05": float(np.nanmean(pv < 0.05)),
        "enrichment_quantiles": {q: float(np.percentile(obs, int(q)))
                                 for q in ("10", "25", "50", "75", "90")},
        "per_structure": [{"id": i, "enrichment": c["enrichment"],
                           "null_mean": c.get("null_enrichment_mean"),
                           "p": c.get("p_value"),
                           "n_region": c.get("n_atoms_in_region")}
                          for i, c in usable],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = load(args.out_dir)
    oversize = [r for r in recs
                if r["status"] == "ok" and r.get("n_atoms", 0) > MAX_ATOMS]
    ok = [r for r in recs
          if r["status"] == "ok" and r.get("n_atoms", 0) <= MAX_ATOMS]
    failed = [r for r in recs if r["status"] == "failed"]
    skipped = [r for r in recs if r["status"] not in ("ok", "failed")]

    errors = {}
    for r in failed:
        kind = r["error"].split(":")[0]
        if "Hydrogen with no neigbors" in r["error"]:
            kind = "reduce2: hydrogen with no neighbours"
        errors[kind] = errors.get(kind, 0) + 1

    data = {
        "corpus": {
            "n_attempted": len(recs), "n_ok": len(ok),
            "n_failed": len(failed), "n_skipped": len(skipped),
            "max_atoms": MAX_ATOMS,
            "n_excluded_oversize_at_reduction": len(oversize),
            "failure_kinds": dict(sorted(errors.items(), key=lambda kv: -kv[1])),
            "skipped_reasons": dict(sorted(
                {r.get("reason", "?"): sum(1 for x in skipped
                                           if x.get("reason") == r.get("reason"))
                 for r in skipped}.items(), key=lambda kv: -kv[1])),
            "seconds_total": round(sum(r.get("seconds", 0) for r in recs), 1),
        },
        "figure_A_operating_point": figure_a(ok),
        "figure_B_spatial_error": figure_b(ok),
        "figure_C_held_out_enrichment": figure_c(ok),
    }
    out = args.out or os.path.join(args.out_dir, "figures.json")
    with open(out, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)

    c = data["corpus"]
    print(f"corpus: {c['n_ok']} ok / {c['n_attempted']} attempted "
          f"({c['n_failed']} failed, {c['n_skipped']} skipped), "
          f"{c['seconds_total']/3600:.2f} h compute")
    print("\nFIGURE A — operating point (heavy-atom universe, concern >= 0.5)")
    for m, a in data["figure_A_operating_point"].items():
        print(f"  {m:5s} n={a['n_structures']:4d}  pooled recall {a['pooled_recall']:.4f}  "
              f"pooled precision {a['pooled_precision']:.3f}  "
              f"recall=1.0 in {a['recall_ecdf']['1.0']*100:.1f}% of structures")
    print("\nFIGURE B — hot voxel to nearest concerning atom (A)")
    for m, b in data["figure_B_spatial_error"].items():
        print(f"  {m:5s} n={b['n_structures']:4d}  {b['n_hot_voxels']:>10,d} voxels  "
              f"median {b['median']:.2f}  p90 {b['p90']:.2f}  p99 {b['p99']:.2f}  "
              f"worst structure max {b['worst_structure_max_a']:.2f}")
    fc = data["figure_C_held_out_enrichment"]
    print("\nFIGURE C — held-out clash enrichment vs spatially matched null")
    if fc.get("n_usable"):
        print(f"  usable {fc['n_usable']}, excluded {fc['n_excluded']} "
              f"({fc['inclusion_rule']})")
        print(f"  observed median {fc['observed_enrichment_median']:.2f}x   "
              f"null median {fc['null_enrichment_median']:.2f}x   "
              f"observed/null {fc['observed_over_null_median']:.2f}x")
        print(f"  enriched over 1.0 in {fc['fraction_enriched_over_1']*100:.1f}% of "
              f"structures; p<0.05 in {fc['fraction_p_below_0.05']*100:.1f}%")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
