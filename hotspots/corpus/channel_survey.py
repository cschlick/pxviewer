"""Survey all nine channels: how often each fires, and whether the family grouping holds.

`../AGGREGATION_PROPOSAL.md` proposes combining channels with `max` *within* a family and
accumulation *across* families, on the argument that channels in a family are redundant
evidence about one thing while families are independent. That argument is a guess from the
physics. This measures it.

Two questions, both at the residue level, which is where a family would be redundant:

* **How often does each channel fire at all?** A channel that marks nothing is dead weight in
  the aggregate; one that marks everything will dominate it.
* **Do channels within a proposed family actually co-occur?** If rama and cablam fire on the
  same residues they are redundant and `max` is right. If they fire on different residues the
  family grouping is wrong, and taking `max` throws away real signal.

    libtbx.python corpus/channel_survey.py IDS.txt OUT_DIR --shard 0/6
    libtbx.python corpus/channel_survey.py IDS.txt OUT_DIR --report

Deliberately reports the correlation matrix over *all* channel pairs, not only within the
proposed families -- a strong pair that crosses a family boundary is exactly the finding that
would falsify the grouping, and looking only where the hypothesis predicts would hide it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hotspots"))

from concern import build_concern_fields, molprobity_concern_events  # noqa: E402
from events import ALL_METRICS, extract_all, load_model  # noqa: E402

MIRROR = "/root/data/pdb_mmcif"
SPACING = 1.0
SIGMA = 2.0
MAX_ATOMS = 50_000

#: The grouping under test. Not assumed anywhere in the measurement — only used to label the
#: report, so a pair that contradicts it is visible rather than silently absorbed.
FAMILIES = {
    "backbone": ("rama", "cablam", "ca_geom", "omega"),
    "sidechain": ("rota",),
    "sterics": ("clash",),
    "covalent": ("bond", "angle", "cbeta"),
}
FAMILY_OF = {m: f for f, ms in FAMILIES.items() for m in ms}


def model_path(pid):
    return os.path.join(MIRROR, pid[1:3], pid + ".cif.gz")


def residue_key(event):
    """Residue this event is about, for cross-channel joins. None when unavailable."""
    r = event.meta.get("residue")
    if r is not None:
        return tuple(r)
    ident = event.meta.get("id")
    return ("id", ident) if ident else None


def survey_one(pid):
    started = time.time()
    rec = {"id": pid}
    try:
        model = load_model(model_path(pid))
        n_atoms = model.get_hierarchy().atoms_size()
        rec["n_atoms"] = int(n_atoms)
        if n_atoms > MAX_ATOMS:
            rec.update(status="skipped", reason=f"n_atoms {n_atoms} > {MAX_ATOMS}")
            rec["seconds"] = round(time.time() - started, 1)
            return rec

        extracted = extract_all(model, use_hydrogens=True, metrics=ALL_METRICS)
        events = molprobity_concern_events(extracted["events"])

        per_channel = defaultdict(lambda: {"n": 0, "n_depositing": 0, "n_outlier": 0})
        # residue -> channel -> worst concern there, for the co-occurrence question
        by_residue = defaultdict(dict)
        for e in events:
            st = per_channel[e.metric]
            st["n"] += 1
            if e.severity > 0:
                st["n_depositing"] += 1
            if e.severity >= 1.0:
                st["n_outlier"] += 1
            key = residue_key(e)
            if key is not None:
                cur = by_residue[key].get(e.metric, 0.0)
                by_residue[key][e.metric] = max(cur, float(e.severity))

        by_metric = {}
        for e in events:
            if e.severity > 0:
                by_metric.setdefault(e.metric, []).append(e)
        hot = {}
        if by_metric:
            fields = build_concern_fields(by_metric, spacing=SPACING, sigma=SIGMA)
            for metric, f in fields.items():
                hot[metric] = float((f.data >= 0.5).mean())

        rec.update(
            status="ok",
            channels={m: dict(v) for m, v in per_channel.items()},
            hot_fraction=hot,
            n_residues=len(by_residue),
            # Sparse residue x channel table, concern only where non-zero: this is what the
            # co-occurrence and correlation questions are computed from at report time.
            residue_concern=[{"r": list(k) if k[0] != "id" else ["id", str(k[1])],
                              "c": {m: round(v, 4) for m, v in ch.items() if v > 0}}
                             for k, ch in by_residue.items()
                             if any(v > 0 for v in ch.values())],
        )
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-1200:])
    rec["seconds"] = round(time.time() - started, 1)
    return rec


def report(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "survey*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no results in", out_dir)
        return
    print("%d structures: %d ok, %d failed, %d skipped" % (
        len(recs), len(ok), sum(1 for r in recs if r["status"] == "failed"),
        sum(1 for r in recs if r["status"] == "skipped")))
    secs = [r["seconds"] for r in ok]
    print("seconds/structure: median %.1f  mean %.1f  max %.1f\n" % (
        np.median(secs), np.mean(secs), max(secs)))

    print("%-9s %-10s %7s %8s %9s %9s %9s" % (
        "channel", "family", "fired", "ev/struct", "deposit%", "outlier%", "hot%"))
    for m in ALL_METRICS:
        rows = [r["channels"].get(m) for r in ok if r.get("channels", {}).get(m)]
        if not rows:
            print("%-9s %-10s %7s" % (m, FAMILY_OF.get(m, "?"), "never"))
            continue
        n = np.array([x["n"] for x in rows], dtype=float)
        dep = np.array([x["n_depositing"] for x in rows], dtype=float)
        out = np.array([x["n_outlier"] for x in rows], dtype=float)
        hotv = [r["hot_fraction"].get(m) for r in ok if r.get("hot_fraction", {}).get(m)]
        print("%-9s %-10s %6d%% %8.0f %8.1f%% %8.2f%% %8.2f%%" % (
            m, FAMILY_OF.get(m, "?"), 100 * len(rows) // len(ok), np.median(n),
            100 * dep.sum() / max(1, n.sum()), 100 * out.sum() / max(1, n.sum()),
            100 * np.median(hotv) if hotv else float("nan")))

    # Residue-level co-occurrence. Jaccard over residues each pair marks, and Spearman over
    # concern where either marks -- the first asks "same places?", the second "same amount?".
    marks = defaultdict(set)
    vals = defaultdict(dict)
    for r in ok:
        for row in r.get("residue_concern", []):
            key = (r["id"], tuple(row["r"]))
            for m, v in row["c"].items():
                marks[m].add(key)
                vals[m][key] = v
    present = [m for m in ALL_METRICS if len(marks[m]) >= 20]
    print("\nresidue-level co-occurrence (Jaccard above diagonal, Spearman rho below)")
    print("%-9s %s" % ("", " ".join("%8s" % m[:8] for m in present)))
    from scipy.stats import spearmanr
    for a in present:
        cells = []
        for b in present:
            if a == b:
                cells.append("%8s" % "-")
            elif ALL_METRICS.index(b) > ALL_METRICS.index(a):
                inter = len(marks[a] & marks[b])
                union = len(marks[a] | marks[b])
                cells.append("%8.3f" % (inter / union if union else 0.0))
            else:
                keys = sorted(marks[a] | marks[b])
                x = [vals[a].get(k, 0.0) for k in keys]
                y = [vals[b].get(k, 0.0) for k in keys]
                rho = spearmanr(x, y).correlation if len(keys) > 10 else float("nan")
                cells.append("%8.3f" % (rho if rho == rho else 0.0))
        print("%-9s %s   %s" % (a, " ".join(cells), FAMILY_OF.get(a, "?")))

    print("\nwithin-family vs across-family Jaccard (the grouping's actual claim):")
    within, across = [], []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            inter = len(marks[a] & marks[b])
            union = len(marks[a] | marks[b])
            j = inter / union if union else 0.0
            (within if FAMILY_OF.get(a) == FAMILY_OF.get(b) else across).append((j, a, b))
    if within:
        print("  within  n=%2d  median %.3f  max %.3f (%s)" % (
            len(within), np.median([x[0] for x in within]), max(within)[0],
            "/".join(max(within)[1:])))
    if across:
        print("  across  n=%2d  median %.3f  max %.3f (%s)" % (
            len(across), np.median([x[0] for x in across]), max(across)[0],
            "/".join(max(across)[1:])))
    if within and across:
        w, a = np.median([x[0] for x in within]), np.median([x[0] for x in across])
        print("  -> within/across ratio %.2f  %s" % (
            w / a if a else float("inf"),
            "grouping supported" if w > 2 * a else "GROUPING NOT SUPPORTED by co-occurrence"))


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
    path = os.path.join(args.out_dir, "survey%s.jsonl" % tag)

    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [i for i in ids if i not in done]
    print("%d structures%s, %d done -> %s" % (
        len(ids), " (shard %d/%d)" % shard if shard else "", len(done), path), flush=True)

    for n, pid in enumerate(todo, 1):
        rec = survey_one(pid)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        print("[%d/%d] %-5s %-8s %6.1fs %s" % (
            n, len(todo), pid, rec["status"], rec["seconds"],
            rec.get("error", rec.get("reason", ""))[:60]), flush=True)


if __name__ == "__main__":
    main()
