"""Is the hotspot field measuring trouble, or measuring how tightly packed the protein is?

A 6 Å ball in a buried core holds more atoms, therefore more validation events, therefore more
severity — from packing alone, with nothing wrong there. This is the same hazard as figure C's
co-localization, and it has to be measured before the density field is believed. Step 2 of
../HOTSPOT_DENSITY_DESIGN.md, and a gate on the rest.

The control is the identical kernel run over every heavy atom with weight 1, so the only
difference between the two fields is what is being summed.

Three questions, in increasing severity:

* **Correlation** between severity intensity and atom density inside the envelope. Some is
  expected and fine — problems can only occur where atoms are.
* **Are hot voxels merely dense voxels?** Compare the atom-density distribution inside the hot
  set against the envelope as a whole. If hot regions are just the core, this field is a
  packing map with extra steps.
* **Does normalizing by packing change what is hot?** Compute trouble-per-atom as well and
  report how much of the hot set survives. If almost all of it does, packing is not driving
  the result; if little does, the per-volume definition is wrong and the per-atom one should
  ship instead. Both are absolute scales, so this is a choice between two defensible
  definitions rather than a retreat to structure-relative normalization.

    libtbx.python corpus/packing_bias.py IDS.txt OUT_DIR --shard 0/4
    libtbx.python corpus/packing_bias.py IDS.txt OUT_DIR --report
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hotspots"))
sys.path.insert(0, HERE)

from concern import molprobity_concern_events  # noqa: E402
from density import DEFAULT_RADIUS, KNEE, atom_density, build_density_fields  # noqa: E402
from events import ALL_METRICS, extract_all, load_model  # noqa: E402
from figure_data import MAX_ATOMS, MAX_VOXELS, heavy_mask, model_path  # noqa: E402

SPACING = 1.0

#: A voxel is "inside the envelope" if the packing control is at least this. Excludes the
#: empty box around the model, which would otherwise dominate every correlation with a huge
#: mass of (0 trouble, 0 atoms) voxels and manufacture a correlation that means nothing.
ENVELOPE_MIN_ATOMS = 1.0


def run_one(pdb_id, spacing=SPACING, radius=DEFAULT_RADIUS):
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
        by_metric = {}
        for e in events:
            if e.severity > 0 and e.atoms_xyz:
                by_metric.setdefault(e.metric, []).append(e)
        if not by_metric:
            rec.update(status="empty", reason="no depositing events")
            return _done(rec, started)

        fields = build_density_fields(by_metric, spacing=spacing, radius=radius)
        if "combined" not in fields:
            rec.update(status="empty", reason="no density")
            return _done(rec, started)
        dens = fields["combined"]
        if dens.data.size > MAX_VOXELS:
            rec.update(status="skipped", reason="grid %d voxels" % dens.data.size)
            return _done(rec, started)

        sites = np.asarray(hierarchy.atoms().extract_xyz()).reshape(-1, 3)
        heavy = heavy_mask(hierarchy)
        packing = atom_density(sites[heavy], spacing=spacing, radius=radius,
                               origin=dens.origin, shape=dens.data.shape)

        env = packing >= ENVELOPE_MIN_ATOMS
        if not env.any():
            rec.update(status="empty", reason="no envelope")
            return _done(rec, started)
        d = dens.data[env]
        a = packing[env]
        hot = d >= KNEE

        from scipy.stats import spearmanr
        rho = spearmanr(d, a).correlation if d.size > 10 else float("nan")

        # trouble per atom: the alternative absolute definition
        per_atom = np.zeros_like(d)
        np.divide(d, a, out=per_atom, where=a > 0)
        # scale so its knee means the same thing: one outlier's worth per typical packing
        med_pack = float(np.median(a))
        per_atom_scaled = per_atom * med_pack
        hot_pa = per_atom_scaled >= KNEE

        # Does the per-atom definition actually decorrelate from packing? Adopting it as the
        # fallback without checking would just be swapping one unexamined normalization for
        # another.
        rho_pa = (spearmanr(per_atom_scaled, a).correlation
                  if per_atom_scaled.size > 10 else float("nan"))

        rec.update(
            status="ok",
            n_env=int(env.sum()),
            spearman_density_vs_packing=float(rho) if rho == rho else None,
            spearman_peratom_vs_packing=float(rho_pa) if rho_pa == rho_pa else None,
            peratom_median_env=float(np.median(per_atom_scaled)),
            peratom_p95_env=float(np.percentile(per_atom_scaled, 95)),
            peratom_p99_env=float(np.percentile(per_atom_scaled, 99)),
            density_p95_env=float(np.percentile(d, 95)),
            hot_fraction=float(hot.mean()),
            hot_fraction_per_atom=float(hot_pa.mean()),
            # is the hot set simply the dense set?
            packing_median_all=med_pack,
            packing_median_hot=float(np.median(a[hot])) if hot.any() else None,
            packing_p90_all=float(np.percentile(a, 90)),
            # how much of the hot set survives normalizing packing away
            overlap_hot_and_peratom=float((hot & hot_pa).sum() / max(1, hot.sum())),
            density_median_env=float(np.median(d)),
            density_p99_env=float(np.percentile(d, 99)),
            density_max=float(d.max()),
        )
    except Exception as exc:
        rec.update(status="failed", error="%s: %s" % (type(exc).__name__, exc),
                   traceback=traceback.format_exc()[-1000:])
    return _done(rec, started)


def _done(rec, started):
    rec["seconds"] = round(time.time() - started, 1)
    return rec


def report(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "packing*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no results in", out_dir)
        return
    print("%d structures: %d ok, %d failed, %d skipped\n" % (
        len(recs), len(ok), sum(1 for r in recs if r["status"] == "failed"),
        sum(1 for r in recs if r["status"] not in ("ok", "failed"))))

    rho = np.array([r["spearman_density_vs_packing"] for r in ok
                    if r.get("spearman_density_vs_packing") is not None])
    print("1. CORRELATION of severity intensity with atom density, inside the envelope")
    print("   spearman rho: median %.3f   p10 %.3f   p90 %.3f" % (
        np.median(rho), np.percentile(rho, 10), np.percentile(rho, 90)))

    pm_all = np.array([r["packing_median_all"] for r in ok])
    pm_hot = np.array([r["packing_median_hot"] for r in ok
                       if r.get("packing_median_hot") is not None])
    ratio = np.array([r["packing_median_hot"] / r["packing_median_all"] for r in ok
                      if r.get("packing_median_hot") and r["packing_median_all"] > 0])
    print("\n2. ARE HOT VOXELS MERELY DENSE VOXELS?")
    print("   atom density, envelope median : %.1f" % np.median(pm_all))
    print("   atom density, hot-set  median : %.1f" % (np.median(pm_hot) if pm_hot.size else float('nan')))
    if ratio.size:
        print("   ratio hot/envelope            : median %.2fx   p90 %.2fx" % (
            np.median(ratio), np.percentile(ratio, 90)))
        verdict = ("packing is NOT driving the hot set" if np.median(ratio) < 1.25
                   else "PACKING BIAS: hot regions are systematically denser")
        print("   -> %s" % verdict)

    rho_pa = np.array([r["spearman_peratom_vs_packing"] for r in ok
                       if r.get("spearman_peratom_vs_packing") is not None])
    if rho_pa.size:
        print("\n   after per-atom normalization, spearman rho: median %.3f" % np.median(rho_pa))
        print("   -> %s" % ("per-atom REMOVES the packing dependence"
                            if abs(np.median(rho_pa)) < 0.3 else
                            "per-atom does NOT remove it; the confound is not packing alone"))

    ov = np.array([r["overlap_hot_and_peratom"] for r in ok])
    hf = np.array([100 * r["hot_fraction"] for r in ok])
    hfp = np.array([100 * r["hot_fraction_per_atom"] for r in ok])
    print("\n3. DOES NORMALIZING PACKING AWAY CHANGE WHAT IS HOT?")
    print("   hot volume, per-volume definition : median %.2f%% of envelope" % np.median(hf))
    print("   hot volume, per-atom  definition  : median %.2f%% of envelope" % np.median(hfp))
    print("   share of per-volume hot set that stays hot per-atom: median %.1f%%" % (
        100 * np.median(ov)))

    dm = np.array([r["density_median_env"] for r in ok])
    d99 = np.array([r["density_p99_env"] for r in ok])
    dmx = np.array([r["density_max"] for r in ok])
    d95 = np.array([r.get("density_p95_env", np.nan) for r in ok])
    print("\n4. WHERE SHOULD THE KNEE ACTUALLY SIT?")
    print("   density (outlier-equivalents) inside envelope:")
    print("     median %.2f   p95 %.2f   p99 %.2f   max %.2f   (current knee %.1f)" % (
        np.median(dm), np.nanmedian(d95), np.median(d99), np.median(dmx), KNEE))
    pam = np.array([r.get("peratom_median_env", np.nan) for r in ok])
    pa95 = np.array([r.get("peratom_p95_env", np.nan) for r in ok])
    pa99 = np.array([r.get("peratom_p99_env", np.nan) for r in ok])
    print("   per-atom (rescaled to median packing):")
    print("     median %.2f   p95 %.2f   p99 %.2f" % (
        np.nanmedian(pam), np.nanmedian(pa95), np.nanmedian(pa99)))
    print("   a knee at the envelope median marks half the protein; it belongs near p95-p99.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids")
    ap.add_argument("out_dir")
    ap.add_argument("--shard", metavar="K/N")
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
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
    path = os.path.join(args.out_dir, "packing%s.jsonl" % tag)
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [i for i in ids if i not in done]
    print("%d structures%s -> %s" % (
        len(ids), " (shard %d/%d)" % shard if shard else "", path), flush=True)

    for n, pid in enumerate(todo, 1):
        rec = run_one(pid, radius=args.radius)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        note = ("rho=%.2f hot=%.2f%%" % (rec.get("spearman_density_vs_packing") or -9,
                                         100 * (rec.get("hot_fraction") or 0))
                if rec["status"] == "ok" else rec.get("error", rec.get("reason", ""))[:50])
        print("[%d/%d] %-5s %-8s %6.1fs  %s" % (
            n, len(todo), pid, rec["status"], rec["seconds"], note), flush=True)


if __name__ == "__main__":
    main()
