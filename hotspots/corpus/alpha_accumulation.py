"""Does faint concern composite into something you can see? The one untested claim.

Every accumulation test in this project asked **does a voxel cross the display threshold**.
A volume render never asks that. It integrates opacity along a view ray:

    alpha = 1 - exp(-k * integral of concern along the ray)

so ten faint concerns that individually reach 0.2 and never cross 0.5 can still composite into
a visibly warm patch. That is the field's remaining justification over MolProbity markers, and
it is the only claim about the field that has never been measured. Measuring it in *field*
space was the wrong test; this measures it in image space.

**The asymmetry that makes it a fair test of markers too.** Markers exist only where a
validator flagged something. So along any ray that never crosses the display threshold, a
marker representation shows *exactly nothing* — not "less", nothing. The question is only
whether the field shows something there, and whether that something is above a stated
visibility floor.

Method, per structure and per view direction:

* project the concern field along the ray axis to get the line integral ``P`` (units of
  concern-angstrom) and the per-ray maximum ``M``;
* keep rays inside the molecular envelope;
* split them: ``M >= 0.5`` (a marker would be there anyway) versus ``M < 0.5``, the
  sub-threshold rays where markers show nothing;
* convert P to alpha and ask what fraction of sub-threshold rays clear a visibility floor,
  against a background of near-empty envelope rays.

``k`` is fixed so that a lone flagged outlier reads alpha = 0.6 — a sane translucent overlay —
rather than tuned to make the answer come out well. Reported for a range of k regardless, since
the conclusion should not hinge on it.

    libtbx.python corpus/alpha_accumulation.py IDS.txt OUT_DIR --shard 0/4
    libtbx.python corpus/alpha_accumulation.py IDS.txt OUT_DIR --report
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

from concern import build_concern_fields, molprobity_concern_events  # noqa: E402
from events import ALL_METRICS, extract_all, load_model  # noqa: E402
from figure_data import MAX_ATOMS, MAX_VOXELS, SIGMA, SPACING, model_path  # noqa: E402

HOT = 0.5

#: Extinction per unit concern-angstrom, fixed so a lone flagged outlier reads alpha = 0.6.
#: A concern-1.0 event with sigma = 2 has a line integral through its peak of
#: 1.0 * sqrt(2*pi) * 2 = 5.01 concern-A, so k = -ln(0.4)/5.01.
K_DEFAULT = 0.183
K_SWEEP = (0.09, 0.183, 0.37)

#: A ray is "inside the envelope" if it passes through any voxel with this much concern, or
#: through the model at all -- taken from the projected atom mask rather than from concern, so
#: an entirely clean region still counts as envelope and can serve as the background.
#:
#: Visibility floor: alpha 0.05 against the background. Deliberately conservative -- large-area
#: luminance JND is nearer 1%, so 5% is several times the threshold at which a difference
#: becomes noticeable, and the claim should not rest on a marginal one.
VISIBLE_ALPHA = 0.05


def alpha_of(line_integral, k=K_DEFAULT):
    return 1.0 - np.exp(-k * np.asarray(line_integral, dtype=float))


def run_one(pdb_id, spacing=SPACING, sigma=SIGMA):
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
        fields = build_concern_fields(by_metric, spacing=spacing, sigma=sigma)
        c = fields["combined"].data
        if c.size > MAX_VOXELS:
            rec.update(status="skipped", reason="grid %d voxels" % c.size)
            return _done(rec, started)

        # Envelope in projection: rays that pass through the model at all. Built from the
        # atoms rather than from concern, so a genuinely clean region still counts and can
        # act as the background a warm patch would be seen against.
        sites = np.asarray(hierarchy.atoms().extract_xyz()).reshape(-1, 3)
        origin = np.asarray(fields["combined"].origin, float)
        idx = np.round((sites - origin) / spacing).astype(int)
        occ = np.zeros(c.shape, dtype=bool)
        ok = np.all((idx >= 0) & (idx < np.array(c.shape)), axis=1)
        occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]] = True

        views = {}
        for axis in (0, 1, 2):
            P = c.sum(axis=axis) * spacing          # line integral, concern-angstrom
            M = c.max(axis=axis)                    # does the ray ever cross the threshold
            E = occ.any(axis=axis)                  # inside the envelope
            sub = E & (M < HOT)                     # markers show NOTHING along these rays
            marked = E & (M >= HOT)
            empty = E & (M < 0.05)                  # near-clean envelope: the background
            if not sub.any() or not empty.any():
                continue
            bg = float(np.median(P[empty])) if empty.any() else 0.0
            entry = {
                "n_env": int(E.sum()),
                "n_sub": int(sub.sum()),
                "n_marked": int(marked.sum()),
                "P_sub_median": float(np.median(P[sub])),
                "P_sub_p90": float(np.percentile(P[sub], 90)),
                "P_bg_median": bg,
                "P_marked_median": float(np.median(P[marked])) if marked.any() else None,
            }
            for k in K_SWEEP:
                # Contrast against the local background, which is what the eye judges:
                # alpha of the ray minus alpha of a clean ray through the same envelope.
                da = alpha_of(P[sub], k) - alpha_of(bg, k)
                entry["k%.3f_dalpha_median" % k] = float(np.median(da))
                entry["k%.3f_dalpha_p90" % k] = float(np.percentile(da, 90))
                entry["k%.3f_frac_visible" % k] = float((da >= VISIBLE_ALPHA).mean())
            views["axis%d" % axis] = entry
        if not views:
            rec.update(status="empty", reason="no usable views")
            return _done(rec, started)
        rec.update(status="ok", views=views)
    except Exception as exc:
        rec.update(status="failed", error="%s: %s" % (type(exc).__name__, exc),
                   traceback=traceback.format_exc()[-1000:])
    return _done(rec, started)


def _done(rec, started):
    rec["seconds"] = round(time.time() - started, 1)
    return rec


def report(out_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "alpha*.jsonl"))):
        with open(p) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no results in", out_dir)
        return
    rows = [v for r in ok for v in r["views"].values()]
    print("%d structures, %d views\n" % (len(ok), len(rows)))

    n_sub = np.array([v["n_sub"] for v in rows], float)
    n_env = np.array([v["n_env"] for v in rows], float)
    print("RAYS WHERE A MARKER SHOWS NOTHING (never crosses the display threshold)")
    print("  share of envelope rays: median %.1f%%" % (100 * np.median(n_sub / n_env)))

    ps = np.array([v["P_sub_median"] for v in rows])
    pb = np.array([v["P_bg_median"] for v in rows])
    pm = np.array([v["P_marked_median"] for v in rows if v.get("P_marked_median")])
    print("\nLINE INTEGRAL of concern (concern-angstrom)")
    print("  clean envelope ray   : median %.2f" % np.median(pb))
    print("  sub-threshold ray    : median %.2f" % np.median(ps))
    print("  ray crossing threshold: median %.2f" % (np.median(pm) if pm.size else float("nan")))

    print("\nCONTRAST AGAINST BACKGROUND, as accumulated alpha")
    print("  %-8s %14s %12s %16s" % ("k", "median dalpha", "p90 dalpha", "rays >= 0.05"))
    for k in K_SWEEP:
        dm = np.array([v["k%.3f_dalpha_median" % k] for v in rows])
        dp = np.array([v["k%.3f_dalpha_p90" % k] for v in rows])
        fv = np.array([v["k%.3f_frac_visible" % k] for v in rows])
        tag = "  (calibrated)" if abs(k - K_DEFAULT) < 1e-9 else ""
        print("  %-8.3f %14.4f %12.4f %15.1f%%%s" % (
            k, np.median(dm), np.median(dp), 100 * np.median(fv), tag))

    dm = np.array([v["k%.3f_dalpha_median" % K_DEFAULT] for v in rows])
    fv = np.array([v["k%.3f_frac_visible" % K_DEFAULT] for v in rows])
    print("\nVERDICT at the calibrated k = %.3f:" % K_DEFAULT)
    print("  the median sub-threshold ray sits %.3f alpha above background" % np.median(dm))
    print("  %.1f%% of them clear the %.2f visibility floor" % (
        100 * np.median(fv), VISIBLE_ALPHA))
    if np.median(dm) >= VISIBLE_ALPHA:
        print("  -> VISIBLE. Faint concern composites into something a marker cannot show,")
        print("     and the median such ray is above the floor, not just the tail.")
    elif np.median(fv) > 0.25:
        print("  -> PARTLY VISIBLE. The median ray is below the floor but a substantial")
        print("     minority clear it; the claim holds for some regions, not generally.")
    else:
        print("  -> NOT VISIBLE. Faint concern does not composite into a discriminable")
        print("     signal, and the field has no advantage over markers on this argument.")


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
    path = os.path.join(args.out_dir, "alpha%s.jsonl" % tag)
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [i for i in ids if i not in done]
    print("%d structures%s -> %s" % (
        len(ids), " (shard %d/%d)" % shard if shard else "", path), flush=True)

    for n, pid in enumerate(todo, 1):
        rec = run_one(pid)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        note = ""
        if rec["status"] == "ok":
            v = next(iter(rec["views"].values()))
            note = "dalpha med %.3f  visible %.0f%%" % (
                v["k%.3f_dalpha_median" % K_DEFAULT],
                100 * v["k%.3f_frac_visible" % K_DEFAULT])
        else:
            note = rec.get("error", rec.get("reason", ""))[:50]
        print("[%d/%d] %-5s %-8s %6.1fs  %s" % (
            n, len(todo), pid, rec["status"], rec["seconds"], note), flush=True)


if __name__ == "__main__":
    main()
