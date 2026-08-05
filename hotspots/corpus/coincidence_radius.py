"""How close do two sub-threshold concerns have to be to make a hotspot — and are they?

The accumulation hypothesis failed (see ../AGGREGATION_PROPOSAL.md). This asks *why*, in a
form that distinguishes the two possible causes, because they have opposite implications:

* the **kernel is too narrow** — the concerns are near each other but sigma = 2 A is not wide
  enough to make their splats overlap above the display threshold. Fixable by widening sigma.
* the **events are too far apart** — no kernel of a defensible width would bring them
  together. Not fixable, and the end of the idea.

Two halves, one analytic and one measured:

**Required separation.** Two peak-normalized Gaussians of concern s1 and s2 at distance d sum
to at most ``max_x [s1 exp(-x^2/2 sigma^2) + s2 exp(-(d-x)^2/2 sigma^2)]`` along the line
between them. The largest d at which that reaches the 0.5 display threshold is the coincidence
radius for that pair. Computed numerically rather than in closed form, because the maximum
moves from the midpoint to the individual peaks as d passes 2 sigma and the closed forms differ
on either side of that.

**Observed separation.** For every depositing sub-threshold event, the distance to the nearest
depositing event of a *different family* — since same-family events combine by max and cannot
accumulate by construction.

Then the exchange rate: what sigma would be needed to bring a given fraction of real pairs into
reach, and what that sigma costs figure B, whose whole claim is that hot voxels sit within
about a residue of something concerning.

    libtbx.python corpus/coincidence_radius.py IDS.txt [--limit 20]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hotspots"))

from concern import FAMILIES, molprobity_concern_events  # noqa: E402
from events import ALL_METRICS, extract_all, load_model  # noqa: E402

MIRROR = "/root/data/pdb_mmcif"
HOT = 0.5
SIGMA = 2.0
FAMILY_OF = {m: f for f, ms in FAMILIES.items() for m in ms}


def peak_of_pair(s1, s2, d, sigma, n=257):
    """Max of two summed peak-normalized Gaussians separated by d, along their axis."""
    x = np.linspace(-sigma, d + sigma, n)
    return float(np.max(s1 * np.exp(-x ** 2 / (2 * sigma ** 2))
                        + s2 * np.exp(-(d - x) ** 2 / (2 * sigma ** 2))))


def coincidence_radius(s1, s2, sigma, hot=HOT, hi=25.0):
    """Largest separation at which this pair can still reach ``hot``. 0 if never."""
    if s1 + s2 < hot:
        return 0.0                      # cannot reach the threshold even coincident
    if peak_of_pair(s1, s2, 0.0, sigma) < hot:
        return 0.0
    lo, hi = 0.0, hi
    for _ in range(40):                 # bisection; the peak is monotone decreasing in d
        mid = 0.5 * (lo + hi)
        if peak_of_pair(s1, s2, mid, sigma) >= hot:
            lo = mid
        else:
            hi = mid
    return lo


def collect(pids, sub_max=HOT):
    """(concern, xyz, family) for every depositing event below ``sub_max``."""
    rows = []
    for pid in pids:
        path = os.path.join(MIRROR, pid[1:3], pid + ".cif.gz")
        try:
            model = load_model(path)
            if model.get_hierarchy().atoms_size() > 50000:
                continue
            events = molprobity_concern_events(
                extract_all(model, use_hydrogens=True, metrics=ALL_METRICS)["events"])
        except Exception as exc:
            print("  %s failed: %s" % (pid, str(exc)[:60]), flush=True)
            continue
        for e in events:
            if not (0 < e.severity < sub_max) or not e.atoms_xyz:
                continue
            xyz = np.asarray(e.atoms_xyz, float).reshape(-1, 3).mean(axis=0)
            rows.append((pid, float(e.severity), xyz, FAMILY_OF.get(e.metric, e.metric)))
        print("  %s: %d sub-threshold depositing events" % (
            pid, sum(1 for r in rows if r[0] == pid)), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    args = ap.parse_args()
    pids = [l.strip() for l in open(args.ids) if l.strip()][: args.limit]

    print("REQUIRED SEPARATION at sigma = %.1f A (equal concerns)" % args.sigma)
    print("  %-10s %s" % ("concern", "coincidence radius"))
    for s in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.49):
        r = coincidence_radius(s, s, args.sigma)
        print("  %-10.2f %s" % (s, "never" if r <= 0 else "%.2f A" % r))

    print("\ncollecting observed events from %d structures..." % len(pids), flush=True)
    rows = collect(pids)
    if not rows:
        print("no events")
        return
    print("\n%d sub-threshold depositing events total" % len(rows))

    # nearest cross-family neighbour, per structure
    recs = []
    for pid in {r[0] for r in rows}:
        sub = [r for r in rows if r[0] == pid]
        xyz = np.array([r[2] for r in sub])
        fam = np.array([r[3] for r in sub])
        con = np.array([r[1] for r in sub])
        tree = cKDTree(xyz)
        # k large enough to escape same-family neighbours; capped at the set size
        k = min(len(sub), 40)
        dists, idx = tree.query(xyz, k=k)
        for i in range(len(sub)):
            for j_pos in range(1, k):
                j = idx[i, j_pos]
                if fam[j] != fam[i]:
                    recs.append((con[i], con[j], float(dists[i, j_pos])))
                    break
    if not recs:
        print("no cross-family pairs")
        return
    d = np.array([r[2] for r in recs])
    s1 = np.array([r[0] for r in recs])
    s2 = np.array([r[1] for r in recs])
    print("\nOBSERVED nearest cross-family separation (n = %d pairs)" % len(d))
    print("  p10 %.2f  p25 %.2f  median %.2f  p75 %.2f  p90 %.2f A" % tuple(
        np.percentile(d, [10, 25, 50, 75, 90])))
    print("  pair concerns: median %.3f and %.3f" % (np.median(s1), np.median(s2)))

    print("\nFRACTION OF REAL PAIRS THAT COULD ACCUMULATE, by sigma")
    print("  %-8s %10s %14s %s" % ("sigma", "reachable", "median radius", "cost to figure B"))
    for sigma in (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        radii = np.array([coincidence_radius(a, b, sigma) for a, b in zip(s1, s2)])
        frac = float((d <= radii).mean())
        med_r = float(np.median(radii))
        # Figure B's hot-voxel spread scales with the kernel: the half-maximum radius of a
        # lone splat is sigma*sqrt(2 ln 2), which is what sets the "within about a residue"
        # claim. Reported so the trade is explicit rather than implied.
        print("  %-8.1f %9.1f%% %13.2f A   half-max radius %.2f A" % (
            sigma, 100 * frac, med_r, sigma * np.sqrt(2 * np.log(2))))

    print("\nwhy pairs fail, at sigma = %.1f" % args.sigma)
    radii = np.array([coincidence_radius(a, b, args.sigma) for a, b in zip(s1, s2)])
    too_weak = radii <= 0
    too_far = (~too_weak) & (d > radii)
    print("  concerns too weak to reach 0.5 even coincident : %5.1f%%" % (100 * too_weak.mean()))
    print("  strong enough but too far apart                : %5.1f%%" % (100 * too_far.mean()))
    print("  actually within reach                          : %5.1f%%" % (
        100 * (~too_weak & (d <= radii)).mean()))
    if too_weak.mean() > 0.5:
        print("\n  -> dominated by WEAKNESS, not distance: widening sigma cannot help,")
        print("     because these pairs cannot reach the threshold at zero separation.")
    elif too_far.mean() > 0.5:
        print("\n  -> dominated by DISTANCE: a wider kernel would reach them, at the")
        print("     cost to figure B shown above.")


if __name__ == "__main__":
    main()
