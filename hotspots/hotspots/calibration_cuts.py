"""The calibration invariant, and the check that keeps it honest.

**Every channel's community outlier cut must evaluate to concern 1.0.**

That single property is what lets channels be combined without inventing weights: if a
flagged rotamer, a flagged clash and a flagged bond all reach exactly 1.0, then treating them
as equally noteworthy is a *display convention* over thresholds other people defined, not a
judgement about which is physically worse. Break it for one channel and that channel is
silently down-weighted everywhere -- which is exactly what had happened (clash and cbeta at
0.5, bond and angle at 0.0) before the corpus run measured it.

Run it:

    libtbx.python hotspots/calibration_cuts.py           # assert the invariant
    libtbx.python hotspots/calibration_cuts.py MODEL     # ... and measure swamping on a model

The second form matters for the deviation channels. Concern *sums* within a metric before it
saturates, so lowering a channel's zero anchor to give it a sub-threshold tail also risks
thousands of mild events overlapping into a field that marks the whole protein. That failure
is invisible in the per-channel cut table and obvious in the hot-volume table, so measure it
rather than reasoning about it -- it has bitten this calibration before (see
GEOMETRY_ZERO_SIGMA in concern.py).
"""
from __future__ import annotations

import sys

import concern as C

#: (channel, description of the community cut, value at that cut, concern function).
#: Each entry names *somebody else's* threshold. Nothing here is our own number.
CUTS = [
    ("rama", "MolProbity general Ramachandran outlier, p = 0.05%", 0.05,
     lambda v: C.log_low_concern(v, C.RAMA_FAVORED_PCT, 0.05)),
    ("rama/cis-Pro", "MolProbity cis-proline outlier, p = 0.20%", 0.20,
     lambda v: C.log_low_concern(v, C.RAMA_FAVORED_PCT, C.RAMA_OUTLIER_PCT[2])),
    ("rota", "MolProbity rotamer outlier, p = 0.30%", 0.30,
     lambda v: C.log_low_concern(v, C.ROTAMER_FAVORED_PCT, C.ROTAMER_OUTLIER_PCT)),
    ("clash", "MolProbity clash, overlap = 0.40 A", 0.40,
     lambda v: C.linear_concern(v, C.CLASH_ZERO_OVERLAP_A, C.CLASH_SATURATION_OVERLAP_A)),
    ("cablam", "CaBLAM outlier contour = 0.01", 0.01,
     lambda v: C.log_low_concern(v, C.CABLAM_FAVORED, C.CABLAM_OUTLIER)),
    ("ca_geom", "CA-geometry outlier contour = 0.005", 0.005,
     lambda v: C.log_low_concern(v, C.CA_GEOM_FAVORED, C.CA_GEOM_OUTLIER)),
    ("cbeta", "MolProbity C-beta deviation = 0.25 A", 0.25,
     lambda v: C.linear_concern(v, C.CBETA_ZERO_A, C.CBETA_SATURATION_A)),
    ("bond", "restraint deviation |Z| = 4 sigma", 4.0,
     lambda v: C.linear_concern(v, C.GEOMETRY_ZERO_SIGMA, C.GEOMETRY_SATURATION_SIGMA)),
    ("angle", "restraint deviation |Z| = 4 sigma", 4.0,
     lambda v: C.linear_concern(v, C.GEOMETRY_ZERO_SIGMA, C.GEOMETRY_SATURATION_SIGMA)),
    ("omega/twisted", "omegalyze twisted peptide, 30 degrees", 30.0,
     lambda v: C._omega_concern({"kind": "twisted", "twist": v})),
    ("omega/cis non-Pro", "omegalyze cis non-proline", None,
     lambda _v: C._omega_concern({"kind": "cis", "is_proline": False})),
]

TOLERANCE = 1e-9


def calibration_cuts():
    """``[(channel, cut description, concern at the cut)]`` for every channel."""
    return [(name, why, fn(value)) for name, why, value, fn in CUTS]


def check(verbose=True):
    """Assert the invariant. Returns the offenders, empty when it holds."""
    bad = []
    if verbose:
        print("%-18s %-46s %s" % ("channel", "community cut", "concern"))
    for name, why, got in calibration_cuts():
        ok = abs(got - 1.0) <= TOLERANCE
        if not ok:
            bad.append((name, got))
        if verbose:
            print("%-18s %-46s %6.3f %s" % (name, why, got, "" if ok else "  <-- WRONG"))
    return bad


def swamping(model_path, spacing=1.0, sigma=2.0):
    """Hot-volume per channel on one model — the check the cut table cannot make.

    A channel whose cut is correctly at 1.0 can still be miscalibrated in the other
    direction: give it a sub-threshold tail that is too generous and thousands of mild events
    overlap into a field marking the whole protein. Reported as the fraction of the box past
    the display threshold, per channel, so a channel that has swallowed the map is obvious.
    """
    import numpy as np

    from concern import build_concern_fields, molprobity_concern_events
    from events import ALL_METRICS, extract_all, load_model

    model = load_model(model_path)
    extracted = extract_all(model, use_hydrogens=True, metrics=ALL_METRICS)
    events = molprobity_concern_events(extracted["events"])
    by_metric = {}
    for e in events:
        if e.severity > 0:
            by_metric.setdefault(e.metric, []).append(e)
    if not by_metric:
        print("no concern events")
        return {}
    fields = build_concern_fields(by_metric, spacing=spacing, sigma=sigma)

    n_all = {}
    for e in events:
        n_all[e.metric] = n_all.get(e.metric, 0) + 1
    print("\n%-10s %8s %8s %10s %8s" % ("channel", "events", "depos", "hot voxels", "% box"))
    out = {}
    for metric in sorted(fields):
        data = fields[metric].data
        hot = int((data >= 0.5).sum())
        frac = 100.0 * hot / data.size
        out[metric] = frac
        print("%-10s %8s %8d %10d %7.2f%%" % (
            metric, n_all.get(metric, "-"), len(by_metric.get(metric, [])), hot, frac))
    return out


def main():
    bad = check()
    if bad:
        print("\nINVARIANT BROKEN for: %s" % ", ".join(n for n, _ in bad))
    else:
        print("\nOK: every community cut evaluates to concern 1.0")
    if len(sys.argv) > 1:
        swamping(sys.argv[1])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
