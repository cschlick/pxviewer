"""Positive control: can the through-space measurement see a spatial halo that is really there?

Every through-space result in this project is a *negative* or near-negative -- rotamer flat at
1.08, the whole off-diagonal at 0.8-1.2. A negative is only worth reporting if the instrument
would have reported a positive had one existed, and nothing here established that. Clash was
the obvious biological control and it does not work: rolled up per residue at MolProbity's
0.40 A contact cut it flags 27% of residues (p90 66%), so the design's precondition -- that
the flagged set is sparse enough to leave a far field -- fails, and its concern saturates at
the cut so the achievable ratio is capped near 3x.

So plant a halo instead. Take a real structure's CA coordinates, choose centres at random,
give every residue background concern, then add a known amplitude to residues within a known
radius *that are also sequence-far from their centre*. Run the identical profile and control
used on the real data. If the recovered curve peaks at the planted radius with roughly the
planted amplitude, the machinery can see through-space co-location and the real negatives
mean what they say. If it does not, every through-space number in this project is void.

This is a test of the instrument, not of biology: the null is the same randomised-centre null,
and the only thing that changes is that the signal is known.

    libtbx.python corpus/synthetic_control.py IDS.txt --n 40
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hotspots"))
sys.path.insert(0, HERE)

from outlier_neighbourhood import (  # noqa: E402
    CONTROL_TRIALS, DIST_EDGES, MIN_SEQ_SEP, _dist_profile_seqfar, _residue_ca,
)
from events import load_model  # noqa: E402
from figure_data import MAX_ATOMS, model_path  # noqa: E402

#: Planted halo. The radius sits in the 4-6 A bin because that is where the real data puts its
#: only surviving through-space signal, so this asks whether the instrument is sensitive
#: exactly where the claim is made. The amplitude is deliberately modest -- the real effect is
#: 1.5-1.6x, so a control planted at 5x would prove only that the instrument sees loud things.
PLANT_RADIUS_A = 5.0
PLANT_AMPLITUDE = 0.15
BACKGROUND = 0.30
BACKGROUND_JITTER = 0.10

#: Fraction of residues used as planted centres, matching the real channels' outlier
#: prevalence (rama 2.2%, cablam 1.9%, rota 3.6%).
CENTRE_FRACTION = 0.025


def plant(ca, rng):
    """Return (concern_by_residue, centres) with a known sequence-far spatial halo."""
    keys = list(ca)
    xyz = np.array([ca[k] for k in keys])
    n_centres = max(3, int(round(CENTRE_FRACTION * len(keys))))
    idx = rng.choice(len(keys), size=min(n_centres, len(keys)), replace=False)
    centres = [keys[i] for i in idx]

    conc = {k: max(0.0, BACKGROUND + BACKGROUND_JITTER * rng.standard_normal())
            for k in keys}
    kres = np.array([k[1] for k in keys], dtype=float)
    chain_id = {}
    for k in keys:
        chain_id.setdefault(k[0], len(chain_id))
    kchain = np.array([chain_id[k[0]] for k in keys])

    for c in centres:
        d = np.linalg.norm(xyz - np.asarray(ca[c], float), axis=1)
        # Plant ONLY on sequence-far residues. A halo planted on chain neighbours too would
        # be recovered by the plain distance profile and prove nothing about the seq-far path,
        # which is the one every through-space claim rests on.
        far = ~((kchain == chain_id[c[0]]) & (np.abs(kres - c[1]) < MIN_SEQ_SEP))
        hit = (d <= PLANT_RADIUS_A) & far
        for i in np.nonzero(hit)[0]:
            conc[keys[i]] += PLANT_AMPLITUDE
    for c in centres:
        conc[c] = 1.0        # centres are the "outliers"; they are excluded from the profile
    return conc, centres


def run_one(pdb_id, seed=0):
    model = load_model(model_path(pdb_id))
    hierarchy = model.get_hierarchy()
    if hierarchy.atoms_size() > MAX_ATOMS:
        return None
    ca = _residue_ca(hierarchy)
    if len(ca) < 60:
        return None
    rng = np.random.default_rng(seed)
    conc, centres = plant(ca, rng)
    excl = set(centres)

    obs = _dist_profile_seqfar(conc, ca, centres, excl)
    pool = [k for k in ca if k not in excl]
    ctl = {i: [] for i in range(len(DIST_EDGES) - 1)}
    for _ in range(CONTROL_TRIALS):
        pick = [pool[i] for i in rng.integers(0, len(pool), size=len(centres))]
        prof = _dist_profile_seqfar(conc, ca, pick, excl)
        for i, (m, _n) in prof.items():
            if m is not None:
                ctl[i].append(m)
    return {
        "id": pdb_id,
        "n_res": len(ca),
        "n_centres": len(centres),
        "obs": {i: obs[i][0] for i in obs},
        "n": {i: obs[i][1] for i in obs},
        "ctl": {i: (float(np.mean(v)) if v else None) for i, v in ctl.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.ids) if l.strip()][:args.n]
    recs = []
    for i, pid in enumerate(ids, 1):
        try:
            r = run_one(pid, seed=i)
            if r:
                recs.append(r)
                print("[%d/%d] %s  %d res, %d centres" % (
                    i, len(ids), pid, r["n_res"], r["n_centres"]), flush=True)
        except Exception as exc:
            print("[%d/%d] %s  FAILED %s" % (i, len(ids), pid, type(exc).__name__), flush=True)

    if not recs:
        print("nothing measured")
        return

    print("\nPLANTED: +%.2f concern within %.0f A of a centre, sequence-far only,"
          % (PLANT_AMPLITUDE, PLANT_RADIUS_A))
    print("         on a background of %.2f. Centres are %.1f%% of residues.\n"
          % (BACKGROUND, 100 * CENTRE_FRACTION))
    expected = (BACKGROUND + PLANT_AMPLITUDE) / BACKGROUND
    print("%-10s %s" % ("dist(A):", "  ".join(
        "%6.0f" % DIST_EDGES[i + 1] for i in range(6))))
    row_o, row_n = [], []
    for i in range(6):
        o = [r["obs"][i] for r in recs if r["obs"].get(i) is not None]
        c = [r["ctl"][i] for r in recs if r["ctl"].get(i) is not None]
        row_o.append(np.mean(o) / np.mean(c) if o and c and np.mean(c) > 0 else np.nan)
        row_n.append(sum(r["n"].get(i, 0) for r in recs))
    print("%-10s %s" % ("obs/ctl:", "  ".join(
        "%6.2f" % v if np.isfinite(v) else "     -" for v in row_o)))
    print("%-10s %s" % ("n resid:", "  ".join("%6d" % n for n in row_n)))
    print("\n%d structures. Expected ratio inside the halo: %.2f" % (len(recs), expected))
    peak = np.nanmax(row_o)
    print("Recovered peak: %.2f in the %.0f-%.0f A bin" % (
        peak, DIST_EDGES[int(np.nanargmax(row_o))], DIST_EDGES[int(np.nanargmax(row_o)) + 1]))
    print("\nVERDICT: %s" % (
        "instrument recovers a planted through-space halo"
        if peak >= 1.0 + 0.5 * (expected - 1.0) else
        "INSTRUMENT IS BLIND -- every through-space number in this project is void"))


if __name__ == "__main__":
    main()
