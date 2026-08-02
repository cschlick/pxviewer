"""Draw a seeded sample from the screened population, and freeze it to a file.

A seed only makes a draw reproducible if the *population* is reproducible too, which is why
`pdb_population.txt` is committed and never regenerated in place (see README.md). Given
that, a plain seeded `random.sample` is exactly reproducible and nothing cleverer is needed.

The drawn list is written out and committed alongside the figure data, so a reader can check
the sample without re-running the draw, and so a later re-draw with a changed population
cannot silently pretend to be the same corpus.

    libtbx.python corpus/draw_sample.py --n 2000 --seed 20260802
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = "/root/data/pdb_mmcif"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--population", default=os.path.join(HERE, "protein_population.txt"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, "sample_%d_seed%d.txt" % (args.n, args.seed))
    ids = [l.strip() for l in open(args.population) if l.strip()]
    random.seed(args.seed)
    drawn = sorted(random.sample(ids, args.n))
    with open(out, "w") as handle:
        handle.write("".join(i + "\n" for i in drawn))

    meta = {
        "population": os.path.abspath(args.population),
        "population_n": len(ids),
        "population_sha256": sha256_of(args.population),
        "seed": args.seed,
        "n": args.n,
        "sample": os.path.abspath(out),
        "sample_sha256": sha256_of(out),
        "mirror": MIRROR,
    }
    with open(out.replace(".txt", ".json"), "w") as handle:
        json.dump(meta, handle, indent=1, sort_keys=True)
    print(json.dumps(meta, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
