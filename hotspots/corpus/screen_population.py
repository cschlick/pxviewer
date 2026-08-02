"""Screen the frozen population down to entries that can carry protein validation.

The hotspot channels are protein validation — Ramachandran, rotamer, clash. A
nucleic-acid-only entry has no phi/psi and no rotamers, so it contributes nothing to any
figure; it only burns time and, if it happens to contain a modified nucleotide, shows up as
a reduce2 failure that looks like a defect and is not one. Two of the thirty structures in
the first test run were exactly this (242d, 6tf0).

So the corpus definition states the requirement rather than discovering it: **an entry is
eligible iff it declares at least one ``polypeptide`` entity.** That is read from the mmCIF
``_entity_poly.type`` record, which sits near the top of the file, so this streams and stops
early instead of decompressing whole entries.

    libtbx.python corpus/screen_population.py            # full population -> eligible list
    libtbx.python corpus/screen_population.py --limit 500  # time it first

Output is ``protein_population.txt`` (the eligible IDs) beside the input, plus
``screen_rejected.jsonl`` recording *why* each rejected entry was dropped. Nothing is
discarded silently: the rejected file plus the eligible file account for every input ID.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = "/root/data/pdb_mmcif"

#: Stop reading an entry after this many decompressed bytes. ``_entity_poly.type`` is header
#: material and sits far inside this on every entry seen so far; anything that does not
#: declare it by here is reported as ``no_entity_poly`` rather than assumed to be protein.
HEADER_BYTES = 400_000


def model_path(pdb_id: str) -> str:
    return os.path.join(MIRROR, pdb_id[1:3], pdb_id + ".cif.gz")


def classify(pdb_id: str) -> dict:
    """``{id, ok, kinds}`` — the polymer kinds an entry declares, read from its header.

    ``ok`` is True iff at least one is a polypeptide. Errors are returned, never raised: a
    screen that dies on one unreadable entry is useless at 200k entries.
    """
    path = model_path(pdb_id)
    try:
        with gzip.open(path, "rt", errors="replace") as handle:
            head = handle.read(HEADER_BYTES)
    except Exception as exc:
        return {"id": pdb_id, "ok": False, "reason": f"unreadable: {type(exc).__name__}"}

    kinds = set()
    for line in head.splitlines():
        s = line.strip()
        # Two layouts: the single-entity `_entity_poly.type  polypeptide(L)` key/value form,
        # and the loop_ form where the type is a bare token on its own row. Scanning for the
        # vocabulary itself catches both without parsing the loop header.
        if s.startswith("_entity_poly.type") or "polypeptide" in s or "polyribo" in s \
                or "polydeoxyribo" in s:
            for kind in ("polypeptide", "polyribonucleotide",
                         "polydeoxyribonucleotide", "polysaccharide"):
                if kind in s:
                    kinds.add(kind)
    if not kinds:
        return {"id": pdb_id, "ok": False, "reason": "no_entity_poly"}
    return {"id": pdb_id, "ok": "polypeptide" in kinds, "kinds": sorted(kinds),
            "reason": None if "polypeptide" in kinds else "no_polypeptide"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--population", default=os.path.join(HERE, "pdb_population.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "protein_population.txt"))
    ap.add_argument("--rejected", default=os.path.join(HERE, "screen_rejected.jsonl"))
    ap.add_argument("--limit", type=int, help="screen only the first N (for timing)")
    ap.add_argument("--procs", type=int, default=10)
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.population) if l.strip()]
    if args.limit:
        ids = ids[: args.limit]
    started = time.time()
    with Pool(args.procs) as pool:
        results = pool.map(classify, ids, chunksize=200)
    elapsed = time.time() - started

    eligible = [r["id"] for r in results if r["ok"]]
    rejected = [r for r in results if not r["ok"]]
    with open(args.out, "w") as handle:
        handle.write("".join(i + "\n" for i in eligible))
    with open(args.rejected, "w") as handle:
        for r in rejected:
            handle.write(json.dumps(r, sort_keys=True) + "\n")

    reasons = {}
    for r in rejected:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print(f"screened {len(ids)} in {elapsed:.1f}s ({len(ids)/max(elapsed,1e-9):.0f}/s)")
    print(f"  eligible (has polypeptide): {len(eligible)}")
    print(f"  rejected: {len(rejected)}  {reasons}")
    print(f"  -> {args.out}")
    assert len(eligible) + len(rejected) == len(ids), "screen lost entries"


if __name__ == "__main__":
    main()
