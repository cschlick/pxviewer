"""Recover readable observations from gzip files truncated by a killed process.

Needed because of the append-across-restarts bug described in ``figure_data._paths``: a
SIGKILLed shard leaves a truncated gzip member, the next shard appends a fresh member after
it, and a normal reader stops dead at the junction — losing everything past the first kill
even though most of it is intact.

A gzip file is a *sequence* of independent members, so the data after a break is still
decompressable if you find where the next member starts. This scans for member headers,
decompresses each independently, and keeps whatever survives.

    libtbx.python corpus/salvage_observations.py OUT_DIR

Writes ``observations.salvaged.jsonl.gz`` (one clean stream, deduplicated by structure and
observation) and reports what was recovered against what the results files say should exist.
This is a repair tool for data already on disk; new runs do not need it.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import zlib

GZIP_MAGIC = b"\x1f\x8b\x08"


CHUNK = 1 << 20


def members(blob: bytes):
    """Yield decompressed bytes for every gzip member that can be read from ``blob``.

    Decompression is fed in chunks rather than one call on purpose. ``decompressobj`` raises
    when it reaches the truncation, and an exception discards the function's return value --
    so decompressing a whole member in one call throws away everything that *did* decode
    before the break. On a file whose first member is the big intact one, that loses almost
    all the recoverable data. Chunking yields each piece as it decodes, so a mid-member error
    costs only the final chunk.
    """
    pos, n = 0, len(blob)
    while pos < n:
        start = blob.find(GZIP_MAGIC, pos)
        if start < 0:
            return
        dec = zlib.decompressobj(31)
        i, ok = start, False
        try:
            while i < n and not dec.eof:
                out = dec.decompress(blob[i:i + CHUNK])
                i += CHUNK
                if out:
                    ok = True
                    yield out
        except zlib.error:
            pass                       # truncated member, or a false header in packed bytes
        if dec.eof and dec.unused_data:
            pos = n - len(dec.unused_data)      # clean end: next member starts exactly here
        else:
            pos = start + 1            # rescan forward for the next plausible header
        del ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    args = ap.parse_args()

    seen, ids_seen = set(), set()
    kept = 0
    dst = os.path.join(args.out_dir, "observations.salvaged.jsonl.gz")
    sources = sorted(p for p in glob.glob(os.path.join(args.out_dir, "observations*.gz"))
                     if not p.endswith("observations.salvaged.jsonl.gz"))

    with gzip.open(dst, "wt") as out:
        for path in sources:
            with open(path, "rb") as handle:
                blob = handle.read()
            recovered = 0
            tail = b""
            for chunk in members(blob):
                data = tail + chunk
                lines = data.split(b"\n")
                tail = lines.pop()          # last element is an incomplete line
                for raw in lines:
                    if not raw.strip():
                        continue
                    try:
                        rec = json.loads(raw)
                    except Exception:
                        continue            # a torn line at a member boundary
                    # Dedupe on the exact line, not on (id, metric, residue): a residue with
                    # alternate conformations legitimately produces two observations sharing
                    # chain/resseq/icode, and the record carries no altloc field to tell them
                    # apart. Keying on the residue would silently drop one of every altloc
                    # pair; an identical line is the only safe definition of "duplicate".
                    key = raw
                    if key in seen:
                        continue
                    seen.add(key)
                    ids_seen.add(rec["id"])
                    out.write(raw.decode("utf-8", "replace") + "\n")
                    recovered += 1
            kept += recovered
            print(f"  {os.path.basename(path):55s} {recovered:>9,d} observations")

    ids = ids_seen
    expected = set()
    for p in glob.glob(os.path.join(args.out_dir, "results*.jsonl")):
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r["status"] == "ok":
                        expected.add(r["id"])
    print(f"\nsalvaged {kept:,d} observations over {len(ids)} structures -> {dst}")
    print(f"results report {len(expected)} ok structures; "
          f"still missing observations for {len(expected - ids)}")


if __name__ == "__main__":
    main()
