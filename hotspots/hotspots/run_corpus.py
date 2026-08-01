"""Run hotspot generation over a corpus of models, and QC every result.

Built for unattended runs on a big machine. Three things matter at corpus scale and none
of them matter for a single model:

* **It must not stop.** One unparseable model, one residue reduce2 refuses, must not end a
  run that has been going for hours. Every model is isolated; failures are recorded and the
  run continues.
* **It must be resumable.** ``--resume`` skips models whose manifest already exists, so a
  killed run picks up where it stopped.
* **Success is not the same as correct.** Every model is checked with
  ``validation_events.check_field_agreement``: the field is sampled back at the atoms and
  compared against the validation it was built from. A run where everything "succeeded" and
  recall quietly sits at 0.6 is the failure worth catching, and it is invisible from exit
  codes.

Parallelism is by **sharding**, not by a process pool: launch N copies with
``--shard k/N``. cctbx state and forked processes interact badly, and a shard that dies
takes nothing else with it.

    libtbx.python hotspots/run_corpus.py MODELS OUT --output-pixel-size 2.0
    for k in $(seq 0 15); do
        libtbx.python hotspots/run_corpus.py MODELS OUT --shard $k/16 --resume &
    done; wait
    libtbx.python hotspots/run_corpus.py MODELS OUT --report      # merge and summarize

``MODELS`` is a directory (searched recursively for .pdb/.cif, optionally .gz) or a text
file listing one model path per line.
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

from events import _load_shared
from make_concern_maps import generate

ve = _load_shared()

MODEL_SUFFIXES = (".pdb", ".cif", ".ent", ".pdb.gz", ".cif.gz", ".ent.gz")

#: Concern at or above which a voxel is "hot" for the QC check. The display contract's
#: yellow anchor -- the point the viewer starts showing anything.
DEFAULT_HOT = 0.5

#: A residue this far from a flagged one can still legitimately carry its signal: the splat
#: is wider than the ~3.8 A between adjacent CA atoms.
DEFAULT_TOLERANCE_A = 4.0


def find_models(source: str):
    """Every model under a directory, or every path listed in a file."""
    if os.path.isfile(source) and not source.lower().endswith(MODEL_SUFFIXES):
        with open(source) as handle:
            return [line.strip() for line in handle
                    if line.strip() and not line.startswith("#")]
    if os.path.isfile(source):
        return [source]
    found = []
    for suffix in MODEL_SUFFIXES:
        found += glob.glob(os.path.join(source, "**", "*" + suffix), recursive=True)
    return sorted(set(found))


def _sample_field_at(field, sites_xyz):
    """Read a Field at Cartesian points, trilinearly — the same thing the viewer does.

    Works from the in-memory Field rather than the written CCP4, so QC costs no extra I/O
    and cannot disagree with what was written.
    """
    from scipy.ndimage import map_coordinates

    sites = np.asarray(sites_xyz, dtype=float).reshape(-1, 3)
    ijk = np.empty((3, sites.shape[0]), dtype=float)
    for axis in range(3):
        ijk[axis] = (sites[:, axis] - field.origin[axis]) / field.spacing
    return map_coordinates(field.data, ijk, order=1, mode="constant", cval=0.0)


def check_model(fields, events, model, *, hot=DEFAULT_HOT, tolerance_a=DEFAULT_TOLERANCE_A):
    """Field-vs-validation agreement for each metric. See validation_events."""
    hierarchy = model.get_hierarchy()
    sites = np.asarray(hierarchy.atoms().extract_xyz()).reshape(-1, 3)
    shared = ve.extract_all(model, metrics=("rama", "rota"))   # cheap; clash needs probe2
    report = {}
    for metric in ("rama", "rota"):
        field = fields.get(metric)
        if field is None:
            continue
        sampled = _sample_field_at(field, sites)
        # Widened past the outlier boolean on purpose: the concern curve rises from the 2%
        # favored boundary, so the field legitimately marks residues MolProbity never flags.
        # Judged against outliers alone this reports correct behaviour as failure.
        r = ve.check_field_agreement(
            shared, sampled, sites, metric=metric, hot_threshold=hot,
            tolerance_a=tolerance_a, concerning=ve.worse_than_percent(2.0))
        report[metric] = {
            "recall": round(r.recall, 4), "explained": round(r.explained, 4),
            "n_outlier_atoms": r.n_outlier_atoms, "n_hot_atoms": r.n_hot_atoms,
            "n_missed": len(r.missed), "n_unexplained": len(r.unexplained),
        }
    return report


def run_one(model_path, out_root, *, spacing, sigma, heavy_atom_clashes, check, hot):
    stem = os.path.basename(model_path).split(".")[0]
    out_dir = os.path.join(out_root, stem)
    started = time.time()
    record = {"model": os.path.abspath(model_path), "stem": stem, "out_dir": out_dir}
    try:
        fields = {}
        manifest = generate(model_path, out_dir, sigma=sigma, spacing=spacing,
                            heavy_atom_clashes=heavy_atom_clashes, fields_out=fields)
        record.update(
            status="ok",
            manifest=manifest.get("manifest_path"),
            event_counts=manifest.get("event_counts"),
            clashscore=(manifest.get("molprobity") or {}).get("clashscore"),
            hydrogens=(manifest.get("molprobity") or {}).get("hydrogens"),
            grid=manifest.get("output_grid"),
            pixel_size=manifest.get("output_pixel_size_angstrom"))
        if check:
            from events import load_model
            record["agreement"] = check_model(
                fields, fields.get("_events"), load_model(model_path), hot=hot)
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc()[-2000:])
    record["seconds"] = round(time.time() - started, 1)
    return record


def _results_path(out_root, shard):
    tag = "" if shard is None else f".shard{shard[0]}of{shard[1]}"
    return os.path.join(out_root, f"corpus_results{tag}.jsonl")


def report(out_root):
    """Merge every shard's results and summarize. Prints the things worth acting on."""
    records = []
    for path in sorted(glob.glob(os.path.join(out_root, "corpus_results*.jsonl"))):
        with open(path) as handle:
            records += [json.loads(line) for line in handle if line.strip()]
    if not records:
        print("no results found in", out_root)
        return records

    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    print(f"{len(records)} models: {len(ok)} ok, {len(failed)} failed")
    if ok:
        secs = [r["seconds"] for r in ok]
        print(f"  time per model: median {np.median(secs):.1f}s  max {max(secs):.1f}s  "
              f"total {sum(r['seconds'] for r in records) / 3600:.2f}h")

    for metric in ("rama", "rota"):
        vals = [r["agreement"][metric]["recall"] for r in ok
                if r.get("agreement", {}).get(metric)]
        if not vals:
            continue
        vals = np.array(vals)
        # Recall is the guarantee the field owes: it must never lose a real outlier.
        print(f"  {metric} recall: min {vals.min():.3f}  median {np.median(vals):.3f}  "
              f"below 1.0 in {(vals < 1.0).sum()}/{len(vals)} models")

    imperfect = sorted(
        (r for r in ok if any(m.get("recall", 1.0) < 1.0
                              for m in (r.get("agreement") or {}).values())),
        key=lambda r: min(m.get("recall", 1.0) for m in r["agreement"].values()))
    if imperfect:
        print("\n  models whose field missed a flagged outlier (investigate these):")
        for r in imperfect[:20]:
            worst = {k: v["recall"] for k, v in r["agreement"].items() if v["recall"] < 1.0}
            print(f"    {r['stem']:20s} {worst}")

    if failed:
        print("\n  failures:")
        for r in failed[:20]:
            print(f"    {r['stem']:20s} {r['error'][:100]}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", help="directory of models, or a file listing model paths")
    parser.add_argument("out_dir", help="root for per-model output directories")
    parser.add_argument("--output-pixel-size", "--spacing", dest="spacing", type=float,
                        default=2.0, metavar="ANGSTROM",
                        help="output voxel size in A (default: 2.0, the fast-viewing setting)")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--heavy-atom-clashes", action="store_true",
                        help="skip H placement; much faster, not the calibrated path")
    parser.add_argument("--shard", metavar="K/N",
                        help="process only shard K of N (0-based); run N of these in parallel")
    parser.add_argument("--limit", type=int, help="stop after this many models")
    parser.add_argument("--resume", action="store_true",
                        help="skip models whose manifest already exists")
    parser.add_argument("--no-check", action="store_true",
                        help="skip the field-vs-validation agreement check")
    parser.add_argument("--hot-threshold", type=float, default=DEFAULT_HOT)
    parser.add_argument("--report", action="store_true",
                        help="merge existing results and summarize; runs nothing")
    args = parser.parse_args()

    if args.report:
        report(args.out_dir)
        return

    shard = None
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not (0 <= k < n):
            parser.error("--shard K/N needs 0 <= K < N")
        shard = (k, n)

    models = find_models(args.models)
    if shard:
        models = [m for i, m in enumerate(models) if i % shard[1] == shard[0]]
    if args.limit:
        models = models[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    results_path = _results_path(args.out_dir, shard)
    done = set()
    if args.resume and os.path.exists(results_path):
        with open(results_path) as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["model"])

    print(f"{len(models)} models"
          f"{f' (shard {shard[0]}/{shard[1]})' if shard else ''}"
          f"{f', {len(done)} already done' if done else ''} -> {results_path}", flush=True)

    for i, path in enumerate(models, 1):
        if args.resume and os.path.abspath(path) in done:
            continue
        record = run_one(path, args.out_dir, spacing=args.spacing, sigma=args.sigma,
                         heavy_atom_clashes=args.heavy_atom_clashes,
                         check=not args.no_check, hot=args.hot_threshold)
        # Append and flush per model: a killed run keeps everything it finished.
        with open(results_path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        note = record.get("error", "")
        if record["status"] == "ok" and record.get("agreement"):
            note = " ".join(f"{m}:recall={v['recall']:.2f}"
                            for m, v in record["agreement"].items())
        print(f"[{i}/{len(models)}] {record['stem']:20s} {record['status']:7s} "
              f"{record['seconds']:6.1f}s  {note[:90]}", flush=True)

    print()
    report(args.out_dir)


if __name__ == "__main__":
    main()
