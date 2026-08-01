"""Write bounded per-metric hotspot maps and a maximum-combined map."""
from __future__ import annotations

import argparse
import json
import os

from concern import (build_concern_fields, molprobity_concern_events,
                     qscore_concern_events, QSCORE_SATURATION_DEFICIT)
from events import extract_all, load_model
from field import write_ccp4
from color_scale import display_contract, empirical_percentile_field


def _read_qscore_json(path):
    payload = json.load(open(path))
    if "flat_results" in payload:
        return payload["flat_results"]
    if "qscore_records" in payload:
        return payload["qscore_records"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Q-score JSON needs flat_results, qscore_records, or a row list")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bounded MolProbity/Q-score hotspot maps, a "
                    "maximum-combined map, and optional percentile outputs.")
    parser.add_argument("model", help="input PDB/mmCIF model, optionally .gz")
    parser.add_argument("out_dir", help="directory for CCP4 maps and manifest")
    parser.add_argument("--qscore-json",
                        help="cctbx Q-score JSON containing per-atom flat_results")
    parser.add_argument("--resolution", type=float,
                        help="map resolution in A for protein expected-Q calibration")
    parser.add_argument("--expected-q", type=float,
                        help="explicit expected Q; overrides resolution regression")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian width in A (default: 2.0)")
    parser.add_argument("--output-pixel-size", "--spacing", dest="spacing",
                        type=float, default=1.0, metavar="ANGSTROM",
                        help="output voxel/pixel size in A (default: 1.0; "
                             "--spacing is a compatibility alias)")
    parser.add_argument("--heavy-atom-clashes", action="store_true",
                        help="skip H addition; labeled preview, not calibrated default")
    args = parser.parse_args()
    if args.spacing <= 0:
        parser.error("--output-pixel-size must be positive")
    if args.sigma <= 0:
        parser.error("--sigma must be positive")
    manifest = generate(
        args.model, args.out_dir, sigma=args.sigma, spacing=args.spacing,
        heavy_atom_clashes=args.heavy_atom_clashes, qscore_json=args.qscore_json,
        resolution=args.resolution, expected_q=args.expected_q)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def generate(model_path, out_dir, *, sigma=2.0, spacing=1.0, heavy_atom_clashes=False,
             qscore_json=None, resolution=None, expected_q=None, fields_out=None):
    """Generate every map and the manifest for one model, and return the manifest.

    The same code path the CLI runs, exposed so a batch driver (see run_corpus.py) does not
    have to reimplement it. ``fields_out``, if given a dict, receives the in-memory
    :class:`Field` objects, which lets a caller check the field against the events it was
    built from without re-reading the CCP4s.
    """
    if spacing <= 0 or sigma <= 0:
        raise ValueError("sigma and output pixel size must be positive")

    class _Args:
        pass
    args = _Args()
    args.model, args.out_dir, args.sigma, args.spacing = model_path, out_dir, sigma, spacing
    args.heavy_atom_clashes = heavy_atom_clashes
    args.qscore_json, args.resolution, args.expected_q = qscore_json, resolution, expected_q

    model = load_model(args.model)
    extracted = extract_all(
        model, use_hydrogens=not args.heavy_atom_clashes)
    mp_events = molprobity_concern_events(extracted["events"])
    by_metric = {
        metric: [e for e in mp_events if e.metric == metric]
        for metric in ("clash", "rama", "rota")
    }
    q_manifest = None
    if args.qscore_json:
        q_events = qscore_concern_events(
            _read_qscore_json(args.qscore_json), resolution=args.resolution,
            expected_q=args.expected_q)
        by_metric["qscore"] = q_events
        q_manifest = {
            "input": os.path.abspath(args.qscore_json),
            "resolution": args.resolution,
            "expected_q": (q_events[0].meta["expected_q"] if q_events else None),
            "saturation_deficit": QSCORE_SATURATION_DEFICIT,
        }

    fields = build_concern_fields(
        by_metric, spacing=args.spacing, sigma=args.sigma)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.basename(args.model).split(".")[0]
    outputs = {}
    color_scaling = {}
    for metric, field in fields.items():
        path = os.path.join(args.out_dir, "%s_%s_hotspot.ccp4" % (stem, metric))
        write_ccp4(field, path)
        percentile, color_metadata = empirical_percentile_field(field.data)
        percentile_field = type(field)(
            percentile, field.origin.copy(), field.spacing, field.sigma,
            field.reference_level)
        percentile_path = os.path.join(
            args.out_dir, "%s_%s_hotspot_percentile.ccp4" % (stem, metric))
        write_ccp4(percentile_field, percentile_path)
        outputs[metric] = {"concern": path, "color_percentile": percentile_path}
        color_scaling[metric] = color_metadata

    manifest = {
        **display_contract(),
        "model": os.path.abspath(args.model),
        "field_semantics": "bounded concern; 0=reassuring, 1=saturated concern",
        "combination": "voxel-wise maximum of capped metric fields",
        "color_scaling": color_scaling,
        "sigma_angstrom": args.sigma,
        "spacing_angstrom": args.spacing,
        "output_pixel_size_angstrom": args.spacing,
        "output_grid": list(fields["combined"].data.shape),
        "calibration": {
            "rama": {
                "transform": "log-low-is-bad",
                "good_percent": 2.0,
                "bad_percent_by_class": {
                    "general": 0.05, "cis_pro": 0.20, "other": 0.10,
                },
            },
            "rota": {
                "transform": "log-low-is-bad",
                "good_percent": 2.0,
                "bad_percent": 0.30,
            },
            "clash": {
                "transform": "linear-high-is-bad",
                "good_overlap_angstrom": 0.0,
                "saturation_overlap_angstrom": 0.80,
            },
        },
        "molprobity": extracted["manifest"],
        "qscore": q_manifest,
        "event_counts": {k: len(v) for k, v in by_metric.items()},
        "outputs": outputs,
    }
    manifest_path = os.path.join(args.out_dir, "%s_hotspots.json" % stem)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest["manifest_path"] = manifest_path
    if fields_out is not None:
        fields_out.update(fields)
        fields_out["_events"] = extracted["events"]
    return manifest


if __name__ == "__main__":
    main()
