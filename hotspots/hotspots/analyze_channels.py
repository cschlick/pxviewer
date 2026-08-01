"""Quantify how much the hotspot field adds beyond a clash-only field.

Run under Phenix Python:
  libtbx.python analyze_channels.py model.cif [model2.cif ...]

The report focuses on the displayed, absolute severity scale:
  * non-clash outliers whose own sites have little clash signal;
  * high combined-field voxels that would not appear in a clash-only map;
  * how strongly high combined-field voxels are dominated by clashes.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.ndimage import map_coordinates

from events import extract_all, load_hierarchy
from field import compute_field


def _on_grid(source, target):
    """Sample source Field on every voxel of target Field."""
    ijk = np.indices(target.data.shape, dtype=float)
    for axis in range(3):
        xyz = target.origin[axis] + ijk[axis] * target.spacing
        ijk[axis] = (xyz - source.origin[axis]) / source.spacing
    return map_coordinates(source.data, ijk, order=1, mode="constant", cval=0.0)


def _site_peak(field, event):
    return max(field.sample(xyz) for xyz in event.atoms_xyz)


def analyze(path, sigma=2.0, spacing=1.0):
    hierarchy = load_hierarchy(path)
    result = extract_all(hierarchy, use_hydrogens=True)
    events = result["events"]
    channels = {
        name: [e for e in events if e.metric == name]
        for name in ("rama", "rota", "clash")
    }
    nonclash = channels["rama"] + channels["rota"]

    combined_field = compute_field(events, spacing=spacing, sigma=sigma)
    clash_field = compute_field(channels["clash"], spacing=spacing, sigma=sigma)
    nonclash_field = compute_field(nonclash, spacing=spacing, sigma=sigma)
    clash_grid = _on_grid(clash_field, combined_field)
    nonclash_grid = _on_grid(nonclash_field, combined_field)
    combined = combined_field.data

    nc_outliers = [e for e in nonclash if e.severity >= 1.0]
    independent = []
    for event in nc_outliers:
        c = _site_peak(clash_field, event)
        n = _site_peak(nonclash_field, event)
        a = _site_peak(combined_field, event)
        if c < 0.5 and a >= 1.0:
            independent.append((event, c, n, a))

    high = combined >= 1.0
    high_n = int(high.sum())
    novel = high & (clash_grid < 0.5)
    clash_dominant = high & (clash_grid > nonclash_grid)
    flat_c = clash_grid.ravel()
    flat_a = combined.ravel()
    occupied = (flat_c > 0.05) | (flat_a > 0.05)
    correlation = (float(np.corrcoef(flat_a[occupied], flat_c[occupied])[0, 1])
                   if occupied.sum() > 2 else None)

    return {
        "model": path,
        "clashscore": result["manifest"]["clashscore"],
        "events": {k: len(v) for k, v in channels.items()},
        "outliers": {
            k: sum(e.severity >= 1.0 for e in v) for k, v in channels.items()
        },
        "field_max": {
            "combined": float(combined.max()),
            "clash": float(clash_field.data.max()),
            "nonclash": float(nonclash_field.data.max()),
        },
        "high_voxels": high_n,
        "high_voxels_without_clash_signal": int(novel.sum()),
        "high_voxels_without_clash_signal_fraction":
            float(novel.sum() / high_n) if high_n else 0.0,
        "high_voxels_clash_dominant_fraction":
            float(clash_dominant.sum() / high_n) if high_n else 0.0,
        "combined_clash_voxel_correlation": correlation,
        "nonclash_outliers": len(nc_outliers),
        "independent_nonclash_regions": len(independent),
        "independent_nonclash_fraction":
            float(len(independent) / len(nc_outliers)) if nc_outliers else 0.0,
        "independent_examples": [
            {
                "metric": e.metric,
                "id": e.meta.get("id", ""),
                "severity": e.severity,
                "clash_field": c,
                "nonclash_field": n,
                "combined_field": a,
            }
            for e, c, n, a in sorted(independent, key=lambda x: x[3], reverse=True)[:8]
        ],
    }


if __name__ == "__main__":
    for model_path in sys.argv[1:]:
        print(json.dumps(analyze(model_path), sort_keys=True))
