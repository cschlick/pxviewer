"""Metric-specific calibration and bounded hotspot-field combination."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.ndimage import map_coordinates

from events import Event
from field import Field, compute_field


RAMA_FAVORED_PCT = 2.0
RAMA_OUTLIER_PCT = {0: 0.05, 2: 0.20}  # general, cis-Pro
RAMA_DEFAULT_OUTLIER_PCT = 0.10         # gly, trans-Pro, pre-Pro, Ile/Val
ROTAMER_FAVORED_PCT = 2.0
ROTAMER_OUTLIER_PCT = 0.30
CLASH_ZERO_OVERLAP_A = 0.0
CLASH_SATURATION_OVERLAP_A = 0.80
QSCORE_EXPECTED_INTERCEPT = 1.1192
QSCORE_EXPECTED_RESOLUTION_SLOPE = -0.1775
QSCORE_SATURATION_DEFICIT = 0.20


def linear_concern(value, good, bad):
    """Map good -> 0 and bad -> 1, allowing either direction."""
    if good == bad:
        raise ValueError("good and bad anchors must differ")
    return float(np.clip((float(value) - good) / (bad - good), 0.0, 1.0))


def log_low_concern(value, good, bad, floor=1e-12):
    """Log interpolation for positive scores where lower is worse."""
    if not (good > bad > 0):
        raise ValueError("log-low calibration requires good > bad > 0")
    value = max(float(value), floor)
    return linear_concern(math.log(value), math.log(good), math.log(bad))


def qscore_expected(resolution):
    """Protein expected-Q regression reported by Pintilie et al. (2020)."""
    return QSCORE_EXPECTED_INTERCEPT + QSCORE_EXPECTED_RESOLUTION_SLOPE * float(resolution)


def molprobity_concern_events(events):
    """Recalibrate extracted MolProbity events onto bounded [0,1] concern."""
    calibrated = []
    for event in events:
        if event.metric == "rama":
            bad = RAMA_OUTLIER_PCT.get(
                event.meta.get("res_type"), RAMA_DEFAULT_OUTLIER_PCT)
            concern = log_low_concern(
                event.meta["score"], RAMA_FAVORED_PCT, bad)
        elif event.metric == "rota":
            concern = log_low_concern(
                event.meta["score"], ROTAMER_FAVORED_PCT,
                ROTAMER_OUTLIER_PCT)
        elif event.metric == "clash":
            overlap = max(0.0, -float(event.meta["overlap"]))
            concern = linear_concern(
                overlap, CLASH_ZERO_OVERLAP_A, CLASH_SATURATION_OVERLAP_A)
        else:
            raise ValueError("unsupported MolProbity metric %r" % event.metric)
        meta = dict(event.meta)
        meta["native_severity"] = event.severity
        meta["concern"] = concern
        calibrated.append(replace(event, severity=concern, meta=meta))
    return calibrated


def qscore_concern_events(records, resolution=None, expected_q=None,
                          saturation_deficit=QSCORE_SATURATION_DEFICIT):
    """Convert cctbx qscore_records/JSON columns or rows to point events."""
    if expected_q is None:
        if resolution is None:
            raise ValueError("Q-score concern requires resolution or expected_q")
        expected_q = qscore_expected(resolution)
    if saturation_deficit <= 0:
        raise ValueError("saturation_deficit must be positive")

    if isinstance(records, dict):
        n = len(records["Q-score"])
        rows = ({key: values[i] for key, values in records.items()}
                for i in range(n))
    else:
        rows = iter(records)

    events = []
    for row in rows:
        q = float(row["Q-score"])
        if not np.isfinite(q):
            continue
        concern = linear_concern(
            q, float(expected_q), float(expected_q) - saturation_deficit)
        xyz = (float(row["x"]), float(row["y"]), float(row["z"]))
        events.append(Event(
            "qscore", concern, [xyz],
            meta={"id": row.get("id", ""), "qscore": q,
                  "expected_q": float(expected_q), "concern": concern}))
    return events


def _sample_on(source, target):
    ijk = np.indices(target.data.shape, dtype=float)
    for axis in range(3):
        xyz = target.origin[axis] + ijk[axis] * target.spacing
        ijk[axis] = (xyz - source.origin[axis]) / source.spacing
    return map_coordinates(source.data, ijk, order=1, mode="constant", cval=0.0)


def build_concern_fields(events_by_metric, spacing=1.0, sigma=2.0):
    """Build capped per-metric fields and their voxel-wise maximum.

    Within one metric, nearby observations add before saturation. Across metrics,
    fields combine by maximum and are never summed.
    """
    events_by_metric = {k: list(v) for k, v in events_by_metric.items() if v}
    if not events_by_metric:
        raise ValueError("at least one non-empty metric is required")
    all_events = [e for events in events_by_metric.values() for e in events]
    target = compute_field(all_events, spacing=spacing, sigma=sigma)
    fields = {}
    arrays = []
    for metric, events in events_by_metric.items():
        raw = compute_field(events, spacing=spacing, sigma=sigma)
        data = np.clip(_sample_on(raw, target), 0.0, 1.0)
        fields[metric] = Field(data, target.origin.copy(), spacing, sigma, 1.0)
        arrays.append(data)
    combined = np.maximum.reduce(arrays)
    fields["combined"] = Field(
        combined, target.origin.copy(), spacing, sigma, 1.0)
    return fields
