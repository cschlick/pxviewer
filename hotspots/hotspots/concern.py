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

# Below this overlap a contact deposits nothing. The original calibration was
# clip(overlap / 0.80) with no floor, which was safe only because the extractor could not
# see sub-threshold contacts: mmtbx.validation.clashscore reports at 0.40 A and no closer,
# so in practice the channel started at concern 0.5. Moving to probe2 supplied the whole
# tail and quietly changed what the same formula means -- 1476 contacts instead of 536 on
# 1TEC, most of them mild, each depositing a little, summing to 10.4% of the box past the
# display threshold and swamping every other channel in the combined map.
#
# A 0.1 A brush is not a modeling problem; 0.40 A is where MolProbity says a contact becomes
# a clash. Gating there restores the calibration's original behaviour exactly -- a reported
# clash still lands at 0.5 and saturates at 0.80 -- while the sub-threshold contacts remain
# in the events for consumers that want them.
CLASH_REPORTING_OVERLAP_A = 0.40

# Covalent geometry. The native value is a deviation from the restraint ideal, so the
# scale-free quantity is Z = |delta| / sigma; MolProbity flags at 4 sigma.
#
# The zero anchor is at the cut itself, not at 0 sigma, and that is not a stylistic choice.
# Unlike every other channel, *every* restraint is an event -- 2589 bonds and 3547 angles on
# a 2737-atom model, against ~300 Ramachandran results -- and concern *sums* within a metric
# before it saturates. Anchored at 0 sigma the median restraint (~1 sigma) deposits ~0.13,
# thousands of those overlap, and the field saturates across the whole protein: measured, it
# put 6695 of 39900 voxels past the display threshold, against 52 for Ramachandran, and the
# combined map stopped meaning "where to look" and started meaning "where the protein is".
#
# Anchoring at the community cut keeps the channel sparse like the others: an unremarkable
# restraint deposits exactly nothing, and only genuine strain accumulates.
GEOMETRY_ZERO_SIGMA = 4.0
GEOMETRY_SATURATION_SIGMA = 8.0

# CaBLAM and C-beta. CaBLAM's score is a probability-like fraction where lower is worse, so
# it is log-interpolated exactly like rama/rota, from the "disfavored" boundary to the
# outlier cut. C-beta deviation is a distance, so it is linear like clash.
CABLAM_FAVORED = 0.05        # above this, unremarkable
CABLAM_OUTLIER = 0.01        # the CaBLAM outlier cut
CA_GEOM_FAVORED = 0.05
CA_GEOM_OUTLIER = 0.005      # the CA-geometry outlier cut
CBETA_ZERO_A = 0.0
CBETA_SATURATION_A = 0.50    # twice the 0.25 A MolProbity cut, as with clash

# Non-trans peptides. omegalyze reports the omega dihedral; a cis or twisted peptide is
# flagged. Concern is the twist away from the nearest ideal (0 for cis, 180 for trans),
# saturating at 30 degrees -- twice the 15 degree boundary omegalyze calls "twisted".
OMEGA_TWIST_SATURATION_DEG = 30.0
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


def _omega_concern(meta):
    """Non-trans peptides. Categorical first, then continuous.

    omegalyze flags every non-trans peptide, so an ordinary cis-proline arrives flagged; it
    is common and legitimate, and treating it as a problem would light up a hotspot on most
    structures. A cis peptide that is *not* proline is a real modeling error and saturates
    regardless of how cleanly cis it is — the twist away from the nearest ideal is near zero
    there, so the continuous measure alone would score it clean.

    Everything else is scored by that twist, saturating at the 30 degree boundary where
    omegalyze stops calling a peptide trans and starts calling it twisted.
    """
    if meta.get("kind") == "cis":
        return 0.0 if meta.get("is_proline") else 1.0
    return linear_concern(meta.get("twist", 0.0), 0.0, OMEGA_TWIST_SATURATION_DEG)


def molprobity_concern_events(events):
    """Recalibrate extracted MolProbity events onto bounded [0,1] concern.

    Two shapes, chosen by what the native quantity is rather than per metric:

    * a **probability-like** score where lower is worse (rama, rota, cablam, ca_geom) is log
      interpolated from its "unremarkable" boundary to its outlier cut, so the community cut
      lands at concern 1.0;
    * a **deviation** with no reference tail behind it (clash, cbeta, bond, angle) is linear
      from zero to a saturation anchor set at twice the community cut, so the cut lands at
      concern 0.5.

    Those two conventions disagree about where an outlier sits, which is inherited rather
    than chosen — it predates this file. It matters because the combined field is a maximum:
    a flagged clash reaches 0.5 while a flagged rotamer reaches 1.0, so geometry outranks
    sterics in the "where to look" map. Worth revisiting; not worth changing silently.
    """
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
            concern = (0.0 if overlap < CLASH_REPORTING_OVERLAP_A else linear_concern(
                overlap, CLASH_ZERO_OVERLAP_A, CLASH_SATURATION_OVERLAP_A))
        elif event.metric == "cablam":
            concern = log_low_concern(
                event.meta["score"], CABLAM_FAVORED, CABLAM_OUTLIER)
        elif event.metric == "ca_geom":
            concern = log_low_concern(
                event.meta["score"], CA_GEOM_FAVORED, CA_GEOM_OUTLIER)
        elif event.metric == "cbeta":
            concern = linear_concern(
                event.meta["deviation"], CBETA_ZERO_A, CBETA_SATURATION_A)
        elif event.metric == "omega":
            concern = _omega_concern(event.meta)
        elif event.metric in ("bond", "angle"):
            concern = linear_concern(
                event.meta["z"], GEOMETRY_ZERO_SIGMA, GEOMETRY_SATURATION_SIGMA)
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
