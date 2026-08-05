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
#: THE INVARIANT: every channel's community outlier cut evaluates to concern **1.0**.
#:
#: The probability-like channels (rama, rota, cablam, ca_geom) always satisfied it: they log
#: interpolate from an "unremarkable" boundary to the cut, so the cut is 1.0 by construction.
#: The deviation channels did not. Each was anchored to saturate at *twice* its community cut
#: -- clash 0.80 for a 0.40 cut, cbeta 0.50 for 0.25, omega 30 for 15, geometry 8 sigma for 4
#: -- which put every one of their cuts at 0.5, and geometry's at 0.0 because its zero anchor
#: sat on the cut itself. So a flagged clash reached half the concern of a flagged rotamer and
#: a flagged bond reached none at all, purely as an inherited convention.
#:
#: That mattered once it was measured: on the 2,000-structure corpus, clash recall at the 0.5
#: display threshold was 0.61 against 1.00 for rama, and since the combined field is a
#: voxel-wise maximum, geometry silently outranked sterics everywhere. Anchoring every cut at
#: 1.0 makes the weighting *inherited from the community* rather than invented here, which is
#: what lets the channels be combined without fitting anything. See ../AGGREGATION_PROPOSAL.md.
#:
#: ``calibration_cuts()`` below asserts the invariant for all nine channels.

#: Contacts below the zero anchor are a brush, not a clash. The anchor sits at 0.30 A rather
#: than at zero overlap so the channel keeps a narrow sub-threshold tail -- which is what lets
#: co-located mild contacts accumulate -- while staying sparse. Chosen from the measured
#: overlap distribution (1TEC/6cg7/3dk2, 4250 contacts: p50 0.14, p90 0.36), not from
#: intuition: 0.20 would have admitted 35% of all contacts and swamped the map, which is the
#: failure described at CLASH_REPORTING_OVERLAP_A below.
CLASH_ZERO_OVERLAP_A = 0.30
CLASH_SATURATION_OVERLAP_A = 0.40

# Below this overlap a contact deposits nothing. The original calibration was
# clip(overlap / 0.80) with no floor, which was safe only because the extractor could not
# see sub-threshold contacts: mmtbx.validation.clashscore reports at 0.40 A and no closer,
# so in practice the channel started at concern 0.5. Moving to probe2 supplied the whole
# tail and quietly changed what the same formula means -- 1476 contacts instead of 536 on
# 1TEC, most of them mild, each depositing a little, summing to 10.4% of the box past the
# display threshold and swamping every other channel in the combined map.
#
# A 0.1 A brush is not a modeling problem; 0.40 A is where MolProbity says a contact becomes
# a clash.
#
# The gate now sits at the ramp's zero anchor (0.20 A) rather than at the cut, so contacts
# between 0.20 and 0.40 A deposit a graded sub-threshold amount instead of nothing. Gating at
# the cut itself made the channel binary once the cut moved to 1.0, which would have excluded
# clash from co-locality accumulation entirely -- the one effect a continuous field can show
# that an outlier list cannot. The swamping this constant exists to prevent is a property of
# the *dense* mild tail below ~0.2 A, not of the 0.2-0.4 A band; that is measured rather than
# assumed in calibration_cuts.py's swamping check.
CLASH_REPORTING_OVERLAP_A = 0.30

# Covalent geometry. The native value is a deviation from the restraint ideal, so the
# scale-free quantity is Z = |delta| / sigma; MolProbity flags at 4 sigma.
#
# The zero anchor must NOT sit at 0 sigma, and that is not a stylistic choice. Unlike every
# other channel, *every* restraint is an event -- 2589 bonds and 3547 angles on a 2737-atom
# model, against ~300 Ramachandran results -- and concern *sums* within a metric before it
# saturates. Anchored at 0 sigma the median restraint (~1 sigma) deposits ~0.13, thousands of
# those overlap, and the field saturates across the whole protein: measured, it put 6695 of
# 39900 voxels past the display threshold, against 52 for Ramachandran, and the combined map
# stopped meaning "where to look" and started meaning "where the protein is".
#
# It previously sat at the cut (4 sigma), which avoided that but made a *flagged* bond or
# angle deposit exactly nothing -- the channel only became visible at 6 sigma and only
# saturated at 8. Both properties cannot hold at once on a bounded scale, so the zero anchor
# moves to 3 sigma, chosen from the measured Z distribution rather than from an assumption of
# normality. Restraint Z is far more heavy-tailed than normal: 2 sigma admits 7.4% of angles
# pooled over three structures and 30% on 1TEC alone, which reproduced the saturation this
# comment exists to prevent. 3 sigma admits ~3% pooled. The check in calibration_cuts.py
# measures it on a real model rather than trusting any of this.
GEOMETRY_ZERO_SIGMA = 3.0
GEOMETRY_SATURATION_SIGMA = 4.0

# CaBLAM and C-beta. CaBLAM's score is a probability-like fraction where lower is worse, so
# it is log-interpolated exactly like rama/rota, from the "disfavored" boundary to the
# outlier cut. C-beta deviation is a distance, so it is linear like clash.
CABLAM_FAVORED = 0.05        # above this, unremarkable
CABLAM_OUTLIER = 0.01        # the CaBLAM outlier cut
CA_GEOM_FAVORED = 0.05
CA_GEOM_OUTLIER = 0.005      # the CA-geometry outlier cut
CBETA_ZERO_A = 0.15
CBETA_SATURATION_A = 0.25    # the MolProbity cut itself, so a flagged C-beta reaches 1.0
#: Zero anchor from the measured deviation distribution (p50 0.04, p90 0.16): at 0.0 every
#: residue with a C-beta deposited something and the channel marked the whole protein.

# Non-trans peptides. omegalyze reports the omega dihedral; a cis or twisted peptide is
# flagged. Concern is the twist away from the nearest ideal (0 for cis, 180 for trans).
#
# The saturation anchor IS omegalyze's own boundary, so this channel already satisfied the
# cut-at-1.0 invariant. Checked against the classifier rather than the previous comment here,
# which claimed the boundary was 15 degrees: mmtbx/validation/omegalyze.py find_omega_type
# calls a peptide trans within 30 degrees of 180 and twisted beyond that, so 30 is the cut.
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
    return _worst_per_residue(calibrated, PER_RESTRAINT_METRICS)


#: Channels with one event per *restraint* rather than one per residue. Everything else in
#: the MolProbity set reports once per residue; these report thousands of times per structure
#: (2589 bonds and 3547 angles on a 279-residue model, against 338 Ramachandran results).
PER_RESTRAINT_METRICS = frozenset({"bond", "angle"})


def _worst_per_residue(events, metrics):
    """Keep only each residue's worst event, for the named per-restraint channels.

    This is Rule 6 -- *roll up to a residue with max, never sum* -- applied where it actually
    bites. Concern sums within a metric before it saturates, which is the property that lets
    co-located problems accumulate; but a channel reporting ~13 restraints per residue sums
    those against each other, so the field marks wherever the protein is dense rather than
    where it is wrong. Measured on 1TEC before this roll-up: the angle channel put 12.1% of
    the box past the display threshold against 0.2% for Ramachandran, and dominated the
    combined map entirely.

    Rolling up by max leaves one event per residue per channel, so the geometry channels
    accumulate *with other residues and other channels* like everything else, and a residue
    with many mildly-strained restraints no longer outranks one with a single bad outlier.
    """
    if not metrics:
        return events
    worst, out = {}, []
    for e in events:
        residue = e.meta.get("residue")
        if e.metric not in metrics or residue is None:
            out.append(e)
            continue
        key = (e.metric, residue)
        if key not in worst or e.severity > worst[key].severity:
            worst[key] = e
    return out + list(worst.values())


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


#: Channels grouped by what they are evidence *about*. Within a family the channels are
#: redundant readings of one property, so they combine by max; families are independent lines
#: of evidence, so they accumulate. See ../AGGREGATION_PROPOSAL.md, and channel_survey.py for
#: the measurement that tests whether the grouping actually holds.
FAMILIES = {
    "backbone": ("rama", "cablam", "ca_geom", "omega"),
    "sidechain": ("rota",),
    "sterics": ("clash",),
    "covalent": ("bond", "angle", "cbeta"),
    "map_fit": ("qscore",),
}

#: p for the cross-family p-norm, (sum s_i^p)^(1/p). p=1 is full accumulation and has an
#: interpretable meaning under the cut-at-1.0 calibration: *two independent half-cut problems
#: in the same place equal one flagged problem*. p -> infinity recovers the plain maximum.
DEFAULT_NORM_P = 1.0


def combine_arrays(arrays_by_metric, mode="max", p=DEFAULT_NORM_P, families=None):
    """Combine per-metric arrays into one. ``mode`` is ``"max"`` or ``"family"``.

    ``"max"`` is the original contract: the voxel-wise maximum, never a sum.

    ``"family"`` takes the max within each family and then the p-norm across families, so
    problems of *different kinds* in the same place reinforce while redundant readings of the
    same kind do not. This is the only operator that can show the effect an outlier list
    structurally cannot: a place where nothing individually crosses its threshold but several
    sub-threshold concerns coincide.

    **The result is still clipped to [0, 1], and that does not blunt the point.** Accumulation
    earns its keep entirely below the threshold -- two concerns of 0.3 reaching 0.6 is the
    whole effect, and it happens inside the bounded range. Above 1.0 the display contract has
    nothing left to say anyway (already maximum concern), so clipping costs only contrast
    between regions that were both already saturated, which no colour ramp was showing.
    """
    arrays = list(arrays_by_metric.values())
    if mode == "max":
        return np.maximum.reduce(arrays)
    if mode != "family":
        raise ValueError("unknown combination mode %r" % (mode,))
    if p <= 0:
        raise ValueError("p-norm requires p > 0")

    families = FAMILIES if families is None else families
    family_of = {m: f for f, ms in families.items() for m in ms}
    by_family = {}
    for metric, data in arrays_by_metric.items():
        # An unfamiliar channel is its own family rather than silently joining another:
        # guessing a grouping for it would be exactly the invented weighting this design
        # exists to avoid.
        fam = family_of.get(metric, "other:%s" % metric)
        by_family[fam] = data if fam not in by_family else np.maximum(by_family[fam], data)
    stack = list(by_family.values())
    if len(stack) == 1:
        return np.clip(stack[0], 0.0, 1.0)
    total = np.zeros_like(stack[0])
    for data in stack:
        total += np.power(data, p)
    return np.clip(np.power(total, 1.0 / p), 0.0, 1.0)


def build_concern_fields(events_by_metric, spacing=1.0, sigma=2.0,
                         combine="max", p=DEFAULT_NORM_P, grid_events=None):
    """Build capped per-metric fields and their combination.

    Within one metric, nearby observations add before saturation. Across metrics the default
    is the voxel-wise maximum; pass ``combine="family"`` for max-within-family and a p-norm
    across families (see :func:`combine_arrays`). The per-metric fields are identical either
    way, so the two combinations are directly comparable on the same run.
    """
    events_by_metric = {k: list(v) for k, v in events_by_metric.items() if v}
    if not events_by_metric:
        raise ValueError("at least one non-empty metric is required")
    # ``grid_events`` fixes the output grid independently of which events are being deposited,
    # so fields built from *subsets* of one structure's events land on the same voxels and can
    # be compared mask-to-mask. Without it a subset derives its own smaller bounding box from
    # its own events, and any voxelwise comparison silently comes out empty.
    all_events = (list(grid_events) if grid_events is not None
                  else [e for events in events_by_metric.values() for e in events])
    target = compute_field(all_events, spacing=spacing, sigma=sigma)
    fields = {}
    arrays = {}
    for metric, events in events_by_metric.items():
        raw = compute_field(events, spacing=spacing, sigma=sigma)
        data = np.clip(_sample_on(raw, target), 0.0, 1.0)
        fields[metric] = Field(data, target.origin.copy(), spacing, sigma, 1.0)
        arrays[metric] = data
    fields["combined"] = Field(
        combine_arrays(arrays, mode=combine, p=p), target.origin.copy(), spacing, sigma, 1.0)
    return fields
