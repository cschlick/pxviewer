"""The hotspot field: severity-weighted intensity over a neighbourhood.

A second, separate quantity from bounded concern. See ../HOTSPOT_DENSITY_DESIGN.md for why
there are two fields; briefly, the concern field answers *where exactly is this problem* and
cannot accumulate beyond ~2 A, which is sub-residue — and something that only accumulates
within a residue does not need to be a field at all. This one answers *which neighbourhoods
carry a concentration of trouble*, and accumulates across residues by construction.

    lambda(x) = sum_i  s_i * K(|x - x_i| / R),     K(u) = 1 - u^2  for u <= 1,  R = 6 A

**Units are flagged-outlier-equivalents.** ``K(0) = 1`` and every channel's community cut is
calibrated to concern 1.0 (see calibration_cuts.py), so a single flagged outlier sitting alone
reads exactly 1.0 and ten mild concerns of 0.2 in one pocket read about 2. The scale is
inherited from whoever defined each community threshold; nothing here is fitted.

Note what this does and does not lift. A *pair* of weak concerns still cannot exceed their sum
(two 0.2s read 0.4, correctly less than one real outlier). What lifts is the ceiling on *many*:
the kernel transmits most of an event's severity across the whole neighbourhood instead of
killing it at 2 A, so a crowd of weak problems genuinely sums. That is the effect an outlier
list structurally cannot reproduce.

**R = 6 A is the one constant in this project not inherited from a community threshold.** It
comes from the measured clustering of sub-threshold events (nearest cross-family neighbour:
median 3.75 A, p75 5.40, p90 7.08). State it, fix it, and do not tune it per figure -- that is
where this design would start becoming a fitted metric.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from field import Field, _splat, compute_field

#: Neighbourhood radius, angstrom. See the module docstring before changing it.
DEFAULT_RADIUS = 6.0

#: Display knee: as much trouble here as one flagged outlier.
KNEE = 1.0
#: Display ceiling for the absolute domain [0, CEILING]. Not a clip on the data.
CEILING = 3.0


def epanechnikov_stencil(radius: float, spacing: float) -> np.ndarray:
    """``K(u) = 1 - u^2`` on a voxel grid, K(0) = 1, compact support at ``radius``.

    Compact support matters: "within 6 A" is then literally true rather than approximately
    true, so the unit can be stated without a tail caveat.
    """
    n = int(np.ceil(radius / spacing))
    ax = np.arange(-n, n + 1) * spacing
    d2 = (ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    u2 = d2 / (radius * radius)
    return np.where(u2 <= 1.0, 1.0 - u2, 0.0)


def _footprint_peak(pts: np.ndarray, radius: float) -> float:
    """Peak of an event's own footprint, ``max_a sum_b K(|a-b|/R)``.

    Divided out so an event peaks at its severity regardless of how many atoms it implicates,
    which is what makes "one flagged outlier reads 1.0" true rather than atom-count dependent.

    It is exact for a single-atom event and slightly generous for a multi-atom one: the
    normalizer is evaluated *at an atom*, while a multi-atom footprint's true maximum can sit
    between atoms. Measured: 1.022 for a compact 5-atom event, against 1.083 for the same
    event through compute_field, so this is the same convention the locator already uses and
    a smaller overshoot than it. Left as-is for consistency; if the anchor ever needs to be
    exact, normalize by the numerical maximum of the footprint rather than its value at an
    atom, and change both fields together.
    """
    if len(pts) == 1:
        return 1.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2) / radius
    k = np.where(d <= 1.0, 1.0 - d * d, 0.0)
    return float(k.sum(axis=1).max())


def compute_density(events, spacing: float = 1.0, radius: float = DEFAULT_RADIUS,
                    grid_events=None, weight=None) -> Optional[Field]:
    """Severity-weighted intensity in flagged-outlier-equivalents.

    ``grid_events`` fixes the output grid independently of what is deposited, so densities
    built from subsets of one structure's events land on the same voxels and can be compared
    mask to mask. ``weight`` overrides the per-event severity -- pass ``lambda e: 1.0`` to get
    a plain event count, or use :func:`atom_density` for the packing-bias control.
    """
    from scipy.signal import fftconvolve

    events = [e for e in events if e.atoms_xyz]
    if not events:
        return None
    grid_events = list(grid_events) if grid_events is not None else events

    # Same box convention as compute_field, so the two fields are voxel-comparable. The
    # padding is the kernel radius here rather than 3 sigma.
    pts_all = np.array([xyz for e in grid_events for xyz in e.atoms_xyz], float)
    lo = np.floor((pts_all.min(axis=0) - radius) / spacing) * spacing
    hi = pts_all.max(axis=0) + radius
    shape = tuple((np.ceil((hi - lo) / spacing).astype(int) + 1).tolist())

    grid = np.zeros(shape, float)
    for e in events:
        s = float(e.severity) if weight is None else float(weight(e))
        if s <= 0:
            continue
        pts = np.asarray(e.atoms_xyz, float).reshape(-1, 3)
        w = s / _footprint_peak(pts, radius)
        for a in pts:
            _splat(grid, (a - lo) / spacing, w)

    data = fftconvolve(grid, epanechnikov_stencil(radius, spacing), mode="same")
    np.clip(data, 0.0, None, out=data)      # fft rounding can produce tiny negatives
    return Field(data=data, origin=lo, spacing=spacing, sigma=radius, reference_level=1.0)


def atom_density(sites_xyz, spacing: float = 1.0, radius: float = DEFAULT_RADIUS,
                 grid_events=None, origin=None, shape=None) -> np.ndarray:
    """The packing control: the same kernel over every heavy atom, weight 1 each.

    A 6 A ball in a buried core holds more atoms, so it holds more validation events, so it
    reads hotter -- from packing alone rather than from being worse. That confound has to be
    measured before the hotspot field is believed, and this is what it is measured against.
    """
    from scipy.signal import fftconvolve

    sites = np.asarray(sites_xyz, float).reshape(-1, 3)
    grid = np.zeros(tuple(shape), float)
    for a in sites:
        _splat(grid, (a - np.asarray(origin, float)) / spacing, 1.0)
    data = fftconvolve(grid, epanechnikov_stencil(radius, spacing), mode="same")
    np.clip(data, 0.0, None, out=data)
    return data


def build_density_fields(events_by_metric: Dict[str, List], spacing: float = 1.0,
                         radius: float = DEFAULT_RADIUS, grid_events=None) -> Dict[str, Field]:
    """Per-family densities and their sum.

    Families combine by **sum** here, not by max: the whole point of this field is that
    independent lines of evidence in one neighbourhood add up. Within a family the members are
    redundant readings of one property (measured: residue-level Jaccard 0.088 within against
    0.000 across, see corpus/channel_survey.py), so they are maxed per residue upstream by the
    concern layer rather than double-counted here.
    """
    from concern import FAMILIES

    family_of = {m: f for f, ms in FAMILIES.items() for m in ms}
    by_family: Dict[str, List] = {}
    for metric, evs in events_by_metric.items():
        by_family.setdefault(family_of.get(metric, "other:%s" % metric), []).extend(evs)
    if not by_family:
        return {}

    all_events = ([e for evs in events_by_metric.values() for e in evs]
                  if grid_events is None else list(grid_events))
    out: Dict[str, Field] = {}
    total = None
    for family, evs in by_family.items():
        f = compute_density(evs, spacing=spacing, radius=radius, grid_events=all_events)
        if f is None:
            continue
        out[family] = f
        total = f.data.copy() if total is None else total + f.data
    if total is not None:
        out["combined"] = Field(total, out[next(iter(out))].origin.copy(), spacing,
                                radius, 1.0)
    return out
