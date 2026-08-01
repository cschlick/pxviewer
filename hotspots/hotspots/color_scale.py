"""Deterministic display transforms for bounded concern fields."""
from __future__ import annotations

import numpy as np


DEFAULT_CONCERN_GATE = 0.05
REPORT_QUANTILES = (0.50, 0.80, 0.95, 0.99)


def display_contract():
    """Authoritative absolute primary display plus optional relative mode."""
    return {
        "primary_display": {
            "field": "concern",
            "domain": [0.0, 1.0],
            "normalization": "none",
            "representation": "direct-volume",
            "opacity": "absolute_concern",
            "hue": "absolute_concern",
            "color_anchors": {
                "0.0": "transparent",
                "0.5": "yellow",
                "0.75": "orange",
                "1.0": "red",
            },
        },
        "percentile_display": {
            "role": "optional_relative_contrast",
            "determines_visibility": False,
        },
    }


def empirical_percentile_field(concern, support_mask=None,
                               concern_gate=DEFAULT_CONCERN_GATE):
    """Return a midrank empirical-CDF field and reproducibility metadata.

    Only finite, supported voxels at or above the fixed concern gate define the
    distribution. Ineligible voxels receive percentile zero. Equal values get
    equal midrank percentiles, making flat/saturated regions deterministic.
    """
    concern = np.asarray(concern, dtype=float)
    eligible = np.isfinite(concern) & (concern >= float(concern_gate))
    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=bool)
        if support_mask.shape != concern.shape:
            raise ValueError("support mask and concern field must have one shape")
        eligible &= support_mask

    values = np.sort(concern[eligible])
    percentile = np.zeros(concern.shape, dtype=float)
    if values.size:
        selected = concern[eligible]
        left = np.searchsorted(values, selected, side="left")
        right = np.searchsorted(values, selected, side="right")
        percentile[eligible] = 0.5 * (left + right) / float(values.size)
        quantiles = {
            "q%02d" % int(round(q * 100)): float(np.quantile(values, q))
            for q in REPORT_QUANTILES
        }
    else:
        quantiles = {"q%02d" % int(round(q * 100)): None
                     for q in REPORT_QUANTILES}
    metadata = {
        "method": "midrank_empirical_cdf",
        "eligibility": "finite AND support_mask AND concern >= gate",
        "concern_gate": float(concern_gate),
        "n_eligible": int(values.size),
        "quantiles": quantiles,
        "hue_breaks_percentile": {
            "transparent_below": 0.50,
            "low": 0.50,
            "medium": 0.80,
            "high": 0.95,
            "very_high": 0.99,
        },
        "role": "optional_relative_contrast",
        "determines_visibility": False,
    }
    return percentile, metadata
