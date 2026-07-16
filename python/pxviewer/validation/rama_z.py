"""Rama-Z (whole-model Ramachandran Z-score) via mmtbx's rama_z.

Unlike the per-residue Ramachandran validator, Rama-Z is a whole-model metric:
``rama_z`` reports a Z-score (and its standard error) for each backbone region
— helix, sheet, loop — plus a combined whole-model score. There is nothing to
anchor per residue, so this validator draws no markers; its output is the
four-region score table. Mirror :mod:`pxviewer.validation.ramachandran` for the
overall module shape.
"""

from __future__ import annotations

from typing import Any, Optional

from . import ValidationResult, register

COLUMNS = ["region", "z_score", "std_err"]

# rama_z region key -> human-readable label, in report order.
_REGIONS = [("H", "Helix"), ("S", "Sheet"), ("L", "Loop"), ("W", "Whole")]


def _fmt(value: Optional[float], ndigits: int) -> str:
    """Format a score/error to ``ndigits`` decimals, blank when None."""
    return "" if value is None else f"{value:.{ndigits}f}"


@register("rama_z", "Rama-Z")
def run(model: Any) -> ValidationResult:
    from libtbx.utils import null_out
    from mmtbx.validation.rama_z import rama_z

    result = rama_z(models=[model], log=null_out())
    scores = result.get_z_scores()  # {region: (z, std_err)}

    rows = []
    for key, label in _REGIONS:
        z, err = scores[key]
        rows.append([label, _fmt(z, 2), _fmt(err, 2)])

    whole_z = scores["W"][0]
    summary = f"whole-model Z = {whole_z:+.2f}"
    return ValidationResult(
        key="rama_z",
        title="Rama-Z",
        columns=COLUMNS,
        rows=rows,
        markers=[],  # whole-model metric: nothing to anchor per residue
        summary=summary,
    )
