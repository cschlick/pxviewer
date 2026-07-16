"""CaBLAM (CA-based low-resolution annotation) validation via mmtbx's cablamalyze.

Run cablamalyze over the model's hierarchy, turn every residue into a table
row, and drop a distinctly-coloured marker on each of the three kinds of flag
CaBLAM raises: a CaBLAM *outlier*, a CaBLAM *disfavored* conformation, and a
*CA-geometry* outlier. Mirrors :mod:`pxviewer.validation.ramachandran`.
"""

from __future__ import annotations

from typing import Any, Optional

from . import ValidationResult, register
from ..kinemage import parse_kinemage

COLUMNS = ["chain", "resid", "res", "cablam", "ca_geom", "type"]


def _fmt(value: Optional[float], ndigits: int) -> str:
    """Format a score to ``ndigits`` decimals, blank when None."""
    return "" if value is None else f"{value:.{ndigits}f}"


def _classify(feedback: Any) -> str:
    """A one-word ``type`` label for the residue, prioritising the flags that
    drive the markers, then falling back to the identified secondary structure."""
    if feedback.cablam_outlier:
        return "CaBLAM outlier"
    if feedback.cablam_disfavored:
        return "CaBLAM disfavored"
    if feedback.c_alpha_geom_outlier:
        return "CA geom outlier"
    if feedback.alpha:
        return "alpha"
    if feedback.beta:
        return "beta"
    if feedback.threeten:
        return "threeten"
    return ""


@register("cablam", "CaBLAM")
def run(model: Any) -> ValidationResult:
    from mmtbx.validation.cablam import cablamalyze
    from libtbx.utils import null_out

    result = cablamalyze(
        pdb_hierarchy=model.get_hierarchy(),
        outliers_only=False,
        out=null_out(),
        quiet=True,
    )

    rows = []
    n_outlier = n_disfavored = n_geom = 0
    for res in result.results:
        fb = res.feedback
        rows.append([
            res.chain_id,
            res.resid,
            res.resname,
            _fmt(res.scores.cablam, 3),
            _fmt(res.scores.c_alpha_geom, 3),
            _classify(fb),
        ])
        n_outlier += bool(fb.cablam_outlier)
        n_disfavored += bool(fb.cablam_disfavored)
        n_geom += bool(fb.c_alpha_geom_outlier)

    summary = (
        f"{n_outlier} CaBLAM outliers, {n_disfavored} disfavored, "
        f"{n_geom} CA-geometry outliers / {len(result.results)} residues"
    )
    return ValidationResult(
        key="cablam",
        title="CaBLAM",
        columns=COLUMNS,
        rows=rows,
        markup=parse_kinemage(result.as_kinemage()),  # outlier/disfavored/CA-geom vectors + wheels
        summary=summary,
    )
