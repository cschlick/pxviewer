"""Rotamer (side-chain chi angle) validation via mmtbx's rotalyze.

Run rotalyze over the model's hierarchy, turn every residue into a table row
(its rotamer name and up to four chi angles), and drop a gold marker on each
outlier's anchor position. Mirrors :mod:`pxviewer.validation.ramachandran`.
"""

from __future__ import annotations

from typing import Any, Optional

from . import ValidationResult, register

# MolProbity gold — the outlier marker colour (a POINT, loc == spike).
_GOLD = (0xFF, 0xD7, 0x00)

COLUMNS = ["chain", "resid", "res", "rotamer", "chi1", "chi2", "chi3", "chi4", "score"]


def _fmt(value: Optional[float], ndigits: int) -> str:
    """Format an angle/score to ``ndigits`` decimals, blank when None."""
    return "" if value is None else f"{value:.{ndigits}f}"


def _chi(chi_angles: Optional[list], index: int) -> str:
    """Format the ``index``-th chi angle, blank when absent or None."""
    if not chi_angles or index >= len(chi_angles):
        return ""
    return _fmt(chi_angles[index], 1)


@register("rotamers", "Rotamers")
def run(model: Any) -> ValidationResult:
    from mmtbx.validation.rotalyze import rotalyze

    result = rotalyze(pdb_hierarchy=model.get_hierarchy(), outliers_only=False)

    rows = []
    markers = []
    for res in result.results:
        rows.append([
            res.chain_id,
            res.resid,
            res.resname,
            res.rotamer_name,
            _chi(res.chi_angles, 0),
            _chi(res.chi_angles, 1),
            _chi(res.chi_angles, 2),
            _chi(res.chi_angles, 3),
            _fmt(res.score, 2),
        ])
        if res.outlier and res.xyz is not None:
            xyz = tuple(res.xyz)
            markers.append((xyz, xyz, _GOLD))  # POINT: loc == spike

    summary = f"{result.n_outliers} outliers, {result.percent_favored:.1f}% favored"
    return ValidationResult(
        key="rotamers",
        title="Rotamers",
        columns=COLUMNS,
        rows=rows,
        markers=markers,
        summary=summary,
    )
