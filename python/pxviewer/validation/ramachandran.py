"""Ramachandran (phi/psi) validation via mmtbx's ramalyze.

Reference validator for the package: run ramalyze over the model's hierarchy,
turn every residue into a table row, and drop a green marker on each outlier's
anchor position. Mirror this module's shape when adding a new validator.
"""

from __future__ import annotations

from typing import Any, Optional

from . import ValidationResult, register
from ..kinemage import parse_kinemage

COLUMNS = ["chain", "resid", "res", "phi", "psi", "type", "score"]


def _fmt(value: Optional[float], ndigits: int) -> str:
    """Format an angle/score to ``ndigits`` decimals, blank when None."""
    return "" if value is None else f"{value:.{ndigits}f}"


@register("ramachandran", "Ramachandran")
def run(model: Any) -> ValidationResult:
    from mmtbx.validation.ramalyze import ramalyze

    result = ramalyze(pdb_hierarchy=model.get_hierarchy(), outliers_only=False)

    rows = []
    for res in result.results:
        rows.append([
            res.chain_id,
            res.resid,
            res.resname,
            _fmt(res.phi, 1),
            _fmt(res.psi, 1),
            res.ramalyze_type(),
            _fmt(res.score, 2),
        ])

    summary = (
        f"{result.n_outliers} outliers, {result.percent_favored:.1f}% favored "
        f"/ {len(result.results)} residues"
    )
    return ValidationResult(
        key="ramachandran",
        title="Ramachandran",
        columns=COLUMNS,
        rows=rows,
        markup=parse_kinemage(result.as_kinemage()),  # green Ca outlier vectors
        summary=summary,
    )
