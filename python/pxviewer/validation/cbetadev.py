"""C-beta deviation validation via mmtbx's cbetadev.

Run cbetadev over the model's hierarchy, turn every residue into a table row,
and draw a magenta displacement vector (Cbeta ideal -> observed) on each
outlier. Mirrors ``ramachandran.py``.
"""

from __future__ import annotations

from typing import Any, Optional

from . import ValidationResult, register
from ..kinemage import parse_kinemage

COLUMNS = ["chain", "resid", "res", "deviation", "dihedral"]


def _fmt(value: Optional[float], ndigits: int) -> str:
    """Format a distance/angle to ``ndigits`` decimals, blank when None."""
    return "" if value is None else f"{value:.{ndigits}f}"


@register("cbetadev", "Cbeta deviation")
def run(model: Any) -> ValidationResult:
    from mmtbx.validation.cbetadev import cbetadev

    result = cbetadev(pdb_hierarchy=model.get_hierarchy(), outliers_only=False)

    rows = []
    for res in result.results:
        rows.append([
            res.chain_id,
            res.resid,
            res.resname,
            _fmt(res.deviation, 3),
            _fmt(res.dihedral_NABB, 1),
        ])

    summary = f"{result.n_outliers} outliers > 0.25 A"
    return ValidationResult(
        key="cbetadev",
        title="Cbeta deviation",
        columns=COLUMNS,
        rows=rows,
        markup=parse_kinemage(result.as_kinemage()),  # magenta ball + dot scatter
        summary=summary,
    )
