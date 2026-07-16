"""Cis/twisted peptide validation via mmtbx's omegalyze.

Run omegalyze over the model's hierarchy, turn every residue into a table row,
and drop a marker on each non-trans peptide's anchor position. Cis-prolines,
cis-nonprolines, and twisted peptides each get a distinct marker colour.
Mirror ramachandran.py's shape.
"""

from __future__ import annotations

from typing import Any, Optional

from . import ValidationResult, register
from ..kinemage import parse_kinemage

COLUMNS = ["chain", "resid", "res", "omega", "type"]


def _fmt(value: Optional[float], ndigits: int) -> str:
    """Format an angle to ``ndigits`` decimals, blank when None."""
    return "" if value is None else f"{value:.{ndigits}f}"


@register("omegalyze", "Cis/twisted peptides")
def run(model: Any) -> ValidationResult:
    from mmtbx.validation import omegalyze

    result = omegalyze.omegalyze(
        pdb_hierarchy=model.get_hierarchy(), nontrans_only=False
    )

    rows = []
    for res in result.results:
        # omega_type: OMEGALYZE_TRANS=0, OMEGALYZE_CIS=1, OMEGALYZE_TWISTED=2.
        if res.omega_type == omegalyze.OMEGALYZE_TWISTED:
            kind = "twisted"
        elif res.omega_type == omegalyze.OMEGALYZE_CIS:
            kind = "cis-Pro" if res.resname.strip() == "PRO" else "cis-nonPro"
        else:
            kind = "trans"

        rows.append([
            res.chain_id,
            res.resid,
            res.resname,
            _fmt(res.omega, 1),
            kind,
        ])

    summary = (
        f"{result.n_cis_proline()} cis-Pro, "
        f"{result.n_cis_general()} cis-nonPro, "
        f"{result.n_twisted_general()} twisted"
    )
    return ValidationResult(
        key="omegalyze",
        title="Cis/twisted peptides",
        columns=COLUMNS,
        rows=rows,
        markup=parse_kinemage(result.as_kinemage()),  # filled cis/twisted triangles
        summary=summary,
    )
