"""Cis/twisted peptide validation via mmtbx's omegalyze.

Run omegalyze over the model's hierarchy, turn every residue into a table row,
and drop a marker on each non-trans peptide's anchor position. Cis-prolines,
cis-nonprolines, and twisted peptides each get a distinct marker colour.
Mirror ramachandran.py's shape.
"""

from __future__ import annotations

from typing import Any, Optional

from . import ValidationResult, register

# Distinct marker colours (POINTs, loc == spike) per non-trans classification.
_CIS_PRO = (0x33, 0x99, 0xFF)     # blue    — cis-proline
_CIS_NONPRO = (0xFF, 0x66, 0x33)  # orange  — cis-nonproline
_TWISTED = (0xFF, 0xDD, 0x33)     # yellow  — twisted

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
    markers = []
    for res in result.results:
        # omega_type: OMEGALYZE_TRANS=0, OMEGALYZE_CIS=1, OMEGALYZE_TWISTED=2.
        if res.omega_type == omegalyze.OMEGALYZE_TWISTED:
            kind, color = "twisted", _TWISTED
        elif res.omega_type == omegalyze.OMEGALYZE_CIS:
            if res.resname.strip() == "PRO":
                kind, color = "cis-Pro", _CIS_PRO
            else:
                kind, color = "cis-nonPro", _CIS_NONPRO
        else:
            kind, color = "trans", None

        rows.append([
            res.chain_id,
            res.resid,
            res.resname,
            _fmt(res.omega, 1),
            kind,
        ])
        if res.is_nontrans and res.xyz is not None:
            xyz = tuple(res.xyz)
            markers.append((xyz, xyz, color))  # POINT: loc == spike

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
        markers=markers,
        summary=summary,
    )
