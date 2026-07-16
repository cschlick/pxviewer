"""C-beta deviation validation via mmtbx's cbetadev.

Run cbetadev over the model's hierarchy, turn every residue into a table row,
and draw a magenta displacement vector (Cbeta ideal -> observed) on each
outlier. Mirrors ``ramachandran.py``.
"""

from __future__ import annotations

from typing import Any, Optional

from . import ValidationResult, register

# MolProbity magenta — the outlier displacement vector (a LINE, ideal -> obs).
_MAGENTA = (0xFF, 0x00, 0xFF)

COLUMNS = ["chain", "resid", "res", "deviation", "dihedral"]


def _fmt(value: Optional[float], ndigits: int) -> str:
    """Format a distance/angle to ``ndigits`` decimals, blank when None."""
    return "" if value is None else f"{value:.{ndigits}f}"


@register("cbetadev", "Cbeta deviation")
def run(model: Any) -> ValidationResult:
    from mmtbx.validation.cbetadev import cbetadev

    result = cbetadev(pdb_hierarchy=model.get_hierarchy(), outliers_only=False)

    rows = []
    markers = []
    for res in result.results:
        rows.append([
            res.chain_id,
            res.resid,
            res.resname,
            _fmt(res.deviation, 3),
            _fmt(res.dihedral_NABB, 1),
        ])
        if res.outlier and res.ideal_xyz is not None and res.xyz is not None:
            loc = tuple(res.ideal_xyz)
            spike = tuple(res.xyz)
            markers.append((loc, spike, _MAGENTA))  # LINE: ideal -> observed

    summary = f"{result.n_outliers} outliers > 0.25 A"
    return ValidationResult(
        key="cbetadev",
        title="Cbeta deviation",
        columns=COLUMNS,
        rows=rows,
        markers=markers,
        summary=summary,
    )
