"""Rotamer (side-chain chi angle) validation via mmtbx's rotalyze.

Run rotalyze over the model's hierarchy, turn every residue into a table row (its
rotamer name and up to four chi angles), and draw MolProbity's gold side-chain
markup on each outlier. rotalyze's own ``as_kinemage`` needs the PDB Chemical
Component Dictionary (absent here), so we draw the same gold side-chain bonds from
interatomic distance instead — same atoms, no extra reference data.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import ValidationResult, register

_GOLD = (0xFF, 0xD7, 0x00)          # MolProbity gold for rotamer-outlier side chains
_MAINCHAIN = {"N", "C", "O", "OXT"}  # excluded so only the side chain (+ Ca-Cb) is drawn
_BOND_MAX = 1.9                      # A: heavy-atom covalent bond cutoff

COLUMNS = ["chain", "resid", "res", "rotamer", "chi1", "chi2", "chi3", "chi4", "score"]


def _sidechain_vectors(hierarchy, outlier_ids) -> list:
    """Gold line segments along each outlier residue's side-chain bonds, inferred
    from interatomic distance (Ca included so the Ca-Cb bond anchors the side chain)."""
    segments = []
    for chain in hierarchy.chains():
        cid = chain.id.strip()
        for rg in chain.residue_groups():
            if (cid, rg.resid().strip()) not in outlier_ids:
                continue
            for ag in rg.atom_groups():
                atoms = [np.array(a.xyz, dtype=float) for a in ag.atoms()
                         if a.element.strip().upper() != "H"
                         and (a.name.strip() == "CA" or a.name.strip() not in _MAINCHAIN)]
                for i in range(len(atoms)):
                    for j in range(i + 1, len(atoms)):
                        if np.linalg.norm(atoms[i] - atoms[j]) < _BOND_MAX:
                            segments.append([atoms[i].tolist(), atoms[j].tolist()])
    return segments


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

    outlier_ids = {(res.chain_id.strip(), res.resid.strip())
                   for res in result.results if res.outlier}
    segments = _sidechain_vectors(model.get_hierarchy(), outlier_ids)
    markup = [{"kind": "vectors", "color": list(_GOLD), "segments": segments}] if segments else []

    summary = f"{result.n_outliers} outliers, {result.percent_favored:.1f}% favored"
    return ValidationResult(
        key="rotamers",
        title="Rotamers",
        columns=COLUMNS,
        rows=rows,
        markup=markup,
        summary=summary,
    )
