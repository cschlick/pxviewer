"""Per-atom Q-score — how well each atom actually sits in the density.

Q-score (Pintilie et al.) places probe points on a series of shells around an atom, reads the
map at them, and correlates that radial profile against the one a well-resolved atom would
give. So it is a per-atom map-model fit measure: 1 is a textbook fit, and low or negative
means the atom is somewhere the density does not support.

Unlike B-factor or occupancy this is not in the file — it has to be computed against a map,
which is why it goes through the viewer's attribute-coloring path (a per-atom array pushed to
the frontend) rather than one of Mol*'s built-in themes.

cctbx does the real work (``cctbx.maptbx.qscore``); this wraps it for the viewer.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

#: cctbx's own defaults for the probe geometry (see cctbx.programs.qscore's phil): 20 shells
#: from 0.1 to 2.0 A, 32 probes each. Matching them means our number is the number
#: phenix.qscore would report, rather than a lookalike computed some other way.
SHELL_RADII: List[float] = [float(r) for r in np.linspace(0.1, 2.0, 20)]
N_PROBES = 32
RTOL = 0.9

#: Q-score runs 0 (no fit) to 1 (textbook), so color it on a fixed scale rather than
#: stretching to whatever this model happens to span — the point of an absolute metric is
#: that the same color means the same quality in every structure you look at.
DOMAIN = (0.0, 1.0)
#: Low is bad, high is good, so the palette has to run that way round.
PALETTE = "red-yellow-green"

_HYDROGEN = {"H", "D"}


def per_atom_qscore(mmm: Any, *, n_probes: int = N_PROBES, rtol: float = RTOL,
                    shells: Optional[List[float]] = None) -> np.ndarray:
    """Q-score for every atom of ``mmm``'s model, as a float array in the model's atom order.

    Hydrogens get ``nan``: cctbx never scores them (it strips them first), and nan is what
    the viewer's attribute theme draws in its "missing" color, so they read as unmeasured
    rather than as a bad fit.

    Runs against a copy of the manager. ``calc_qscore`` removes hydrogens from the model it
    is handed and calls ``set_model`` back onto the manager — on the live one that would swap
    the model the viewer is streaming out from under it, mid-session.
    """
    from cctbx.maptbx.qscore import calc_qscore

    elements = [e.strip().upper() for e in mmm.model().get_hierarchy().atoms().extract_element()]
    scored_atoms = np.array([e not in _HYDROGEN for e in elements], dtype=bool)

    result = calc_qscore(mmm.deep_copy(), shells=list(shells or SHELL_RADII),
                         n_probes=n_probes, rtol=rtol, nproc=1)
    scored = np.asarray(result["qscore_per_atom"], dtype=float).reshape(-1)

    expected = int(scored_atoms.sum())
    if scored.size != expected:
        raise ValueError(
            f"q-score returned {scored.size} values for {expected} non-hydrogen atoms")

    values = np.full(len(elements), np.nan, dtype=float)
    values[scored_atoms] = scored
    return values
