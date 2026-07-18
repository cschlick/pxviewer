"""Place a small-molecule ligand from the monomer library, and fit it into density.

The ideal coordinates come from geostd — the same CIFs that carry the restraints — so any
of its ~54,000 components can be dropped in centred on a chosen point (a marker), and,
where a map is available, settled into the local density with a large radius of
convergence via explode-and-refine (``mmtbx.refinement.real_space.explode_and_refine`` —
the engine inside phenix/lifi's fit, but license-clean and needing no boxing).

Everything here is cctbx (BSD-3-Clause-LBNL); no phenix. See :mod:`pxviewer.desktop` for
the marker wiring.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np

__all__ = ["available", "ideal_atoms", "build_ligand_model", "fit_into_density"]


def _monomer_root() -> Optional[str]:
    from .geometry import monomer_library_root

    return monomer_library_root()


def _cif_path(code: str) -> Optional[str]:
    """geostd buckets a component by its lowercased first character: ``g/data_GOL.cif``."""
    root = _monomer_root()
    code = (code or "").strip().upper()
    if not root or not code:
        return None
    path = os.path.join(root, code[0].lower(), f"data_{code}.cif")
    return path if os.path.isfile(path) else None


def available(code: str) -> bool:
    """Whether the monomer library has this component with ideal coordinates."""
    return _cif_path(code) is not None


def ideal_atoms(code: str) -> Tuple[List[str], List[str], np.ndarray]:
    """``(names, elements, xyz)`` ideal coordinates for a monomer, straight from geostd.

    ``xyz`` is an ``(N, 3)`` array. Raises ValueError if the component is not in the
    library or carries no coordinates.
    """
    import iotbx.cif

    path = _cif_path(code)
    if path is None:
        raise ValueError(f"no monomer '{code.upper()}' in the library")
    block = iotbx.cif.reader(file_path=path).model()[f"comp_{code.upper()}"]
    try:
        names = list(block["_chem_comp_atom.atom_id"])
        elements = list(block["_chem_comp_atom.type_symbol"])
        xyz = np.array(
            [[float(x), float(y), float(z)] for x, y, z in zip(
                block["_chem_comp_atom.x"], block["_chem_comp_atom.y"],
                block["_chem_comp_atom.z"])],
            dtype=float)
    except KeyError as exc:  # pragma: no cover - malformed / coordinate-free entry
        raise ValueError(f"monomer '{code.upper()}' has no ideal coordinates") from exc
    return names, elements, xyz


def build_ligand_model(code: str, center, *, crystal_symmetry: Any = None,
                       data_manager: Any = None) -> Any:
    """A restraints-ready cctbx model of ``code``, its centroid moved to ``center``.

    ``crystal_symmetry`` should be the frame the model will live/refine in (e.g. the
    paired map's) so a later fit indexes the density correctly; without one a loose P1
    box around the ligand is used, which is fine for placing but not for fitting.
    """
    from . import cctbx_io

    names, elements, xyz = ideal_atoms(code)
    center = np.asarray(center, dtype=float).reshape(3)
    placed = xyz - xyz.mean(axis=0) + center  # centroid -> center

    code = code.upper()
    n = len(names)
    # One residue (all atoms share a resseq); the residue name is the code, so cctbx
    # finds its restraints in the same library the coordinates came from.
    base = cctbx_io.model_from_sites(
        placed, elements=elements, names=names,
        resnames=[code] * n, chains=["A"] * n, resseqs=[900] * n,
        label=code, data_manager=data_manager)

    import mmtbx.model

    model = mmtbx.model.manager(
        model_input=None,
        pdb_hierarchy=base.get_hierarchy(),
        crystal_symmetry=crystal_symmetry or base.crystal_symmetry(),
        log=None)
    model.process(make_restraints=True)
    return model


def fit_into_density(model: Any, map_data: Any, *, resolution: float = 3.0,
                     number_of_trials: int = 20, nproc: int = 1) -> Any:
    """Fit ``model`` into ``map_data`` with a large radius of convergence.

    ``mmtbx.refinement.real_space.explode_and_refine``: many trials of a big random
    perturbation ('explode') then real-space refine, scored by map correlation, best
    kept — so orientation and conformation are searched, not just nudged. ``map_data``
    and ``model`` must share a crystal symmetry (frame). Returns the fitted sites as an
    ``(N, 3)`` numpy array (also written back onto ``model``). No boxing needed.
    """
    from mmtbx.refinement.real_space import explode_and_refine

    ear = explode_and_refine.run(
        xray_structure=model.get_xray_structure(),
        pdb_hierarchy=model.get_hierarchy(),
        map_data=map_data,
        restraints_manager=model.get_restraints_manager(),
        resolution=float(resolution),
        number_of_trials=int(number_of_trials),
        nproc=int(nproc),
        # Keep the single best-by-correlation trial. The default "merge_models" scoring
        # averages the ensemble, and its merge step is broken under Python 3 in this
        # mmtbx (an uncomparable-object sort); "cc" takes the best pose, which is what a
        # ligand fit wants anyway.
        score_method=["cc"],
        show=False,
        log=None)
    sites = ear.xray_structure.sites_cart()
    model.set_sites_cart(sites)
    return np.asarray(sites)
