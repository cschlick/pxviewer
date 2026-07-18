"""Place a small-molecule ligand — from the monomer library or a SMILES string — and fit
it into density.

For a library component the ideal coordinates come from geostd (the same CIFs that carry
the restraints), so any of its ~54,000 entries can be dropped in centred on a chosen point
(a marker). For anything else, a SMILES string is embedded to a 3D conformer by rdkit and
that conformer's geometry is written into a monomer restraint CIF on the fly, so a novel
ligand is placed and restrained the same way. Either way, where a map is available the
placed model is settled into the local density with a large radius of convergence via
explode-and-refine (``mmtbx.refinement.real_space.explode_and_refine`` — the engine inside
phenix/lifi's fit, but license-clean and needing no boxing).

Everything here is cctbx + rdkit (both BSD); no phenix. See :mod:`pxviewer.desktop` for
the marker wiring.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from typing import Any, List, Optional, Tuple

import numpy as np

__all__ = ["available", "ideal_atoms", "build_ligand_model",
           "build_ligand_from_smiles", "coarse_orient", "fit_into_density"]


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
    names, elements, xyz = ideal_atoms(code)
    center = np.asarray(center, dtype=float).reshape(3)
    placed = xyz - xyz.mean(axis=0) + center  # centroid -> center
    # One residue whose name is the code, so cctbx finds its restraints in the same
    # library the coordinates came from — no restraint CIF needed.
    return _assemble_model(placed, names, elements, code.upper(),
                           crystal_symmetry=crystal_symmetry, data_manager=data_manager)


def build_ligand_from_smiles(smiles: str, code: str, center, *,
                             crystal_symmetry: Any = None,
                             data_manager: Any = None) -> Any:
    """A restraints-ready cctbx model of an arbitrary ``smiles`` ligand, centroid at
    ``center`` — for anything not in the monomer library.

    rdkit parses the SMILES, adds hydrogens and embeds a 3D conformer (cleaned up with
    MMFF); that single conformer supplies both the coordinates and — measured off it — the
    ideal bond lengths and angles, which are written into a monomer restraint CIF that
    cctbx reads to build the geometry. Coordinates and restraints therefore come from the
    same source, so the placed model is immediately fit-ready, exactly like
    :func:`build_ligand_model`. ``code`` is the (<=3-char) residue name it is filed under.

    Raises ValueError if rdkit cannot parse or embed the SMILES.
    """
    code = (code or "LIG").strip().upper()[:3] or "LIG"
    names, elements, xyz, cif_object = _smiles_restraints(smiles, code)
    center = np.asarray(center, dtype=float).reshape(3)
    placed = xyz - xyz.mean(axis=0) + center  # centroid -> center
    return _assemble_model(placed, names, elements, code,
                           crystal_symmetry=crystal_symmetry, data_manager=data_manager,
                           restraint_objects=[(f"{code}.cif", cif_object)])


def _assemble_model(placed: np.ndarray, names: List[str], elements: List[str], code: str,
                    *, crystal_symmetry: Any = None, data_manager: Any = None,
                    restraint_objects: Any = None) -> Any:
    """A processed, restraints-ready one-residue model from placed coordinates.

    ``crystal_symmetry`` should be the frame the model will live/refine in (e.g. the paired
    map's) so a later fit indexes the density correctly; without one a loose P1 box around
    the ligand is used, fine for placing but not for fitting. ``restraint_objects``, when
    given, carries the ligand's own restraint CIF (the SMILES path); otherwise the residue
    name must resolve in the monomer library.
    """
    from . import cctbx_io
    import mmtbx.model

    n = len(names)
    base = cctbx_io.model_from_sites(
        placed, elements=elements, names=names,
        resnames=[code] * n, chains=["A"] * n, resseqs=[900] * n,
        label=code, data_manager=data_manager)
    model = mmtbx.model.manager(
        model_input=None,
        pdb_hierarchy=base.get_hierarchy(),
        crystal_symmetry=crystal_symmetry or base.crystal_symmetry(),
        restraint_objects=restraint_objects,
        log=None)
    model.process(make_restraints=True)
    return model


def _smiles_restraints(smiles: str, code: str
                       ) -> Tuple[List[str], List[str], np.ndarray, Any]:
    """``(names, elements, xyz, cif_object)`` for a SMILES ligand.

    The atom names are generated once here and used for both the coordinates and the
    restraint CIF, so cctbx maps the two together. ``cif_object`` is an in-memory
    ``iotbx.cif`` monomer block (``comp_list`` + ``comp_<code>``) with ideal bond/angle
    values read straight off the embedded conformer.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from mmtbx.ligands import rdkit_utils
    import iotbx.cif

    if not (smiles or "").strip():
        raise ValueError("empty SMILES")
    try:
        # embed3d + addHs: a hydrogen-complete 3D conformer.
        mol = rdkit_utils.mol_from_smiles(smiles.strip(), embed3d=True)
    except Exception as exc:
        raise ValueError(f"rdkit could not build a 3D model from {smiles!r}: {exc}") from exc
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"rdkit could not embed a conformer for {smiles!r}")
    try:  # tidy the geometry so measured ideals are sensible; not fatal if it can't
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:  # pragma: no cover - force field just not parameterised
        pass

    conf = mol.GetConformer()
    xyz = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    # Unique PDB-style names: element symbol + a per-element running index (C1, C2, O1…).
    counts: dict = {}
    names = []
    for el in elements:
        counts[el] = counts.get(el, 0) + 1
        names.append(f"{el}{counts[el]}")

    cif_object = iotbx.cif.reader(
        input_string=_monomer_cif_text(mol, code, names, elements, xyz)).model()
    return names, elements, xyz, cif_object


_BOND_TYPES = None


def _monomer_cif_text(mol: Any, code: str, names: List[str], elements: List[str],
                      xyz: np.ndarray) -> str:
    """A CCP4-monomer restraint CIF for ``mol`` — ideal bond/angle values off its
    conformer. Enough for cctbx to build a full geometry restraints manager: atoms (with
    the element as its own generic energy type, which the energy library accepts), bonds,
    and every bond–bond angle."""
    from rdkit import Chem

    global _BOND_TYPES
    if _BOND_TYPES is None:
        _BOND_TYPES = {Chem.BondType.SINGLE: "single", Chem.BondType.DOUBLE: "double",
                       Chem.BondType.TRIPLE: "triple", Chem.BondType.AROMATIC: "aromatic"}

    def angle(i: int, j: int, k: int) -> float:
        u, v = xyz[i] - xyz[j], xyz[k] - xyz[j]
        c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    n_all = mol.GetNumAtoms()
    n_nh = sum(1 for e in elements if e != "H")
    out = [
        "data_comp_list", "loop_",
        "_chem_comp.id", "_chem_comp.three_letter_code", "_chem_comp.name",
        "_chem_comp.group", "_chem_comp.number_atoms_all",
        "_chem_comp.number_atoms_nh", "_chem_comp.desc_level",
        f"{code} {code} 'ligand from SMILES' non-polymer {n_all} {n_nh} .",
        f"data_comp_{code}", "loop_",
        "_chem_comp_atom.comp_id", "_chem_comp_atom.atom_id", "_chem_comp_atom.type_symbol",
        "_chem_comp_atom.type_energy", "_chem_comp_atom.x", "_chem_comp_atom.y",
        "_chem_comp_atom.z",
    ]
    for i in range(n_all):
        out.append(f"{code} {names[i]} {elements[i]} {elements[i]} "
                   f"{xyz[i][0]:.4f} {xyz[i][1]:.4f} {xyz[i][2]:.4f}")

    out += ["loop_", "_chem_comp_bond.comp_id", "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2", "_chem_comp_bond.type",
            "_chem_comp_bond.value_dist", "_chem_comp_bond.value_dist_esd"]
    neighbours: dict = {i: set() for i in range(n_all)}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        neighbours[i].add(j)
        neighbours[j].add(i)
        dist = float(np.linalg.norm(xyz[i] - xyz[j]))
        out.append(f"{code} {names[i]} {names[j]} "
                   f"{_BOND_TYPES.get(b.GetBondType(), 'single')} {dist:.4f} 0.020")

    out += ["loop_", "_chem_comp_angle.comp_id", "_chem_comp_angle.atom_id_1",
            "_chem_comp_angle.atom_id_2", "_chem_comp_angle.atom_id_3",
            "_chem_comp_angle.value_angle", "_chem_comp_angle.value_angle_esd"]
    for j in range(n_all):  # every pair of bonds sharing atom j is an angle about j
        ns = sorted(neighbours[j])
        for a in range(len(ns)):
            for b in range(a + 1, len(ns)):
                i, k = ns[a], ns[b]
                out.append(f"{code} {names[i]} {names[j]} {names[k]} "
                           f"{angle(i, j, k):.3f} 3.0")
    return "\n".join(out) + "\n"


def coarse_orient(model: Any, map_data: Any, *, step_deg: int = 30) -> Any:
    """Rotate the rigid ligand about its centroid to the best-scoring orientation in the
    density, centroid held where it is. Gives explode-and-refine a good starting
    orientation — the thing it can otherwise get trapped away from for a compact ligand,
    since it perturbs and refines but does not itself do a global rotation search.

    A coarse Euler grid, each orientation scored by summed density at the atoms
    (``maptbx.real_space_target_simple``). Rigid + a C++ score, so a few hundred–thousand
    evaluations stay well under a second. Writes the winning orientation onto ``model``.
    """
    from cctbx import maptbx
    from cctbx.array_family import flex
    from scitbx.math import euler_angles

    unit_cell = model.crystal_symmetry().unit_cell()
    sites = model.get_sites_cart()
    centered = sites.as_numpy_array() - sites.as_numpy_array().mean(axis=0)
    center = sites.as_numpy_array().mean(axis=0)

    def score(sc):
        return maptbx.real_space_target_simple(
            unit_cell=unit_cell, density_map=map_data, sites_cart=sc)

    best_score, best_sites = score(sites), sites  # baseline: leave it as placed
    for a in range(0, 360, step_deg):
        for b in range(0, 181, step_deg):
            for c in range(0, 360, step_deg):
                rot = np.array(euler_angles.xyz_matrix(a, b, c)).reshape(3, 3)
                cand = flex.vec3_double(
                    np.ascontiguousarray(centered @ rot.T + center))
                s = score(cand)
                if s > best_score:
                    best_score, best_sites = s, cand
    model.set_sites_cart(best_sites)
    return model


def fit_into_density(model: Any, map_data: Any, *, resolution: float = 3.0,
                     number_of_trials: int = 20, nproc: int = 1,
                     presearch: bool = True) -> Any:
    """Fit ``model`` into ``map_data`` with a large radius of convergence.

    First a coarse rotational pre-search (``presearch``) rotates the rigid ligand to the
    best-scoring orientation in the density, then
    ``mmtbx.refinement.real_space.explode_and_refine`` does many trials of a big random
    perturbation ('explode') then real-space refine, scored by map correlation, best
    kept — so orientation *and* conformation are searched. The pre-search matters because
    explode perturbs from wherever it starts; a badly-oriented compact ligand can trap
    it. ``map_data`` and ``model`` must share a crystal symmetry (frame). Returns the
    fitted sites as an ``(N, 3)`` numpy array (also written back onto ``model``). No
    boxing needed.
    """
    if presearch:
        coarse_orient(model, map_data)

    from mmtbx.refinement.real_space import explode_and_refine

    # explode_and_refine writes scratch PDBs (merged.pdb, …) to the current directory, so
    # run it in a throwaway temp dir. The app's own file I/O uses absolute paths, so the
    # process-wide chdir does not disturb it, and the result is read from memory below.
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="pxviewer-ligand-")
    try:
        os.chdir(tmp)
        ear = explode_and_refine.run(
            xray_structure=model.get_xray_structure(),
            pdb_hierarchy=model.get_hierarchy(),
            map_data=map_data,
            restraints_manager=model.get_restraints_manager(),
            resolution=float(resolution),
            number_of_trials=int(number_of_trials),
            nproc=int(nproc),
            # Keep the single best-by-correlation trial. The default "merge_models"
            # scoring averages the ensemble, and its merge step is broken under Python 3
            # in this mmtbx (an uncomparable-object sort); "cc" takes the best pose, which
            # is what a ligand fit wants anyway.
            score_method=["cc"],
            show=False,
            log=None)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    sites = ear.xray_structure.sites_cart()
    model.set_sites_cart(sites)
    return np.asarray(sites)
