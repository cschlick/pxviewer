"""Add explicit hydrogens to a model with reduce2 (MolProbity).

Contact and clash analysis is far more reliable with real hydrogens: probe2 can
then use actual H positions and directionality instead of guessing from heavy
atoms, so no vdW/H-bond heuristics are needed. reduce2 places and optimizes the
hydrogens (including Asn/Gln/His flips) exactly as MolProbity expects.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any


def hydrogens_available() -> bool:
    """True when the monomer library reduce2 needs is present."""
    from .geometry import monomer_library_available

    return monomer_library_available()


def add_hydrogens(model: Any, *, flip: bool = True) -> Any:
    """Return a new model with explicit H placed and optimized by reduce2.

    Runs reduce2 ``approach=add`` with ``n_terminal_charge=no_charge`` — the mode
    that avoids the N-terminal propeller placement which otherwise crashes on these
    inputs — and, when ``flip`` is set, ``add_flip_movers=True`` for the MolProbity
    Asn/Gln/His flip and H-bond-network optimization. cctbx exposes no in-memory
    entry point, so the result is written to a temp file and reloaded.
    """
    from iotbx.cli_parser import run_program
    from iotbx.data_manager import DataManager
    from mmtbx.programs import reduce2

    workdir = tempfile.mkdtemp(prefix="pxviewer-reduce2-")
    in_path = os.path.join(workdir, "in.pdb")
    out_path = os.path.join(workdir, "in_H.pdb")
    with open(in_path, "w") as fh:
        fh.write(model.model_as_pdb())

    args = [in_path, "approach=add", "n_terminal_charge=no_charge",
            # Ions and other het residues have no hydrogens (so no H restraints);
            # without this reduce2 stops with "Restraints were not found for: NA CA".
            "ignore_missing_restraints=True",
            f"output.filename={out_path}", "output.overwrite=True"]
    if flip:
        args.append("add_flip_movers=True")
    with open(os.devnull, "w") as devnull:
        run_program(program_class=reduce2.Program, args=args, logger=devnull)

    dm = DataManager()
    dm.process_model_file(out_path)
    return dm.get_model()
