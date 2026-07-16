"""Run cctbx's probe2 (MolProbity all-atom contacts) and extract its dot surface.

probe2 produces a flat list of contact "dots": each has a surface position
(``loc``), a spike tip (``spike``; for an overlap it points into the clash), an
interaction ``type`` (wide/close contact, H-bond, small/bad overlap), and the
``gap`` between atoms. We turn that into drawable dots — a position, a spike tip,
and a MolProbity colour — for the viewer to render as a point cloud plus clash
spikes.

probe2 wants explicit hydrogens for a full analysis; when the model has none we
fall back to a heavy-atom run (``ignore_lack_of_explicit_hydrogens``).
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from typing import Any, List, Tuple

# MolProbity gap -> RGB (mirrors probe2._color_for_gap); H-bonds get green-tint.
_HBOND_RGB = (0x8C, 0xFF, 0xB4)


def _dot_rgb(dtype: str, gap: float) -> Tuple[int, int, int]:
    if dtype == "hb":
        return _HBOND_RGB
    if gap > 0.35:
        return (0x40, 0x40, 0xFF)  # blue (wide contact)
    if gap > 0.25:
        return (0x60, 0xA8, 0xFF)  # sky
    if gap > 0.15:
        return (0x00, 0xD0, 0xB0)  # sea
    if gap > 0.0:
        return (0x30, 0xD0, 0x30)  # green
    if gap > -0.1:
        return (0xD8, 0xD8, 0x60)  # yellowtint
    if gap > -0.2:
        return (0xF5, 0xF5, 0x00)  # yellow
    if gap > -0.3:
        return (0xFF, 0x9A, 0x00)  # orange
    if gap > -0.4:
        return (0xFF, 0x30, 0x30)  # red
    return (0xFF, 0x66, 0xB4)      # hotpink (bad clash)


def _model_has_hydrogens(model: Any) -> bool:
    elements = model.get_hierarchy().atoms().extract_element()
    return any(e.strip() == "H" for e in elements)


def run_probe_dots(model: Any) -> List[dict]:
    """Run probe2 on a cctbx model; return the raw ``flat_results`` dot list."""
    from iotbx.cli_parser import run_program
    from mmtbx.programs import probe2

    workdir = tempfile.mkdtemp(prefix="pxviewer-probe-")
    model_path = os.path.join(workdir, "model.pdb")
    with open(model_path, "w") as fh:
        fh.write(model.model_as_pdb())
    out_path = os.path.join(workdir, "probe.json")

    args = [model_path, "approach=self", "source_selection=all",
            "output.format=json", f"output.file_name={out_path}"]
    if not _model_has_hydrogens(model):
        args.append("ignore_lack_of_explicit_hydrogens=True")

    with open(os.devnull, "w") as devnull:
        run_program(program_class=probe2.Program, args=args, logger=devnull)
    with open(out_path) as fh:
        return json.load(fh)["flat_results"]


def probe_dots(model: Any) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]]:
    """``[(loc, spike, rgb), ...]`` — drawable dots with MolProbity colours."""
    dots = []
    for row in run_probe_dots(model):
        dots.append((tuple(row["loc"]), tuple(row["spike"]), _dot_rgb(row["type"], row["gap"])))
    return dots


def encode_dots(dots) -> bytes:
    """Pack dots for the wire: ``[u32 n][per dot: f32 loc xyz, f32 spike xyz, u32 rgb]``."""
    parts = [struct.pack("<I", len(dots))]
    for (lx, ly, lz), (sx, sy, sz), (r, g, b) in dots:
        rgb = (int(r) << 16) | (int(g) << 8) | int(b)
        parts.append(struct.pack("<6fI", lx, ly, lz, sx, sy, sz, rgb))
    return b"".join(parts)
