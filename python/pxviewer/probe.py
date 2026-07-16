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
import struct
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


def run_probe_dots(model: Any, *, data_manager: Any = None) -> List[dict]:
    """Run probe2 on a cctbx model in memory; return the raw ``flat_results`` dots.

    No disk round-trip: the model goes straight into a DataManager (no PDB write) and
    probe2's JSON is captured from ``run()``'s return value with
    ``output.write_files=False`` (nothing written or read back). ``run_program`` isn't
    used because it discards that return value — probe2's ``get_results`` is commented
    out — so the Program is built and run directly, mirroring how the CLI parser
    assembles the master phil (probe2's scope plus the base output scope).
    """
    import iotbx.phil
    from libtbx.program_template import ProgramTemplate
    from libtbx.utils import null_out
    from mmtbx.programs import probe2

    from .cctbx_io import data_manager as _dm

    dm = _dm(data_manager)
    dm.add_model("model", model)

    master = iotbx.phil.parse(probe2.Program.master_phil_str, process_includes=True)
    master.adopt_scope(iotbx.phil.parse(ProgramTemplate.output_phil_str))
    params = master.extract()
    params.approach = "self"
    params.source_selection = "all"
    params.output.format = "json"
    params.output.write_files = False        # capture the JSON from run()'s return, not a file
    params.output.filename = "probe.json"    # set (never written) so validate() stays quiet
    if not _model_has_hydrogens(model):
        params.ignore_lack_of_explicit_hydrogens = True

    task = probe2.Program(dm, params, master_phil=master, logger=null_out())
    task.validate()
    _results, out_string = task.run()
    return json.loads(out_string)["flat_results"]


# probe2 dot types: wc/cc wide+close contact, hb hydrogen bond, so small overlap,
# bo bad overlap. MolProbity "clashes" are the bad overlaps (overlap > 0.4 A).
_CLASH_TYPES = {"bo"}


def probe_dots(
    model: Any, *, only_clashes: bool = False, data_manager: Any = None,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]]:
    """``[(loc, spike, rgb), ...]`` — drawable dots with MolProbity colours.

    With ``only_clashes`` the result is limited to bad-overlap dots (the MolProbity
    clashes), for drawing a clash-only overlay separate from the full surface.
    """
    dots = []
    for row in run_probe_dots(model, data_manager=data_manager):
        if only_clashes and row["type"] not in _CLASH_TYPES:
            continue
        dots.append((tuple(row["loc"]), tuple(row["spike"]), _dot_rgb(row["type"], row["gap"])))
    return dots


def probe_dots_split(model: Any, *, data_manager: Any = None):
    """Run probe2 once and return ``(contacts, clashes)`` — the full dot surface and
    the bad-overlap (clash) subset — so both overlays come from a single run."""
    contacts, clashes = [], []
    for row in run_probe_dots(model, data_manager=data_manager):
        dot = (tuple(row["loc"]), tuple(row["spike"]), _dot_rgb(row["type"], row["gap"]))
        contacts.append(dot)
        if row["type"] in _CLASH_TYPES:
            clashes.append(dot)
    return contacts, clashes


def encode_dots(dots) -> bytes:
    """Pack dots for the wire: ``[u32 n][per dot: f32 loc xyz, f32 spike xyz, u32 rgb]``."""
    parts = [struct.pack("<I", len(dots))]
    for (lx, ly, lz), (sx, sy, sz), (r, g, b) in dots:
        rgb = (int(r) << 16) | (int(g) << 8) | int(b)
        parts.append(struct.pack("<6fI", lx, ly, lz, sx, sy, sz, rgb))
    return b"".join(parts)
