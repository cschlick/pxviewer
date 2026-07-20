"""Classify user files and stage volumes for the browser viewer.

Atomic models are read by cctbx and streamed through a live session — never
parsed in the browser (see :mod:`pxviewer.cctbx_io`). So this module only builds a
browser scene for *volumes*, which still load as MVSJ + MRC. It also holds the
file-kind detection used to route a dropped file, and the bundled sample.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .volume import Volume, create_volume_view

__all__ = [
    "MODEL_FORMATS",
    "VOLUME_FORMATS",
    "FILE_DIALOG_FILTER",
    "REFLECTION_FORMATS",
    "SAMPLE_STRUCTURE",
    "file_kind",
    "sample_structure_path",
    "create_volume_file_view",
]

# Ubiquitin (PDB 1UBQ) — a small, iconic single-chain protein shipped inside the
# package (pxviewer/data) so the desktop app always has something real to open,
# and the map+model demo can compute a density from it.
SAMPLE_STRUCTURE = ("1ubq.pdb", "Ubiquitin (1UBQ)")

# Every bundled sample. 1TEC is deliberately imperfect: it trips every MolProbity
# validator — Ramachandran, rotamer, Cbeta and CaBLAM outliers plus a cis-proline
# *and* a cis-nonproline — so the Validation tab has a structure that shows all of
# its markup. (1UBQ is near-pristine and shows almost none.)
SAMPLES = [
    SAMPLE_STRUCTURE,
    ("1tec.pdb", "Thermitase-eglin (1TEC) — validation demo"),
    ("zn_site.pdb", "Metal site — Zn coordination (restraint edits)"),
]


def sample_structure_path(filename: str | None = None) -> Path | None:
    """Path to a bundled sample model (package data), or None if it isn't there.
    Defaults to :data:`SAMPLE_STRUCTURE`; pass a filename from :data:`SAMPLES`."""
    path = Path(__file__).resolve().parent / "data" / (filename or SAMPLE_STRUCTURE[0])
    return path if path.is_file() else None


# Model formats cctbx's DataManager reads (streamed live, not browser-parsed).
MODEL_FORMATS = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "mmcif",
    ".mmcif": "mmcif",
}

# Volume formats, still staged as MVSJ + MRC for the browser.
VOLUME_FORMATS = {
    ".mrc": "map",
    ".map": "map",
    ".ccp4": "map",
}

# Reflection formats. Nothing here is drawable: these carry the density a viewer shows,
# but only as coefficients (see pxviewer.reflections). mmCIF structure factors are not
# listed because .cif already means "model" above and the suffix cannot tell them apart —
# cctbx has to read the file to know, which a suffix table cannot do.
REFLECTION_FORMATS = {
    ".mtz": "mtz",
}


def _filter(label: str, suffixes) -> str:
    patterns = " ".join(f"*{s}" for s in sorted(suffixes))
    return f"{label} ({patterns})"


FILE_DIALOG_FILTER = ";;".join(
    [
        _filter("All supported",
                list(MODEL_FORMATS) + list(VOLUME_FORMATS) + list(REFLECTION_FORMATS)),
        _filter("Models", MODEL_FORMATS),
        _filter("Volumes", VOLUME_FORMATS),
        _filter("Reflections", REFLECTION_FORMATS),
        "All files (*)",
    ]
)


def file_kind(path: str | Path) -> str:
    """Classify a path as ``"model"``, ``"volume"`` or ``"reflections"`` by its suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in MODEL_FORMATS:
        return "model"
    if suffix in VOLUME_FORMATS:
        return "volume"
    if suffix in REFLECTION_FORMATS:
        return "reflections"
    known = ", ".join(sorted(set(MODEL_FORMATS) | set(VOLUME_FORMATS) | set(REFLECTION_FORMATS)))
    raise ValueError(f"unsupported file type '{suffix or path}'. Supported: {known}")


def create_volume_file_view(path: str | Path, *, out_dir: str | Path) -> Path:
    """Copy a volume file into ``out_dir`` and write an MVSJ scene that loads it.

    The scene refers to the copy by bare filename, so both are served from the
    same directory. Returns the MVSJ path. Models are not handled here — they load
    through cctbx into a live session — so a non-volume path is rejected.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"no such file: {src}")
    if file_kind(src) != "volume":
        raise ValueError(f"{src.name} is not a volume; atomic models are loaded via cctbx")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    copy = out / src.name
    shutil.copyfile(src, copy)

    mvsj = create_volume_view(
        volumes=[Volume(url=copy.name, ref="volume-0", isosurface_kind="relative", isosurface_value=2.0)],
        title=src.name,
    )
    mvsj_path = out / "scene.mvsj"
    mvsj_path.write_text(mvsj)
    return mvsj_path
