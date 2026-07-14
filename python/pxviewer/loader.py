"""Load a structure or volume file from the user's filesystem into a scene.

The desktop app can only show files the frontend can fetch over HTTP, so opening
a local file means: copy it into the served directory, write an MVSJ scene beside
it that points at the copy, and hand the scene's URL to the viewer.

This module holds the file-kind detection and scene building so both are usable
(and testable) without Qt.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

from .api import create_view
from .volume import Volume, create_volume_view

__all__ = [
    "STRUCTURE_FORMATS",
    "VOLUME_FORMATS",
    "FILE_DIALOG_FILTER",
    "file_kind",
    "structure_format",
    "create_file_view",
]

# Suffix -> the format name MolViewSpec's `parse` expects.
STRUCTURE_FORMATS = {
    ".pdb": "pdb",
    ".ent": "pdb",
    ".cif": "mmcif",
    ".mmcif": "mmcif",
    ".bcif": "bcif",
}

# Suffix -> the format name MolViewSpec's volume node expects.
VOLUME_FORMATS = {
    ".mrc": "map",
    ".map": "map",
    ".ccp4": "map",
}

# The default representation for an opened structure: cartoon for anything Mol*
# classifies as polymer, ball-and-stick for the rest. A small molecule has no
# polymer chain, so it shows up entirely through the ligand component.
DEFAULT_COMPONENTS: List[dict] = [
    {"selector": "polymer", "representation": "cartoon", "color": "#4577b2"},
    {"selector": "ligand", "representation": "ball_and_stick", "color": "#cc3399"},
]


def _filter(label: str, suffixes) -> str:
    patterns = " ".join(f"*{s}" for s in sorted(suffixes))
    return f"{label} ({patterns})"


FILE_DIALOG_FILTER = ";;".join(
    [
        _filter("Structures and volumes", list(STRUCTURE_FORMATS) + list(VOLUME_FORMATS)),
        _filter("Structures", STRUCTURE_FORMATS),
        _filter("Volumes", VOLUME_FORMATS),
        "All files (*)",
    ]
)


def file_kind(path: str | Path) -> str:
    """Classify a path as ``"structure"`` or ``"volume"`` by its suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in STRUCTURE_FORMATS:
        return "structure"
    if suffix in VOLUME_FORMATS:
        return "volume"
    known = ", ".join(sorted(set(STRUCTURE_FORMATS) | set(VOLUME_FORMATS)))
    raise ValueError(f"unsupported file type '{suffix or path}'. Supported: {known}")


def structure_format(path: str | Path) -> str:
    """The MolViewSpec parse format for a structure path."""
    suffix = Path(path).suffix.lower()
    try:
        return STRUCTURE_FORMATS[suffix]
    except KeyError:
        raise ValueError(f"'{suffix}' is not a structure format") from None


def create_file_view(path: str | Path, *, out_dir: str | Path) -> Tuple[Path, str]:
    """Copy a user file into ``out_dir`` and write an MVSJ scene that loads it.

    The scene refers to the copy by bare filename, so both files must be served
    from the same directory. Returns ``(mvsj_path, kind)``.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"no such file: {src}")
    kind = file_kind(src)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    copy = out / src.name
    shutil.copyfile(src, copy)

    if kind == "volume":
        mvsj = create_volume_view(
            volumes=[Volume(url=copy.name, ref="volume-0", isosurface_kind="relative", isosurface_value=2.0)],
            title=src.name,
        )
    else:
        mvsj = create_view(
            copy.name,
            format=structure_format(src),
            components=DEFAULT_COMPONENTS,
            title=src.name,
        )

    mvsj_path = out / "scene.mvsj"
    mvsj_path.write_text(mvsj)
    return mvsj_path, kind
