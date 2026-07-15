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
    "SAMPLE_STRUCTURE",
    "file_kind",
    "sample_structure_path",
    "create_volume_file_view",
]

# Hen egg-white lysozyme (PDB 1AKI) — a small, familiar structure kept in the test
# data so the desktop app has something real to open without a file dialog.
SAMPLE_STRUCTURE = ("1aki.pdb", "Lysozyme (1AKI)")


def sample_structure_path() -> Path | None:
    """Path to the bundled lysozyme file, or None when it isn't on disk.

    It lives under ``tests/data``, which a wheel doesn't ship, so callers must
    cope with it being absent rather than assume a checkout.
    """
    path = Path(__file__).resolve().parents[1] / "tests" / "data" / SAMPLE_STRUCTURE[0]
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


def _filter(label: str, suffixes) -> str:
    patterns = " ".join(f"*{s}" for s in sorted(suffixes))
    return f"{label} ({patterns})"


FILE_DIALOG_FILTER = ";;".join(
    [
        _filter("Models and volumes", list(MODEL_FORMATS) + list(VOLUME_FORMATS)),
        _filter("Models", MODEL_FORMATS),
        _filter("Volumes", VOLUME_FORMATS),
        "All files (*)",
    ]
)


def file_kind(path: str | Path) -> str:
    """Classify a path as ``"model"`` or ``"volume"`` by its suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in MODEL_FORMATS:
        return "model"
    if suffix in VOLUME_FORMATS:
        return "volume"
    known = ", ".join(sorted(set(MODEL_FORMATS) | set(VOLUME_FORMATS)))
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
