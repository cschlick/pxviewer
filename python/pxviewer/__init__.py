"""pxviewer: Python API for building pxviewer Mol* scenes and data."""

import os as _os
import sys as _sys
from pathlib import Path as _Path

# Conda puts dependency DLLs (Qt6, OpenSSL, zmq…) in <env>/Library/bin, which
# activation normally adds to PATH. The bundled Windows app launches
# pythonw.exe with no activation — register the dir here, before any pyd that
# links those DLLs (ssl via websockets, PySide6) gets imported.
if _sys.platform == "win32":
    _dll_dir = _Path(_sys.prefix) / "Library" / "bin"
    if _dll_dir.is_dir():
        _os.add_dll_directory(str(_dll_dir))
del _os, _sys, _Path

__version__ = "0.1.0"

# Make sure cctbx can find the monomer library shipped by the `chem_data` package before
# any restraints are built, so minimization/validation work out of the box on a conda
# install. Usually this sets nothing at all -- cctbx cascades through both chem_data
# directories by itself, and an MMTBX_CCP4_MONOMER_LIB redirect would narrow it to one.
# It clears a stale redirect and, on a layout cctbx cannot see, sets one as a fallback.
# The reasoning lives in geometry.configure_monomer_library.
from .geometry import configure_monomer_library as _configure_monomer_library

_configure_monomer_library()
del _configure_monomer_library

from .api import (
    AtomArrays,
    Volume,
    VolumeStyle,
    create_example_view,
    create_view,
    create_volume_view,
    create_volume_view_from_data,
    encode_bcif_arrays,
    read_volume,
    set_volume_color,
    set_volume_opacity,
    set_volume_style,
    write_volume,
)
from .live import ATOM_IDENTITY_CONTRACT, ComponentExpression, LiveSession, Primitive, Selection

__all__ = [
    "AtomArrays",
    "Volume",
    "VolumeStyle",
    "create_example_view",
    "create_view",
    "create_volume_view",
    "create_volume_view_from_data",
    "set_volume_color",
    "set_volume_opacity",
    "set_volume_style",
    "encode_bcif_arrays",
    "read_volume",
    "write_volume",
    "LiveSession",
    "Selection",
    "Primitive",
    "ComponentExpression",
    "ATOM_IDENTITY_CONTRACT",
]
