"""pxviewer: Python API for building pxviewer Mol* scenes and data."""

__version__ = "0.1.0"

# Point cctbx at the monomer library shipped by the `chem_data` package (if present)
# before any restraints are built, so minimization/validation work out of the box on a
# conda install. Cheap and side-effect-free when chem_data is absent or the variable is
# already set; the real logic (and env-var precedence) lives in geometry.monomer_library_root.
from .geometry import monomer_library_root as _monomer_library_root

_monomer_library_root()
del _monomer_library_root

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
