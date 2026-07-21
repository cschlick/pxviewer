"""cctbx-native volume I/O — the map counterpart to :mod:`pxviewer.cctbx_io`.

All map reading goes through cctbx (``DataManager`` / ``map_manager``), exactly as
model reading goes through cctbx. We never import ``mrcfile`` ourselves: cctbx
already owns that, and hands the grid over as ``map_data().as_numpy_array()``.

Two things fall out of using cctbx for I/O:

* A **map + model loaded together** arrives as a cctbx ``map_model_manager``, which
  *is* the group — we don't have to guess whether files belong together, cctbx
  tells us. :func:`split_map_model_manager` splits one into a :class:`ModelData`
  and its :class:`VolumeData` maps.
* The numpy grid is a round-trip through flex. That copy is the price of letting
  cctbx do all the I/O, so we take it lazily — metadata and re-writing the map for
  the browser need only the ``map_manager``, never the numpy array.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from .cctbx_io import ModelData

# MVS renders isosurfaces at a *relative* level in sigma units; 1.5σ is a sane
# default for both cryo-EM and crystallographic maps.
DEFAULT_ISO_SIGMA = 1.5


class VolumeData:
    """A single map: the native cctbx ``map_manager`` plus a lazy numpy view.

    The ``map_manager`` is the authority — grid metadata is read from it and the
    map is re-written from it to serve to the browser. ``array`` materialises the
    grid as numpy only on first access (and caches it), so viewing a large map
    never forces the flex→numpy copy.
    """

    def __init__(self, map_manager: Any, *, name: str = "map", map_id: str = "map_manager"):
        self.map_manager = map_manager
        self.name = name
        self.map_id = map_id
        self._array: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_map_manager(cls, map_manager: Any, *, name: str = "map", map_id: str = "map_manager") -> "VolumeData":
        return cls(map_manager, name=name, map_id=map_id)

    @classmethod
    def from_map_file(cls, path: Any, *, data_manager: Any = None) -> "VolumeData":
        """Read a map file (MRC/MAP/CCP4/…) through a cctbx ``DataManager``."""
        from .cctbx_io import data_manager as _dm

        dm = _dm(data_manager)
        dm.process_real_map_file(str(path))
        return cls.from_map_manager(dm.get_real_map(str(path)), name=Path(path).name)

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        *,
        spacing: Any = 1.0,
        origin: Tuple[int, int, int] = (0, 0, 0),
        name: str = "map",
    ) -> "VolumeData":
        """Wrap a numpy ``[x, y, z]`` grid in a cctbx ``map_manager`` (P1).

        Used to bring generated grids (e.g. demos) through cctbx like everything
        else, so nothing needs to touch ``mrcfile`` directly. ``spacing`` is the
        voxel size (a scalar, or a per-axis ``(sx, sy, sz)``); ``origin`` is the
        integer grid offset of the box (cctbx models origin in grid units).
        """
        from cctbx import crystal
        from iotbx.map_manager import map_manager
        from scitbx.array_family import flex

        arr = np.ascontiguousarray(array, dtype=np.float64)
        nx, ny, nz = arr.shape
        sx, sy, sz = (float(spacing),) * 3 if np.isscalar(spacing) else tuple(float(s) for s in spacing)
        ox, oy, oz = (int(round(o)) for o in origin)
        grid = flex.double(arr.reshape(-1))
        grid.reshape(flex.grid((ox, oy, oz), (ox + nx, oy + ny, oz + nz)))
        symmetry = crystal.symmetry(
            unit_cell=(nx * sx, ny * sy, nz * sz, 90, 90, 90),
            space_group_symbol="P1",
        )
        mm = map_manager(
            map_data=grid,
            unit_cell_grid=(nx, ny, nz),
            unit_cell_crystal_symmetry=symmetry,
            wrapping=False,
        )
        return cls.from_map_manager(mm, name=name)

    # -- grid access -----------------------------------------------------

    @property
    def array(self) -> np.ndarray:
        """The grid as a numpy array (``map_data().as_numpy_array()``), cached."""
        with self._lock:
            if self._array is None:
                self._array = self.map_manager.map_data().as_numpy_array()
            return self._array

    # -- metadata (read from the map_manager; no array copy) -------------

    @property
    def grid(self) -> Tuple[int, int, int]:
        """The grid dimensions of this map's box (``map_data().all()``)."""
        return tuple(self.map_manager.map_data().all())

    @property
    def origin(self) -> Tuple[int, int, int]:
        """The grid origin of the box (nonzero for a boxed/cut-out map)."""
        return tuple(self.map_manager.map_data().origin())

    @property
    def unit_cell(self) -> Tuple[float, ...]:
        return tuple(self.map_manager.unit_cell().parameters())

    @property
    def unit_cell_grid(self) -> Tuple[int, int, int]:
        return tuple(self.map_manager.unit_cell_grid)

    @property
    def pixel_sizes(self) -> Tuple[float, ...]:
        return tuple(self.map_manager.pixel_sizes())

    @property
    def space_group(self) -> str:
        return self.map_manager.crystal_symmetry().space_group().type().lookup_symbol()

    def stats(self) -> dict:
        """Basic grid statistics (min/max/mean/std) — forces the numpy copy."""
        a = self.array
        return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}

    def suggested_iso(self) -> float:
        """A reasonable default isosurface level, in sigma (relative)."""
        return DEFAULT_ISO_SIGMA

    # -- output ----------------------------------------------------------

    def write_map(self, path: Any, *, working_frame: bool = False) -> None:
        """Write the map out via cctbx.

        By default cctbx writes a map back in the frame it was *read* in: the CCP4
        header carries the original origin, so a reader puts the map back where it came
        from. That is what someone saving a file wants.

        ``working_frame`` writes the map where it currently *is* instead. Pairing a map
        with a model shifts both into a common frame — cctbx's convention is to work
        shifted and shift back on output — and the viewer draws the model at its shifted
        coordinates. So the copy the viewer renders has to be written in that same frame,
        or the model is drawn away from its own density.
        """
        mm = self.map_manager
        if working_frame and tuple(mm.shift_cart()) != (0, 0, 0):
            mm = mm.deep_copy()  # only when shifted: maps are large
            mm.set_original_origin_and_gridding(original_origin=(0, 0, 0))
        mm.write_map(str(path))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"VolumeData(name={self.name!r}, map_id={self.map_id!r}, grid={self.grid})"


def split_map_model_manager(mmm: Any, *, name: Optional[str] = None) -> Tuple[Optional[ModelData], List[VolumeData]]:
    """Split a cctbx ``map_model_manager`` into a model + its maps (the group).

    Returns ``(model_data_or_None, [VolumeData, ...])``. The maps keep their cctbx
    ids (``map_manager``, ``map_manager_1``/``map_manager_2`` for half-maps, …), so
    a multi-map manager yields one :class:`VolumeData` per map.
    """
    model = mmm.model()
    model_data = ModelData.from_model(model) if model is not None else None

    volumes: List[VolumeData] = []
    for map_id in mmm.map_id_list():
        mm = mmm.get_map_manager_by_id(map_id)
        if mm is None:
            continue
        label = map_id if name is None else f"{name}:{map_id}"
        volumes.append(VolumeData.from_map_manager(mm, name=label, map_id=map_id))
    return model_data, volumes


def masked_map_copy(mmm: Any, map_id: str, radius: float) -> Any:
    """A copy of ``mmm``'s ``map_id`` with density further than ``radius`` A from the
    model removed.

    A *copy*, emphatically: cctbx's ``mask_all_maps_around_atoms`` masks the manager's
    maps in place, and the map being refined against must not quietly acquire holes
    because someone tidied up the view. The mask and the copy are cctbx's own, and both
    scratch maps are removed from the manager on the way out so repeated masking does
    not pile them up.

    Needs a paired model — which is the manager's whole point, and why this takes one.
    """
    made = []
    try:
        mmm.create_mask_around_atoms(model=mmm.model(), mask_atoms_atom_radius=float(radius))
        made = list(mmm.create_masked_copies_of_maps(map_id_list=[map_id], mask_id="mask"))
        if not made:
            raise ValueError("cctbx produced no masked copy")
        return mmm.get_map_manager_by_id(made[0]).deep_copy()
    finally:
        for scratch in made + ["mask"]:
            try:
                mmm.remove_map_manager_by_id(scratch)
            except Exception:  # pragma: no cover - nothing to remove
                pass


def encode_map_box(map_manager: Any, *, level: float = 3.0, is_difference: bool = True) -> bytes:
    """Serialise a (boxed) map as a self-contained density window for the live viewer.

    A small binary header followed by the raw f32 grid, holding everything the browser
    needs to place and contour the box without any crystallography of its own: the box
    dimensions, the Cartesian position of grid point ``(0, 0, 0)``, and the three Cartesian
    voxel *step-vectors* (so a non-orthogonal cell simply gives non-axis-aligned steps).
    This is the density counterpart of the raw-f32 coordinate frame — no parse, just an
    affine and a float buffer the frontend drops onto a Mol* volume.

    ``level`` is a contour in sigma of the (sigma-scaled) map; a difference map is drawn at
    ``+level`` (green) and ``-level`` (red). Grid values are C-order: index ``(i, j, k)`` is
    at ``data[(i*ny + j)*nz + k]`` and sits at Cartesian ``origin + i*step0 + j*step1 + k*step2``.

    Layout (little-endian; the sender prepends the u32 message tag):
        u32 flags (bit 0 = is_difference); f32 level;
        i32 nx, ny, nz; f32 origin[3]; f32 step0[3], step1[3], step2[3]; f32 data[nx*ny*nz]
    """
    import struct

    md = map_manager.map_data()
    nx, ny, nz = md.all()
    data = np.ascontiguousarray(md.as_numpy_array(), dtype="<f4")
    ortho = np.array(
        map_manager.crystal_symmetry().unit_cell().orthogonalization_matrix(), dtype="float64"
    ).reshape(3, 3)
    # The map_manager's own cell spans exactly this grid (for a boxed map, the box's cell over
    # the box's dims), so the Cartesian step along grid axis j is column j of the frac->cart
    # matrix divided by that axis's grid count. (unit_cell_grid still refers to the full cell,
    # so it must NOT be used here.)
    steps = ortho @ np.diag([1.0 / nx, 1.0 / ny, 1.0 / nz])
    shift = map_manager.shift_cart()  # translation that moved the box to a zero origin
    origin = (-shift[0], -shift[1], -shift[2])  # so grid (0,0,0) sits back at its Cartesian place

    header = struct.pack("<If", int(bool(is_difference)), float(level))
    header += struct.pack("<iii", int(nx), int(ny), int(nz))
    header += struct.pack("<fff", *origin)
    for j in range(3):
        header += struct.pack("<fff", float(steps[0, j]), float(steps[1, j]), float(steps[2, j]))
    return header + data.tobytes()


def map_model_manager_from_files(
    model_file: Optional[Any] = None,
    map_files: Any = (),
    *,
    ignore_symmetry_conflicts: bool = True,
    data_manager: Any = None,
) -> Any:
    """Build a cctbx ``map_model_manager`` from files — cctbx decides the grouping.

    Note that ``get_map_model_manager`` *consumes* its inputs: it removes the model and
    maps from the DataManager on the way out, since building the manager shifts them.
    The returned manager is therefore the only record that these files belong together.
    """
    from .cctbx_io import data_manager as _dm

    dm = _dm(data_manager)
    maps = [str(p) for p in (map_files or ())]
    return dm.get_map_model_manager(
        model_file=str(model_file) if model_file is not None else None,
        map_files=maps,
        ignore_symmetry_conflicts=ignore_symmetry_conflicts,
    )
