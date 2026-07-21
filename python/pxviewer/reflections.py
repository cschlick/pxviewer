"""Read X-ray reflections with cctbx.

Reflections are the one thing here that cannot be drawn. What a viewer shows is the
density they imply, which is an FFT away — and, for a file carrying only amplitudes, a
model away too, since the phases have to come from somewhere. So this module reads a
reflection file and says what is in it; turning that into a map is a separate step with
its own requirements.

Which of the two kinds of file it is, is cctbx's call, not ours. ``map_coefficients`` is
a child datatype of ``miller_array`` in the DataManager, so cctbx already separates a
refinement file (2FOFCWT/PH2FOFCWT — coefficients, ready to transform) from a data file
(F/SIGF — amplitudes, needing a model for phases). We ask it rather than reading column
names ourselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

__all__ = [
    "DEFAULT_RESOLUTION_FACTOR",
    "MAP_STYLE",
    "DIFFERENCE_MAP_TYPES",
    "PHASED_MAP_TYPES",
    "phased_maps",
    "LiveDifferenceMap",
    "ReflectionData",
    "is_difference_map",
    "map_from_coefficients",
    "root_label",
]

#: Grid spacing for transforming coefficients, as a fraction of the resolution. cctbx's
#: own default, and the same spacing as Coot's default sampling rate of 1.5 (clipper
#: states it as a rate r, which is this at 1/(2r)) — so the map is gridded the way a
#: crystallographer expects. 0.25 is smoother at ~2.2x the voxels, which we pay for
#: twice: writing the file and isosurfacing it on the GPU.
DEFAULT_RESOLUTION_FACTOR = 1.0 / 3.0

#: How each kind of map is shown, following crystallographic convention (and Coot):
#: ``(colour, level, negative colour)``. A 2Fo-Fc map is blue at 1.5 sigma and has no
#: negative side worth drawing. A difference map is read at both signs at once — green
#: where the density wants more than the model has, red where it wants less — so it is
#: drawn twice, at +level and -level, and is only half a map without both.
MAP_STYLE = {
    False: ("dodgerblue", 1.5, None),  # a regular map
    True: ("green", 3.0, "red"),       # a difference map
}

# Map-coefficient label conventions. cctbx knows an array *is* coefficients but not what
# they mean — no format records "this is a difference map" — so, like every other viewer,
# we keep the table. phenix writes 2FOFCWT/FOFCWT, refmac FWT/DELFWT.
#
# Matched on the whole root, never as a substring: "2FOFCWT" contains "FOFCWT", so a
# substring test calls the 2Fo-Fc map a difference map. test_2fofc_is_not_a_difference_map.
_DIFFERENCE_ROOTS = frozenset({
    "FOFCWT", "FOFCWT_NO_FILL",  # phenix
    "DELFWT",                     # refmac
})


def root_label(label: str) -> str:
    """The amplitude column's root, e.g. '2FOFCWT,PHI2FOFCWT' -> '2FOFCWT'.

    What a crystallographer calls the map, and short enough for the object list.
    """
    return label.split(",")[0].strip()


def _root_label(label: str) -> str:
    return root_label(label).upper()


def is_difference_map(label: str) -> bool:
    """Whether a map-coefficient label names a difference map, by convention.

    A guess, and unavoidably so — the file does not say. It decides how the map is
    contoured and coloured, not what it contains, so being wrong is cosmetic.
    """
    return _root_label(label) in _DIFFERENCE_ROOTS


def map_from_coefficients(
    coefficients: Any, *, resolution_factor: float = DEFAULT_RESOLUTION_FACTOR
) -> Any:
    """Transform map coefficients into a cctbx ``map_manager``.

    Sigma-scaled, so a contour level means the same thing here as it does everywhere
    else in crystallography: "1.5 sigma" is 1.5 standard deviations of this map.
    """
    from iotbx.map_manager import map_manager

    fft = coefficients.fft_map(resolution_factor=resolution_factor)
    fft.apply_sigma_scaling()
    return map_manager(
        map_data=fft.real_map_unpadded(),
        unit_cell_grid=fft.n_real(),
        unit_cell_crystal_symmetry=coefficients.crystal_symmetry(),
        wrapping=True,
    )


#: The maps to compute from amplitudes and a model: the density, and what the model does
#: not account for. Both, because a difference map is how you find what the density says
#: and the model does not — reading one without the other is half the job.
PHASED_MAP_TYPES = ("2mFo-DFc", "mFo-DFc")

#: Which of those is a difference map. We name these, rather than reading them off a
#: file, so this is knowledge and not a guess like _DIFFERENCE_ROOTS. Spelled as a set
#: rather than a prefix test for the same reason that table matches whole roots:
#: "2mFo-DFc" ends with "mFo-DFc".
DIFFERENCE_MAP_TYPES = frozenset({"mFo-DFc"})


def phased_maps(
    model: Any,
    reflection_file: Any,
    *,
    map_types: Sequence[str] = PHASED_MAP_TYPES,
    resolution_factor: float = DEFAULT_RESOLUTION_FACTOR,
    scattering_table: str = "n_gaussian",
    data_manager: Any = None,
) -> dict:
    """Compute density from amplitudes and a model, which is where the phases come from.

    Returns ``{"maps": {map_type: map_manager}, "r_work": float, "r_free": float}``.

    ``model`` is a live ``mmtbx.model.manager``, not a filename: the model the viewer
    holds often exists nowhere on disk — reduce2 built it, or Minimize moved it — and
    recomputing density after the model moves is the point of keeping the reflections
    around at all. The DataManager takes it directly (``add_model``), and ``get_fmodel``
    then gathers it with the diffraction data.

    ``update_all_scales`` is not optional. ``get_fmodel`` returns an unscaled fmodel —
    no bulk solvent, no overall scaling — and 2mFo-DFc computed from that is wrong for
    real data, in a way that looks plausible rather than broken.
    """
    from .cctbx_io import data_manager as _dm

    dm = _dm(data_manager)
    dm.add_model("model", model)
    dm.process_miller_array_file(str(reflection_file))

    fmodel = dm.get_fmodel(scattering_table=scattering_table)
    fmodel.update_all_scales()
    density = fmodel.electron_density_map()
    maps = {
        map_type: map_from_coefficients(
            density.map_coefficients(map_type=map_type), resolution_factor=resolution_factor)
        for map_type in map_types
    }
    return {"maps": maps, "r_work": fmodel.r_work(), "r_free": fmodel.r_free()}


class LiveDifferenceMap:
    """Recompute an mFo-DFc difference map fast, as a model moves — the *warm* path.

    A full :func:`phased_maps` re-derives bulk solvent and rescales every call (~0.1-1 s),
    far too slow to follow a drag. This builds the scaled ``fmodel`` **once** and then, per
    update, recomputes *only* f_calc from the moved atoms and re-FFTs the difference
    coefficients — 5-40x faster: interactive (tens of Hz) for a small protein, a few Hz for
    a few-thousand-atom model. See ``scripts/bench_live_maps.py`` for numbers.

    Two deliberate crystallographic choices, both about honesty rather than speed:

    * It recomputes the **difference** map (mFo-DFc), not 2mFo-DFc. The difference map shows
      where the model disagrees with the data, so watching it flatten as you fit is real
      feedback. Recomputing 2mFo-DFc live would instead just echo the moving model back —
      model bias, the one thing a crystallographer is trained to distrust.
    * Scales and the bulk-solvent mask are **frozen** at construction (``update_f_calc``
      only, never ``update_all_scales`` per frame): the map answers to the model you are
      moving, not to a re-fit of the experiment to that model, and :attr:`r_free` stays a
      fixed reference that dragging cannot flatter. The frozen mask does go stale under
      large rearrangements — call :meth:`rescale` (the expensive step) to refresh it.
    """

    def __init__(
        self,
        model: Any,
        reflection_file: Any,
        *,
        resolution_factor: float = DEFAULT_RESOLUTION_FACTOR,
        scattering_table: str = "n_gaussian",
        data_manager: Any = None,
    ) -> None:
        from .cctbx_io import data_manager as _dm

        dm = _dm(data_manager)
        dm.add_model("model", model)
        dm.process_miller_array_file(str(reflection_file))
        self._fmodel = dm.get_fmodel(scattering_table=scattering_table)
        self._fmodel.update_all_scales()  # once: bulk solvent + overall scaling (the slow part)
        self._resolution_factor = resolution_factor

    def recompute(
        self,
        *,
        model: Any = None,
        sites_cart: Any = None,
        xray_structure: Any = None,
        map_type: str = "mFo-DFc",
    ) -> Any:
        """A fresh difference ``map_manager`` for the moved atoms.

        Give the new conformation as a cctbx ``model``, an ``xray_structure``, or a
        ``sites_cart`` array (flex ``vec3_double`` or an ``(N, 3)`` / flat numpy array).
        Only f_calc is recomputed; the frozen scales and mask are reused.
        """
        xrs = xray_structure
        if xrs is None and model is not None:
            xrs = model.get_xray_structure()
        if xrs is None:
            xrs = self._fmodel.xray_structure.deep_copy_scatterers()
            if sites_cart is not None:
                xrs.set_sites_cart(_as_vec3(sites_cart))
        self._fmodel.update_xray_structure(xray_structure=xrs, update_f_calc=True)
        coefficients = self._fmodel.electron_density_map().map_coefficients(map_type=map_type)
        return map_from_coefficients(coefficients, resolution_factor=self._resolution_factor)

    def recompute_local(
        self,
        center: Any,
        *,
        radius: float = 6.0,
        model: Any = None,
        sites_cart: Any = None,
        xray_structure: Any = None,
        map_type: str = "mFo-DFc",
    ) -> Any:
        """A small difference-map box (~``2*radius`` A on a side) centred on ``center``
        (Cartesian ``(x, y, z)`` in A) — the map to stream while tugging *there*.

        The full map is recomputed and only then cropped: the FFT is over all of reciprocal
        space, so a local window does not cut the compute (see ``scripts/bench_live_maps.py``)
        — what it cuts is *delivery*. A 5 A window is a ~20-grid-point box of tens of KB
        instead of the whole-cell megabytes, cheap to ship and redraw every frame. Reach for
        a full :meth:`recompute` behind a deliberate "recompute" action for the whole model.

        Returns a boxed ``map_manager`` whose origin records where the window sits in the cell.
        """
        import math

        from iotbx.map_model_manager import map_model_manager

        full = self.recompute(
            model=model, sites_cart=sites_cart, xray_structure=xray_structure, map_type=map_type)
        unit_cell = full.crystal_symmetry().unit_cell()
        n_grid = full.unit_cell_grid
        edges = unit_cell.parameters()[:3]
        frac = unit_cell.fractionalize(tuple(float(c) for c in center))
        lower, upper = [], []
        for axis in range(3):
            half = radius / (edges[axis] / n_grid[axis])   # window half-width in grid points
            middle = frac[axis] * n_grid[axis]
            lower.append(int(math.floor(middle - half)))
            upper.append(int(math.ceil(middle + half)))
        mmm = map_model_manager(map_manager=full)  # `full` is discarded, so no copy needed
        mmm.box_all_maps_with_bounds_and_shift_origin(lower_bounds=lower, upper_bounds=upper)
        return mmm.map_manager()

    def rescale(self) -> None:
        """Re-derive bulk solvent and overall scales — the expensive step frozen per frame.
        Call after a large rearrangement so the stale mask stops distorting the map."""
        self._fmodel.update_all_scales()

    @property
    def r_work(self) -> float:
        return self._fmodel.r_work()

    @property
    def r_free(self) -> float:
        """R-free against the frozen scales — a fixed reference, not re-fit to the model."""
        return self._fmodel.r_free()


def _as_vec3(sites: Any) -> Any:
    """Coerce ``(N,3)`` / flat numpy coordinates (or a flex array) to flex ``vec3_double``."""
    from scitbx.array_family import flex

    if isinstance(sites, flex.vec3_double):
        return sites
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(sites, dtype="float64")).reshape(-1, 3)
    return flex.vec3_double(arr)


class ReflectionData:
    """A reflection file, as cctbx read it: the arrays plus what to say about them."""

    def __init__(
        self,
        arrays: Sequence[Any],
        labels: Sequence[str],
        map_coefficient_labels: Sequence[str] = (),
        *,
        name: str = "reflections",
        path: Optional[Any] = None,
    ):
        self.arrays: List[Any] = list(arrays)
        self.labels: List[str] = list(labels)
        # Empty unless the file already carries map coefficients (see the module note).
        self.map_coefficient_labels: List[str] = list(map_coefficient_labels)
        self.name = name
        self.path = str(path) if path is not None else None

    @classmethod
    def from_file(cls, path: Any, *, data_manager: Any = None) -> "ReflectionData":
        """Read a reflection file (MTZ, or mmCIF structure factors) through cctbx."""
        from .cctbx_io import data_manager as _dm

        dm = _dm(data_manager)
        name = str(path)
        dm.process_miller_array_file(name)
        coefficients = dm.get_map_coefficients_labels(name) if dm.has_map_coefficients() else []
        return cls(
            dm.get_miller_arrays(filename=name),
            dm.get_miller_array_labels(name),
            coefficients,
            name=Path(name).name,
            path=name,
        )

    # -- what the file is ------------------------------------------------

    @property
    def has_map_coefficients(self) -> bool:
        """True when the file carries map coefficients, so density needs no model.

        The fork the whole feature turns on: with coefficients a map is one transform
        away; without them the phases have to be computed against a model.
        """
        return bool(self.map_coefficient_labels)

    def map_coefficient_arrays(self) -> List[Any]:
        """The arrays that are map coefficients (empty for a data file)."""
        wanted = set(self.map_coefficient_labels)
        return [a for a in self.arrays if _label_of(a) in wanted]

    # -- metadata (read from the arrays; cctbx is the authority) ---------

    @property
    def crystal_symmetry(self) -> Any:
        return self.arrays[0].crystal_symmetry() if self.arrays else None

    @property
    def resolution_range(self) -> Optional[tuple]:
        """``(d_max, d_min)`` in Angstrom across every array, or None if empty."""
        ranges = [a.d_max_min() for a in self.arrays if a.size()]
        if not ranges:
            return None
        return max(r[0] for r in ranges), min(r[1] for r in ranges)

    @property
    def n_reflections(self) -> int:
        return max((a.size() for a in self.arrays), default=0)

    def summary(self) -> str:
        """One line for the status bar / object list."""
        parts = [f"{len(self.labels)} array(s)", f"{self.n_reflections} reflections"]
        span = self.resolution_range
        if span is not None:
            parts.append(f"{span[0]:.1f}-{span[1]:.2f} A")
        parts.append("map coefficients" if self.has_map_coefficients else "amplitudes")
        return ", ".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ReflectionData(name={self.name!r}, labels={self.labels!r})"


def _label_of(array: Any) -> str:
    """The label cctbx knows an array by, matching DataManager's label strings."""
    info = array.info()
    return info.label_string() if info is not None else ""
