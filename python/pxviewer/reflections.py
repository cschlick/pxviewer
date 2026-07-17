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

__all__ = ["ReflectionData"]


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
