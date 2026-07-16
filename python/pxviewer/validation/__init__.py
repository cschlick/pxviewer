"""MolProbity validation: a registry of validators over a cctbx model.

Each validator lives in its own submodule and defines a ``run(model)`` that
returns a :class:`ValidationResult` — a table (columns + rows) plus a set of
3-D :data:`Marker` s to draw in the viewport. Validators announce themselves
with the :func:`register` decorator, so adding a validator is just dropping a
new submodule in this package; the desktop's Validation tab is data-driven from
the registry and picks it up with no tab-code changes.

Markers reuse the probe-dot wire (``LiveSession.show_probe_dots(dots,
channel=N)``): a marker is a ``(loc, spike, rgb)`` tuple, drawn as a POINT when
``loc == spike`` and as a LINE (``loc`` -> ``spike``) otherwise. Each validator
gets its own stable channel via :func:`channel_for` so overlays toggle
independently and never collide with the probe2 contact/clash channels (0/1).

See :mod:`pxviewer.validation.ramachandran` for the reference validator; mirror
its structure when writing a new one.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple, Tuple

# A drawable marker: (loc_xyz, spike_xyz, rgb). POINT when loc == spike, else a
# LINE from loc to spike. Same shape as a probe dot, so it rides the probe-dot wire.
Marker = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]


@dataclass
class ValidationResult:
    """One validator's output: a labelled table plus 3-D markers.

    ``key``      stable id of the validator (matches its :func:`register` key).
    ``title``    human-readable name, shown as the section/group-box title.
    ``columns``  table column headers.
    ``rows``     one ``list`` of cell values per row (len == len(columns)).
    ``markers``  :data:`Marker` s to draw, typically one per flagged residue.
    ``summary``  a one-line summary string (counts / percentages).
    """

    key: str
    title: str
    columns: List[str]
    rows: List[list]
    markers: List[Marker]
    summary: str


class ValidatorSpec(NamedTuple):
    """A registered validator: its id, display title, and ``run(model)`` callable."""

    key: str
    title: str
    run: Callable[[Any], ValidationResult]


# Registry keyed by validator key; iteration is stabilised by sorting on the key
# (see :func:`validators`), so channel/ordering is independent of import order.
_REGISTRY: dict[str, ValidatorSpec] = {}

# Validator marker channels start here, clear of the probe2 contact/clash
# channels (0 and 1) that share the same probe-dot wire.
CHANNEL_BASE = 10


def register(key: str, title: str) -> Callable[[Callable[[Any], ValidationResult]], Callable[[Any], ValidationResult]]:
    """Decorator: register a ``run(model) -> ValidationResult`` under ``key``/``title``."""

    def deco(run_fn: Callable[[Any], ValidationResult]) -> Callable[[Any], ValidationResult]:
        _REGISTRY[key] = ValidatorSpec(key, title, run_fn)
        return run_fn

    return deco


def _discover() -> None:
    """Import every submodule so its :func:`register` runs. Idempotent — repeated
    imports are served from ``sys.modules``, so this is cheap to call anywhere."""
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")


def validators() -> List[ValidatorSpec]:
    """All registered validators, in a stable order (sorted by key)."""
    _discover()
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def channel_for(key: str) -> int:
    """The probe-dot channel for a validator's markers: ``CHANNEL_BASE`` + its
    index in the stable validator order, so every validator owns a distinct one."""
    order = [spec.key for spec in validators()]
    return CHANNEL_BASE + order.index(key)


def run_all(model: Any) -> List[ValidationResult]:
    """Run every registered validator on ``model``, in stable order."""
    return [spec.run(model) for spec in validators()]
