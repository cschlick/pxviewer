"""Imported bounded-concern fields — the Hotspots generator's display product.

This is deliberately a separate module from :mod:`pxviewer.hotspots`, and the separation is
the point. Two incompatible value systems exist:

* :mod:`pxviewer.hotspots` computes an **unbounded surprisal severity** on a fixed ``[0, 4]``
  scale where ``1.0`` is the community outlier cut. pxviewer computes it itself.
* This module imports **bounded concern** in ``[0, 1]`` written by the sibling Hotspots
  generator, where ``0`` is reassuring and ``1`` is saturated concern.

They are not convertible. A Ramachandran score of 3.84% is severity 0.43 and concern 0.000 —
the first says "less likely than the outlier cut by some decades", the second says "inside the
2% favored boundary, nothing to see". Presenting one where the other is expected is what makes
a table disagree with the map beside it, so nothing here converts, rescales, or falls back to
severity: a caller either has an import (and uses this module) or a computed score (and uses
:mod:`pxviewer.hotspots`).

The authoritative display contract travels *in the manifest*, under ``primary_display``, and
:func:`display_anchors` reads it rather than assuming it. The viewer's job is to honour it:
absolute concern drives both hue and opacity on one fixed ``[0, 1]`` domain, with no
per-field percentile, min/max, sigma, or viewport-relative normalization — so the same concern
value looks the same in every structure and every metric. Percentile maps are an optional
relative-contrast product; they are imported when present but never decide what is visible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: Concern is bounded, and displayed on exactly this domain everywhere.
DOMAIN = (0.0, 1.0)

#: The fallback display contract, used only when a manifest does not carry ``primary_display``
#: (a hand-opened CCP4 pair, or an older generator). Matches the generator's documented
#: anchors: transparent at 0, yellow at the half-concern mark, orange, red at saturation.
DEFAULT_ANCHORS: Dict[str, float] = {"yellow": 0.5, "orange": 0.75, "red": 1.0}

#: Hex colours for the warm anchors. Shared with the isosurface path so a contour drawn at
#: 0.75 is the same orange the direct volume shows at 0.75.
ANCHOR_COLORS: Dict[str, str] = {"yellow": "#FFD400", "orange": "#F46D43", "red": "#B2182B"}

#: Voxels at or below this are treated as "no observation here" — the generator's own
#: eligibility gate for percentile ranking, reused so a residue with no events is absent from
#: the table rather than present with a 0.00.
EPSILON = 1e-6


@dataclass
class ConcernField:
    """One metric's bounded concern map, and its optional percentile companion."""

    name: str
    concern: Any                      # VolumeData
    values: np.ndarray                # the concern grid, validated to [0, 1]
    percentile: Optional[Any] = None  # VolumeData, when the generator wrote one
    percentile_values: Optional[np.ndarray] = None
    color_scaling: dict = _field(default_factory=dict)


@dataclass
class ConcernImport:
    """Every field read from one manifest (or one hand-opened map), plus its contract."""

    fields: Dict[str, ConcernField]
    anchors: Dict[str, float]
    source: Path
    #: ``molprobity`` disclosure from the manifest: which metrics the generator could not run.
    omitted_metrics: List[str] = _field(default_factory=list)
    omission_reason: str = ""

    @property
    def primary(self) -> str:
        """The field to show first: the combined map when there is one."""
        return "combined" if "combined" in self.fields else next(iter(self.fields))


def display_anchors(payload: dict) -> Dict[str, float]:
    """Read the display contract out of a manifest's ``primary_display``.

    The generator states where yellow, orange and red fall; the viewer follows rather than
    hardcoding, so changing the contract upstream does not silently leave the viewer painting
    the old scale. Anything missing falls back to :data:`DEFAULT_ANCHORS`.
    """
    anchors = dict(DEFAULT_ANCHORS)
    declared = (payload.get("primary_display") or {}).get("color_anchors")
    if isinstance(declared, dict):
        by_color = {}
        for position, name in declared.items():
            try:
                by_color[str(name).strip().lower()] = float(position)
            except (TypeError, ValueError):
                continue
        for name in anchors:
            if name in by_color:
                anchors[name] = by_color[name]
    # A non-monotonic contract would produce an incoherent ramp; keep the declared values only
    # while they still ascend.
    if not (0.0 <= anchors["yellow"] < anchors["orange"] < anchors["red"] <= 1.0):
        return dict(DEFAULT_ANCHORS)
    return anchors


def concern_color(value: float, anchors: Optional[Dict[str, float]] = None) -> str:
    """The colour of a constant-concern isosurface, interpolated between the anchors.

    Below the yellow anchor there is no warm colour to give — the contract makes that range
    transparent — so it clamps to yellow rather than inventing a cool end.
    """
    anchors = anchors or DEFAULT_ANCHORS
    stops = [(anchors["yellow"], ANCHOR_COLORS["yellow"]),
             (anchors["orange"], ANCHOR_COLORS["orange"]),
             (anchors["red"], ANCHOR_COLORS["red"])]
    value = float(value)
    if value <= stops[0][0]:
        return stops[0][1]
    for (lo, low), (hi, high) in zip(stops, stops[1:]):
        if value <= hi:
            fraction = (value - lo) / (hi - lo) if hi > lo else 0.0
            channels = []
            for offset in (1, 3, 5):
                a, b = int(low[offset:offset + 2], 16), int(high[offset:offset + 2], 16)
                channels.append(round(a + (b - a) * fraction))
            return "#" + "".join(f"{channel:02X}" for channel in channels)
    return stops[-1][1]


# -- reading ------------------------------------------------------------------


def _resolve(value: Any, manifest: Path) -> Path:
    """Find a map the manifest points at, whether it recorded an absolute or a relative path.

    Manifests are routinely moved next to the maps they describe, so a recorded absolute path
    that no longer exists is not an error while the named file sits beside the manifest.
    """
    candidate = Path(str(value))
    for choice in (candidate, manifest.parent / candidate, manifest.parent / candidate.name):
        if choice.exists():
            return choice
    raise FileNotFoundError(f"map listed in the manifest was not found: {value}")


def _pairs_from_manifest(path: Path) -> Tuple[Dict[str, Tuple[Path, Optional[Path], dict]], dict]:
    payload = json.loads(path.read_text())
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("hotspot manifest has no outputs mapping")

    scaling = payload.get("color_scaling") or {}
    pairs: Dict[str, Tuple[Path, Optional[Path], dict]] = {}

    def percentile_of(output: dict) -> Optional[Path]:
        # Optional by contract: percentile is relative-contrast analysis, not required to draw.
        listed = output.get("color_percentile")
        if not listed:
            return None
        try:
            return _resolve(listed, path)
        except FileNotFoundError:
            return None

    # A concern-map manifest nests one entry per metric; the local-CC manifest is a single
    # scientific field and puts its maps directly in outputs.
    if outputs.get("concern"):
        pairs["local_cc"] = (_resolve(outputs["concern"], path), percentile_of(outputs),
                             scaling if isinstance(scaling, dict) else {})
    else:
        for metric, output in outputs.items():
            if not isinstance(output, dict) or not output.get("concern"):
                continue
            pairs[str(metric)] = (_resolve(output["concern"], path), percentile_of(output),
                                  scaling.get(metric, {}) if isinstance(scaling, dict) else {})
    if not pairs:
        raise ValueError("manifest lists no concern maps")
    # Combined is the "where to look" field and belongs first; components keep manifest order.
    ordered = {}
    for name in sorted(pairs, key=lambda n: (n != "combined", list(pairs).index(n))):
        ordered[name] = pairs[name]
    return ordered, payload


#: Map extensions, longest first so ``.map.gz`` is recognised before ``.gz``. Splitting on
#: every dot instead (``Path.suffixes``) would mangle a name that merely contains one.
_MAP_SUFFIXES = (".map.gz", ".mrc.gz", ".ccp4.gz", ".ccp4", ".map", ".mrc")


def _split_map_name(path: Path) -> Tuple[str, str]:
    """``foo_hotspot.map.gz`` -> ``('foo_hotspot', '.map.gz')``."""
    lowered = path.name.lower()
    for suffix in _MAP_SUFFIXES:
        if lowered.endswith(suffix):
            return path.name[: -len(suffix)], path.name[-len(suffix):]
    return path.stem, path.suffix


def _pair_from_map(path: Path) -> Dict[str, Tuple[Path, Optional[Path], dict]]:
    """A hand-opened CCP4: discover the ``_percentile`` sibling if it happens to be there."""
    stem, suffix = _split_map_name(path)
    if stem.endswith("_percentile"):
        stem = stem[: -len("_percentile")]
        concern_path = path.with_name(stem + suffix)
        percentile_path: Optional[Path] = path
        if not concern_path.exists():
            raise FileNotFoundError(
                f"concern map not found beside the percentile map: {concern_path.name}")
    else:
        concern_path = path
        sibling = path.with_name(stem + "_percentile" + suffix)
        percentile_path = sibling if sibling.exists() else None
    # The generator names its files "<model>_<metric>_hotspot"; the metric is what the field
    # selector shows, so drop the decoration rather than listing "1tec_rama_hotspot".
    metric = stem[: -len("_hotspot")] if stem.endswith("_hotspot") else stem
    return {metric or "concern": (concern_path, percentile_path, {})}


def read_fields(path: Any, volume_cls: Any) -> ConcernImport:
    """Import every concern field a manifest describes, or a single hand-opened concern map.

    Validates what the display contract depends on and nothing else: values really are in
    ``[0, 1]``, and a percentile companion — when one exists — really is on the same grid, so
    the two can be read against each other. A missing percentile map is not an error; it is
    optional analysis, and the primary display never consults it.
    """
    path = Path(path)
    payload: dict = {}
    if path.suffix.lower() == ".json":
        pairs, payload = _pairs_from_manifest(path)
    else:
        pairs = _pair_from_map(path)

    fields: Dict[str, ConcernField] = {}
    for metric, (concern_path, percentile_path, scaling) in pairs.items():
        concern = volume_cls.from_map_file(concern_path)
        values = np.asarray(concern.array, dtype=np.float32)
        if values.ndim != 3 or not values.size:
            raise ValueError(f"{metric}: concern map must be a non-empty 3-D grid")
        if not np.isfinite(values).all():
            raise ValueError(f"{metric}: concern map contains non-finite values")
        if values.min() < -1e-5 or values.max() > 1.00001:
            raise ValueError(
                f"{metric}: concern must be bounded to [0, 1] — this map spans "
                f"[{values.min():.3f}, {values.max():.3f}], which is not a concern field")

        percentile = percentile_values = None
        if percentile_path is not None:
            percentile = volume_cls.from_map_file(percentile_path)
            percentile_values = np.asarray(percentile.array, dtype=np.float32)
            if percentile_values.shape != values.shape:
                raise ValueError(f"{metric}: concern and percentile grids differ in shape")
            if concern.origin != percentile.origin or not np.allclose(
                    concern.pixel_sizes, percentile.pixel_sizes, rtol=0, atol=1e-6):
                raise ValueError(f"{metric}: concern and percentile maps are not co-registered")
            if not np.allclose(concern.map_manager.shift_cart(),
                               percentile.map_manager.shift_cart(), rtol=0, atol=1e-5):
                raise ValueError(f"{metric}: concern and percentile maps are placed differently")
            if not np.isfinite(percentile_values).all():
                raise ValueError(f"{metric}: percentile map contains non-finite values")
            percentile_values = np.ascontiguousarray(np.clip(percentile_values, 0.0, 1.0))

        fields[metric] = ConcernField(
            name=metric, concern=concern,
            values=np.ascontiguousarray(np.clip(values, 0.0, 1.0)),
            percentile=percentile, percentile_values=percentile_values,
            color_scaling=scaling if isinstance(scaling, dict) else {})

    molprobity = payload.get("molprobity") or {}
    return ConcernImport(
        fields=fields, anchors=display_anchors(payload), source=path,
        omitted_metrics=[str(m) for m in (molprobity.get("omitted_metrics") or [])],
        omission_reason=str(molprobity.get("reason") or ""))


# -- the residue table --------------------------------------------------------


def residue_columns(metrics: List[str]) -> List[str]:
    """Columns for :func:`residue_rows`.

    Every value column is labelled ``concern`` and every one of them is in ``[0, 1]``. Native
    validation numbers (a Ramachandran percentage, a clash overlap in A) are a different
    quantity on a different scale; if they are ever shown they belong in their own columns
    under their own names, never merged into one labelled ``concern``.
    """
    return ["chain", "resid", "res"] + [f"{metric} concern" for metric in metrics]


#: What a sampled row does and does not mean. Shown with the table, because the distinction
#: is not guessable from the numbers.
TABLE_CAVEAT = (
    "Values are the concern field read at each residue's atoms, so they rank neighborhoods, "
    "not residues: the generator splats each observation with a ~2 Å Gaussian, which is wider "
    "than the spacing between neighboring residues, so a residue next to a hotspot carries a "
    "fraction of it. A high row means look here, not that this residue is the outlier.")


def residue_rows(model: Any, fields: Dict[str, ConcernField], *, primary: str,
                 threshold: float = 0.0, limit: int = 200) -> List[list]:
    """The imported concern fields read back per residue, worst first.

    Values come from the *maps themselves*, sampled at each atom, so the table cannot drift
    from what the viewport draws — which is the whole point: a table on one scale beside a map
    on another is what made a favored residue read as a Ramachandran problem.

    Rolled up over a residue's atoms with a **max**, never a sum: the generator splats one
    observation across the atoms it implicates, so summing would count a single phi/psi four
    times and rank a large residue above a small one for no reason but size.

    **This is a field readout, not an attribution** (see :data:`TABLE_CAVEAT`). The splat is
    deliberately smooth and wider than the ~3.8 Å between adjacent Ca atoms, so concern
    deposited on one residue is genuinely present at its neighbours' atoms and there is no way
    to unmix it here. Exact per-residue concern exists upstream, before the splat; recovering
    it from the grid afterwards is not possible, and this module does not pretend otherwise by
    inventing a de-blurring rule. A coarse ``--output-pixel-size`` smooths the peaks down
    further, so sampled values sit below the deposited concern.

    Rows are ranked by ``primary`` (the field on screen) and filtered by the same absolute
    ``threshold`` that drives the display, so the table lists what is actually visible.
    """
    from .volume_io import sample_at_sites

    hierarchy = model.get_hierarchy()
    sites = np.asarray(hierarchy.atoms().extract_xyz(), dtype="float64").reshape(-1, 3)

    metrics = list(fields)
    sampled = {
        metric: sample_at_sites(fields[metric].concern.map_manager, sites,
                                array=fields[metric].values)
        for metric in metrics
    }

    by_residue: Dict[tuple, List[int]] = {}
    for i, atom in enumerate(hierarchy.atoms_with_labels()):
        key = (atom.chain_id.strip(), atom.resid().strip(), atom.resname.strip())
        by_residue.setdefault(key, []).append(i)

    cut = max(float(threshold), EPSILON)
    scored = []
    for key, indices in by_residue.items():
        parts = [float(sampled[metric][indices].max()) for metric in metrics]
        rank = parts[metrics.index(primary)] if primary in metrics else max(parts)
        if rank < cut:
            continue
        scored.append((rank, key, parts))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [[chain, resid, resname]
            + [f"{part:.3f}" if part > EPSILON else "" for part in parts]
            for _rank, (chain, resid, resname), parts in scored[:limit]]
