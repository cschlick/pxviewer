"""A categorized, self-describing map of the ``session``/``app`` API.

Bound as ``api`` in the console namespace, this is the "how do I do X without
reading the docs" entry point: type ``api`` for every command grouped by topic
with a one-line description, ``api.find("color")`` to filter, and — since it is a
real IPython shell — ``session.<name>?`` for full help or ``session.<Tab>`` to
explore. The layered overview → filter → detail flow beats a flat dump.

It covers both bound objects, because a capability an agent cannot find is a
capability that does not exist to it: ``session`` verbs are the per-model
surface (LiveSession), ``app`` verbs the scene level (AppCore) — loading,
volumes, clips, jobs.
"""

from __future__ import annotations

import inspect
import re
from typing import List, Optional, Tuple

# `:class:`Selection`` -> `Selection`; then drop stray reST backticks.
_RST_ROLE = re.compile(r":[a-z]+:`([^`]+)`")

Row = Tuple[str, str, str]  # (name, prefixed compact signature, one-line doc)
Group = Tuple[str, List[Row]]

# Topic -> ordered method names, per bound object. Anything public but unlisted
# lands in "Other", so a newly added API method still shows up (just
# uncategorized).
_SESSION_CATEGORIES: List[Tuple[str, List[str]]] = [
    ("Selecting atoms", [
        "select", "select_by", "highlight", "focus", "clear_selection",
        "enable_mouse_selection", "enable_measure_mode", "disable_mouse_selection",
        "wait_for_selection", "on_selection", "on_pick", "on_measurement",
    ]),
    ("Representations & color", [
        "set_representation", "add_representation", "remove_representation",
        "clear_representations", "color_by",
    ]),
    ("Per-atom attributes", [
        "set_attribute", "attributes", "load_attributes", "load_attribute_text", "write_cif",
    ]),
    ("Measurements", [
        "add_distance", "add_angle", "add_dihedral", "add_label",
        "remove_primitive", "clear_primitives",
    ]),
    ("Interactions & clashes", [
        "set_interactions", "clear_interactions",
        "set_computed_interactions", "show_computed_interactions", "hide_computed_interactions",
        "set_clashes", "clear_clashes",
    ]),
    ("Probe contacts (with hydrogens)", [
        "show_probe_dots", "clear_probe_dots",
    ]),
    ("Volumes", [
        "set_volume_color", "set_volume_opacity", "set_volume_style", "set_volume_position",
        "set_volume_negative_iso", "set_volume_negative_color", "set_volume_negative_style",
        "set_volume_negative_opacity", "set_volume_negative_visible",
        "reload_volume", "show_map_box", "clear_map_box", "set_localres_domain",
    ]),
    ("Display, camera & clipping", [
        "set_clip", "set_camera", "set_background",
    ]),
    ("Coordinates", ["push"]),
    ("Session", ["model", "diff", "start", "stop", "wait_for_client"]),
]

_APP_CATEGORIES: List[Tuple[str, List[str]]] = [
    ("Loading & objects", [
        "load_file", "load_files", "fetch_and_load", "add_map_file",
        "set_active_model", "remove_model", "remove_volume",
    ]),
    ("Selection", [
        "select_by_expression", "selection_description", "clear_selection",
        "show_selected", "hide_selected",
    ]),
    ("Regions & clipping", [
        "set_model_sphere", "set_model_clip", "model_sphere_state",
        "set_volume_sphere", "set_volume_clip", "volume_sphere_state",
        "set_volume_box", "set_volume_downsample", "set_volume_mask",
        "set_view_radius",
    ]),
    ("Volumes", [
        "set_volume_iso", "set_volume_style", "set_volume_color",
        "set_volume_opacity", "set_volume_visible", "set_volume_scroll_target",
        "set_volume_negative_iso", "set_volume_negative_color",
        "set_volume_negative_style", "set_volume_negative_opacity",
        "set_volume_negative_visible",
        "set_color_by_resolution", "fit_localres_domain",
    ]),
    ("Models", [
        "set_model_visible", "set_model_color", "set_model_representation",
        "set_model_type_hidden", "model_appearance", "volume_appearance",
        "color_model_by_property", "color_model_by_mapcc",
        "color_model_by_qscore", "color_model_by_hotspots",
    ]),
    ("Refinement & validation", [
        "minimize_model", "stop_minimization", "run_validation",
        "compute_real_space_difference", "set_live_difference_map",
        "compute_resolution_map", "make_maps", "update_maps",
        "fit_ligand_at_marker", "fit_ligand_from_smiles_at_marker",
    ]),
    ("Output", ["save_screenshot", "write_object"]),
    ("Jobs", ["run_background", "job_state"]),
]


def _one_line(obj) -> str:
    doc = (inspect.getdoc(obj) or "").split("\n", 1)[0].strip()
    doc = _RST_ROLE.sub(r"\1", doc)  # strip Sphinx roles like :class:`Selection`
    return doc.replace("``", "").replace("`", "")


def _compact_signature(name: str, obj) -> str:
    try:
        params = [p for p in inspect.signature(obj).parameters if p != "self"]
    except (TypeError, ValueError):
        return f"{name}(…)"
    return f"{name}(…)" if params else f"{name}()"


def _row(name: str, obj, prefix: str) -> Optional[Row]:
    if isinstance(obj, property):
        return (name, f"{prefix}.{name}", _one_line(obj.fget))
    if callable(obj):
        return (name, f"{prefix}.{_compact_signature(name, obj)}", _one_line(obj))
    return None


def build_groups(cls, categories=None, prefix: str = "session") -> List[Group]:
    """Introspect ``cls`` into ``[(category, [(name, signature, doc)])]``."""
    if categories is None:
        categories = _APP_CATEGORIES if prefix == "app" else _SESSION_CATEGORIES
    groups: List[Group] = []
    seen = set()
    for category, names in categories:
        rows = []
        for name in names:
            row = _row(name, getattr(cls, name, None), prefix)
            if row is not None:
                rows.append(row)
                seen.add(name)
        if rows:
            groups.append((category, rows))
    extra = [
        row
        for name, obj in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_") and name not in seen
        for row in (_row(name, obj, prefix),)
        if row is not None
    ]
    if extra:
        groups.append(("Other", sorted(extra)))
    return groups


class ApiGuide:
    """A self-printing overview of the API (bound as ``api`` in the console)."""

    def __init__(self, cls=None, groups: Optional[List[Group]] = None, title: str = "pxviewer API"):
        if groups is None:
            if cls is None:
                from .live import LiveSession

                cls = LiveSession
            groups = build_groups(cls, prefix=self._prefix_for(cls))
        self._groups = groups
        self._title = title

    @staticmethod
    def _prefix_for(cls) -> str:
        return "app" if cls.__name__ == "AppCore" else "session"

    @classmethod
    def full(cls) -> "ApiGuide":
        """Both bound objects: ``session`` (LiveSession) then ``app`` (AppCore).

        A capability that only lives on ``app`` must still be findable —
        session-only coverage is exactly how an agent ends up reinventing
        region-carving with files instead of finding ``set_volume_box``.
        """
        from .appcore import AppCore
        from .live import LiveSession

        return cls(groups=(
            build_groups(LiveSession, _SESSION_CATEGORIES, "session")
            + build_groups(AppCore, _APP_CATEGORIES, "app")))

    def find(self, text: str) -> "ApiGuide":
        """A filtered guide: rows whose name or description contains ``text``."""
        t = text.lower()
        filtered = [
            (category, matched)
            for category, rows in self._groups
            for matched in ([r for r in rows if t in r[0].lower() or t in r[2].lower()],)
            if matched
        ]
        return ApiGuide(groups=filtered, title=f"pxviewer API — matching {text!r}")

    # `api("color")` reads as nicely as `api.find("color")`.
    def __call__(self, text: str) -> "ApiGuide":
        return self.find(text)

    def __repr__(self) -> str:
        lines = [self._title, "call: session.<name>(…) / app.<name>(…)   help: <name>?"]
        for category, rows in self._groups:
            lines.append("")
            lines.append(f"  {category}")
            for name, sig, doc in rows:
                lines.append(f"    {sig:<32} {doc}")
        if not self._groups:
            lines.append("")
            lines.append("  (nothing matched)")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        parts = [
            f'<div style="font-family:sans-serif">'
            f'<div style="font-weight:600;margin-bottom:2px">{self._title}</div>'
            f'<div style="color:#666;font-size:90%;margin-bottom:8px">'
            f'call <code>session.name(…)</code> or <code>app.name(…)</code> &middot; '
            f'help <code>name?</code></div>'
        ]
        for category, rows in self._groups:
            parts.append(f'<div style="font-weight:600;margin-top:8px">{category}</div>')
            parts.append('<table style="border-collapse:collapse">')
            for name, sig, doc in rows:
                parts.append(
                    '<tr>'
                    f'<td style="padding:1px 12px 1px 8px;white-space:nowrap">'
                    f'<code>{sig}</code></td>'
                    f'<td style="padding:1px 0;color:#333">{doc}</td>'
                    '</tr>'
                )
            parts.append('</table>')
        if not self._groups:
            parts.append('<div style="color:#999">(nothing matched)</div>')
        parts.append('</div>')
        return "".join(parts)
