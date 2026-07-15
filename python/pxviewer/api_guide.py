"""A categorised, self-describing map of the ``LiveSession`` API.

Bound as ``api`` in the console namespace, this is the "how do I do X without
reading the docs" entry point: type ``api`` for every command grouped by topic
with a one-line description, ``api.find("color")`` to filter, and — since it is a
real IPython shell — ``session.<name>?`` for full help or ``session.<Tab>`` to
explore. The layered overview → filter → detail flow beats a flat dump.
"""

from __future__ import annotations

import inspect
import re
from typing import List, Optional, Tuple

# `:class:`Selection`` -> `Selection`; then drop stray reST backticks.
_RST_ROLE = re.compile(r":[a-z]+:`([^`]+)`")

Row = Tuple[str, str, str]  # (name, compact signature, one-line doc)
Group = Tuple[str, List[Row]]

# Topic -> ordered method names. Anything public but unlisted lands in "Other",
# so a newly added API method still shows up (just uncategorised).
_CATEGORIES: List[Tuple[str, List[str]]] = [
    ("Selecting atoms", [
        "select", "select_by", "highlight", "focus", "clear_selection",
        "enable_mouse_selection", "enable_measure_mode", "disable_mouse_selection",
        "wait_for_selection", "on_selection", "on_pick", "on_measurement",
    ]),
    ("Representations & colour", [
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
        "detect_clashes", "set_clashes", "show_clashes", "clear_clashes",
    ]),
    ("Volumes", [
        "set_volume_color", "set_volume_opacity", "set_volume_style", "set_volume_position",
    ]),
    ("Display & camera", ["set_axis"]),
    ("Coordinates", ["push"]),
    ("Session", ["model", "diff", "start", "stop", "wait_for_client"]),
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


def _row(name: str, obj) -> Optional[Row]:
    if isinstance(obj, property):
        return (name, name, _one_line(obj.fget))
    if callable(obj):
        return (name, _compact_signature(name, obj), _one_line(obj))
    return None


def build_groups(cls) -> List[Group]:
    """Introspect ``cls`` into ``[(category, [(name, signature, doc)])]``."""
    groups: List[Group] = []
    seen = set()
    for category, names in _CATEGORIES:
        rows = []
        for name in names:
            row = _row(name, getattr(cls, name, None))
            if row is not None:
                rows.append(row)
                seen.add(name)
        if rows:
            groups.append((category, rows))
    extra = [
        row
        for name, obj in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_") and name not in seen
        for row in (_row(name, obj),)
        if row is not None
    ]
    if extra:
        groups.append(("Other", sorted(extra)))
    return groups


class ApiGuide:
    """A self-printing overview of the API (bound as ``api`` in the console)."""

    def __init__(self, cls=None, groups: Optional[List[Group]] = None, title: str = "pxviewer API"):
        if cls is None:
            from .live import LiveSession

            cls = LiveSession
        self._cls = cls
        self._groups = build_groups(cls) if groups is None else groups
        self._title = title

    def find(self, text: str) -> "ApiGuide":
        """A filtered guide: rows whose name or description contains ``text``."""
        t = text.lower()
        filtered = [
            (category, matched)
            for category, rows in self._groups
            for matched in ([r for r in rows if t in r[0].lower() or t in r[2].lower()],)
            if matched
        ]
        return ApiGuide(self._cls, filtered, title=f"pxviewer API — matching {text!r}")

    # `api("color")` reads as nicely as `api.find("color")`.
    def __call__(self, text: str) -> "ApiGuide":
        return self.find(text)

    def __repr__(self) -> str:
        lines = [self._title, "call: session.<name>(...)   help: session.<name>?"]
        for category, rows in self._groups:
            lines.append("")
            lines.append(f"  {category}")
            for name, sig, doc in rows:
                lines.append(f"    session.{sig:<26} {doc}")
        if not self._groups:
            lines.append("")
            lines.append("  (nothing matched)")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        parts = [
            f'<div style="font-family:sans-serif">'
            f'<div style="font-weight:600;margin-bottom:2px">{self._title}</div>'
            f'<div style="color:#666;font-size:90%;margin-bottom:8px">'
            f'call <code>session.name(...)</code> &middot; help <code>session.name?</code></div>'
        ]
        for category, rows in self._groups:
            parts.append(f'<div style="font-weight:600;margin-top:8px">{category}</div>')
            parts.append('<table style="border-collapse:collapse">')
            for name, sig, doc in rows:
                parts.append(
                    '<tr>'
                    f'<td style="padding:1px 12px 1px 8px;white-space:nowrap">'
                    f'<code>session.{sig}</code></td>'
                    f'<td style="padding:1px 0;color:#333">{doc}</td>'
                    '</tr>'
                )
            parts.append('</table>')
        if not self._groups:
            parts.append('<div style="color:#999">(nothing matched)</div>')
        parts.append('</div>')
        return "".join(parts)
