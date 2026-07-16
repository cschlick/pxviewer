"""Parse MolProbity kinemage markup into drawable primitives.

The MolProbity validators emit their markup as kinemage text: ``@vectorlist``
(line segments), ``@dotlist`` (points), ``@balllist`` (spheres) and
``@trianglelist`` (filled triangles). We parse that into plain, JSON-serialisable
primitives the viewer can render directly.

Kinemage point syntax (one or more points per physical line)::

    {label} <flags> x, y, z       flags: P = pen-up (move / start), L = line-to,
    {label} r=0.322 x, y, z       r= gives a ball radius; coords are comma or space
                                  separated and come AFTER the {label} (labels can
                                  themselves contain numbers).

A primitive is a dict: always ``kind`` + ``color`` ([r,g,b]); geometry per kind:
``vectors``->``segments`` [[p,p],..], ``dots``->``points`` [p,..],
``balls``->``balls`` [[p,radius],..], ``triangles``->``triangles`` [[p,p,p],..].
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Kinemage named colours -> RGB (the palette MolProbity validation markup uses).
_KIN_COLORS = {
    "green": (0x00, 0xFF, 0x00), "gold": (0xFF, 0xD7, 0x00), "magenta": (0xFF, 0x00, 0xFF),
    "sea": (0x00, 0xFF, 0xC0), "lime": (0x80, 0xFF, 0x00), "yellow": (0xFF, 0xFF, 0x00),
    "purple": (0xA0, 0x20, 0xF0), "red": (0xFF, 0x00, 0x00), "white": (0xFF, 0xFF, 0xFF),
    "pink": (0xFF, 0x8C, 0xC0), "hotpink": (0xFF, 0x00, 0x80), "blue": (0x00, 0x00, 0xFF),
    "sky": (0x00, 0xB0, 0xFF), "orange": (0xFF, 0x80, 0x00), "cyan": (0x00, 0xFF, 0xFF),
    "deadblack": (0x00, 0x00, 0x00), "black": (0x00, 0x00, 0x00), "gray": (0x80, 0x80, 0x80),
    "grey": (0x80, 0x80, 0x80), "peach": (0xFF, 0xB0, 0x70), "yellowtint": (0xFF, 0xFF, 0xC0),
    "greentint": (0xA0, 0xFF, 0xA0),
}
_DEFAULT_COLOR = (0xFF, 0xFF, 0xFF)
_DEFAULT_BALL_RADIUS = 0.2

# Pen flags: P starts a new stroke/strip (pen-up move), the rest just continue it.
_FLAGS = {"P", "L", "U", "M", "X"}

_LIST_KINDS = {
    "@vectorlist": "vectors", "@dotlist": "dots",
    "@balllist": "balls", "@trianglelist": "triangles",
}

_COLOR_RE = re.compile(r"color=\s*(\w+)")
_POINT_RE = re.compile(r"\{[^}]*\}([^{]*)")  # capture the text after each {label}


def _parse_point(chunk: str):
    """Parse the text after one ``{label}`` into (flags, xyz, radius, colour). xyz is
    None if the chunk has no coordinate triple; colour is None unless the point names
    one (a per-point colour overrides the list's, e.g. CaBLAM's score-coded wheels)."""
    flags = set()
    radius = None
    color = None
    coords: List[float] = []
    for tok in chunk.replace(",", " ").split():
        if tok.startswith("r="):
            try:
                radius = float(tok[2:])
            except ValueError:
                pass
        elif tok.startswith("width="):
            continue
        elif tok in _FLAGS:
            flags.add(tok)
        else:
            try:
                coords.append(float(tok))
            except ValueError:
                if tok.lower() in _KIN_COLORS:
                    color = _KIN_COLORS[tok.lower()]
    xyz = tuple(coords[-3:]) if len(coords) >= 3 else None
    return flags, xyz, radius, color


def _points_in_line(line: str):
    """Yield (flags, xyz, radius, colour) for every ``{label} … x y z`` point."""
    for chunk in _POINT_RE.findall(line):
        flags, xyz, radius, color = _parse_point(chunk)
        if xyz is not None:
            yield flags, xyz, radius, color


def _resolve_colors(points, header_color):
    """Give each point its effective colour: the list's, overridden by a per-point
    colour which then persists for the following points (kinemage's rule)."""
    resolved = []
    current = header_color
    for flags, xyz, radius, color in points:
        if color is not None:
            current = color
        resolved.append((flags, xyz, radius, current))
    return resolved


def _segments(points) -> dict:
    """colour -> line segments. P is a pen-up move; others draw from the previous
    point (the segment takes the colour of the point it draws to)."""
    segments: dict = {}
    prev = None
    for flags, xyz, _r, color in points:
        if "P" in flags or prev is None:
            prev = xyz
        else:
            segments.setdefault(color, []).append([list(prev), list(xyz)])
            prev = xyz
    return segments


def _triangles(points) -> dict:
    """colour -> triangles. P starts a new strip; within a strip each new point makes
    a triangle with the previous two."""
    triangles: dict = {}
    strip: List[tuple] = []
    for flags, xyz, _r, color in points:
        if "P" in flags:
            strip = [xyz]
        else:
            strip.append(xyz)
            if len(strip) >= 3:
                a, b, c = strip[-3], strip[-2], strip[-1]
                triangles.setdefault(color, []).append([list(a), list(b), list(c)])
    return triangles


def parse_kinemage(text: str) -> List[dict]:
    """Parse kinemage ``text`` into a list of drawable primitive dicts."""
    prims: List[dict] = []
    kind = None
    color = _DEFAULT_COLOR
    points: List[tuple] = []

    def flush():
        """Emit one primitive per distinct colour in the list just parsed."""
        if kind is None or not points:
            return
        resolved = _resolve_colors(points, color)
        if kind == "dots":
            by: dict = {}
            for _f, xyz, _r, c in resolved:
                by.setdefault(c, []).append(list(xyz))
            for c, pts in by.items():
                prims.append({"kind": "dots", "color": list(c), "points": pts})
        elif kind == "balls":
            by = {}
            for _f, xyz, r, c in resolved:
                by.setdefault(c, []).append([list(xyz), r if r is not None else _DEFAULT_BALL_RADIUS])
            for c, balls in by.items():
                prims.append({"kind": "balls", "color": list(c), "balls": balls})
        elif kind == "vectors":
            for c, segs in _segments(resolved).items():
                prims.append({"kind": "vectors", "color": list(c), "segments": segs})
        elif kind == "triangles":
            for c, tris in _triangles(resolved).items():
                prims.append({"kind": "triangles", "color": list(c), "triangles": tris})

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):
            head = line.split(None, 1)[0]
            if head in _LIST_KINDS:
                flush()
                kind = _LIST_KINDS[head]
                points = []
                m = _COLOR_RE.search(line)
                color = _KIN_COLORS.get(m.group(1).lower(), _DEFAULT_COLOR) if m else _DEFAULT_COLOR
            else:  # @subgroup / @group / @master etc. end the current list
                flush()
                kind = None
                points = []
            continue
        if kind is not None:
            points.extend(_points_in_line(line))
    flush()
    return prims
