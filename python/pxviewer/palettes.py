"""Default-color palettes for objects as they open.

A bundled set of named four-color palettes (``data/palettes.json``). One :class:`PaletteCycler`
lives on the app and hands each new object an opening color: it picks a *random* palette group
when the app starts, gives each object a random color from that group, and rolls to a fresh
random group every fourth object. So colors differ from run to run (not a fixed sequence),
objects that open close together share a group and therefore contrast rather than clash, and
across a session the palette keeps changing.

Everything stays a *default* — the moment a color is changed by hand it sticks; this only
decides what a fresh object looks like before anyone touches it. Difference maps are exempt
(they keep their conventional green/red) and so never draw a palette color.
"""

from __future__ import annotations

import colorsys
import json
import random
from pathlib import Path
from typing import List, Optional

_PALETTES_PATH = Path(__file__).resolve().parent / "data" / "palettes.json"


def load_palettes() -> List[List[str]]:
    """Every bundled palette as a list of ``[c0, c1, c2, c3]`` hex-color lists.

    Returns a single neutral fallback if the data file is somehow missing, so coloring
    never fails.
    """
    try:
        data = json.loads(_PALETTES_PATH.read_text())
    except Exception:  # pragma: no cover - the file ships with the package
        return [["#4C78A8", "#F58518", "#54A24B", "#B279A2"]]
    return [list(colors) for colors in data.values() if len(colors) >= 4]


#: How many objects share one random group before a new one is rolled.
_GROUP_SIZE = 4


def _hue(color: str) -> float:
    """A ``#RRGGBB`` color's hue in [0, 1); anything unparseable sorts first."""
    try:
        r, g, b = (int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    except (ValueError, IndexError):  # pragma: no cover - the bundled file is hex
        return -1.0
    return colorsys.rgb_to_hsv(r, g, b)[0]


def suggested_colors(count: int = 8) -> List[str]:
    """A short list of distinct colors to offer when picking one by hand.

    Drawn from the same bundled palettes the automatic defaults come from, so a color set
    deliberately sits in the same family as the ones objects open in, rather than looking
    imported from somewhere else. Spread around the hue circle and de-duplicated, so
    neighboring swatches read apart instead of being three versions of the same pink.
    """
    seen: List[str] = []
    for palette in load_palettes():
        for color in palette:
            if color not in seen:
                seen.append(color)
    if not seen:  # pragma: no cover - load_palettes always yields its fallback
        return []
    by_hue = sorted(seen, key=_hue)
    if count >= len(by_hue):
        return by_hue
    step = len(by_hue) / count
    return [by_hue[int(i * step)] for i in range(count)]


class PaletteCycler:
    """Hands out random opening colors (see the module docstring).

    A random group is chosen when the cycler is created; each :meth:`next_color` returns a
    random color from it, avoiding the one just handed out so two objects in a row are never
    identical; every ``_GROUP_SIZE`` colors a fresh random group is rolled. Seeded from
    system entropy, so a session's colors differ from the last.
    """

    def __init__(self, *, seed: Optional[int] = None) -> None:
        self._palettes = load_palettes()
        self._rng = random.Random(seed)  # seed only for tests; None -> entropy, different runs
        self._group = list(self._rng.choice(self._palettes))
        self._assigned = 0
        self._last: Optional[str] = None

    def next_color(self) -> str:
        """A random color for the next object — see the class docstring."""
        if self._assigned and self._assigned % _GROUP_SIZE == 0:
            self._group = list(self._rng.choice(self._palettes))  # a new group every four
        choices = [c for c in self._group if c != self._last] or list(self._group)
        color = self._rng.choice(choices)
        self._assigned += 1
        self._last = color
        return color
