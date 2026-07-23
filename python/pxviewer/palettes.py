"""Default-colour palettes for objects as they open.

A bundled set of named four-colour palettes (``data/palettes.json``). One :class:`PaletteCycler`
lives on the app and hands each new object an opening colour: it picks a *random* palette group
when the app starts, gives each object a random colour from that group, and rolls to a fresh
random group every fourth object. So colours differ from run to run (not a fixed sequence),
objects that open close together share a group and therefore contrast rather than clash, and
across a session the palette keeps changing.

Everything stays a *default* — the moment a colour is changed by hand it sticks; this only
decides what a fresh object looks like before anyone touches it. Difference maps are exempt
(they keep their conventional green/red) and so never draw a palette colour.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

_PALETTES_PATH = Path(__file__).resolve().parent / "data" / "palettes.json"


def load_palettes() -> List[List[str]]:
    """Every bundled palette as a list of ``[c0, c1, c2, c3]`` hex-colour lists.

    Returns a single neutral fallback if the data file is somehow missing, so colouring
    never fails.
    """
    try:
        data = json.loads(_PALETTES_PATH.read_text())
    except Exception:  # pragma: no cover - the file ships with the package
        return [["#4C78A8", "#F58518", "#54A24B", "#B279A2"]]
    return [list(colours) for colours in data.values() if len(colours) >= 4]


#: How many objects share one random group before a new one is rolled.
_GROUP_SIZE = 4


class PaletteCycler:
    """Hands out random opening colours (see the module docstring).

    A random group is chosen when the cycler is created; each :meth:`next_colour` returns a
    random colour from it, avoiding the one just handed out so two objects in a row are never
    identical; every ``_GROUP_SIZE`` colours a fresh random group is rolled. Seeded from
    system entropy, so a session's colours differ from the last.
    """

    def __init__(self, *, seed: Optional[int] = None) -> None:
        self._palettes = load_palettes()
        self._rng = random.Random(seed)  # seed only for tests; None -> entropy, different runs
        self._group = list(self._rng.choice(self._palettes))
        self._assigned = 0
        self._last: Optional[str] = None

    def next_colour(self) -> str:
        """A random colour for the next object — see the class docstring."""
        if self._assigned and self._assigned % _GROUP_SIZE == 0:
            self._group = list(self._rng.choice(self._palettes))  # a new group every four
        choices = [c for c in self._group if c != self._last] or list(self._group)
        colour = self._rng.choice(choices)
        self._assigned += 1
        self._last = colour
        return colour
