"""Default-colour palettes for objects as they open.

A bundled set of named four-colour palettes (``data/palettes.json``). Each new *family* —
a model together with the maps phased from it, or a standalone object — is handed the next
palette, and its objects draw their opening colours from it: the model, its 2mFo-DFc/regular
maps, and so on. Everything stays a *default* — the moment a colour is changed by hand it
sticks; this only decides what a fresh object looks like before anyone touches it.

The palettes are deliberately loud and mutually-contrasting within a set, so a structure and
its maps read apart while still looking like a coordinated family, and successive structures
get visibly different palettes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

_PALETTES_PATH = Path(__file__).resolve().parent / "data" / "palettes.json"


def load_palettes() -> List[List[str]]:
    """Every bundled palette as a list of ``[c0, c1, c2, c3]`` hex-colour lists.

    Order follows the file (insertion order), so the cycle is stable across runs. Returns a
    single neutral fallback if the data file is somehow missing, so colouring never fails.
    """
    try:
        data = json.loads(_PALETTES_PATH.read_text())
    except Exception:  # pragma: no cover - the file ships with the package
        return [["#4C78A8", "#F58518", "#54A24B", "#B279A2"]]
    return [list(colours) for colours in data.values() if len(colours) >= 4]


class PaletteCycler:
    """Hands out palettes one after another, wrapping at the end.

    Stateful and deterministic: the Nth call to :meth:`next` returns palette ``N % count``,
    so the first structure of a session gets the first palette, the next the second, and so
    on. One cycler lives on the app; each family calls :meth:`next` once when it forms.
    """

    def __init__(self) -> None:
        self._palettes = load_palettes()
        self._cursor = 0

    def next(self) -> List[str]:
        palette = self._palettes[self._cursor % len(self._palettes)]
        self._cursor += 1
        return list(palette)
