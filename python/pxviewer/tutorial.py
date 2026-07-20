"""Guided walkthroughs for the desktop app — a non-modal 'coach' that steps the user
through a use case and advances itself when each task is actually done.

A tutorial is a list of :class:`Step`. Each step carries the instruction text, an optional
``done`` predicate the coach polls against live app state (so the step ticks itself off when
the user really does it, not when they click a button), and an optional ``action`` — a
labelled shortcut that performs the step for them (useful for setup like loading the sample).

Predicates and actions receive the ``ControlsWindow`` (``cw``), so they can read app state
via ``cw._desktop`` and, for actions, drive the UI. Keeping the content here — plain data —
means adding another walkthrough is just another list. The coach widget lives in
:mod:`pxviewer.desktop`.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple


class Step:
    def __init__(self, text: str, *, done: Optional[Callable[[Any], bool]] = None,
                 action: Optional[Tuple[str, Callable[[Any], None]]] = None) -> None:
        self.text = text
        self.done = done        # (cw) -> bool; when True the coach auto-advances
        self.action = action    # (label, (cw) -> None): a "do it for me" shortcut


class Tutorial:
    def __init__(self, title: str, steps: List[Step]) -> None:
        self.title = title
        self.steps = steps


def _sample_path() -> str:
    from .loader import sample_structure_path

    return str(sample_structure_path())


def _active(cw: Any) -> Optional[str]:
    return cw._desktop._active_model_id


def _selection_count(cw: Any) -> int:
    mid = _active(cw)
    return len(cw._desktop._scene_selection.get(mid, [])) if mid else 0


def _edit_count(cw: Any) -> int:
    mid = _active(cw)
    return len(cw._desktop.model_edits(mid)) if mid else 0


def restraint_edits_tutorial() -> Tutorial:
    """Author a custom restraint edit end to end — the feature's whole loop."""
    return Tutorial("Custom restraint edits", [
        Step(
            "Restraint **edits** add bonds and angles the monomer library can't know on its "
            "own — a covalent-ligand link, or a metal coordination bond. Let's author one on "
            "a sample structure.\n\nLoad the sample model to begin.",
            done=lambda cw: bool(cw._desktop._models),
            action=("Load the sample model", lambda cw: cw._desktop.load_files([_sample_path()])),
        ),
        Step(
            "Turn on atom picking, then click **two atoms in different residues** in the "
            "viewport — each click adds to the selection (pick different residues so the two "
            "aren't already bonded). Click empty space to start over.",
            done=lambda cw: _selection_count(cw) >= 2,
            action=("Turn on atom picking", lambda cw: cw._pick_btn.setChecked(True)),
        ),
        Step(
            "Open the **Tools** tab and find the **Restraint edits** panel (below Measure). "
            "Click **Bond** — it reads the current distance as the target and adds the "
            "restraint. Watch it appear in the list.",
            done=lambda cw: _edit_count(cw) >= 1,
        ),
        Step(
            "That's it — the custom bond now governs this app's minimize and drag. Use "
            "**Save…** in that panel to write it as a phenix `geometry_restraints.edits` "
            "file (or **Load…** one back).\n\nThe whole loop: point at atoms → author a "
            "restraint → refine here and hand the same restraint to phenix.refine.",
        ),
    ])


def all_tutorials() -> List[Tutorial]:
    """Every walkthrough offered, in menu order."""
    return [restraint_edits_tutorial()]
