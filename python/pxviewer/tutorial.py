"""Guided walkthroughs for the desktop app — a non-modal 'coach' that steps the user
through a use case and advances itself when each task is actually done.

A tutorial is a list of :class:`Step`. Each step carries the instruction text, an optional
``done`` predicate the coach polls against live app state (so the step ticks itself off when
the user really does it, not when they click a button), and an optional ``target`` — the
widget the step is about. The coach never does the task; it only offers a "Show me where"
button that flashes ``target`` (revealing its tab first) so the user can find the control
and do it themselves.

Predicates and targets receive the ``ControlsWindow`` (``cw``), so they read app state via
``cw._desktop`` and return a widget from ``cw``. Keeping the content here — plain data —
means adding another walkthrough is just another list. The coach widget lives in
:mod:`pxviewer.desktop`.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional


class Step:
    def __init__(self, text: str, *, done: Optional[Callable[[Any], bool]] = None,
                 target: Optional[Callable[[Any], Any]] = None) -> None:
        self.text = text
        self.done = done        # (cw) -> bool; when True the coach auto-advances
        self.target = target    # (cw) -> QWidget the "Show me where" button flashes


class Tutorial:
    def __init__(self, title: str, steps: List[Step]) -> None:
        self.title = title
        self.steps = steps


def _active(cw: Any) -> Optional[str]:
    return cw._desktop._active_model_id


def _selection_count(cw: Any) -> int:
    mid = _active(cw)
    return len(cw._desktop._scene_selection.get(mid, [])) if mid else 0


def _edit_count(cw: Any) -> int:
    mid = _active(cw)
    return len(cw._desktop.model_edits(mid)) if mid else 0


def load_edits_tutorial() -> Tutorial:
    """Load a shared restraint-edits file onto a structure — the reading half of the loop."""
    return Tutorial("Load restraint edits", [
        Step(
            "Restraint **edits** — custom bonds/angles the monomer library can't know — can "
            "be shared as a phenix PHIL file. Let's load one onto a metal site.\n\nOpen the "
            "example: click **Demos** and pick **Metal site — Zn coordination**.",
            done=lambda cw: bool(cw._desktop._models),
            target=lambda cw: cw._demos_btn,
        ),
        Step(
            "On the **Tools** tab, in the **Restraint edits** panel (below Measure), click "
            "**Load…** and open the sample file (`zn_site_edits.phil`, already selected). It "
            "adds the **Zn–water** coordination bond — the one cctbx doesn't restrain on its "
            "own — so watch it appear in the list.",
            done=lambda cw: _edit_count(cw) >= 1,
            target=lambda cw: cw._edit_load_btn,
        ),
        Step(
            "Loaded! That Zn–water restraint now governs this app's minimize and drag, and "
            "it came straight from a phenix `geometry_restraints.edits` file — the same file "
            "phenix.refine reads.\n\nNext, try **Custom restraint edits** to author one "
            "yourself.",
        ),
    ])


def restraint_edits_tutorial() -> Tutorial:
    """Author a custom restraint edit end to end — the writing half of the loop."""
    return Tutorial("Custom restraint edits", [
        Step(
            "Now let's author a restraint by hand. A metal's coordination is a good case: "
            "cctbx guesses the Zn–His bonds, but not the water in the fourth site.\n\nOpen "
            "the example: click **Demos** and pick **Metal site — Zn coordination**.",
            done=lambda cw: bool(cw._desktop._models),
            target=lambda cw: cw._demos_btn,
        ),
        Step(
            "Turn on atom picking with the **Pick** button, then click the **zinc** and the "
            "**water oxygen** beside it — the pair that isn't already coordinated. Each click "
            "adds to the selection; click empty space to start over.",
            done=lambda cw: _selection_count(cw) >= 2,
            target=lambda cw: cw._pick_btn,
        ),
        Step(
            "On the **Tools** tab, in the **Restraint edits** panel, click **Bond**. It takes "
            "the current Zn–water distance as the target and adds the restraint — watch it "
            "appear in the list. (If it says the bond already exists, you picked two atoms "
            "cctbx already coordinated — pick the zinc and the lone water instead.)",
            done=lambda cw: _edit_count(cw) >= 1,
            target=lambda cw: cw._edit_bond_btn,
        ),
        Step(
            "That's the whole loop — the custom bond now governs this app's minimize and "
            "drag. Use **Save…** to write it as a phenix `geometry_restraints.edits` file "
            "(exactly the kind the Load tutorial reads), for phenix.refine.",
            target=lambda cw: cw._edit_save_btn,
        ),
    ])


def all_tutorials() -> List[Tutorial]:
    """Every walkthrough offered, in menu order — reading before writing."""
    return [load_edits_tutorial(), restraint_edits_tutorial()]
