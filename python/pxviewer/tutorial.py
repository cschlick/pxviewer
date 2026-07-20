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


def _validation_ran(cw: Any) -> bool:
    mid = _active(cw)
    entry = cw._desktop._model_entry(mid) if mid else None
    return bool(entry and entry.get("validation"))


def validation_tutorial() -> Tutorial:
    """Run MolProbity validation and read the results — find what looks wrong in a model."""
    return Tutorial("Validate a structure", [
        Step(
            "MolProbity **validation** flags the parts of a model that look wrong — bad "
            "rotamers, Ramachandran and C-beta outliers, backbone (CaBLAM) problems, odd "
            "cis-peptides. Let's run it on a structure built to trip every check.\n\nOpen it: "
            "click **Demos** and pick **Thermitase-eglin (1TEC)**.",
            done=lambda cw: bool(cw._desktop._models),
            target=lambda cw: cw._demos_btn,
        ),
        Step(
            "Open the **Validation** tab and click **Run validation**. It runs every "
            "validator on the active model in the background — give it a moment.",
            done=_validation_ran,
            target=lambda cw: cw._validate_btn,
        ),
        Step(
            "Each validator now has its own sub-tab: a summary, a table of outliers, and a "
            "**Markers** switch that draws the problems right in the viewport. Click any row "
            "in a table to select and zoom to that residue.\n\nThat's the loop — find the "
            "outliers, see them in 3D, fix them (drag or minimize), and re-run.",
        ),
    ])


def ligand_fitting_tutorial() -> Tutorial:
    """Fit a ligand into difference density — pxviewer's take on Phenix's ligand-fitting
    tutorial, self-contained (no phenix, no external data)."""
    return Tutorial("Fit a ligand into density", [
        Step(
            "Phenix's ligand-fitting tutorial fits a flexible ligand into a difference map. "
            "Let's do the same, straight from data.\n\nOpen the example: click **Demos** and "
            "pick **Ligand fitting** — a ligand-free model plus reflections that secretly "
            "contain an ATP.",
            done=lambda cw: bool(cw._desktop._models) and bool(cw._desktop._reflections),
            target=lambda cw: cw._demos_btn,
        ),
        Step(
            "Compute the maps: in the **Scene** list select the **reflections** object and "
            "click **Make maps** in its panel. That phases the data against the model — and "
            "the **mFo-DFc** difference map lights up a green blob where the model is missing "
            "atoms: the ATP.",
            done=lambda cw: cw._desktop.map_for_model() is not None,
        ),
        Step(
            "Mark the blob. Contour the **mFo-DFc** map (scroll the wheel over the viewport) "
            "and rotate to the green density near the protein. Then on the **Tools** tab, in "
            "**Ligand placement**, click **Place ligand marker** and click the blob in the "
            "viewport to drop a marker there.",
            done=lambda cw: len(cw._desktop._markers) >= 1,
            target=lambda cw: cw._lig_place_btn,
        ),
        Step(
            "Build and fit: in the Ligand placement panel type **ATP** in the monomer-code "
            "box, tick **Fit into density**, and click **Fit ligand here**. It builds ATP and "
            "settles it into the density (explode-and-refine).",
            done=lambda cw: any("ligand" in m["name"].lower() for m in cw._desktop._models),
            target=lambda cw: cw._lig_fit_btn,
        ),
        Step(
            "Done — ATP is now modelled in the density that was empty. That is the whole "
            "ligand-fitting loop, the same as Phenix's tutorial: difference map → place → "
            "build → fit — with no phenix and no downloaded dataset.",
        ),
    ])


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
    """Every walkthrough offered, in menu order — validate, fit a ligand, then the restraint-
    edits pair (reading before writing)."""
    return [validation_tutorial(), ligand_fitting_tutorial(),
            load_edits_tutorial(), restraint_edits_tutorial()]
