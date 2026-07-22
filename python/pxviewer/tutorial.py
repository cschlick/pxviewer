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


def _minimizing(cw: Any) -> bool:
    return not cw._desktop._minimize_idle.is_set()


def cryo_em_refinement_tutorial() -> Tutorial:
    """Real-space refine a model into a cryo-EM density — pxviewer's take on Phenix's
    real_space_refine, self-contained (map computed from the model, no external data)."""
    return Tutorial("Real-space refine into cryo-EM density", [
        Step(
            "Cryo-EM refinement (phenix's `real_space_refine`) slides a model into a 3D "
            "density map — a gradient-driven minimization, not against reflections but "
            "against the map itself.\n\nOpen the example: click **Demos** and pick **Cryo-EM "
            "— real-space refine a model into density**. It loads a model that sits slightly "
            "*off* its own density, waiting to be pushed back in.",
            done=lambda cw: cw._desktop.map_for_model() is not None,
            target=lambda cw: cw._demos_btn,
        ),
        Step(
            "Real-space refine it: on the **Tools** tab, in **Minimize**, tick **Use map** "
            "(so the minimizer pulls toward the density, not just ideal geometry) and click "
            "**Minimize**. Watch the model creep into the map — that *is* real-space "
            "refinement, streaming live.",
            done=_minimizing,
            target=lambda cw: cw._minimize_btn,
        ),
        Step(
            "When the model stops shifting it has settled into the density — click **Stop**. "
            "You just did what `phenix.real_space_refine` does: minimized an atomic model into "
            "a cryo-EM map, no reflections and no phenix. Re-run **Make maps** isn't needed — "
            "the map here is the target, fixed.",
            target=lambda cw: cw._minimize_map_check,
        ),
    ])


def _live_difference_seen(cw: Any) -> bool:
    return cw._desktop._diff_boxes > 0


def xray_refinement_tutorial() -> Tutorial:
    """Refine against X-ray data and watch the difference map answer back — break the fit by
    hand, see mFo-DFc light up live under the pointer, then minimize it back."""
    return Tutorial("X-ray: refine with a live difference map", [
        Step(
            "X-ray refinement judges a model against **data**, not against a map someone "
            "already made. The honest reporter is the **mFo-DFc difference map**: green where "
            "the data wants density the model does not explain, red where the model puts "
            "atoms the data will not support.\n\nOpen the example: click **Demos** and pick "
            "**X-ray — model + reflections (make density)**. It loads a model alongside "
            "amplitudes computed from that same model, so the two start in exact agreement — "
            "which gives us a flat difference map to break on purpose.",
            done=lambda cw: bool(cw._desktop._models) and bool(cw._desktop._reflections),
            target=lambda cw: cw._demos_btn,
        ),
        Step(
            "Phase the data: in the **Objects** list select the **reflections**, then click "
            "**Make maps** in the panel below. That computes **2mFo-DFc** — the map you build "
            "into — and **mFo-DFc**, the difference map, and pairs both with the model so "
            "they share a frame.\n\nCheck the R-work it reports: essentially zero, because "
            "this data came from this model. Contour the difference map and it has nothing to "
            "say — which is a difference map doing its job.",
            done=lambda cw: cw._desktop.map_for_model() is not None,
        ),
        Step(
            "Now arm the live feedback. On the **Settings** tab, in **Drag atoms**, tick "
            "**Live difference map**.\n\nFrom here on every drag re-phases mFo-DFc in a small "
            "box around the atom you are holding and streams it to the viewport as you move. "
            "Only that window updates — the whole-structure maps are deliberately left alone, "
            "so what you see is the data disagreeing with you, not a stale map echoing the "
            "model back.",
            done=lambda cw: cw._desktop._live_diff,
            target=lambda cw: cw._tug_livemap_check,
        ),
        Step(
            "Break the fit: **Shift-drag** an atom in the viewport and pull it out of its "
            "density.\n\nWatch the box that follows your pointer. **Red** blooms where you "
            "have just parked atoms the data does not support, and **green** stays behind in "
            "the density they left — the difference map recomputing as fast as you can drag. "
            "Let go and the window clears, leaving the model genuinely wrong.",
            done=_live_difference_seen,
            target=lambda cw: cw._tug_livemap_check,
        ),
        Step(
            "Refine it back. On the **Tools** tab, in **Minimization**, tick **Use map** and "
            "click **Minimize**.\n\nThe minimizer pulls the model toward the density while "
            "the geometry restraints keep bonds and angles honest — the two targets X-ray "
            "refinement always balances. Watch the atom slide home, streaming live.",
            done=_minimizing,
            target=lambda cw: cw._minimize_btn,
        ),
        Step(
            "When it stops moving click **Stop**, then select the **reflections** again and "
            "click **Update maps** to re-phase against the corrected model. The difference "
            "density you created is gone.\n\nThat is the whole X-ray loop, and why the "
            "difference map is the one to trust: it shows the error, you fix it — by hand or "
            "by minimizing — then re-phase and look again.",
            target=lambda cw: cw._minimize_stop_btn,
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
    """Every walkthrough offered, in menu order — validate, fit a ligand, then the two
    refinements (real-space into cryo-EM density, then X-ray against reflections), then the
    restraint-edits pair (reading before writing)."""
    return [validation_tutorial(), ligand_fitting_tutorial(), cryo_em_refinement_tutorial(),
            xray_refinement_tutorial(), load_edits_tutorial(), restraint_edits_tutorial()]
