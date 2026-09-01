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
    """A walkthrough, and the data it walks through.

    ``loader`` is how the tutorial gets its own example on screen: ``(desktop) -> None``,
    run when the tutorial starts. A tutorial that had to *ask* for its data spent its first
    step on "click Get and pick X" -- navigation, not crystallography -- and, worse, could
    not tell whether that had happened: the step's ``done`` predicate could only ask "is a
    model loaded?", which is already true for anyone with their own work open. Starting
    such a tutorial skipped straight to step 2 and then described a structure that was not
    the one on screen. Loading its own data makes the precondition true by construction.
    """

    def __init__(self, title: str, steps: List[Step],
                 loader: Optional[Callable[[Any], None]] = None) -> None:
        self.title = title
        self.steps = steps
        self.loader = loader


def _load_bundled(desktop: Any, filename: str) -> None:
    """Load one of the bundled sample structures by name."""
    from .loader import sample_structure_path

    path = sample_structure_path(filename)
    if path is None:
        raise FileNotFoundError("the bundled sample %s is missing" % filename)
    desktop.load_file(str(path))


#: The worked example: p53 bound to the nucleosome, a 4.2 A cryo-EM reconstruction whose
#: periphery is markedly softer than its core -- which is the whole point of looking at
#: local resolution rather than the single number on the entry page. Its half-maps are
#: deposited (many entries' are not), which is what makes the calculation possible at all.
LOCAL_RESOLUTION_PDB_ID = "9r04"
LOCAL_RESOLUTION_EMDB = "53478"


def _fetch_local_resolution(desktop: Any) -> None:
    """Start the download-and-compute for the local-resolution tutorial.

    Returns as soon as the background job is running. The loader is called on the GUI
    thread (see MainWindow._load_tutorial_data), so it must not block: this is ~160 MB of
    map to fetch and a minute or two of computation, and doing it inline would freeze the
    window for the duration with no way to tell that from a hang. The tutorial's first
    step waits on :func:`_resolution_ready` instead, so the walkthrough is honest about
    what it is waiting for.
    """
    desktop.fetch_and_compute_resolution(
        pdb_id=LOCAL_RESOLUTION_PDB_ID, emdb_number=LOCAL_RESOLUTION_EMDB,
        with_model=True, color=True, reuse_existing=True)


def _resolution_ready(cw: Any) -> bool:
    """Whether a coloured-by-resolution surface is actually drawn and usable.

    Gated on the viewport's own acknowledgement (``localres_drawn``), not on the map
    being pinned: pinning happens when Python has *streamed* the payload, seconds before
    the browser finishes building the surface, and a tutorial that advanced then was
    describing a map that was not on screen yet.
    """
    # _volumes is a list of entry dicts. An earlier version called .values() on it, and
    # the coach's defensive except around done-predicates swallowed the AttributeError --
    # so the step never auto-advanced and nobody saw an error. Predicates fail silent by
    # design; that makes them the one place a type mistake survives unnoticed.
    return any(entry.get("localres_drawn")
               for entry in getattr(cw._desktop, "_volumes", []) or [])


def _colouring_by_resolution(cw: Any) -> bool:
    return any(entry.get("color_by_resolution")
               for entry in getattr(cw._desktop, "_volumes", []) or [])


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


def open_model_tutorial() -> Tutorial:
    """The starting point: a model on screen and the three gestures that drive it."""
    return Tutorial("Open a model", [
        Step(
            "**1UBQ is loaded** — ubiquitin, a small well-behaved protein and the classic "
            "first structure.\n\nThe viewport is direct: **drag** to rotate, **scroll** to "
            "zoom, **click** an atom to select it (its details land in the status line).",
        ),
        Step(
            "How it is drawn lives in the **Scene** tab: select the model in the object "
            "list and its appearance pane opens — representation (cartoon, sticks, …), "
            "colouring, and which atom types are shown.\n\nColourings that map a number "
            "(B-factor, occupancy) get a **Range** control there, so the scale is yours.",
        ),
        Step(
            "That's the loop for any structure: load it (the **Open** button holds both "
            "**Open file(s)…** and **Fetch from PDB / EMDB…**), look at it, style it."
            "\n\nThe other tutorials each start from a scene like this one and add one "
            "skill.",
        ),
    ], loader=lambda d: _load_bundled(d, "1ubq.pdb"))


def map_model_tutorial() -> Tutorial:
    """A model paired with density — the everyday working scene."""
    return Tutorial("A model with its map", [
        Step(
            "**1UBQ is loaded with a map computed from it** — a stand-in for the "
            "experimental density you would normally have. Model and map arrive paired, "
            "so tools that need both (refinement, Q-score, tugging) know which map "
            "belongs to which model.",
        ),
        Step(
            "The map's surface is a contour: select the map in the object list and drag "
            "**Level** — or hover the viewport and **scroll** — to move it. Higher shows "
            "only the strongest density; the slider's right end always clears the map "
            "entirely.",
        ),
        Step(
            "The rest of the map's look lives in the same pane: opacity, surface or "
            "mesh, clipping, and colourings — a cryo-EM map with half-maps can be "
            "coloured by local resolution from its **Color** dropdown.",
        ),
    ], loader=lambda d: d.load_map_model_demo())


def _active_model_entry(cw: Any):
    mid = _active(cw)
    return cw._desktop._model_entry(mid) if mid else None


def _rep_is_ball_and_stick(cw: Any) -> bool:
    entry = _active_model_entry(cw)
    reps = (entry.get("reps") or [entry.get("rep")]) if entry else []
    return "ball-and-stick" in reps


def _tyr29_selected(cw: Any) -> bool:
    """Whether the selection includes TYR 29 -- however the user made it."""
    entry = _active_model_entry(cw)
    model = getattr(entry["session"], "model", None) if entry else None
    if model is None:
        return False
    indices = cw._desktop._scene_selection.get(entry["id"]) or []
    if not indices:
        return False
    atoms = model.get_hierarchy().atoms()
    for index in indices:
        if 0 <= index < len(atoms):
            if atoms[index].parent().parent().resseq_as_int() == 29:
                return True
    return False


def _conformer_picked(cw: Any) -> bool:
    entry = _active_model_entry(cw)
    return bool(entry and entry.get("conformer"))


def _conformer_back_to_all(cw: Any) -> bool:
    # Meaningful because the coach only evaluates the *current* step: this one is
    # reached by picking a conformer first, so None here is a deliberate return trip,
    # not the untouched default.
    entry = _active_model_entry(cw)
    return bool(entry) and entry.get("conformer") is None


def _coloured_by_occupancy(cw: Any) -> bool:
    entry = _active_model_entry(cw)
    return bool(entry and entry.get("color") == "occupancy")


def altlocs_tutorial() -> Tutorial:
    """Alternate conformations: one residue, several refined positions. Every doing step
    waits for the user to actually do it, and the route deliberately passes through two
    general tools -- the Representation dropdown and the Selection box -- before the
    conformer machinery, so the walkthrough teaches them on the way."""
    return Tutorial("Alternate conformations", [
        Step(
            "**3NIR is loaded** — crambin at 0.48 Å, sharp enough that many side chains "
            "were refined in **two or more positions** (alternate conformations, "
            "\"altlocs\"), each with its own occupancy. You will not see them yet: the "
            "cartoon abstracts side chains away.\n\nSo, first thing: in the **Objects** "
            "list, click the **3nir.pdb** row — its appearance controls open below — and "
            "set **Representation** to **Ball & stick**. Every atom is drawn, and the "
            "doubled side chains appear (as fuzz, at this scale — next we zoom in on "
            "one).",
            done=_rep_is_ball_and_stick,
        ),
        Step(
            "Type `resseq 29` into the **Selection** box below the object list and "
            "press **Enter** (or the arrow button). That is cctbx's selection language — "
            "the same strings Phenix uses — and it selects **tyrosine 29**, which was "
            "refined in **three** positions.\n\nThe camera moves to it and frames it "
            "the standard way — backbone **N on the left, C on the right, side chain "
            "up** — so every residue you select reads the same. (The **Focus on "
            "selection** box below turns the moving off.)",
            done=_tyr29_selected,
            target=lambda cw: cw._select_expr,
        ),
        Step(
            "Look at the highlighted tyrosine: **three complete side-chain positions**, "
            "labelled A, B and C in the model. Now isolate one: in the model's "
            "appearance pane, set the **Conformer** dropdown to **A** (or B, or C).\n\n"
            "The ring settles into a single position — one self-consistent model.",
            done=_conformer_picked,
        ),
        Step(
            "Set **Conformer** back to **All** and watch the three positions return."
            "\n\n**All** is the honest picture — the deposited model *is* the "
            "ensemble — and the single-letter views are for working on one conformation "
            "at a time.",
            done=_conformer_back_to_all,
        ),
        Step(
            "The occupancies behind the split are numbers on the atoms. In the model's "
            "**Color** dropdown, pick **By occupancy**.\n\nTyr 29's three rings each "
            "hold a fraction of an atom's worth of electrons, and now they stand apart "
            "from the full-occupancy backbone — blue is low, red is high, and the "
            "**Range** control that appears lets you set what the ramp spans.",
            done=_coloured_by_occupancy,
        ),
        Step(
            "That's the whole skill — and two tools you will reuse everywhere: "
            "**Representation** to choose what is drawn, the **Selection** box to name "
            "atoms precisely, **Conformer** to isolate one model, and **By occupancy** "
            "to see how the refinement split the density.",
        ),
    ], loader=lambda d: _load_bundled(d, "3nir.pdb"))


def validation_tutorial() -> Tutorial:
    """Run MolProbity validation and read the results — find what looks wrong in a model."""
    return Tutorial("Validate a structure", [
        Step(
            "MolProbity **validation** flags the parts of a model that look wrong — bad "
            "rotamers, Ramachandran and C-beta outliers, backbone (CaBLAM) problems, odd "
            "cis-peptides.\n\n**1TEC is loaded** — a structure that trips every one of "
            "those checks. Let's run validation on it.",
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
    ], loader=lambda d: _load_bundled(d, "1tec.pdb"))


def ligand_fitting_tutorial() -> Tutorial:
    """Fit a ligand into difference density — pxviewer's take on Phenix's ligand-fitting
    tutorial, self-contained (no phenix, no external data)."""
    return Tutorial("Fit a ligand into density", [
        Step(
            "Phenix's ligand-fitting tutorial fits a flexible ligand into a difference map. "
            "Let's do the same, straight from data.\n\n**Loaded:** a ligand-free model, "
            "plus reflections that secretly contain an ATP. The model cannot explain that "
            "density — which is exactly what a difference map is for.",
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
    ], loader=lambda d: d.load_ligand_fitting_demo())


def _minimizing(cw: Any) -> bool:
    return not cw._desktop._minimize_idle.is_set()


def cryo_em_refinement_tutorial() -> Tutorial:
    """Real-space refine a model into a cryo-EM density — pxviewer's take on Phenix's
    real_space_refine, self-contained (map computed from the model, no external data)."""
    return Tutorial("Real-space refine into cryo-EM density", [
        Step(
            "Cryo-EM refinement (phenix's `real_space_refine`) slides a model into a 3D "
            "density map — a gradient-driven minimization, not against reflections but "
            "against the map itself.\n\n**Loaded:** a model sitting slightly *off* its "
            "own density, waiting to be pushed back in.",
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
    ], loader=lambda d: d.load_real_space_refinement_demo())


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
            "atoms the data will not support.\n\n**Loaded:** a model alongside amplitudes "
            "computed from that same model, so the two start in exact agreement — which "
            "gives us a flat difference map to break on purpose.",
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
            "Break the fit: enable **Refine drag** on the Tools tab, then drag an atom in "
            "the viewport and pull it out of its "
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
    ], loader=lambda d: d.load_xray_demo())


def load_edits_tutorial() -> Tutorial:
    """Load a shared restraint-edits file onto a structure — the reading half of the loop."""
    return Tutorial("Load restraint edits", [
        Step(
            "Restraint **edits** — custom bonds/angles the monomer library can't know — can "
            "be shared as a phenix PHIL file.\n\n**Loaded:** a zinc site. cctbx works out "
            "the Zn–His bonds on its own, but not the water in the fourth coordination "
            "position. Let's supply that one from a file.",
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
    ], loader=lambda d: _load_bundled(d, "zn_site.pdb"))


def restraint_edits_tutorial() -> Tutorial:
    """Author a custom restraint edit end to end — the writing half of the loop."""
    return Tutorial("Custom restraint edits", [
        Step(
            "Now let's author a restraint by hand rather than read one from a file. A "
            "metal's coordination is a good case: cctbx guesses the Zn–His bonds, but not "
            "the water in the fourth site.\n\n**Loaded:** the same zinc site, with no "
            "edits on it.",
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
    ], loader=lambda d: _load_bundled(d, "zn_site.pdb"))


def local_resolution_tutorial() -> Tutorial:
    """Colour a cryo-EM map by local resolution — where the map is trustworthy, and where
    it is not. The one tutorial whose data is fetched rather than bundled: half-maps are
    too large to ship, and the calculation needs them."""
    return Tutorial("Look at local resolution", [
        Step(
            "A cryo-EM entry quotes **one** resolution — 4.2 Å for this one. That number "
            "is an average over the whole reconstruction, and almost no map is uniform: a "
            "rigid core can be far better than the quoted figure while a flexible "
            "periphery is far worse.\n\n**Local resolution** answers the question the "
            "single number cannot — *how much should I trust the density right here?* — "
            "and it is the difference between building a side chain with confidence and "
            "inventing one.",
        ),
        Step(
            "This needs the two **half-maps**: independent reconstructions from half the "
            "particles each. Where they agree out to fine detail the resolution is high; "
            "where they diverge early it is low. cctbx computes the local half-map FSC "
            "throughout the map and records where it falls through 0.143.\n\n"
            "**EMD-53478 and its model 9R04 are downloading now** — about 160 MB into your "
            "working directory (`~/pxviewer-data` unless you have changed it), then a minute "
            "or two to compute. Watch the status bar.\n\nBoth are kept: re-running this "
            "tutorial reuses the downloads *and* the computed resolution map, so the wait "
            "is first-time only.",
            done=_resolution_ready,
        ),
        Step(
            "The map is now **coloured by local resolution** rather than by a flat colour: "
            "the resolution map is pinned underneath it, hidden, and drives the colour.\n\n"
            "Look at the difference between the middle and the edges. The nucleosome core "
            "is the best-ordered part; the p53 that binds it, and the DNA ends, are softer. "
            "That variation is invisible in the single quoted number.",
            done=_colouring_by_resolution,
        ),
        Step(
            "The colouring is a switch on the map itself: open the **Loaded** panel and "
            "look at the map's own controls, where **Colour by resolution** can be turned "
            "off and on. With it off you are back to one colour and no idea which parts "
            "earned it.\n\nThat's the loop: fetch the half-maps, compute once, and let the "
            "map say where it can be believed. You can run it on your own maps from the "
            "map menu's **Colour by local resolution**, with local files or another entry.",
        ),
    ], loader=_fetch_local_resolution)


def all_tutorials() -> List[Tutorial]:
    """Every walkthrough offered, in menu order — looking before judging before changing:
    the three viewing ones (open a model, a model with its map, alternate conformations),
    then validation, then the fitting/refinement group, then the restraint-edits pair
    (reading before writing). There is no separate examples list: every bundled example
    is the opening scene of the tutorial that explains it."""
    return [open_model_tutorial(), map_model_tutorial(), altlocs_tutorial(),
            validation_tutorial(), ligand_fitting_tutorial(), cryo_em_refinement_tutorial(),
            local_resolution_tutorial(), xray_refinement_tutorial(), load_edits_tutorial(),
            restraint_edits_tutorial()]
