"""The figure-spec corpus for the figfuzz harness.

A spec is a *goal*, not a script — the maker agent must discover the API and
produce the figure through MCP, which is the point: task language a real user
would write is what we want exercising the software. If a spec names a verb,
it is because the semantic ("only the density near the ligand") is the thing
under test, not the method.

Each spec is a dict:

    id       stable run-dir name ("f03-boxed-site")
    title    one line for the report
    task     the paragraph the maker agent is given
    stage    files the runner materializes into the run dir first:
             {"path": "map.mrc", "kind": "blob"|"big"|"diff", "shape": (n,n,n)}
    expect   declarative checks for the programmatic critic (see critic.py):
             models / volumes      -- minimum counts
             models_exact          -- exact counts (removal specs)
             volumes_exact
             volume_props          -- attrs every volume must show
             grid_deliberate       -- drawn grid is full-res or boxed
             regioned              -- some clip/box/mask state exists
             negative_visible      -- a difference map's minus contour is on
             volume_hidden         -- at least N maps not drawn
             primitives            -- at least N drawing primitives exist
             model_clipped         -- the model carries a clip
             selection_nonempty    -- a selection is active
             reflections           -- at least N reflection sets loaded
             conformer_set         -- a model has a conformer chosen
             background            -- the viewport's background hex
             markers               -- at least N markers placed
             kernel                -- a Jupyter kernel reports attached
             edits                 -- at least N custom restraint edits
             nonblank              -- the screenshot is not empty (default on)
             no_job_errors         -- status() reports none (default on)
    fetch    needs the network (skipped under --no-fetch)
"""

from __future__ import absolute_import, division, print_function

import random

#: Local models, relative to python/pxviewer/data. heme_site carries a ligand;
#: zn_site a metal site; the others are ordinary proteins.
MODELS = ["1ubq.pdb", "1tec.pdb", "3nir.pdb", "heme_site.pdb", "zn_site.pdb"]


def _spec(id_, title, task, *, stage=(), expect=None, fetch=False, anchor=None):
    expect = {"nonblank": True, "no_job_errors": True, **(expect or {})}
    return {"id": id_, "title": title, "task": task,
            "stage": list(stage), "expect": expect, "fetch": fetch,
            "anchor": anchor}


def corpus(data_dir, seed=0):
    """The spec list with its draws resolved under ``seed`` — which model each
    ``{model}`` placeholder lands on, so a full-corpus run is reproducible."""
    rng = random.Random(seed)
    p = lambda name: f"{data_dir}/{name}"
    m = lambda: p(rng.choice(MODELS))
    # Model+map specs name the model once — the maps are staged with their
    # box centered on that model's centroid, or the "density near the site"
    # task would be unreachable (the fixture map has its own frame).
    f03m = m()
    f06m = m()
    f14m = m()
    h04m = m()
    h05m = m()
    i09m = m()
    j01m = m()
    specs = [
        _spec("f01-ligand-site", "Heme site with its environment",
              f"Load {p('heme_site.pdb')} and make a figure of the heme site: "
              "the protein as a cartoon, the ligand and the residues within "
              "~4 Å of it drawn in more detail, and the camera framed on the "
              "site so it fills the view.",
              expect={"models": 1}),

        _spec("f02-bfactor", "Cartoon colored by B-factor",
              f"Load {p('3nir.pdb')} and produce a figure of the structure "
              "colored by atomic B-factor — something a reader could take a "
              "B-factor ramp from.",
              expect={"models": 1}),

        _spec("f03-boxed-site", "Density carved around a site",
              f"Load {f03m} and the map file sitediff.mrc from this directory. "
              "Make a figure showing the density only within ~15 Å of the "
              "ligand — as a mesh at a sensible contour. The point is the "
              "surrounding density must not appear.",
              stage=[{"path": "sitediff.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              anchor=f03m,
              expect={"models": 1, "volumes": 1, "regioned": True}),

        _spec("f04-bigmap", "Publishable mesh from an oversized map",
              "Load bigmap.mrc from this directory and produce a clean, "
              "publication-quality figure of the density as a mesh at a "
              "sensible contour. Pay attention to what the rendered grid "
              "actually is before you save.",
              stage=[{"path": "bigmap.mrc", "kind": "big", "shape": (256, 256, 256)}],
              expect={"volumes": 1, "grid_deliberate": True}),

        _spec("f05-diffmap", "Difference map, both signs",
              "Load diffmap.mrc from this directory and make a figure of the "
              "difference map — both positive and negative contours must be "
              "visible in the frame.",
              stage=[{"path": "diffmap.mrc", "kind": "diff", "shape": (96, 96, 96)}],
              expect={"volumes": 1, "negative_visible": True}),

        _spec("f06-masked", "Map masked to the model footprint",
              f"Load {f06m} and blob.mrc from this directory. Make a figure of "
              "the map cut down to the model's footprint — density well away "
              "from any atom should not show.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              anchor=f06m,
              expect={"models": 1, "volumes": 1, "regioned": True}),

        _spec("f07-slab", "A slab through the structure",
              f"Load {m()} and make a figure of a thin slab through the "
              "middle of the molecule — the structure outside the slab must "
              "not be drawn.",
              expect={"models": 1, "model_clipped": True}),

        _spec("f08-clip-survives-rep", "Clip held across a rep change",
              f"Load {m()}, restrict the drawn structure to a ~20 Å sphere "
              "around its center, then change the representation to "
              "ball-and-stick and make the figure — the clip must still hold "
              "in the final image.",
              expect={"models": 1, "model_clipped": True}),

        _spec("f09-two-models", "Two models, one hidden",
              f"Load {p('1ubq.pdb')} and {p('1tec.pdb')}. Make a figure that "
              "shows exactly one of them — the other must not appear.",
              expect={"models": 2}),

        _spec("f10-measure", "A labeled distance",
              f"Load {m()} and produce a figure that measures and labels the "
              "distance between two clearly identifiable atoms of your "
              "choice, with the structure in context.",
              expect={"models": 1, "primitives": 1}),

        _spec("f11-fetch-pdb-map", "Fetched model inside its map",
              "Fetch PDB entry 7BV2 together with its map and make a figure "
              "of the model sitting in its electron density at a sensible "
              "contour.",
              expect={"models": 1, "volumes": 1}, fetch=True),

        _spec("f12-fetch-emdb", "Fetched cryo-EM map as a clean mesh",
              "Fetch the EMD-30210 map and produce a clean figure of it as a "
              "mesh at a sensible contour for its resolution.",
              expect={"volumes": 1}, fetch=True),

        _spec("f13-quality", "A final-quality render",
              f"Load {m()} and produce the best-looking still the viewer "
              "can make: supersampled, richer lighting if available. The "
              "frame must show the whole structure.",
              expect={"models": 1}),

        _spec("f14-transparent", "Transparent surface over the model",
              f"Load {f14m} and blob.mrc from this directory. Make a figure "
              "with the map as a transparent surface around the cartoon "
              "model — the model must read through the surface.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              anchor=f14m,
              expect={"models": 1, "volumes": 1}),

        # ---- round 2: machinery the first corpus never touched -------------

        _spec("g01-express-color", "Selection-grammar coloring",
              f"Load {m()} and make a figure where the lysines stand out "
              "from the rest of the protein — use the app's typed/expression "
              "selection machinery to pick them, not a hand-built atom list.",
              expect={"models": 1, "selection_nonempty": True}),

        _spec("g02-surface", "Molecular surface",
              f"Load {m()} and make a figure of the molecule's surface — "
              "not its cartoon — with any bound ligand or metal drawn "
              "through it in enough detail to identify.",
              expect={"models": 1}),

        _spec("g03-demo-pair", "Demo pair: model in its density",
              "The app has a built-in demo that loads a model together with "
              "a density map computed for it — find it (look for a "
              "load_*_demo method on the app object). Then make a figure of "
              "the model sitting inside its own density.",
              expect={"models": 1, "volumes": 1}),

        _spec("g04-xray-maps", "Maps from the x-ray demo",
              "The app has a demo that loads a model together with real or "
              "synthesized diffraction data unpaired — find it (a "
              "load_xray_demo-style method). Then make the maps from the "
              "reflections and produce a figure of the model inside the "
              "2mFo-DFc-style density.",
              expect={"models": 1, "volumes": 1, "reflections": 1}),

        _spec("g05-remove", "Scene mutation: remove the map",
              f"Load {m()} and blob.mrc from this directory, then remove the "
              "map from the scene entirely. The figure is the model alone — "
              "verify no density is left drawn.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              expect={"models": 1, "volumes_exact": 0}),

        _spec("g06-model-sphere", "Spherical model clip",
              f"Load {m()}, pick a residue near its middle, and make a "
              "figure where only the structure within ~15 Å of that residue "
              "is drawn — a spherical neighborhood, not a slab.",
              expect={"models": 1, "model_clipped": True}),

        _spec("g07-rsr-stream", "Live real-space refinement",
              "The app has a demo that loads a model shaken away from its "
              "own density — find it (a load_real_space_refinement_demo-style "
              "method). Run its real-space refinement ('minimize' into the "
              "density) and make the figure of the model sitting correctly "
              "inside the density again.",
              expect={"models": 1, "volumes": 1}),

        _spec("g08-two-volumes", "Two maps, one hidden",
              "Load blob.mrc and diffmap.mrc from this directory. Make a "
              "figure that shows exactly one of them — your choice — with "
              "the other not drawn.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)},
                     {"path": "diffmap.mrc", "kind": "diff", "shape": (96, 96, 96)}],
              expect={"volumes": 2, "volume_hidden": 1}),

        _spec("g09-validation", "Validation outliers on screen",
              f"Load {m()}, run the app's geometry validation, and make a "
              "figure that lets a reader find the outliers it reports.",
              expect={"models": 1}),

        _spec("g10-iso-choice", "A contour level, chosen twice",
              "Load blob.mrc from this directory. First look at it at a low "
              "contour, then make the final figure at 3.0 sigma — the saved "
              "frame must be at 3.0, not the level you started from.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              expect={"volumes": 1, "volume_props": {"iso": 3.0}}),

        _spec("g11-focus-residue", "Residue-level framing",
              f"Load {m()} and use the app's residue-navigation machinery to "
              "walk to a residue near the middle of the structure — the "
              "figure frames on that residue, centered and filling a good "
              "part of the view.",
              expect={"models": 1}),

        _spec("g12-conformers", "One conformer of several",
              "Load multiconf.pdb from this directory — a multi-conformer "
              "model — and make a figure that shows exactly one conformer, "
              "with the others not drawn.",
              stage=[{"path": "multiconf.pdb", "kind": "multiconf",
                      "from": p("1ubq.pdb"), "copies": 3}],
              expect={"models": 1, "conformer_set": True}),

        _spec("g13-white-background", "Publication figure on white",
              f"Load {m()} and make a figure on a white background with the "
              "structure in a dark color — the kind of figure that goes on a "
              "white slide.",
              expect={"models": 1, "background": "#ffffff"}),

        _spec("g14-volume-clip", "A slab of density only",
              "Load blob.mrc from this directory and make a figure of a thin "
              "slab of the density — the map outside the slab must not be "
              "drawn. Clip the volume, not the camera.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              expect={"volumes": 1, "regioned": True}),

        # -- round 3: the machinery the first two rounds never touched -------

        _spec("h01-measure-edits", "Measurements and a custom restraint",
              f"Load {m()}. Draw a distance between two atoms and an angle "
              "across three — labels on screen, visible in the frame. Then "
              "author a bond edit between two of the atoms (taking the current "
              "geometry as the target), confirm the model lists the edit, and "
              "save the edits to edits.phil in this directory.",
              expect={"models": 1, "primitives": 2, "edits": 1,
                      "selection_nonempty": True}),

        _spec("h02-volume-sphere", "Map that follows the eye",
              "Load blob.mrc from this directory. Put the map on a spherical "
              "bound that follows where the camera is pointed, with a radius "
              "around 25 Å — then move the view and show the density has "
              "moved with it.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              expect={"volumes": 1, "regioned": True}),

        _spec("h03-interactions", "Contact overlays on a fold",
              f"Load {m()} and turn on its non-covalent interaction overlays — "
              "the hydrogen-bond / contact markup, not a cartoon color change. "
              "The figure should show the fold with its contact lines drawn.",
              expect={"models": 1}),

        _spec("h04-qscore", "Atoms colored by their fit to density",
              f"Load {h04m} and the map qmap.mrc from this directory — it is "
              "sited on the model. Color the model's atoms by how well each "
              "fits the density (Q-score or map correlation), so a reader sees "
              "which regions disagree with the data.",
              stage=[{"path": "qmap.mrc", "kind": "blob", "shape": (64, 64, 64)}],
              expect={"models": 1, "volumes": 1},
              anchor=h04m),

        _spec("h05-replay-stress", "Everything survives a viewport reload",
              f"Load {h05m} and the map blob.mrc from this directory. Select a "
              "handful of residues, contour the map to a level you choose, and "
              "frame the site. THEN reload the viewport and confirm every "
              "piece came back — models, map, contour, selection, and camera. "
              "Report what status() shows after the reload.",
              stage=[{"path": "blob.mrc", "kind": "blob", "shape": (96, 96, 96)}],
              expect={"models": 1, "volumes": 1, "selection_nonempty": True},
              anchor=h05m),

        _spec("h06-kernel", "An attached Jupyter kernel",
              "Attach pxviewer's Jupyter kernel, push a small Python object "
              "into it, and confirm the kernel reports alive in status. Then "
              "shut it down cleanly. Report what each step returned.",
              # No scene, no figure — and the spec ENDS with the kernel
              # detached, so a post-hoc kernel/nonblank check can't hold.
              expect={"nonblank": False}),

        _spec("h07-hide-show", "Hide it, then bring it back",
              f"Load {m()}. Select a loop or a few residues and hide just "
              "those atoms — then show them again and finish with the whole "
              "molecule visible in the frame.",
              # Ends with the selection plausibly cleared for a clean figure —
              # a post-hoc selection check can't hold.
              expect={"models": 1}),

        _spec("h08-error-honesty", "Bad input must fail loudly",
              f"Probe the API's error behavior, then make a figure. In order: "
              "(1) try to load /nonexistent/model.pdb; (2) try a nonsense "
              "selection expression like 'resname banana and and'; (3) try to "
              "set a contour level on a volume id that does not exist; (4) try "
              "a camera state that is not valid numbers. For each, record "
              "whether you got an honest error or a silent no-op — report them "
              f"all. Then load {m()} and frame it so the final screenshot has "
              "content.",
              expect={"models": 1}),

        _spec("h09-picked-residue", "Build a selection atom by atom",
              f"Load {m()}. Using the picked-atom API (not selection "
              "expressions): pick two or three atoms across different "
              "residues, toggle one residue in and out of the selection, and "
              "finish with a non-empty selection focused in the frame.",
              expect={"models": 1, "selection_nonempty": True}),

        _spec("h10-markers", "Ligand markers placed and managed",
              f"Load {m()}. Place two ligand markers on the structure — arm "
              "the marker tool and, if the viewport needs a click you cannot "
              "give it, drive the marker callback directly. Show both in "
              "status, hide one, remove it, and leave one visible marker in "
              "the final frame.",
              expect={"models": 1, "markers": 1}),

        _spec("h11-residue-walk", "Walk down a helix",
              f"Load {m()}. Frame one residue, then step along the chain "
              "residue by residue a few times — the camera should follow each "
              "step. Report the residue and camera target at each stop, and "
              "finish framed on wherever you ended.",
              expect={"models": 1}),

        _spec("h12-pair-map", "Pair a model to an off-origin map",
              f"Load {m()} and the map offmap.mrc from this directory — the "
              "map is deliberately NOT co-located with the model. Pair the "
              "model to the map and align them so the model sits inside the "
              "density. The figure should show them together.",
              stage=[{"path": "offmap.mrc", "kind": "blob", "shape": (48, 48, 48)}],
              expect={"models": 1, "volumes": 1}),

        # -- round 4: the machinery rounds 1-3 left untouched -------------

        _spec("i01-edits-roundtrip", "Edits saved and restored",
              f"Load {m()}. Add a custom bond or angle edit between two atoms "
              "that are close but not bonded (a coordination bond is the "
              "classic case — check distances with the hierarchy or "
              "measure_selection first). List the edits, save them to a file "
              "in this directory, clear them, then load the file back and "
              "confirm the edit is restored and drawn. Keep it visible in "
              "the final frame.",
              expect={"models": 1, "edits": 1}),

        _spec("i02-marker-move", "Move a placed marker",
              f"Load {m()}. Place a marker on the structure (place_marker or "
              "the arm-the-tool path), move it a few Angstrom along one axis "
              "through its move callback (a Shift-drag in the viewport lands "
              "on the same handler), confirm the new position in status, "
              "then clear all markers. Report what each step returned.",
              expect={"models": 1, "nonblank": False}),

        _spec("i03-model-slab", "Clip a slab through the model",
              f"Load {m()}. Clip the model down to a thin slab through its "
              "middle — a few Angstrom deep — so the interior packing shows "
              "in cross-section, then release the clip so the whole "
              "molecule returns. Report what the slab controls were called.",
              expect={"models": 1}),

        _spec("i04-ordered-measure", "Angle vertex order is semantic",
              f"Load {m()}. Build an angle measurement by picking three "
              "atoms in a specific order (use the picked-atom API or an "
              "explicit atoms list), then pick the SAME three atoms in a "
              "different order and compare the reported angles — the vertex "
              "must follow pick order, not sorted index order. Then measure "
              "a dihedral across four atoms. Report what each call "
              "returned.",
              expect={"models": 1}),

        _spec("i05-export-roundtrip", "Write the model out and reload it",
              f"Load {m()}. Write the loaded model's coordinates back out "
              "to a PDB file in this directory through the app's export "
              "verb (not by copying the input). Remove the model, load the "
              "file you wrote, and confirm the atom count and frame look "
              "identical. Finish on the reloaded model.",
              expect={"models": 1}),

        _spec("i06-push-stream", "Move atoms live through the session",
              f"Load {m()}. Move a residue or two a few Angstrom along an "
              "axis by editing the model's coordinates through the cctbx "
              "model and pushing the new conformation to the viewport "
              "(the session's push/stream path — the same one minimization "
              "uses). Confirm the atoms visibly moved and that the model "
              "data and the drawn atoms agree afterward.",
              expect={"models": 1}),

        _spec("i07-ligand-restraints", "Restraints for an unknown ligand",
              f"Load {p('heme_site.pdb')}. Ask the app which residues it "
              "cannot build geometry restraints for, generate restraints "
              "for what it reports, and export the ligand's cif to this "
              "directory. Report which residues needed them and what each "
              "verb returned.",
              expect={"models": 1}),

        _spec("i08-bad-edits", "Measure and edit calls refuse bad input",
              f"Probe the measurement and edit APIs' error behavior, then "
              f"make a figure. In order: (1) measure_selection or "
              "add_edit_from_selection with too few atoms (e.g. 1); "
              "(2) measure with FIVE atoms — more than any measurement "
              "takes; (3) an edit naming atom indices that do not exist in "
              "the model; (4) remove_edit on an edit id that does not "
              "exist. Record honest error vs silent no-op for each. Then "
              f"load {m()} and finish on a framed figure.",
              expect={"models": 1}),

        _spec("i09-group-load", "Model and map in one group",
              f"Load {i09m} and the map blob.mrc from this directory "
              "TOGETHER as a group (one load_files call or the group's "
              "load path), so they arrive in a shared frame. Confirm they "
              "share a group in status and that the model sits in the "
              "density without any manual pairing.",
              stage=[{"path": "blob.mrc", "kind": "blob",
                      "shape": (96, 96, 96)}],
              anchor=i09m,
              expect={"models": 1, "volumes": 1}),

        # Round 5 — machinery still untouched after 46 specs
        _spec("j01-model-sphere", "Model clip sphere, every mode",
              f"Load {j01m}. Exercise app.set_model_sphere through its modes "
              "in order: 'view' at radius ~15, then 'point', then 'selection' "
              "after making a real selection, then 'off'. After each, call "
              "app.model_sphere_state and confirm the read-back matches what "
              "you asked for — and that 'off' restores the whole model. "
              "Finish on a figure with a ~15 Å view sphere active.",
              expect={"models": 1}),

        _spec("j02-two-models", "Two models, active-model switching",
              f"Load {p('1ubq.pdb')} AND {p('3nir.pdb')} (two separate "
              "models, not a group). Switch the active model with "
              "app.set_active_model and confirm status agrees each time; "
              "check which model app.measure_selection atoms refer to under "
              "each active id; then app.remove_model one of them and confirm "
              "the survivor is fully intact. Finish on a figure of the "
              "surviving model.",
              expect={"models": 1}),

        _spec("j03-minimize", "Live minimization lifecycle",
              f"Load {p('zn_site.pdb')} and run app.minimize_model(). Watch "
              "the run via status/busy, let it settle, then "
              "app.stop_minimization(). Verify: job_errors stays empty, the "
              "model's coordinates actually differ from the file's (compare "
              "a few atoms), and the app stays usable after stop. Finish on "
              "a framed figure of the minimized model.",
              expect={"models": 1}),

        _spec("j04-orient-views", "Cardinal views and figure framing",
              f"Load {m()}. Use app.orient_camera to take three orthographic "
              "views (down X, down Y, down Z — pick target/up/direction that "
              "achieve it) and app.set_view_radius/app.zoom_view to frame "
              "each. Check app.camera_state() after each orient reports the "
              "orientation you asked for. End on the best view on a white "
              "background (session.set_background).",
              expect={"models": 1}),

        _spec("j05-type-visibility", "Hide/show whole structure types",
              f"Load {p('1ubq.pdb')}. List types with "
              "app.model_structure_types, hide the waters "
              "with app.set_model_type_hidden, verify status reports them "
              "hidden AND the viewport drops them, then show them again. "
              "Probe a type the model does not have and a misspelled label — "
              "report honest error vs silent no-op for each. Finish on a "
              "figure with only the water hidden.",
              expect={"models": 1}),

        _spec("j06-conformer-switch", "Live conformer switching",
              "Load multiconf.pdb from this directory — a model with "
              "altlocs A and B. Call app.model_conformers, switch to 'A', "
              "verify only A draws (status/scene), switch to 'B', then "
              "back to all conformers. Also probe a conformer that does not "
              "exist and report whether it fails honestly.",
              stage=[{"path": "multiconf.pdb", "kind": "multiconf",
                      "from": p("1ubq.pdb"), "copies": 3}],
              expect={"models": 1}),

        _spec("j07-validation-markers", "Validation overlays, per-kind",
              f"Load {m()} and run app.run_validation with the per-residue "
              "validators plus hotspots=True. When it finishes, toggle each "
              "validator's MolProbity markup off and on with "
              "app.set_validation_markers and confirm it redraws without a "
              "re-run (no new busy job). Report what each validator drew.",
              expect={"models": 1}),

        _spec("j08-write-objects", "write_object round trips",
              f"Load {m()} and the map blob.mrc from this directory as a "
              "group. Use app.write_object to save the model as mmCIF and "
              "the volume as a map file, then load both written files back "
              "and verify the reloaded model's coordinates match the "
              "original's atom-for-atom and the map's grid matches.",
              stage=[{"path": "blob.mrc", "kind": "blob",
                      "shape": (64, 64, 64)}],
              anchor=i09m,
              expect={"models": 2, "volumes": 2}),
    ]
    return specs


def generate(data_dir, count, seed):
    """``count`` specs sampled from the corpus — repeat a spec's draws under
    different seeds and the same task exercises different inputs."""
    rng = random.Random(seed)
    picked = []
    for i in range(count):
        spec = dict(rng.choice(corpus(data_dir, seed=rng.randrange(2 ** 31))))
        spec["id"] = f"{i + 1:02d}-{spec['id']}"
        picked.append(spec)
    return picked
