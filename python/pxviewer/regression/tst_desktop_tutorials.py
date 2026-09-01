"""The guided tutorials and the bundled examples they run on.

The coach is non-modal and advances itself when each step's task is *actually done* --
a model really loaded, atoms really selected, validation really cached -- rather than on
a button press. So each exercise here drives the app the way a user would and then asks
the coach whether it noticed, which means the tutorials cannot drift away from the app
they narrate.

These are the slowest exercises in the suite: phasing, ligand fitting and real-space
refinement all run for real.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import sys
import time

from pxviewer.regression.tst_utils import (
    closing_modals, data_path, dispose, have, process_events, qt_application,
    shipped_defaults, skip)

if not have("PySide6.QtWebEngineWidgets", "websockets", "iotbx.data_manager"):
    skip("PySide6 QtWebEngine / websockets / iotbx.data_manager not available")

qt_application()

from PySide6.QtWidgets import (                    # noqa: E402
    QApplication, QLabel, QPushButton, QTabWidget, QWidgetAction)

from pxviewer import tutorial                      # noqa: E402
from pxviewer.desktop import DesktopApp            # noqa: E402
from pxviewer.loader import sample_structure_path  # noqa: E402

LOAD_TIMEOUT_S = 30
PHASE_TIMEOUT_S = 120
REFINE_TIMEOUT_S = 90
FIT_TIMEOUT_S = 150

#: The tutorials, in the order the Get menu offers them: looking, judging, changing.
TUTORIAL_TITLES = [
    "Open a model",
    "A model with its map",
    "Alternate conformations",
    "Validate a structure",
    "Fit a ligand into density",
    "Real-space refine into cryo-EM density",
    "Look at local resolution",
    "X-ray: refine with a live difference map",
    "Load restraint edits",
    "Custom restraint edits",
]


@contextlib.contextmanager
def desktop():
    app = DesktopApp(port=0)
    app._webapp.start()
    with closing_modals():
        try:
            yield app
        finally:
            dispose(app)


def pump_until(predicate, what, timeout=LOAD_TIMEOUT_S):
    deadline = time.time() + timeout
    while time.time() < deadline and not predicate():
        process_events()
        time.sleep(0.02)
    process_events()
    assert predicate(), what


def pump_while(predicate, timeout):
    """Run the loop while ``predicate`` holds. Used where timing out is acceptable --
    letting a minimization settle, for instance, which has no definite end."""
    deadline = time.time() + timeout
    while time.time() < deadline and predicate():
        process_events()
        time.sleep(0.05)
    process_events()


def stop_minimizing(app):
    app.stop_minimization()
    pump_until(lambda: app._minimize_idle.is_set(), "minimization never went idle",
               timeout=10)


def monomer_library():
    from pxviewer.geometry import monomer_library_available

    return monomer_library_available()


def progress(app):
    return app._viewport.coach_progress.text()


# -- the coach itself ---------------------------------------------------------


def exercise_the_coach_advances_when_each_step_is_actually_done():
    """Hidden until started, then advancing on real app state rather than on Next.

    "Show me where" is the other half: it points at a control and never acts, so a user
    who presses it has not accidentally completed the step it is explaining.
    """
    if not have("rdkit", "mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: rdkit / pdb_interpretation not available)")
        return

    with desktop() as app:
        controls, coach = app._controls, app._viewport
        assert coach.coach_bar.isHidden()             # not shown until one starts

        controls._start_tutorial(tutorial.restraint_edits_tutorial())
        assert not coach.coach_bar.isHidden()
        assert progress(app) == "Step 1 / 4"

        # Starting it loaded its own example, so the tutorial opens on a step about a
        # structure it knows is there rather than asking for one.
        assert [m["name"] for m in app._models] == ["zn_site.pdb"]

        # That first step is orientation: nothing to do but read it, so it offers Next
        # rather than a control to be shown.
        assert coach.coach_next.text() == "Next"
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 1 / 4"          # no predicate: it waits
        controls._tutorial_next()
        assert progress(app) == "Step 2 / 4"

        # Step 2: selecting two atoms.
        mid = app._active_model_id
        app._scene_selection[mid] = [0, 1]
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 3 / 4"

        # Step 3: authoring an edit between two atoms in different residues.
        atoms = app._model_entry(mid)["session"].model.get_hierarchy().atoms()
        resseqs = [a.parent().parent().resseq for a in atoms]
        other = next(k for k in range(len(atoms)) if resseqs[k] != resseqs[0])
        app._scene_selection[mid] = [0, other]
        app.add_edit_from_selection(mid, "bond")
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 4 / 4"
        assert coach.coach_next.text() == "Finish"

        controls._tutorial_next()
        assert coach.coach_bar.isHidden()
        assert controls._tutorial is None


def exercise_starting_a_tutorial_loads_its_own_example():
    """Every tutorial brings its data with it, so no step has to say "go and open this".

    The step that did could not tell whether it had happened: its predicate asked "is a
    model loaded?", which is already true for anyone with their own work open, so the
    tutorial skipped its own setup and opened on a step describing a structure that was
    not on screen.
    """
    with desktop() as app:
        for build in tutorial.all_tutorials():
            assert build.loader is not None, build.title
        assert not app._models


def exercise_a_tutorial_asks_before_disturbing_loaded_work():
    """Only when there is something to lose. The dialog offers to clear or to add
    alongside, and cancelling leaves both the scene and the coach untouched.

    ``closing_modals`` rejects the dialog, which is the Cancel path -- so this also pins
    that a refused dialog aborts rather than loading anything.
    """
    with desktop() as app:
        controls = app._controls
        app.load_file(data_path("1ubq.pdb"))
        process_events()
        assert [m["name"] for m in app._models] == ["1ubq.pdb"]

        controls._start_tutorial(tutorial.validation_tutorial())
        process_events()

        assert [m["name"] for m in app._models] == ["1ubq.pdb"]   # untouched
        assert controls._tutorial is None                          # and not started


def exercise_every_tutorial_step_points_at_a_control_that_exists():
    """A dead target flashes nothing, which reads as a broken tutorial rather than a
    broken pointer. Checked across every tutorial, not just the one being walked."""
    with desktop() as app:
        controls = app._controls
        for build in tutorial.all_tutorials():
            for i, step in enumerate(build.steps):
                if step.target is not None:
                    assert step.target(controls) is not None, (build.title, i)


def exercise_the_altlocs_tutorial_follows_the_users_hands():
    """Five doing steps, each waiting for the actual state change -- and the route
    passes through Representation and the Selection box before the conformer
    machinery, so two general tools are taught on the way. The very first message
    already asks for the Ball & stick switch (a leading page-turn step felt
    sluggish), so there is no unconditional Next anywhere before the work."""
    with desktop() as app:
        controls = app._controls
        controls._start_tutorial(tutorial.altlocs_tutorial())
        process_events()
        assert [m["name"] for m in app._models] == ["3nir.pdb"]
        assert progress(app) == "Step 1 / 6"
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 1 / 6", "advanced without the user doing anything"

        mid = app._active_model_id
        app.set_model_representation(mid, "ball-and-stick")
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 2 / 6"
        # The selection step points at a control that exists from startup.
        assert tutorial.altlocs_tutorial().steps[1].target(controls) is not None
        # Focus-on-selection is the pane's default, which is what the step relies on.
        assert controls._focus_on_select.isChecked()

        # A selection elsewhere is not enough: the predicate wants the example residue.
        app.select_by_expression("resseq 5")
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 2 / 6", "any selection satisfied the residue step"

        count = app.select_by_expression("resseq 29")
        assert count > 0, "the example residue selected nothing"
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 3 / 6"

        app.set_model_conformer(mid, "A")
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 4 / 6"

        app.set_model_conformer(mid, None)
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 5 / 6"

        app.set_model_color(mid, "occupancy")
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 6 / 6"


def exercise_a_single_residue_selection_gets_the_oriented_framing():
    """Typing one residue frames it the standard way -- N left, C right, side chain up
    -- via the same orientation the space-bar navigation uses. Anything that is not
    exactly one amino acid falls back to the plain centre-and-frame focus."""
    import numpy as np

    with desktop() as app:
        app.load_file(data_path("3nir.pdb"))
        process_events()
        entry = app._model_entry(app._active_model_id)
        session = entry["session"]

        calls = []
        session.orient_camera = lambda *a: calls.append(("orient", a))
        session.focus = lambda atoms: calls.append(("focus", list(atoms)))
        session.set_clip = lambda front, back, radius=None, ref=None: calls.append(
            ("clip", front, back, radius))

        app.select_by_expression("resseq 29")
        orients = [c for c in calls if c[0] == "orient"]
        assert orients, [c[0] for c in calls]
        target, up, direction, radius, extents = orients[-1][1]
        assert all(e >= 2.0 for e in extents), extents
        # The vectors are a real frame: unit, orthogonal, aimed at the residue.
        assert abs(np.linalg.norm(up) - 1.0) < 1e-6
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6
        assert abs(np.dot(up, direction)) < 1e-6
        assert radius > 0
        atoms = session.model.get_hierarchy().atoms()
        # Centred on the selection's mass, not on CA: with the side chain as "up", CA
        # sits at the bottom of the picture, and a CA-centred view rides high by half a
        # side chain. The camera target must be the middle of what is shown.
        sel_xyz = np.array([a.xyz for a in atoms
                            if a.parent().parent().resseq_as_int() == 29])
        assert np.linalg.norm(np.array(target) - sel_xyz.mean(axis=0)) < 1e-6, (
            "not centred on the selection's centre of mass")

        # Everything else frames by its principal axes: a residue range reads with the
        # chain running left-to-right (first selected atom leftward of the last)...
        app.select_by_expression("resseq 5:14")
        orients = [c for c in calls if c[0] == "orient"]
        assert len(orients) >= 2, "a multi-residue selection was not framed"
        target, up, direction, radius, extents = orients[-1][1]
        assert abs(np.dot(up, direction)) < 1e-6
        # Ten residues of chain run mostly along the long (right) axis: the width the
        # camera must frame dwarfs the slab depth, which is the number pair a single
        # bounding radius could not express.
        assert extents[0] > extents[2], extents
        span_atoms = [np.array(a.xyz) for a in atoms
                      if 5 <= a.parent().parent().resseq_as_int() <= 14]
        right = np.cross(np.asarray(direction), np.asarray(up))
        right = right / np.linalg.norm(right)
        chain_span = span_atoms[-1] - span_atoms[0]
        assert np.dot(right, chain_span) > 0, "the chain does not run left-to-right"

        # Every focused selection gets the camera-following isolation sphere: sized to
        # the selection plus context, and lifted again when the selection is cleared.
        clips = [c for c in calls if c[0] == "clip"]
        assert clips and clips[-1][3] is not None and clips[-1][3] > 4.0, clips[-1:]
        assert (clips[-1][1], clips[-1][2]) == (0.0, 1.0), "the slab handles moved"

        # ...while a shape with no frame at all keeps the plain centre-and-frame focus.
        app.select_by_expression("resseq 5 and name CA")
        focus_calls = [c for c in calls if c[0] == "focus"]
        assert focus_calls, "a lone atom cannot be oriented"
        assert calls[-1] == ("clip", 0.0, 1.0, calls[-1][3]) and calls[-1][3] >= 4.0, (
            "even a plain focus should isolate the neighbourhood")

        app.select_by_expression("")
        assert calls[-1] == ("clip", 0.0, 1.0, None), (
            "clearing the selection must lift the isolation sphere")


def exercise_the_validation_tutorial_advances_when_validation_runs():
    """Load the demo, run validation, read the results."""
    with desktop() as app:
        controls = app._controls
        controls._start_tutorial(tutorial.validation_tutorial())
        process_events()
        assert progress(app) == "Step 1 / 3"
        assert controls._validate_btn.text() == "Run validation"    # the step-2 target

        # 1TEC arrives with the tutorial: the structure the steps talk about is the one
        # on screen, which is the whole reason the loader exists.
        assert [m["name"] for m in app._models] == ["1tec.pdb"]
        controls._tutorial_next()                     # orientation step
        assert progress(app) == "Step 2 / 3"

        # Step 2 advances once validation has cached results. Stood in for rather than
        # run: a real MolProbity pass is exercised by tst_validation.py, and repeating it
        # here would add a minute to say nothing new about the coach.
        mid = app._active_model_id
        assert not app._model_entry(mid).get("validation")
        app._model_entry(mid)["validation"] = {"rotalyze": object()}
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 3 / 3"
        assert app._viewport.coach_next.text() == "Finish"

        controls._tutorial_next()
        assert app._viewport.coach_bar.isHidden()


def exercise_the_cryo_em_tutorial_refines_a_shaken_model_into_its_density():
    """The demo loads a model sitting *off* a density computed from it, paired as one
    group. Real-space refinement settles it back in, and the map-model correlation climbs
    -- which is the claim the whole demo exists to make."""
    if not have("mmtbx.monomer_library.pdb_interpretation", "iotbx.map_model_manager"):
        print("    (skipped: pdb_interpretation / map_model_manager not available)")
        return

    with desktop() as app:
        controls = app._controls
        # Starting it loads the demo -- the shaken model and the density it belongs to.
        controls._start_tutorial(tutorial.cryo_em_refinement_tutorial())
        process_events()
        assert progress(app) == "Step 1 / 3"

        assert len(app._models) == 1
        assert len(app._volumes) == 1
        gid = app._models[0]["group"]
        mmm = app.group_mmm(gid)
        assert gid is not None and mmm is not None
        assert app.map_for_model() is not None
        controls._tutorial_next()                     # orientation step
        assert progress(app) == "Step 2 / 3"

        mmm.set_resolution(3.0)
        cc_before = float(mmm.map_model_cc())

        statuses = []
        app.bridge.status_changed.connect(statuses.append)
        app.minimize_model(use_map=True)
        pump_until(lambda: not app._minimize_idle.is_set(),
                   "minimization never started", timeout=REFINE_TIMEOUT_S)
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 3 / 3"

        # Let it settle into the density before reading the correlation back.
        pump_while(
            lambda: not any("holding" in s or "rmsd" in s for s in statuses),
            timeout=REFINE_TIMEOUT_S)
        stop_minimizing(app)

        assert float(mmm.map_model_cc()) > cc_before


def exercise_the_xray_tutorial_walks_the_difference_map_loop():
    """Load model and reflections, phase them, arm the live difference map, see one, then
    minimize into the density. Every step's predicate ticks off against real app state."""
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: pdb_interpretation not available)")
        return

    with desktop() as app:
        controls = app._controls
        controls._start_tutorial(tutorial.xray_refinement_tutorial())
        process_events()
        assert progress(app) == "Step 1 / 6"

        # The model and its amplitudes come with the tutorial.
        assert len(app._models) == 1
        assert len(app._reflections) == 1
        assert app.map_for_model() is None                  # not phased yet
        controls._tutorial_next()                           # orientation step
        assert progress(app) == "Step 2 / 6"

        mid, rid = app._models[0]["id"], app._reflections[0]["id"]
        app.make_maps(rid, mid)
        pump_until(lambda: app.map_for_model(mid) is not None,
                   "phasing never produced a map", timeout=PHASE_TIMEOUT_S)
        assert any("mFo-DFc" in v["name"] for v in app._volumes)
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 3 / 6"

        assert not app._live_diff
        app.set_live_difference_map(True)
        assert app._live_diff
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 4 / 6"

        # Step 4 waits for a live difference window to reach the viewport. Driving a real
        # refine drag needs the viewer, so the count is bumped directly -- the point is
        # that the step keys off _diff_boxes, which only _diff_worker raises and nothing
        # ever resets.
        assert app._diff_boxes == 0
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 4 / 6"                # no drag yet, so it waits
        app._diff_boxes += 1
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 5 / 6"

        app.minimize_model(use_map=True)
        pump_until(lambda: not app._minimize_idle.is_set(),
                   "minimization never started", timeout=REFINE_TIMEOUT_S)
        controls._maybe_advance_tutorial()
        assert progress(app) == "Step 6 / 6"
        stop_minimizing(app)


# -- the bundled examples -----------------------------------------------------


def exercise_the_metal_example_and_its_sample_edits_file():
    """The bundled metal site loads -- Zn, water, histidines -- and the bundled edits file
    applies to it, adding the Zn-water coordination bond cctbx does not restrain on its
    own."""
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: pdb_interpretation not available)")
        return

    site = sample_structure_path("zn_site.pdb")
    edits = sample_structure_path("zn_site_edits.phil")
    assert site is not None and edits is not None

    with desktop() as app:
        app.load_files([str(site)])
        pump_until(lambda: app._models, "the metal site never loaded")

        mid = app._active_model_id
        names = set(a.name.strip() for a in
                    app._model_entry(mid)["session"].model.get_hierarchy().atoms())
        assert {"ZN", "O", "NE2"} <= names   # metal, water, coordinating His nitrogen
        # Isolated histidines are not a polymer, so it draws ball-and-stick rather than
        # an empty cartoon.
        assert app._model_entry(mid)["rep"] == "ball-and-stick"

        assert app.model_edits(mid) == []
        assert app.load_edits(mid, str(edits)) == 1        # the Zn-water bond
        loaded = app.model_edits(mid)
        assert len(loaded) == 1
        assert loaded[0]["kind"] == "bond"


def exercise_the_ligand_fitting_demo_makes_maps_and_fits_atp():
    """A ligand-free model plus reflections that contain ATP. Making maps yields a paired
    2mFo-DFc and a difference map, and ATP fits back into the blob it came from -- the
    Phenix ligand-fitting tutorial, self-contained."""
    if not have("mmtbx.monomer_library.pdb_interpretation", "numpy"):
        print("    (skipped: pdb_interpretation / numpy not available)")
        return
    if not monomer_library():
        print("    (skipped: no monomer library, so no ATP restraints)")
        return
    import numpy as np

    with desktop() as app:
        app.load_ligand_fitting_demo()
        process_events()
        assert len(app._models) == 1
        assert len(app._reflections) == 1
        assert app.map_for_model() is None                 # not phased yet

        labels = [a.text() for a in app._controls._build_get_menu().actions()]
        assert any("ligand" in text.lower() for text in labels)  # its tutorial is offered

        mid, rid = app._models[0]["id"], app._reflections[0]["id"]
        app.make_maps(rid, mid)
        pump_until(lambda: app.map_for_model(mid) is not None,
                   "phasing never produced a map", timeout=PHASE_TIMEOUT_S)
        assert any("mFo-DFc" in v["name"] for v in app._volumes)

        # A marker at the blob, then build and fit ATP into it.
        center = app._LIGAND_FITTING_CENTER
        app._markers.append({"id": "marker-1", "name": "m", "position": list(center),
                             "atom": None, "visible": True})
        before = set(m["id"] for m in app._models)

        app.fit_ligand_at_marker("marker-1", "ATP", fit=True, trials=8)
        pump_until(lambda: set(m["id"] for m in app._models) != before,
                   "the ligand never appeared", timeout=FIT_TIMEOUT_S)

        ligand = next(m for m in app._models if m["id"] not in before)
        fitted = ligand["session"].model.get_sites_cart().as_numpy_array().mean(0)
        assert np.linalg.norm(fitted - np.array(center)) < 4.0     # into the blob


# -- where they are offered ---------------------------------------------------


def heading_text(action):
    """The visible text of a menu section heading, or None if it is not one."""
    if not isinstance(action, QWidgetAction):
        return None
    widget = action.defaultWidget()
    return widget.text() if isinstance(widget, QLabel) else None


def exercise_help_and_the_two_menus():
    """Help is a placeholder; Open holds the ways data gets on screen, Tutorials the
    walkthroughs -- fetching lives with opening, where a reader looks for it."""
    with desktop() as app:
        controls = app._controls
        controls._on_help()
        assert "documentation" in controls._status_label.text().lower()

        open_labels = [a.text() for a in controls._build_open_menu().actions()]
        assert any("Open file" in text for text in open_labels)
        assert any("PDB / EMDB" in text for text in open_labels)

        get_labels = [a.text() for a in controls._build_get_menu().actions()]
        assert not any("PDB / EMDB" in text for text in get_labels), (
            "fetch is still in the tutorials menu")
        assert any("restraint" in text.lower() for text in get_labels)


def exercise_every_menu_item_a_tutorial_names_actually_exists():
    """A tutorial that says "click **X**" must be naming something the menu has.

    Nothing checked this, and it drifted: four steps sent the reader to a **Demos**
    button, which had been renamed **Get** long enough ago that the old name appeared
    nowhere else. A tutorial is the one place in the app where a stale label is not a
    cosmetic problem -- the reader cannot proceed, and assumes they have misunderstood
    rather than that the instruction is wrong.

    Only bolded phrases containing a parenthesis are checked, which is what a menu entry
    looks like here; bolding is also used for emphasis, and this is not a spell-checker.
    """
    import re

    from pxviewer import tutorial

    with desktop() as app:
        labels = {a.text().strip() for a in app._controls._build_get_menu().actions()}
        labels |= {a.text().strip() for a in app._controls._build_open_menu().actions()}
        for tut in tutorial.all_tutorials():
            for index, step in enumerate(tut.steps):
                for phrase in re.findall(r"\*\*([^*]+)\*\*", step.text or ""):
                    if "(" not in phrase:
                        continue
                    assert phrase in labels, (
                        "%s step %d names a menu item that does not exist: %r"
                        % (tut.title, index, phrase))


def exercise_the_get_menu_lists_the_online_examples_and_tutorials():
    """The Tutorials menu is exactly the tutorials, in their settled order -- no
    headings left to explain, since fetching moved under Open and there is no separate
    examples section (every bundled example is a tutorial's opening scene)."""
    with desktop() as app:
        controls = app._controls
        actions = controls._build_get_menu().actions()
        tutorials = [a.text() for a in actions
                     if not a.isSeparator() and a.text().strip()]
        assert [t.lower() for t in tutorials] == [t.lower() for t in TUTORIAL_TITLES]

        # One tutorials button (capped, with the menu) and one Open button whose menu
        # holds both ways of getting data on screen.
        buttons = controls.widget().findChildren(QPushButton)
        tut_buttons = [b for b in buttons if b.toolTip().startswith("Guided tutorials")]
        assert len(tut_buttons) == 1
        assert tut_buttons[0].menu() is not None
        open_buttons = [b for b in buttons if b.toolTip().startswith("Open a structure")]
        assert len(open_buttons) == 1 and open_buttons[0].menu() is not None
        assert not any(b.toolTip().startswith("Fetch an entry") for b in buttons)

        tabs = controls.widget().findChild(QTabWidget)
        # Tabs are icon-only, so the label lives in the tooltip.
        assert [tabs.tabToolTip(i) for i in range(5)] == [
            "Scene", "Tools", "Validation", "Hotspots", "Geometry"]
        assert all(not tabs.tabIcon(i).isNull() for i in range(tabs.count()))


def exercise_the_tutorials_are_offered_in_a_settled_order():
    """Loading-first, writing-second. The Get menu and ``all_tutorials`` must agree, or
    the menu offers one thing and the coach starts another."""
    assert [t.title for t in tutorial.all_tutorials()] == TUTORIAL_TITLES


def run():
    # Every exercise here builds a DesktopApp, which reads its defaults from QSettings --
    # so the whole file runs against a fresh install's preferences, not the user's.
    with shipped_defaults():
        for name, fn in sorted(globals().items()):
            if name.startswith("exercise"):
                print("  %s" % name)
                sys.stdout.flush()
                fn()
    print("OK")


if __name__ == "__main__":
    run()
