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

#: The tutorials, in the order the Get menu offers them: loading-first, writing-second.
TUTORIAL_TITLES = [
    "Validate a structure",
    "Fit a ligand into density",
    "Real-space refine into cryo-EM density",
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

        # Step 1 targets a control, so "Show me where" is offered -- and only points.
        assert not coach.coach_show.isHidden()
        controls._on_coach_show_me()                  # flashes Get; loads nothing
        assert not app._models

        # The user loads a structure themselves, and the predicate then advances.
        app.load_files([str(sample_structure_path())])
        pump_until(lambda: app._models, "the structure never loaded")
        controls._maybe_advance_tutorial()
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


def exercise_every_tutorial_step_points_at_a_control_that_exists():
    """A dead target flashes nothing, which reads as a broken tutorial rather than a
    broken pointer. Checked across every tutorial, not just the one being walked."""
    with desktop() as app:
        controls = app._controls
        for build in tutorial.all_tutorials():
            for i, step in enumerate(build.steps):
                if step.target is not None:
                    assert step.target(controls) is not None, (build.title, i)


def exercise_the_validation_tutorial_advances_when_validation_runs():
    """Load the demo, run validation, read the results."""
    with desktop() as app:
        controls = app._controls
        controls._start_tutorial(tutorial.validation_tutorial())
        assert progress(app) == "Step 1 / 3"
        assert controls._validate_btn.text() == "Run validation"    # the step-2 target

        app.load_files([str(sample_structure_path("1tec.pdb"))])
        pump_until(lambda: app._models, "the demo never loaded")
        controls._maybe_advance_tutorial()
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
        controls._start_tutorial(tutorial.cryo_em_refinement_tutorial())
        assert progress(app) == "Step 1 / 3"

        app.load_real_space_refinement_demo(shake=0.6)
        process_events()
        assert len(app._models) == 1
        assert len(app._volumes) == 1
        gid = app._models[0]["group"]
        mmm = app.group_mmm(gid)
        assert gid is not None and mmm is not None
        assert app.map_for_model() is not None
        controls._maybe_advance_tutorial()
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
        assert progress(app) == "Step 1 / 6"

        app.load_xray_demo()
        process_events()
        assert len(app._models) == 1
        assert len(app._reflections) == 1
        assert app.map_for_model() is None                  # not phased yet
        controls._maybe_advance_tutorial()
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
        assert any("Ligand fitting" in text for text in labels)

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


def exercise_help_and_the_get_menu():
    """Help is a placeholder; remote data, examples and tutorials share the Get menu."""
    with desktop() as app:
        controls = app._controls
        controls._on_help()
        assert "documentation" in controls._status_label.text().lower()

        actions = controls._build_get_menu().actions()
        labels = [a.text() for a in actions]
        assert any("PDB / EMDB" in text for text in labels)          # remote data
        assert any("1UBQ" in text for text in labels)                # a sample
        assert any("restraint" in text.lower() for text in labels)   # a tutorial

        headings = [heading_text(a) for a in actions if heading_text(a)]
        assert headings == ["Online", "Examples", "Tutorials"]


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
        for tut in tutorial.all_tutorials():
            for index, step in enumerate(tut.steps):
                for phrase in re.findall(r"\*\*([^*]+)\*\*", step.text or ""):
                    if "(" not in phrase:
                        continue
                    assert phrase in labels, (
                        "%s step %d names a menu item that does not exist: %r"
                        % (tut.title, index, phrase))


def exercise_the_get_menu_lists_the_online_examples_and_tutorials():
    """Get combines online retrieval, bundled examples and guided tutorials.

    The headings are asserted through their rendered widget rather than through
    ``QAction.text()``. They used to be ``QMenu.addSection``, whose text macOS silently
    drops -- so the menu showed two unlabelled dividers while a test reading ``.text()``
    passed happily.
    """
    with desktop() as app:
        controls = app._controls
        actions = controls._build_get_menu().actions()

        headings = [(i, heading_text(a)) for i, a in enumerate(actions)
                    if heading_text(a)]
        assert [text for _i, text in headings] == ["Online", "Examples", "Tutorials"]
        online_i, examples_i, tutorials_i = (item[0] for item in headings)
        # Drawn, not merely set: an unparented label with no text renders as nothing.
        assert all(heading_text(actions[i]).strip() for i, _t in headings)

        def entries(start, stop=None):
            block = actions[start:stop] if stop is not None else actions[start:]
            return [a.text() for a in block
                    if not a.isSeparator() and a.text().strip()]

        assert entries(online_i + 1, examples_i) == ["Fetch from PDB / EMDB…"]

        examples = entries(examples_i + 1, tutorials_i)
        assert len(examples) == 8
        # Named for the task, not the protein: every entry leads with what it is for, so
        # a reader picking one does not have to already know which structure demonstrates
        # what. The PDB code follows in parentheses for the ones that have one.
        for expected in ("Model only", "Map + model", "Validation", "X-ray maps",
                         "Ligand fitting", "Real-space refinement", "Restraint edits",
                         "Alternate conformations"):
            assert any(text.startswith(expected) for text in examples), expected

        tutorials = entries(tutorials_i + 1)
        assert [t.lower() for t in tutorials] == [t.lower() for t in TUTORIAL_TITLES]

        # Get is the single icon-only menu button for content the user did not supply.
        buttons = controls.widget().findChildren(QPushButton)
        get_buttons = [b for b in buttons if b.toolTip().startswith("Get data from")]
        assert len(get_buttons) == 1
        assert get_buttons[0].menu() is not None
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
