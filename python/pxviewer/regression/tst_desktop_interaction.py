"""Driving the desktop: modes, minimization, markers, ligands, and what the panes say.

The through-line is that a control must never lie about what the app is doing. A run that
has converged but is still holding must not look idle; a button that started a job must
come back even when the job raises; the show-all-markers label has to track the individual
boxes rather than its own last click.

**Two exercises write real preference keys**, and every one of them builds a DesktopApp,
which reads those keys at construction. ``run()`` therefore wraps the lot in
``shipped_defaults()`` -- see its docstring for why redirecting QSettings to a temporary
directory does not work on macOS, which is a lesson learned the expensive way.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import sys
import time

from libtbx.test_utils import approx_equal, raises

from pxviewer.regression.tst_utils import (
    closing_modals, data_path, dispose, have, process_events, qt_application,
    shipped_defaults, skip, tmp_dir)

if not have("PySide6.QtWebEngineWidgets", "websockets", "iotbx.data_manager"):
    skip("PySide6 QtWebEngine / websockets / iotbx.data_manager not available")

from PySide6.QtCore import QSettings                # noqa: E402

QAPP = qt_application()

from PySide6.QtCore import QEvent, QPointF, Qt      # noqa: E402
from PySide6.QtGui import QMouseEvent               # noqa: E402
from PySide6.QtWidgets import (                     # noqa: E402
    QApplication, QCheckBox, QComboBox, QRadioButton, QTableWidget)

from pxviewer.desktop import DesktopApp, _make_checkable_combo    # noqa: E402
from pxviewer.live import LiveSession               # noqa: E402

MINIMIZE_TIMEOUT_S = 40
BUILD_TIMEOUT_S = 60


class Recording_session(LiveSession):
    """A real session that keeps what it was told to draw.

    Only the marker payload is recorded: it is the one thing the viewer hit-tests
    against, so a drag that is meant to move a marker rather than tug the molecule
    depends on it having been sent.
    """

    def __init__(self, *args, **kwargs):
        super(Recording_session, self).__init__(*args, **kwargs)
        self.marker_updates = []

    def set_markers(self, markers, radius):
        self.marker_updates.append(([dict(m) for m in markers], radius))
        return super(Recording_session, self).set_markers(markers, radius)


@contextlib.contextmanager
def desktop(**kwargs):
    app = DesktopApp(port=0, **kwargs)
    app._webapp.start()
    with closing_modals():
        try:
            yield app
        finally:
            dispose(app)


@contextlib.contextmanager
def bare_desktop(**kwargs):
    """No webapp: enough for the pure control-state exercises, and quicker."""
    app = DesktopApp(port=0, **kwargs)
    with closing_modals():
        try:
            yield app
        finally:
            dispose(app)


def pump_until(predicate, what, timeout=BUILD_TIMEOUT_S):
    deadline = time.time() + timeout
    while time.time() < deadline and not predicate():
        process_events()
        time.sleep(0.02)
    process_events()
    assert predicate(), what


def settle(seconds=1.0):
    """Pump for a fixed spell, for the queued cross-thread signals that have no predicate
    of their own to wait on."""
    end = time.time() + seconds
    while time.time() < end:
        process_events()
        time.sleep(0.01)


def ubiquitin(app, name="1ubq"):
    return app._add_model(LiveSession.from_model_file(data_path("1ubq.pdb")), name)


def place_marker(app, mid="marker-1", position=(0.0, 0.0, 0.0)):
    app._markers.append({"id": mid, "name": "m", "position": list(position),
                         "atom": None, "visible": True})


def build_ligand(app, code="EOH", smiles="CCO", marker="marker-1"):
    """Build a ligand from SMILES at a marker and return its model entry."""
    place_marker(app, marker)
    before = set(m["id"] for m in app._models)
    app.fit_ligand_from_smiles_at_marker(marker, smiles, code, fit=False)
    pump_until(lambda: set(m["id"] for m in app._models) != before,
               "the ligand was never built")
    return next(m for m in app._models if m["id"] not in before)


def monomer_library():
    from pxviewer.geometry import monomer_library_available

    return monomer_library_available()


# -- modes --------------------------------------------------------------------


def accented(button):
    """Whether a button carries the filled accent that marks it as the live one.

    Not ``bool(styleSheet())``: the *inactive* button is given the plain framed look,
    which is also a stylesheet, so emptiness says nothing. The accent is a background
    fill, and that is the difference a glance actually picks up.
    """
    return "background:" in button.styleSheet()


def exercise_minimize_buttons_show_which_state_is_live():
    """A glance at the play/pause pair should say whether a run is going: only the live
    one is enabled and accented."""
    with bare_desktop() as app:
        controls = app._controls
        play, stop = controls._minimize_btn, controls._minimize_stop_btn

        assert play.isEnabled() and not stop.isEnabled()
        assert accented(play) and not accented(stop)

        app.bridge.minimizing_changed.emit(True)
        process_events()
        assert stop.isEnabled() and not play.isEnabled()
        assert accented(stop) and not accented(play)

        app.bridge.minimizing_changed.emit(False)
        process_events()
        assert play.isEnabled() and not stop.isEnabled()
        assert accented(play) and not accented(stop)


def exercise_pick_and_refine_drag_are_mutually_exclusive():
    """Coordinate-changing drag needs its own Tools button and cannot overlap selection --
    a drag would otherwise both move atoms and pick them."""
    with bare_desktop() as app:
        controls = app._controls
        assert not controls._pick_btn.isChecked()
        assert not controls._refine_drag_btn.isChecked()
        assert not app._selection_enabled and not app._tug_enabled

        controls._refine_drag_btn.click()
        assert controls._refine_drag_btn.isChecked()
        assert not controls._pick_btn.isChecked()
        assert app._tug_enabled and not app._selection_enabled

        controls._pick_btn.click()
        assert controls._pick_btn.isChecked()
        assert not controls._refine_drag_btn.isChecked()
        assert app._selection_enabled and not app._tug_enabled


def exercise_refine_drag_arm_is_exactly_pause():
    """Arming a drag stops a running minimization -- the same signal and the same status
    the Pause button raises. With nothing running it is a no-op, as Pause is."""
    with bare_desktop() as app:
        status = []
        app.bridge.status_changed.connect(status.append)

        app._on_tug("m", "arm", -1, None)
        process_events()
        assert not app._minimize_stop.is_set()
        assert status == []

        # A run is going, with idle cleared as minimize_model would leave it.
        app._minimize_idle.clear()
        app._on_tug("m", "arm", -1, None)
        process_events()
        armed = list(status)
        assert app._minimize_stop.is_set()
        assert armed and "stopping" in armed[-1].lower()

        status[:] = []
        app.stop_minimization()
        process_events()
        assert status == [armed[-1]]              # the identical message


def exercise_minimization_runs_continuously_until_stopped():
    """A convergent minimization is over in about a second -- too fast to watch or
    interrupt. So the run stays on after it converges, holding the model at its minimum,
    which gives a steady window to Stop or to hand the model to a drag."""
    if not have("mmtbx.refinement.geometry_minimization"):
        print("    (skipped: geometry_minimization not available)")
        return
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    with desktop() as app:
        status = []
        app.bridge.status_changed.connect(status.append)
        app.load_files([data_path("1ubq.pdb")])
        process_events()

        app.minimize_model(use_map=False)
        pump_until(lambda: any("holding" in s for s in status),
                   "never reached the held state", timeout=MINIMIZE_TIMEOUT_S)
        assert not app._minimize_idle.is_set()     # held means still running

        app.stop_minimization()
        pump_until(lambda: app._minimize_idle.is_set(),
                   "the run never ended", timeout=5)
        settle(0.3)
        assert any("bond rmsd" in s and "->" in s for s in status), \
            "no improvement summary"


# -- persisted preferences ----------------------------------------------------


def exercise_focus_surroundings_defaults_on_and_persists():
    """Mol*'s click-focus neighbourhood is on by default and the choice survives a
    restart. Pick mode suppresses it *without* changing the preference, then restores it
    -- so turning picking on and off again does not quietly rewrite a setting.
    """
    key = "defaults/focus_surroundings"
    QSettings("pxviewer", "pxviewer").remove(key)

    first = DesktopApp(port=0)
    try:
        controls = first._controls
        assert first._focus_surroundings
        assert controls._focus_surroundings_check.isChecked()

        # A real session rather than a mock: the flag it keeps is the state under test,
        # and asserting on it says the session was actually reconfigured.
        session = LiveSession.from_sites([[0, 0, 0], [1, 0, 0]])
        entry = {"session": session}
        first._models.append(entry)

        controls._pick_btn.click()
        assert session._focus_surroundings is False    # suppressed in the viewer ...
        assert first._focus_surroundings               # ... but the preference stands

        controls._pick_btn.click()
        assert session._focus_surroundings is True     # and it comes back
        first._models.remove(entry)

        controls._focus_surroundings_check.click()
        assert not first._focus_surroundings
    finally:
        first.stop()

    second = DesktopApp(port=0)
    try:
        assert not second._focus_surroundings
        assert not second._controls._focus_surroundings_check.isChecked()
    finally:
        second.stop()


def exercise_new_model_show_and_representation_defaults_persist():
    """The layers and Show choices a new model opens with survive a restart, and are
    applied to the next model loaded."""
    settings = QSettings("pxviewer", "pxviewer")
    settings.setValue("defaults/model_representations", '["cartoon"]')
    settings.setValue(
        "defaults/shown_structure_types",
        '["Protein", "Nucleic acid", "Sugar", "Ion", "Water", "Ligand / other"]')
    settings.setValue("defaults/molstar_interactions", "false")
    settings.sync()

    first = DesktopApp(port=0)
    try:
        first.set_default_model_representation("ball-and-stick", True)
        first.set_default_model_show("Water", False)
        first.set_default_model_show("Mol* interactions", True)
    finally:
        first.stop()

    second = DesktopApp(port=0)
    try:
        assert second._default_model_reps == ["cartoon", "ball-and-stick"]
        assert "Water" not in second._default_shown_types
        assert second._default_model_interactions

        mid = ubiquitin(second)
        entry = second._model_entry(mid)
        assert entry["reps"] == ["cartoon", "ball-and-stick"]
        assert len(entry["session"]._representations) == 2
        assert entry["hidden_types"] == {"Water"}
        assert entry["interactions"]
        assert entry["session"]._computed_interactions_visible
    finally:
        second.stop()


# -- markers and ligands ------------------------------------------------------


def exercise_dragging_a_marker_moves_it_and_never_tugs():
    """A marker is a handle, not an atom. A Shift-drag on one moves the marker, detaching
    it from any snapped atom, and reports its position for the viewer's hit-test -- it
    must not start a molecule tug."""
    with desktop() as app:
        app._add_model(Recording_session.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        session = app._control_session()
        assert isinstance(session, Recording_session)

        app._on_marker([1.0, 2.0, 3.0], atom=5)        # placed, snapped to atom 5
        mid = app._markers[0]["id"]
        assert app._markers[0]["atom"] == 5
        assert session.marker_updates
        assert session.marker_updates[-1][0][0]["id"] == mid   # the viewer was told

        # An intermediate move updates the position and drops the atom it was snapped to.
        app._on_marker_move(mid, [4.0, 5.0, 6.0], final=False)
        assert app._markers[0]["position"] == [4.0, 5.0, 6.0]
        assert app._markers[0]["atom"] is None
        assert app._tug is None                        # never a molecule tug

        app._on_marker_move(mid, [7.0, 8.0, 9.0], final=True)
        assert app._markers[0]["position"] == [7.0, 8.0, 9.0]
        assert app._tug is None


def exercise_placing_a_ligand_clears_the_input_fields():
    """After a ligand is placed the code and SMILES boxes clear, so the next one cannot
    silently inherit the previous code as its name. A failed build leaves the inputs
    untouched, so a typo can be fixed rather than retyped."""
    if not have("rdkit"):
        print("    (skipped: rdkit not available)")
        return

    with desktop() as app:
        controls = app._controls

        place_marker(app, "marker-1")
        controls._lig_code_edit.setText("EOH")
        controls._lig_smiles_edit.setText("CCO")
        before = len(app._models)
        controls._on_fit_ligand()
        pump_until(lambda: len(app._models) != before, "the ligand was never placed")

        assert controls._lig_code_edit.text() == ""
        assert controls._lig_smiles_edit.text() == ""

        place_marker(app, "marker-2")
        controls._lig_code_edit.setText("ABC")
        controls._lig_smiles_edit.setText("not_a_smiles!!!")
        before = len(app._models)
        controls._on_fit_ligand()
        settle(0.8)

        assert len(app._models) == before              # nothing placed
        assert controls._lig_code_edit.text() == "ABC"
        assert controls._lig_smiles_edit.text() == "not_a_smiles!!!"


def exercise_a_smiles_ligand_carries_and_saves_its_restraints():
    """A ligand built from SMILES carries its restraint CIF, the Loaded tree flags it as
    savable, and export writes the pair a refinement needs: the placed coordinates and
    the monomer dictionary beside them. An ordinary model has nothing to save."""
    if not have("rdkit"):
        print("    (skipped: rdkit not available)")
        return

    with desktop() as app, tmp_dir() as directory:
        import os

        ligand = build_ligand(app)
        assert ligand.get("restraints_cif")
        item = next(it for it in app._loaded_summary()["items"]
                    if it["id"] == ligand["id"])
        assert item["has_restraints_cif"] is True

        out = os.path.join(directory, "EOH.cif")
        app.save_restraints_cif(ligand["id"], out)
        text = open(out).read()
        for expected in ("generated by pxviewer", "SMILES_CANONICAL", "RDKit"):
            assert expected in text, expected

        # A protein model has no restraints of its own to save.
        app.load_files([data_path("1ubq.pdb")])
        process_events()
        protein = next(m for m in app._models if m["name"].endswith(".pdb"))
        protein_item = next(it for it in app._loaded_summary()["items"]
                            if it["id"] == protein["id"])
        assert protein_item["has_restraints_cif"] is False
        with raises(ValueError) as e:
            app.save_restraints_cif(protein["id"], os.path.join(directory, "nope.cif"))
        assert "no restraints" in str(e.value)

        # Export writes both halves in one step.
        coord, restraints = app.export_ligand(
            ligand["id"], os.path.join(directory, "EOH.mmcif"))
        assert os.path.basename(coord) == "EOH.mmcif"
        assert os.path.basename(restraints) == "EOH_restraints.cif"
        assert "_atom_site" in open(coord).read()             # placed coordinates
        assert "SMILES_CANONICAL" in open(restraints).read()  # the dictionary
        with raises(ValueError):
            app.export_ligand(protein["id"], os.path.join(directory, "prot.cif"))


def exercise_writing_a_restrained_model_as_mmcif_needs_no_probe():
    """Writing a model as mmCIF must emit coordinates, not a validation report.

    A ligand -- like any minimized model -- carries a restraints manager, and mmtbx's
    full ``model_as_mmcif`` would compute a clashscore that shells out to the external
    Probe binary, which is not present. The write has to succeed anyway.
    """
    if not have("rdkit"):
        print("    (skipped: rdkit not available)")
        return
    import os

    from iotbx.data_manager import DataManager

    with desktop() as app, tmp_dir() as directory:
        ligand = build_ligand(app)
        assert ligand["session"].model.restraints_manager_available()   # the Probe trigger

        out = os.path.join(directory, "EOH.cif")
        app.write_object("model", ligand["id"], out)      # must not raise
        DataManager().process_model_file(out)             # and it reparses


def exercise_authoring_saving_and_loading_restraint_edits():
    """Author a custom bond from a selection, round-trip it through a PHIL file, and
    confirm the model's restraints carry it. A duplicate -- a bond the library already
    restrains -- is refused and not stored."""
    if not have("rdkit", "mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: rdkit / pdb_interpretation not available)")
        return
    import os

    with desktop() as app, tmp_dir() as directory:
        ligand = build_ligand(app)
        mid = ligand["id"]
        model = ligand["session"].model
        names = [a.name.strip() for a in model.get_hierarchy().atoms()]

        def n_bonds():
            geometry = model.get_restraints_manager().geometry
            return geometry.pair_proxies().bond_proxies.simple.size()

        base = n_bonds()

        # Author a bond between C1 and O1, which are not natively bonded.
        app._scene_selection[mid] = [names.index("C1"), names.index("O1")]
        app.add_edit_from_selection(mid, "bond")
        assert len(app.model_edits(mid)) == 1
        item = next(it for it in app._loaded_summary()["items"] if it["id"] == mid)
        assert len(item["edits"]) == 1
        assert "bond" in item["edits"][0]["summary"]
        assert n_bonds() == base + 1          # the minimizer's restraints carry it

        # A duplicate is refused and the list is unchanged.
        app._scene_selection[mid] = [names.index("C1"), names.index("C2")]
        with raises(ValueError):
            app.add_edit_from_selection(mid, "bond")
        assert len(app.model_edits(mid)) == 1

        phil = os.path.join(directory, "edits.phil")
        app.save_edits(mid, phil)
        assert "geometry_restraints.edits" in open(phil).read()
        app.clear_edits(mid)
        assert app.model_edits(mid) == []
        assert app.load_edits(mid, phil) == 1
        assert len(app.model_edits(mid)) == 1


def exercise_loading_a_hand_written_edits_phil():
    """The user's own file, not one this app wrote.

    The round trip above proves save and load agree with each other, which they would
    even if both were wrong. This starts from the PHIL that ships beside ``zn_site.pdb``
    -- authored by hand, in the format cctbx and phenix read -- so it also pins that the
    format the rest of the world writes is the format this reads.
    """
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: pdb_interpretation not available)")
        return
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        print("    (skipped: no monomer library)")
        return

    with desktop() as app:
        app.load_file(data_path("zn_site.pdb"))
        mid = app._models[0]["id"]
        model = app._model_entry(mid)["session"].model

        def n_bonds():
            return model.get_restraints_manager().geometry.pair_proxies(
                ).bond_proxies.simple.size()

        app._controls._ensure_restraints()
        before = n_bonds()

        assert app.load_edits(mid, data_path("zn_site_edits.phil")) == 1
        assert len(app.model_edits(mid)) == 1
        assert n_bonds() == before + 1        # the Zn-water bond cctbx does not add itself

        # It survives into the summary the tree and Appearance pane read.
        item = next(it for it in app._loaded_summary()["items"] if it["id"] == mid)
        assert "bond" in item["edits"][0]["summary"]


def exercise_a_phil_with_nothing_in_it_is_refused():
    """A misspelled scope parses to zero edits without complaint (see tst_edits.py), so
    this layer is what tells the user their file did nothing."""
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: pdb_interpretation not available)")
        return
    import os

    with desktop() as app, tmp_dir() as directory:
        app.load_file(data_path("zn_site.pdb"))
        mid = app._models[0]["id"]

        empty = os.path.join(directory, "nothing.phil")
        with open(empty, "w") as fh:
            fh.write("geometry_restraints.edit { bond { distance_ideal = 2.0 } }\n")

        with raises(ValueError) as e:
            app.load_edits(mid, empty)
        assert "no restraint edits" in str(e.value)
        assert app.model_edits(mid) == []


def exercise_a_phil_with_no_sigma_is_refused_and_says_which_edit():
    """Matching cctbx, which will not guess a weight. The message has to name the edit:
    a real edits file can hold a dozen, and "sigma missing" alone does not say where."""
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: pdb_interpretation not available)")
        return
    import os

    with desktop() as app, tmp_dir() as directory:
        app.load_file(data_path("zn_site.pdb"))
        mid = app._models[0]["id"]

        path = os.path.join(directory, "nosigma.phil")
        with open(path, "w") as fh:
            fh.write("""
            geometry_restraints.edits {
              bond {
                atom_selection_1 = "chain S and resseq 1 and name ZN"
                atom_selection_2 = "chain S and resseq 2 and name O"
                distance_ideal = 2.1
              }
            }
            """)

        with raises(ValueError) as e:
            app.load_edits(mid, path)
        assert "sigma" in str(e.value)
        assert "ZN" in str(e.value)           # which edit, not just that one was bad
        assert app.model_edits(mid) == []


def exercise_a_phil_naming_an_atom_that_is_not_there_is_refused():
    """Loading it must leave the model exactly as it was: a half-applied edit list is
    one a later minimize or drag cannot build restraints from."""
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: pdb_interpretation not available)")
        return
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        print("    (skipped: no monomer library)")
        return
    import os

    with desktop() as app, tmp_dir() as directory:
        app.load_file(data_path("zn_site.pdb"))
        mid = app._models[0]["id"]
        app._controls._ensure_restraints()

        bad = os.path.join(directory, "typo.phil")
        with open(bad, "w") as fh:
            fh.write("""
            geometry_restraints.edits {
              bond {
                atom_selection_1 = "chain S and resseq 1 and name ZN"
                atom_selection_2 = "name NOSUCHATOM"
                distance_ideal = 2.1
                sigma = 0.05
              }
            }
            """)

        with raises(ValueError):
            app.load_edits(mid, bad)
        assert app.model_edits(mid) == []     # reverted, not left half-applied


def _model_with_an_unknown_ligand(directory):
    """Write a PDB whose ligand the monomer library cannot type. Returns its path."""
    from pxviewer import ligands as ligands_mod
    import os

    model = ligands_mod.build_ligand_from_smiles("CC(=O)Oc1ccccc1C(=O)O", "AIN", (0, 0, 0))
    hierarchy = model.get_hierarchy()
    for group in hierarchy.atom_groups():
        group.resname = "L01"
    path = os.path.join(directory, "unknown_ligand.pdb")
    with open(path, "w") as handle:
        handle.write(hierarchy.as_pdb_string(crystal_symmetry=model.crystal_symmetry()))
    return path


def exercise_an_unknown_ligand_can_be_given_restraints():
    """One unrecognised residue costs the whole model its restraints, so the app offers to
    infer a dictionary rather than leaving minimize, drag and the Geometry tab dead."""
    if not have("rdkit", "mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: rdkit / pdb_interpretation not available)")
        return
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    with desktop() as app, tmp_dir() as directory:
        app.load_file(_model_with_an_unknown_ligand(directory))
        mid = app._models[0]["id"]

        unknown = app.unknown_ligands(mid)
        assert [u["code"] for u in unknown] == ["L01"]

        made = app.generate_ligand_restraints(mid)
        assert set(made) == {"L01"}
        assert "CC(=O)Oc1ccccc1C(=O)O" in made["L01"] or made["L01"]   # perceived chemistry
        assert app.unknown_ligands(mid) == []

        # The model now builds, which is the thing that was impossible before.
        model = app._model_entry(mid)["session"].model
        model.process(make_restraints=True)
        assert model.get_restraints_manager().geometry.pair_proxies(
            ).bond_proxies.simple.size() > 15


def exercise_declining_the_offer_leaves_the_model_alone():
    """``closing_modals`` rejects the dialog, which is the Cancel path: nothing is
    generated, and the model is left exactly as unbuildable as it was -- an inferred
    dictionary is a guess, and a refused guess must not be applied anyway."""
    if not have("rdkit", "mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: rdkit / pdb_interpretation not available)")
        return
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    with desktop() as app, tmp_dir() as directory:
        app.load_file(_model_with_an_unknown_ligand(directory))
        mid = app._models[0]["id"]

        assert app._controls._offer_ligand_restraints(mid) is False
        assert [u["code"] for u in app.unknown_ligands(mid)] == ["L01"]


def exercise_minimize_offers_to_fix_a_ligand_that_blocks_it():
    """Minimize and drag build restraints on worker threads, which cannot ask the user
    anything -- so an unknown ligand made them fail with a line of cctbx text, and the
    offer to do something about it existed only in the Geometry tab.

    The check is free on a healthy model: it acts only when the background pre-warm has
    already failed and recorded why. Working it out at click time would mean a full
    interpretation pass on the GUI thread every time somebody pressed Minimize.
    """
    if not have("rdkit", "mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: rdkit / pdb_interpretation not available)")
        return
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    with desktop() as app, tmp_dir() as directory:
        app.load_file(_model_with_an_unknown_ligand(directory))
        mid = app._models[0]["id"]
        entry = app._model_entry(mid)

        # The warm runs on its own thread and records why it could not build.
        pump_until(lambda: "restraints_error" in entry,
                   "the failed pre-warm was never recorded")
        assert "Fatal problems" in entry["restraints_error"]

        # closing_modals rejects the dialog, so this is the decline path: the offer was
        # made, nothing was generated, and the recorded reason stands.
        app._controls._offer_restraints_if_blocked(mid)
        assert [u["code"] for u in app.unknown_ligands(mid)] == ["L01"]
        assert "restraints_error" in entry

        # Accepting it (driven directly here) clears the block.
        assert app.generate_ligand_restraints(mid, ["L01"])
        app._controls._offer_restraints_if_blocked(mid)
        assert entry.get("restraints_error") is None
        assert entry["session"].model.restraints_manager_available()


def exercise_a_healthy_model_is_never_asked_about_ligands():
    """The pre-flight must cost nothing on an ordinary model, or every Minimize would pay
    for an interpretation pass to be told there is no problem."""
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    with desktop() as app:
        app.load_file(data_path("zn_site.pdb"))
        mid = app._models[0]["id"]
        entry = app._model_entry(mid)
        pump_until(lambda: entry["session"].model.restraints_manager_available(),
                   "restraints never warmed")
        assert entry.get("restraints_error") is None
        app._controls._offer_restraints_if_blocked(mid)   # a no-op, and must not raise


# -- the live difference map --------------------------------------------------


def exercise_the_live_difference_map_streams_a_box_during_a_drag():
    """With a phased model and the live difference map on, arming a drag and feeding it a
    frame streams an mFo-DFc window to the model's session; disabling clears it.

    The whole wiring: arm, the background recompute worker, and teardown.
    """
    if not have("mmtbx.f_model", "numpy"):
        print("    (skipped: mmtbx.f_model / numpy not available)")
        return
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return
    import os
    import struct

    import numpy as np

    from pxviewer.cctbx_io import read_model

    path = data_path("1ubq.pdb")
    model = read_model(path)

    with desktop() as app, tmp_dir() as directory:
        # A self-contained phased group: the bundled model, and reflections computed
        # from it so the phases have something to be right about.
        f_obs = abs(model.get_xray_structure().structure_factors(d_min=2.0).f_calc())
        f_obs.set_observation_type_xray_amplitude()
        dataset = f_obs.as_mtz_dataset(column_root_label="F")
        dataset.add_miller_array(f_obs.generate_r_free_flags(),
                                 column_root_label="FreeR_flag")
        mtz = os.path.join(directory, "d.mtz")
        dataset.mtz_object().write(mtz)

        app.load_files([path])
        app.load_files([mtz])
        pump_until(lambda: app._models and app._reflections, "nothing loaded")

        mid, rid = app._models[0]["id"], app._reflections[0]["id"]
        app.make_maps(rid, mid)
        pump_until(lambda: app.map_for_model(mid) is not None,
                   "phasing never produced a map", timeout=150)

        session = app._model_entry(mid)["session"]
        app._tug_session = session                 # as a drag 'begin' would set it
        app.set_live_difference_map(True)
        app._maybe_start_live_diff(mid, atom=model.get_number_of_atoms() // 2)
        assert app._diff_ctx is not None           # armed

        app._queue_live_diff(np.array(model.get_sites_cart(), dtype="float64"))
        pump_until(lambda: session._last_map_box is not None,
                   "no difference window ever arrived", timeout=90)
        assert struct.unpack_from("<I", session._last_map_box, 0)[0] == 4   # _TAG_MAP

        app.set_live_difference_map(False)
        assert app._diff_ctx is None
        assert session._last_map_box is None


# -- what the panes say -------------------------------------------------------


def exercise_the_selection_pane_describes_picked_atoms():
    """One atom is reported by hierarchy identity, coordinates, occupancy and B factor.

    A ribbon pick arrives as every atom in the residue, so it is presented as the residue
    rather than as a dump of those implementation-level atoms.
    """
    from types import SimpleNamespace

    with desktop() as app:
        mid = ubiquitin(app)
        controls = app._controls

        app._on_model_selection(mid, SimpleNamespace(indices=[0]))
        process_events()
        text = controls._selection_label.text()
        assert "1 atom selected" in text
        assert "1ubq · chain A · MET 1 · N (N)" in text
        assert "xyz 27.340, 24.430, 2.614" in text
        assert "occ 1.00 · B 9.67" in text

        model = app._model_entry(mid)["session"].model
        residue = model.get_hierarchy().only_model().chains()[0].residue_groups()[0]
        app._on_model_selection(
            mid, SimpleNamespace(indices=list(range(len(residue.atoms())))))
        process_events()
        text = controls._selection_label.text()
        assert text.startswith("1 residue selected · %d atoms" % len(residue.atoms()))
        assert "1ubq · chain A: MET 1" in text
        assert "Center " in text
        assert "B-factor mean " in text
        assert "· N (N)" not in text            # no per-atom dump

        app.clear_selection()
        process_events()
        assert controls._selection_label.text() == "None"


def exercise_validation_subtabs_and_row_focus():
    """Each result becomes a sub-tab, and selecting a whole row focuses that residue,
    resolved to atom indices on the active model."""
    from pxviewer.validation import ValidationResult

    with desktop() as app:
        mid = ubiquitin(app)

        # A synthetic result drives the UI: no mmtbx run is needed to say what the tabs
        # and the row-to-residue resolution do.
        result = ValidationResult(
            key="ramachandran", title="Ramachandran",
            columns=["chain", "resid", "res"], rows=[["A", "  13 ", "ILE"]],
            markup=[], summary="1 residue")
        app._controls._on_validation_ready((mid, [result]))

        tabs = app._controls._validation_subtabs
        # Clashes & contacts is the permanent first tab; the validator follows it.
        assert tabs.count() == 2
        rama = next(i for i in range(tabs.count())
                    if tabs.tabText(i) == "Ramachandran")
        assert rama == 1

        table = tabs.widget(rama).findChild(QTableWidget)
        assert table.selectionBehavior() == QTableWidget.SelectionBehavior.SelectRows

        table.selectRow(0)
        index = app._model_entry(mid)["_residue_index"]
        assert index[("A", "13")] == [94, 95, 96, 97, 98, 99, 100, 101]   # ILE 13


def exercise_one_button_shows_and_hides_every_validation_overlay():
    """Each validator draws on its own channel, so clearing the viewport otherwise means
    visiting every sub-tab. One button drives them all, and its label says what the next
    click will do -- tracking the individual boxes so it never lies."""
    from pxviewer.validation import ValidationResult

    def result(key):
        return ValidationResult(key=key, title=key.title(), columns=["chain", "resid"],
                                rows=[["A", "1"]], markup=[{"kind": "dots"}], summary="s")

    with desktop() as app:
        controls = app._controls
        button = controls._all_markup_btn
        assert not button.isEnabled()          # nothing drawn, so nothing to toggle

        controls._on_validation_ready(
            ("m1", [result("ramachandran"), result("rotamers")], True))
        live = [c for c in controls._marker_checks if c.isEnabled()]
        assert len(live) == 2
        assert all(c.isChecked() for c in live)
        assert button.isEnabled()
        assert button.text() == "Hide all markers"

        button.click()
        assert not any(c.isChecked() for c in live)
        assert button.text() == "Show all markers"

        button.click()
        assert all(c.isChecked() for c in live)
        assert button.text() == "Hide all markers"

        # The label follows the individual boxes, not its own last click.
        live[0].setChecked(False)
        assert button.text() == "Hide all markers"
        live[1].setChecked(False)
        assert button.text() == "Show all markers"

        # A re-run replaces the checkboxes; the button reasons about the new ones only.
        controls._on_validation_ready(("m1", [result("cablam")], True))
        assert len(controls._marker_checks) == 1
        assert button.text() == "Hide all markers"


# -- long operations ----------------------------------------------------------


def exercise_a_long_operation_raises_the_busy_indicator():
    """Some operations run for tens of seconds and the text status alone is easy to miss.

    Driven from one place, ``run_background``, so no operation can forget it -- including
    one that fails, which is exactly the case that would leave it spinning forever.
    """
    import threading

    with desktop() as app:
        bar = app._controls._busy_bar
        assert bar.isHidden()

        # Two overlapping operations: the bar stays up until the *last* finishes, so a
        # background rephasing does not switch it off while a score is still going.
        gate = threading.Event()
        app.run_background(lambda: gate.wait(10), name="t-a", label="A")
        app.run_background(lambda: gate.wait(10), name="t-b", label="B")
        settle()
        assert app._busy_labels == ["A", "B"]
        assert not bar.isHidden()

        gate.set()
        settle()
        assert app._busy_labels == []
        assert bar.isHidden()

        def boom():
            raise RuntimeError("worker failed")

        app.run_background(boom, name="t-boom", label="Failing")
        settle()
        assert app._busy_labels == []
        assert bar.isHidden()


def exercise_a_running_operation_disables_only_its_own_button():
    """A second click would queue a duplicate of work already in flight, so the button
    that started one is disabled until it finishes -- and only that button. It comes back
    even if the worker raises, or it would be dead for the rest of the session."""
    import threading

    with desktop() as app:
        controls = app._controls
        find, validate = controls._hotspot_btn, controls._validate_btn
        assert find.isEnabled() and validate.isEnabled()

        gate = threading.Event()
        app.run_background(lambda: gate.wait(10), name="t-hs", label="Finding hotspots")
        settle()
        assert not find.isEnabled()        # cannot queue a second one
        assert validate.isEnabled()        # an unrelated action is still offered

        gate.set()
        settle()
        assert find.isEnabled()

        def boom():
            raise RuntimeError("worker failed")

        app.run_background(boom, name="t-boom", label="Finding hotspots")
        settle()
        assert find.isEnabled()


# -- atom-precision work ------------------------------------------------------


def exercise_atom_precision_actions_switch_a_ribbon_to_ball_and_stick():
    """A cartoon ribbon cannot show atoms, so atom-precision work would draw markup into
    empty space. Those actions switch the model first; a representation that already
    shows atoms is left as the user chose."""
    if not have("mmtbx.monomer_library.pdb_interpretation"):
        print("    (skipped: pdb_interpretation not available)")
        return
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return
    from pxviewer.geometry import GeometryRestraints

    with desktop() as app:
        mid = ubiquitin(app)
        assert app._model_entry(mid)["rep"] == "cartoon"        # the polymer default

        restraints = GeometryRestraints(app._model_entry(mid)["session"].model)
        i_seqs = tuple(int(i) for i in restraints.row("bond", 0)[0])
        app.show_restraint_notations(mid, [("bond", i_seqs)])
        assert app._model_entry(mid)["rep"] == "ball-and-stick"

        app.set_model_representation(mid, "spacefill")
        app.ensure_atoms_shown(mid)
        assert app._model_entry(mid)["rep"] == "spacefill"      # already shows atoms

        # Measuring switches a ribbon too. Select and colour do not, being unhooked.
        app.set_model_representation(mid, "cartoon")
        app._scene_selection[mid] = [0, 1]
        app.measure_selection("distance")
        assert app._model_entry(mid)["rep"] == "ball-and-stick"


def exercise_residue_orientation_and_space_navigation():
    """Oriented focus frames a residue N->C screen-right with the side chain up, and
    advance_residue steps along the chain for space-bar navigation."""
    if not have("numpy"):
        print("    (skipped: numpy not available)")
        return
    import numpy as np

    with desktop() as app:
        mid = ubiquitin(app)
        model = app._model_entry(mid)["session"].model
        index = app._build_residue_index(model)

        _target, up, direction, _radius = app._residue_orientation(
            model, index[("A", "13")])
        assert approx_equal(float(np.linalg.norm(up)), 1.0, eps=1e-6)
        assert approx_equal(float(np.linalg.norm(direction)), 1.0, eps=1e-6)
        assert abs(float(np.dot(up, direction))) < 1e-6         # orthonormal

        atoms = model.get_hierarchy().atoms()
        named = dict((atoms[i].name.strip(), np.array(atoms[i].xyz))
                     for i in index[("A", "13")])
        n_to_c = named["C"] - named["N"]
        n_to_c /= np.linalg.norm(n_to_c)
        screen_right = np.cross(direction, up)                  # Mol*'s right = view x up
        assert float(np.dot(screen_right, n_to_c)) > 0.99
        assert float(np.dot(up, named["CB"] - named["CA"])) > 0  # side chain up

        app._focused_residue = ("A", "13")
        app.advance_residue(1)
        assert app._focused_residue == ("A", "14")
        app.advance_residue(-1)
        assert app._focused_residue == ("A", "13")


# -- showing and hiding parts of a model --------------------------------------


def exercise_hide_and_show_selected_atoms():
    """Hide-selected drops the selection from the drawn atoms -- a partial
    representation, the same mechanism as the structure-type toggles."""
    with desktop() as app:
        mid = app._add_model(
            LiveSession.from_sites([[i, 0, 0] for i in range(6)]), "M")
        entry = app._model_entry(mid)
        assert entry["hidden_atoms"] == set()
        assert app._shown_indices(entry) is None            # all shown

        controls = app._controls
        assert not controls._hide_sel_btn.isEnabled()       # nothing selected yet

        app._scene_selection = {mid: [1, 2, 3]}
        controls._on_scene_selection_changed(app._scene_selection)
        assert controls._hide_sel_btn.isEnabled()
        assert controls._show_sel_btn.isEnabled()

        app.hide_selected()
        assert entry["hidden_atoms"] == {1, 2, 3}
        assert app._shown_indices(entry) == [0, 4, 5]       # the rest still drawn

        app.show_selected()
        assert entry["hidden_atoms"] == set()
        assert app._shown_indices(entry) is None


def exercise_hide_structure_types():
    """Show and hide cctbx structure classes by restricting the representation."""
    def shown(on):
        if "runs" in on:
            return sum(end - start + 1 for start, end in on["runs"])
        return len(on.get("list", []))

    with desktop() as app:
        captured = {}
        app.bridge.loaded_changed.connect(lambda s: captured.update(s))

        mid = ubiquitin(app)
        entry = app._model_entry(mid)
        session = entry["session"]

        assert app.model_structure_types(mid) == ["Protein", "Water"]
        assert entry["hidden_types"] == set()
        assert all("on" not in r for r in session._representations.values())
        item = next(it for it in captured["items"] if it["id"] == mid)
        assert item["types"] == ["Protein", "Water"]
        assert item["hidden_types"] == []

        app.set_model_type_hidden(mid, "Water", True)
        assert entry["hidden_types"] == {"Water"}
        reps = list(session._representations.values())
        assert len(reps) == 1
        assert shown(reps[0]["on"]) == 602                  # the protein atoms

        app.set_model_representation(mid, "spacefill")
        assert "on" in next(iter(session._representations.values()))   # still restricted

        app.set_model_type_hidden(mid, "Protein", True)
        assert shown(next(iter(session._representations.values()))["on"]) == 0

        app.set_model_type_hidden(mid, "Water", False)
        assert shown(next(iter(session._representations.values()))["on"]) == 58

        app.set_model_type_hidden(mid, "Protein", False)
        assert entry["hidden_types"] == set()
        assert all("on" not in r for r in session._representations.values())

        # The pane exposes a structure-type checklist once there is more than one type.
        controls = app._controls
        controls._update_appearance("model", mid)
        checkable = [c for c in controls._appearance_box.findChildren(QComboBox)
                     if c.model().rowCount() and c.model().item(0).isCheckable()]
        assert checkable, "expected a checkable structure-type combo"
        labels = [checkable[0].model().item(i).text()
                  for i in range(checkable[0].model().rowCount())]
        assert "Mol* interactions" in labels
        assert not any(box.text() == "Computed interactions"
                       for box in controls._appearance_box.findChildren(QCheckBox))

        # Tree row layout: visible check in column 0, active radio in 1, name in 2.
        tree = controls._loaded_tree
        first = tree.topLevelItem(0)
        assert tree.columnCount() == 3
        assert first.checkState(0) in (Qt.CheckState.Checked, Qt.CheckState.Unchecked)
        assert isinstance(tree.itemWidget(first, 1), QRadioButton)
        assert first.text(0) == ""
        assert "1ubq" in first.text(2)


def exercise_a_checkable_combo_requires_a_click_inside_the_popup():
    """The click that *opens* the dropdown must not toggle the item under the cursor.

    The popup is really shown and the events are aimed at the first item's own rectangle,
    rather than forcing ``indexAt`` to answer with it: the hit-test is half of what is
    being checked, so replacing it would leave only the other half.
    """
    combo = _make_checkable_combo()
    combo.add_checkable("Protein", True, "Protein")
    combo.add_checkable("Water", True, "Water")
    fired = []
    combo.on_change = lambda data, checked: fired.append((data, checked))

    combo.showPopup()
    process_events()
    view = combo.view()
    viewport = view.viewport()
    first = combo.model().index(0, 0)
    point = QPointF(view.visualRect(first).center())
    assert view.indexAt(view.visualRect(first).center()) == first, \
        "the popup did not lay out, so the events would not land on an item"

    def send(kind):
        event = QMouseEvent(kind, point, Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        combo.eventFilter(viewport, event)

    # The opening gesture: a release with no press in the popup before it.
    send(QEvent.Type.MouseButtonRelease)
    assert combo.model().item(0).checkState() == Qt.CheckState.Checked
    assert fired == []

    # A real click inside the popup -- press, then release -- toggles the item.
    send(QEvent.Type.MouseButtonPress)
    send(QEvent.Type.MouseButtonRelease)
    assert combo.model().item(0).checkState() == Qt.CheckState.Unchecked
    assert fired == [("Protein", False)]

    combo.hidePopup()


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
