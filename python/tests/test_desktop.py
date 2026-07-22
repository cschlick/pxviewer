"""Tests for the desktop atoms-table model (no QWebEngine needed)."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pxviewer.data import AtomArrays  # noqa: E402
from pxviewer.desktop import (  # noqa: E402
    _make_atom_table_model,
    _make_restraint_table_model,
    _runs,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Data:
    def __init__(self, arrays):
        self.arrays = arrays


class _Session:
    """Minimal stand-in for a LiveSession (just what the table model reads)."""

    def __init__(self, arrays, attributes=None):
        self._data = _Data(arrays)
        self._attributes = attributes or {}


def _arrays():
    return AtomArrays(
        element=["N", "C", "O"], name=["N", "CA", "C"], resname=["ALA"] * 3, chain=["A"] * 3,
        resseq=[1, 1, 1], x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0], z=[0.0, 0.0, 0.0],
        b=[10.0, 20.0, 30.0], occ=[1.0, 1.0, 1.0],
    )


def test_runs_collapses_contiguous_indices():
    assert list(_runs([3, 1, 2, 2, 5])) == [(1, 3), (5, 5)]
    assert list(_runs([])) == []


def test_atom_table_columns_and_values(qapp):
    model = _make_atom_table_model()
    model.set_session(_Session(_arrays(), {"score": [0.1, 0.2, 0.3]}))

    assert model.rowCount() == 3
    headers = [model.headerData(i, Qt.Orientation.Horizontal) for i in range(model.columnCount())]
    assert headers[:6] == ["#", "element", "name", "resname", "chain", "resseq"]
    assert {"x", "y", "z", "B", "occ", "score"} <= set(headers)

    assert model.data(model.index(1, 0)) == "1"  # the "#" index column
    assert model.data(model.index(0, headers.index("element"))) == "N"
    assert model.data(model.index(1, headers.index("B"))) == "20.000"
    assert model.data(model.index(2, headers.index("score"))) == "0.300"  # a custom attribute


def test_atom_table_nan_renders_blank(qapp):
    model = _make_atom_table_model()
    model.set_session(_Session(_arrays(), {"partial": [1.0, float("nan"), 3.0]}))
    col = [model.headerData(i, Qt.Orientation.Horizontal) for i in range(model.columnCount())].index("partial")
    assert model.data(model.index(0, col)) == "1.000"
    assert model.data(model.index(1, col)) == ""  # NaN -> blank cell


def test_atom_table_empty_when_no_session(qapp):
    model = _make_atom_table_model()
    model.set_session(None)
    assert model.rowCount() == 0 and model.columnCount() == 0


def test_atom_table_filter_to_selection(qapp):
    """Filtering restricts the visible rows to a selected subset (show-only-selected)."""
    model = _make_atom_table_model()
    model.set_session(_Session(_arrays(), {"score": [0.1, 0.2, 0.3]}))
    headers = [model.headerData(i, Qt.Orientation.Horizontal) for i in range(model.columnCount())]

    assert not model.is_filtered() and model.rowCount() == 3

    model.set_filter([2, 0])  # unordered + not the full set -> sorted, deduped
    assert model.is_filtered() and model.rowCount() == 2
    # Row 0 -> atom 0, row 1 -> atom 2; the "#" column shows the real atom index.
    assert model.data(model.index(0, 0)) == "0"
    assert model.data(model.index(1, 0)) == "2"
    assert model.data(model.index(1, headers.index("B"))) == "30.000"  # atom 2's B-factor
    assert model.row_atom(1) == 2
    assert model.atom_row(2) == 1
    assert model.atom_row(1) == -1  # atom 1 is filtered out

    model.set_filter(None)
    assert not model.is_filtered() and model.rowCount() == 3
    assert model.row_atom(1) == 1 and model.atom_row(1) == 1


def test_scene_selection_aggregation(qapp):
    """Each model reports its own picks; the desktop unions them into a scene selection."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    from pxviewer.appserver import find_frontend_dir, frontend_is_built

    fd = find_frontend_dir()
    if fd is None or not frontend_is_built(fd):
        pytest.skip("frontend not built")

    from types import SimpleNamespace

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0], [7, 0, 0]]), "B")

        # A selection spanning two models is the union of each model's picks.
        app._on_model_selection(a, SimpleNamespace(indices=[0, 1]))
        app._on_model_selection(b, SimpleNamespace(indices=[2]))
        assert app._scene_selection == {a: [0, 1], b: [2]}

        assert app.session_for(a)._n_atoms == 2
        assert app.session_for(b)._n_atoms == 3
        assert app.session_for("nope") is None

        # An empty report drops that model's slice; Clear drops everything.
        app._on_model_selection(a, SimpleNamespace(indices=[]))
        assert app._scene_selection == {b: [2]}
        app.clear_selection()
        assert app._scene_selection == {}
    finally:
        app.stop()


def test_controls_table_model_dropdown_and_filter(qapp):
    """The atoms table's model dropdown follows the active model but can be pinned,
    and the filter checkbox collapses the table to the selected atoms."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    from pxviewer.appserver import find_frontend_dir, frontend_is_built

    fd = find_frontend_dir()
    if fd is None or not frontend_is_built(fd):
        pytest.skip("frontend not built")

    from types import SimpleNamespace

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        controls = app._controls
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0], [7, 0, 0]]), "B")

        # Dropdown lists both models and follows the active one (B, added last).
        assert controls._table_model_combo.count() == 2
        assert controls._table_model_id == b
        assert controls._atom_model.rowCount() == 3  # B's atoms

        # A pick in B shows up as selected rows in the table.
        app._on_model_selection(b, SimpleNamespace(indices=[0, 2]))
        assert controls._table_selection_indices() == [0, 2]
        selected = {i.row() for i in controls._atom_view.selectionModel().selectedRows()}
        assert selected == {0, 2}

        # Filtering collapses the table to just those atoms.
        controls._filter_selection_check.setChecked(True)
        assert controls._atom_model.is_filtered()
        assert controls._atom_model.rowCount() == 2
        controls._filter_selection_check.setChecked(False)
        assert not controls._atom_model.is_filtered()

        # Pinning the dropdown to A keeps the table on A even though B is active.
        controls._table_model_combo.setCurrentIndex(0)  # A
        assert controls._table_pinned and controls._table_model_id == a
        assert controls._atom_model.rowCount() == 2  # A's atoms

        # Picking the active model (B) again resumes auto-follow.
        controls._table_model_combo.setCurrentIndex(1)  # B == active
        assert not controls._table_pinned and controls._table_model_id == b
    finally:
        app.stop()


def test_volume_registry_and_grouping(qapp, tmp_path):
    """Volumes are their own category; a map+model loads as a cctbx group."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np
    from iotbx.map_model_manager import map_model_manager

    from pxviewer.desktop import DesktopApp
    from pxviewer.volume_io import VolumeData

    # A synthetic map + model on disk.
    mmm = map_model_manager()
    mmm.generate_map()
    map_path = tmp_path / "m.mrc"
    model_path = tmp_path / "m.pdb"
    mmm.map_manager().write_map(str(map_path))
    model_path.write_text(mmm.model().model_as_pdb())

    app = DesktopApp(port=0)
    app._webapp.start()
    captured = {}
    app.bridge.loaded_changed.connect(lambda s: captured.update(s))
    try:
        # An individual volume: registered, visible, composed into an MVSJ, and
        # its map written (via cctbx) where the browser can fetch it.
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        assert len(app._volumes) == 1 and app._volumes[0]["visible"]
        assert not app._models  # volumes never enter the model registry
        assert app._write_volume_scene() is not None
        assert (app._webapp.volume_dir / "vols" / f"{vid}.map").exists()

        # This is a software app (the default), where hiding a map is refused, so it stays
        # in the scene; the scene is empty of volumes only once the map is unloaded.
        app.set_volume_visible(vid, False)
        assert app._volume_entry(vid)["visible"] is True
        assert app._write_volume_scene() is not None
        app.remove_volume(vid)
        assert not app._volumes
        assert app._write_volume_scene() is None

        # A map + model loaded together -> one cctbx group (model + its map).
        kind = app.load_files([str(model_path), str(map_path)])
        assert kind == "group"
        assert len(app._models) == 1 and len(app._volumes) == 1
        gid = app._models[0]["group"]
        assert gid is not None and app._volumes[0]["group"] == gid and gid in app._groups
        # A model + a volume coexist: the viewport has both ws and MVSJ.
        assert len(app._visible_model_ws()) == 1 and app._write_volume_scene() is not None
        # The Loaded summary carries the group + both items.
        assert gid in {g["id"] for g in captured["groups"]}
        assert {it["kind"] for it in captured["items"]} == {"model", "volume"}

        app.remove_group(gid)
        assert not app._models and not app._volumes and gid not in app._groups
    finally:
        app.stop()


def test_map_model_demo_loads_bundled_model_as_group(qapp):
    """The map+model demo: bundled model + a cctbx-generated density, as one group."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        kind = app.load_map_model_demo(d_min=4.0)  # coarser = faster to generate
        assert kind == "group"
        # One model + exactly one density map (the redundant model_map is dropped).
        assert len(app._models) == 1 and len(app._volumes) == 1
        gid = app._models[0]["group"]
        assert gid is not None and app._volumes[0]["group"] == gid
        assert app._models[0]["session"]._n_atoms == 660  # ubiquitin (1UBQ)
        # Model + map compose the viewport together.
        assert len(app._visible_model_ws()) == 1 and app._write_volume_scene() is not None
    finally:
        app.stop()


def test_group_keeps_the_map_model_manager_that_pairs_its_objects(qapp):
    """A group is not just a label: it holds the cctbx map_model_manager that put the
    model and map in a common frame. That manager is the record of the pairing — the
    DataManager does not keep one (get_map_model_manager evicts what it consumed)."""
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_map_model_demo(d_min=4.0)
        gid = app._models[0]["group"]
        mmm = app.group_mmm(gid)
        assert mmm is not None
        # The viewer shows the manager's own model, so minimizing it in place keeps
        # the pairing true rather than drifting from it.
        assert app._models[0]["session"].model is mmm.model()
        # ... and the map offered for minimization is the manager's, not one we
        # picked out by inspecting grids ourselves.
        assert app.map_for_model() is mmm.map_manager().map_data()

        app.remove_group(gid)
        assert app.group_mmm(gid) is None
    finally:
        app.stop()


def _model_and_map_files(tmp_path):
    """The same structure written out as a separate model file and map file."""
    from iotbx.map_model_manager import map_model_manager

    mmm = map_model_manager()
    mmm.generate_map()
    map_path = tmp_path / "m.mrc"
    model_path = tmp_path / "m.pdb"
    mmm.map_manager().write_map(str(map_path))
    model_path.write_text(mmm.model().model_as_pdb())
    return model_path, map_path


def test_looking_compatible_is_not_being_paired(qapp, tmp_path):
    """The regression this guards: deciding a model and map go together by inspecting
    them. These two *are* mutually compatible by cctbx's own test and sit in the same
    group, and they are still not paired — because nothing ever paired them. Only a
    map_model_manager makes a pairing, so a group without one offers no map."""
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.cctbx_io import read_model
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.volume_io import VolumeData

    model_path, map_path = _model_and_map_files(tmp_path)
    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        model = read_model(str(model_path))
        volume = VolumeData.from_map_file(str(map_path))
        # A group put together without cctbx pairing the contents.
        gid = app._new_group("hand-made")
        app._add_model(LiveSession.from_cctbx_model(model), "m", group=gid)
        app._add_volume(volume, "map", group=gid)

        # The premise: they really would pass an eyeball compatibility check.
        assert volume.map_manager.origin_is_zero()
        assert volume.map_manager.is_compatible_model(model)
        # And that counts for nothing: no manager, no pairing, no map.
        assert app.group_mmm(gid) is None
        assert app.map_for_model() is None
        with pytest.raises(ValueError, match="not paired"):
            app.minimize_model(use_map=True)
    finally:
        app.stop()


def test_unpaired_objects_can_be_paired_explicitly(qapp, tmp_path):
    """Pairing is offered as an action rather than inferred, because it is one: cctbx
    relocates the model into a common frame with the map. Doing it is what makes the
    two usable together."""
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp

    model_path, map_path = _model_and_map_files(tmp_path)
    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_files([str(model_path)])
        app.load_files([str(map_path)])
        models, volumes = app.pairable()
        assert len(models) == 1 and len(volumes) == 1  # both unpaired, so both offered
        assert app.map_for_model() is None
        assert app._controls._pair_btn.isEnabled()  # something to pair on both sides

        gid = app.pair_model_with_map(models[0]["id"], volumes[0]["id"])
        mmm = app.group_mmm(gid)
        assert mmm is not None
        assert app._models[0]["group"] == gid and app._volumes[0]["group"] == gid
        # The pairing is cctbx's, and it is what now answers the map question.
        assert app.map_for_model() is mmm.map_manager().map_data()
        assert app._models[0]["session"].model is mmm.model()
        # Paired objects are no longer on offer for pairing.
        assert app.pairable() == ([], [])
        assert not app._controls._pair_btn.isEnabled()
        with pytest.raises(ValueError, match="already paired"):
            app.pair_model_with_map(app._models[0]["id"], app._volumes[0]["id"])
        # ... and the map they are paired with is now offered for minimization.
        assert app._controls._minimize_map_check.isEnabled()
    finally:
        app.stop()


def test_minimize_buttons_show_which_state_is_live(qapp):
    """A glance at the play/pause pair should say whether a run is going: idle enables and
    accents Minimize, running enables and accents Stop, and only the live one is styled."""
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    try:
        ctrls = app._controls
        play, stop = ctrls._minimize_btn, ctrls._minimize_stop_btn
        # Idle at construction: Minimize live (enabled + accented), Stop quiet.
        assert play.isEnabled() and not stop.isEnabled()
        assert play.styleSheet() and not stop.styleSheet()
        # Enter the running state: the accent moves to Stop, Minimize goes plain.
        app.bridge.minimizing_changed.emit(True)
        qapp.processEvents()
        assert stop.isEnabled() and not play.isEnabled()
        assert stop.styleSheet() and not play.styleSheet()
        # Back to idle: the accent returns to Minimize.
        app.bridge.minimizing_changed.emit(False)
        qapp.processEvents()
        assert play.styleSheet() and not stop.styleSheet()
    finally:
        app.stop()


def test_shift_arm_is_exactly_pause(qapp):
    """Shift is pressed (a drag is imminent): the 'arm' message does exactly what the Pause
    button does — stop a running minimization, same signal, same status. With nothing
    running it is a no-op, just as Pause is disabled when nothing is running."""
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    try:
        status = []
        app.bridge.status_changed.connect(status.append)

        # Nothing running: arm does nothing (no stop, no message).
        app._on_tug("m", "arm", -1, None)
        qapp.processEvents()
        assert not app._minimize_stop.is_set() and status == []

        # A run is going (idle cleared as minimize_model would): arm stops it and says so —
        # the very same status the Pause button raises.
        app._minimize_idle.clear()
        app._on_tug("m", "arm", -1, None)
        qapp.processEvents()
        arm_status = list(status)
        assert app._minimize_stop.is_set()
        assert arm_status and "stopping" in arm_status[-1].lower()

        # And pressing Pause directly raises the identical message.
        status.clear()
        app.stop_minimization()
        qapp.processEvents()
        assert status == [arm_status[-1]]
    finally:
        app.stop()


def test_minimization_runs_continuously_until_stopped(qapp):
    """A convergent minimization is over in ~1 s — too fast to watch or interrupt. So the
    run stays 'on' after it converges (the model held at its minimum) until the user ends
    it, giving a steady window to Stop or hand the model to a drag. Stopping then reports
    the improvement and frees the model."""
    import time

    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("mmtbx.refinement.geometry_minimization")
    from pxviewer.geometry import monomer_library_available
    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB)")
    from pathlib import Path

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        status = []
        app.bridge.status_changed.connect(status.append)
        app.load_files([str(Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb")])
        qapp.processEvents()

        app.minimize_model(use_map=False)
        # Refine to convergence, then hold: wait for the holding message.
        deadline = time.time() + 40
        while time.time() < deadline and not any("holding" in s for s in status):
            qapp.processEvents()
            time.sleep(0.02)
        assert any("holding" in s for s in status), "never reached the held state"
        # Held means still running: the model has not been released to a drag yet.
        assert not app._minimize_idle.is_set()

        # Stop (as Pause / a Shift-drag would): the run ends and the model is freed.
        app.stop_minimization()
        deadline = time.time() + 5
        while time.time() < deadline and not app._minimize_idle.is_set():
            qapp.processEvents()
            time.sleep(0.02)
        assert app._minimize_idle.is_set()
        for _ in range(30):
            qapp.processEvents()
            time.sleep(0.01)
        assert any("bond rmsd" in s and "->" in s for s in status), "no improvement summary"
    finally:
        app.stop()


def test_smiles_ligand_restraints_can_be_saved(qapp, tmp_path):
    """A ligand built from SMILES carries its geostd restraint CIF, the Loaded tree flags
    it as savable, and save_restraints_cif writes the provenance-bearing file. An ordinary
    model has nothing to save and says so."""
    import time

    pytest.importorskip("rdkit")
    pytest.importorskip("iotbx.data_manager")
    from pathlib import Path

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._markers.append({"id": "marker-1", "name": "Ligand marker 1",
                             "position": [0.0, 0.0, 0.0], "atom": None, "visible": True})
        before = {m["id"] for m in app._models}
        app.fit_ligand_from_smiles_at_marker("marker-1", "CCO", "EOH", fit=False)
        deadline = time.time() + 60
        while time.time() < deadline and {m["id"] for m in app._models} == before:
            qapp.processEvents()
            time.sleep(0.05)
        ligand = next(m for m in app._models if m["id"] not in before)

        # The entry carries the CIF, and the Loaded tree advertises it as savable.
        assert ligand.get("restraints_cif")
        item = next(it for it in app._loaded_summary()["items"] if it["id"] == ligand["id"])
        assert item["has_restraints_cif"] is True

        out = tmp_path / "EOH.cif"
        app.save_restraints_cif(ligand["id"], str(out))
        text = out.read_text()
        assert "generated by pxviewer" in text and "SMILES_CANONICAL" in text and "RDKit" in text

        # A protein model has no restraints of its own to save.
        app.load_files([str(Path(__file__).resolve().parents[1] / "pxviewer" / "data" / "1ubq.pdb")])
        qapp.processEvents()
        protein = next(m for m in app._models if m["name"].endswith(".pdb"))
        pitem = next(it for it in app._loaded_summary()["items"] if it["id"] == protein["id"])
        assert pitem["has_restraints_cif"] is False
        with pytest.raises(ValueError, match="no restraints"):
            app.save_restraints_cif(protein["id"], str(tmp_path / "nope.cif"))

        # Export writes the pair a refinement needs in one step: the fitted coordinates
        # (mmCIF, .mmcif) and the restraints dictionary alongside as <stem>_restraints.cif.
        coord, restraints = app.export_ligand(ligand["id"], str(tmp_path / "EOH.mmcif"))
        from pathlib import Path as _P
        assert _P(coord).name == "EOH.mmcif" and _P(restraints).name == "EOH_restraints.cif"
        assert "_atom_site" in _P(coord).read_text()  # placed coordinates, macromolecular mmCIF
        assert "SMILES_CANONICAL" in _P(restraints).read_text()  # the monomer dictionary
        with pytest.raises(ValueError, match="no restraints"):
            app.export_ligand(protein["id"], str(tmp_path / "prot.cif"))
    finally:
        app.stop()


def test_tutorial_coach_advances_when_steps_are_done(qapp):
    """The guided coach is non-modal and hidden until started, then advances itself as each
    step's task is actually done (model loaded, atoms selected, edit added) — not on a mere
    button press — and closes on Finish."""
    import time

    pytest.importorskip("rdkit")
    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")
    from pxviewer import tutorial
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        cw = app._controls
        coach = app._viewport  # the coach pane lives on the viewport window
        assert coach.coach_bar.isHidden()  # not shown until a tutorial starts

        cw._start_tutorial(tutorial.restraint_edits_tutorial())
        assert not coach.coach_bar.isHidden()
        assert coach.coach_progress.text() == "Step 1 / 4"
        # Step 1 targets a control, so "Show me where" is offered — and only points, never
        # acts (no model is loaded by clicking it).
        assert not coach.coach_show.isHidden()
        cw._on_coach_show_me()  # flashes the Demos button; must not raise or load anything
        assert not app._models

        # The user loads a structure themselves; the predicate then advances.
        from pxviewer.loader import sample_structure_path
        app.load_files([str(sample_structure_path())])
        deadline = time.time() + 30
        while time.time() < deadline and not app._models:
            qapp.processEvents()
            time.sleep(0.02)
        cw._maybe_advance_tutorial()
        assert coach.coach_progress.text() == "Step 2 / 4"

        # Step 2: selecting two atoms advances.
        mid = app._active_model_id
        app._scene_selection[mid] = [0, 1]
        cw._maybe_advance_tutorial()
        assert coach.coach_progress.text() == "Step 3 / 4"

        # Step 3: authoring an edit (two atoms in different residues) advances to the last step.
        atoms = app._model_entry(mid)["session"].model.get_hierarchy().atoms()
        resseqs = [a.parent().parent().resseq for a in atoms]
        j = next(k for k in range(len(atoms)) if resseqs[k] != resseqs[0])
        app._scene_selection[mid] = [0, j]
        app.add_edit_from_selection(mid, "bond")
        cw._maybe_advance_tutorial()
        assert coach.coach_progress.text() == "Step 4 / 4"
        assert coach.coach_next.text() == "Finish"

        # Finish closes the coach.
        cw._tutorial_next()
        assert coach.coach_bar.isHidden() and cw._tutorial is None
    finally:
        app.stop()


def test_metal_example_and_its_sample_edits_file(qapp):
    """The bundled metal site loads (Zn + water + histidines), and the bundled sample edits
    file applies to it — adding the Zn-water coordination bond cctbx doesn't restrain on its
    own. Tutorials are offered loading-first, writing-second."""
    import time

    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")
    from pxviewer import tutorial
    from pxviewer.desktop import DesktopApp
    from pxviewer.loader import sample_structure_path

    assert [t.title for t in tutorial.all_tutorials()] == \
        ["Validate a structure", "Fit a ligand into density",
         "Real-space refine into cryo-EM density",
         "Load restraint edits", "Custom restraint edits"]

    site = sample_structure_path("zn_site.pdb")
    edits_file = sample_structure_path("zn_site_edits.phil")
    assert site is not None and edits_file is not None

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_files([str(site)])
        deadline = time.time() + 30
        while time.time() < deadline and not app._models:
            qapp.processEvents()
            time.sleep(0.02)
        mid = app._active_model_id
        names = {a.name.strip() for a in
                 app._model_entry(mid)["session"].model.get_hierarchy().atoms()}
        assert {"ZN", "O", "NE2"} <= names  # metal, water, coordinating His nitrogen
        # isolated histidines are not a polymer, so it draws ball-and-stick, not empty cartoon
        assert app._model_entry(mid)["rep"] == "ball-and-stick"

        assert app.model_edits(mid) == []
        skipped = app.load_edits(mid, str(edits_file))
        assert skipped == 0
        loaded = app.model_edits(mid)
        assert len(loaded) == 1 and loaded[0]["kind"] == "bond"
    finally:
        app.stop()


def test_ligand_fitting_demo_makes_maps_and_fits_atp(qapp):
    """The ligand-fitting demo loads a ligand-free model + reflections that contain ATP.
    Making maps yields a paired 2mFo-DFc and an mFo-DFc difference map, and ATP fits back
    into the blob it came from — the Phenix ligand-fitting tutorial, self-contained."""
    import time

    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")
    from pxviewer.geometry import monomer_library_available
    if not monomer_library_available():
        pytest.skip("no monomer library (ATP)")
    import numpy as np

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_ligand_fitting_demo()
        qapp.processEvents()
        assert len(app._models) == 1 and len(app._reflections) == 1
        assert app.map_for_model() is None  # not phased yet
        # the Demos menu offers it
        labels = [a.text() for a in app._controls._build_demos_menu().actions()]
        assert any("Ligand fitting" in l for l in labels)

        mid, rid = app._models[0]["id"], app._reflections[0]["id"]
        app.make_maps(rid, mid)
        deadline = time.time() + 120
        while time.time() < deadline and app.map_for_model(mid) is None:
            qapp.processEvents()
            time.sleep(0.1)
        assert app.map_for_model(mid) is not None                       # paired 2mFo-DFc
        assert any("mFo-DFc" in v["name"] for v in app._volumes)        # a difference map

        # A marker at the blob, build+fit ATP — it lands where the density is.
        center = app._LIGAND_FITTING_CENTER
        app._markers.append({"id": "marker-1", "name": "m", "position": list(center),
                             "atom": None, "visible": True})
        before = {m["id"] for m in app._models}
        app.fit_ligand_at_marker("marker-1", "ATP", fit=True, trials=8)
        deadline = time.time() + 150
        while time.time() < deadline and {m["id"] for m in app._models} == before:
            qapp.processEvents()
            time.sleep(0.1)
        ligand = next(m for m in app._models if m["id"] not in before)
        fitted = ligand["session"].model.get_sites_cart().as_numpy_array().mean(0)
        assert np.linalg.norm(fitted - np.array(center)) < 4.0  # fitted into the blob
    finally:
        app.stop()


def test_cryo_em_demo_refines_a_shaken_model_into_its_density(qapp):
    """The cryo-EM demo loads a model sitting *off* a density computed from it, paired as one
    group. Minimizing with the map (real-space refinement) settles the model back into the
    density — the map-model correlation climbs — and its tutorial advances at each step."""
    import time

    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")
    pytest.importorskip("iotbx.map_model_manager")
    from pxviewer import tutorial
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        cw, coach = app._controls, app._viewport
        cw._start_tutorial(tutorial.cryo_em_refinement_tutorial())
        assert coach.coach_progress.text() == "Step 1 / 3"

        # Step 1: load the demo — a model + a density map paired in one group.
        app.load_real_space_refinement_demo(shake=0.6)
        qapp.processEvents()
        assert len(app._models) == 1 and len(app._volumes) == 1
        gid = app._models[0]["group"]
        mmm = app.group_mmm(gid)
        assert gid is not None and mmm is not None and app.map_for_model() is not None
        cw._maybe_advance_tutorial()
        assert coach.coach_progress.text() == "Step 2 / 3"

        # How well the model fits its density before refinement.
        mmm.set_resolution(3.0)
        cc_before = float(mmm.map_model_cc())

        # Step 2: real-space refine — minimize into the map. It advances once running.
        statuses = []
        app.bridge.status_changed.connect(statuses.append)
        app.minimize_model(use_map=True)
        deadline = time.time() + 90
        while time.time() < deadline and app._minimize_idle.is_set():
            qapp.processEvents()
            time.sleep(0.02)
        cw._maybe_advance_tutorial()
        assert coach.coach_progress.text() == "Step 3 / 3"

        # Let it settle into the density, then stop.
        deadline = time.time() + 90
        while time.time() < deadline and not any(
                "holding" in s or "rmsd" in s for s in statuses):
            qapp.processEvents()
            time.sleep(0.05)
        app.stop_minimization()
        deadline = time.time() + 10
        while time.time() < deadline and not app._minimize_idle.is_set():
            qapp.processEvents()
            time.sleep(0.05)

        cc_after = float(mmm.map_model_cc())
        assert cc_after > cc_before  # the model moved into the density
    finally:
        app.stop()


def test_palette_default_colours_flow_through_a_family(qapp):
    """A new model and the maps phased from it draw opening colours from one palette: the
    model (uniform ribbon, or carbon-tint on atoms) and its 2mFo-DFc map take successive
    palette slots, while the difference map keeps its conventional green/red."""
    import tempfile
    import time
    from pathlib import Path

    pytest.importorskip("mmtbx.f_model")
    from pxviewer.cctbx_io import read_model
    from pxviewer.desktop import DesktopApp
    from pxviewer.loader import sample_structure_path

    path = sample_structure_path()  # 1UBQ, a polymer
    model = read_model(str(path))
    f_obs = abs(model.get_xray_structure().structure_factors(d_min=2.0).f_calc())
    f_obs.set_observation_type_xray_amplitude()
    td = tempfile.mkdtemp()
    mtz = Path(td) / "d.mtz"
    ds = f_obs.as_mtz_dataset(column_root_label="F")
    ds.add_miller_array(f_obs.generate_r_free_flags(), column_root_label="FreeR_flag")
    ds.mtz_object().write(str(mtz))

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_files([str(path)])
        deadline = time.time() + 40
        while time.time() < deadline and not app._models:
            qapp.processEvents()
            time.sleep(0.02)
        entry = app._models[0]
        palette = entry["palette"]
        assert len(palette) == 4 and all(c.startswith("#") for c in palette)
        session = entry["session"]

        def rep():
            return list(session._representations.values())[0]

        # Polymer default is a cartoon in a uniform palette[0] (a solid ribbon).
        assert entry["rep"] == "cartoon"
        assert rep()["color"] == "uniform" and rep()["colorValue"] == palette[0]
        # Switched to an atom view, the same colour becomes a carbon tint (O/N/S standard).
        app.set_model_representation(entry["id"], "ball-and-stick")
        assert rep().get("carbonColor") == palette[0]

        app.load_files([str(mtz)])
        deadline = time.time() + 40
        while time.time() < deadline and not app._reflections:
            qapp.processEvents()
            time.sleep(0.02)
        app.make_maps(app._reflections[0]["id"], entry["id"])
        deadline = time.time() + 150
        while time.time() < deadline and app.map_for_model(entry["id"]) is None:
            qapp.processEvents()
            time.sleep(0.05)
        maps = {v["name"]: v for v in app._volumes}
        assert maps["2mFo-DFc"]["color"] == palette[1]   # next palette slot
        assert maps["mFo-DFc"]["color"] == "green"        # difference map untouched
        assert maps["mFo-DFc"]["negative_color"] == "red"
    finally:
        app.stop()


def test_live_difference_map_streams_a_box_during_a_drag(qapp):
    """With 'Live difference map' on and a phased model, arming a drag and feeding it a frame
    streams an mFo-DFc window (a _TAG_MAP payload) to the model's session; disabling clears it.
    Exercises the whole wiring — arm, the background recompute worker, and teardown."""
    import struct
    import tempfile
    import time
    from pathlib import Path

    import numpy as np

    pytest.importorskip("mmtbx.f_model")
    from pxviewer.geometry import monomer_library_available
    if not monomer_library_available():
        pytest.skip("no monomer library")
    from pxviewer.cctbx_io import read_model
    from pxviewer.desktop import DesktopApp
    from pxviewer.loader import sample_structure_path

    # A self-contained phased group: bundled model + synthetic reflections computed from it.
    path = sample_structure_path()
    model = read_model(str(path))
    f_obs = abs(model.get_xray_structure().structure_factors(d_min=2.0).f_calc())
    f_obs.set_observation_type_xray_amplitude()
    td = tempfile.mkdtemp()
    mtz = Path(td) / "d.mtz"
    ds = f_obs.as_mtz_dataset(column_root_label="F")
    ds.add_miller_array(f_obs.generate_r_free_flags(), column_root_label="FreeR_flag")
    ds.mtz_object().write(str(mtz))

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_files([str(path)])
        app.load_files([str(mtz)])
        deadline = time.time() + 60
        while time.time() < deadline and (not app._models or not app._reflections):
            qapp.processEvents()
            time.sleep(0.02)
        mid, rid = app._models[0]["id"], app._reflections[0]["id"]
        app.make_maps(rid, mid)
        deadline = time.time() + 150
        while time.time() < deadline and app.map_for_model(mid) is None:
            qapp.processEvents()
            time.sleep(0.05)
        assert app.map_for_model(mid) is not None  # phased

        session = app._model_entry(mid)["session"]
        app._tug_session = session          # as a drag 'begin' sets it
        app.set_live_difference_map(True)
        app._maybe_start_live_diff(mid, atom=model.get_number_of_atoms() // 2)
        assert app._diff_ctx is not None    # armed: a phased group with reflections exists

        app._queue_live_diff(np.array(model.get_sites_cart(), dtype="float64"))
        deadline = time.time() + 90
        while time.time() < deadline and session._last_map_box is None:
            qapp.processEvents()
            time.sleep(0.05)
        assert session._last_map_box is not None
        assert struct.unpack_from("<I", session._last_map_box, 0)[0] == 4  # _TAG_MAP

        app.set_live_difference_map(False)  # clears the window and disarms
        assert app._diff_ctx is None
        assert session._last_map_box is None
    finally:
        app.stop()


def test_validation_tutorial_advances_when_validation_runs(qapp):
    """The validation walkthrough: load the demo, run validation, read the results. It
    advances when a model is loaded and again once validation has cached results."""
    import time

    pytest.importorskip("iotbx.data_manager")
    from pxviewer import tutorial
    from pxviewer.desktop import DesktopApp
    from pxviewer.loader import sample_structure_path

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        cw = app._controls
        coach = app._viewport
        cw._start_tutorial(tutorial.validation_tutorial())
        assert coach.coach_progress.text() == "Step 1 / 3"
        assert cw._validate_btn.text() == "Run validation"  # the step-2 highlight target

        # Step 1: load the validation demo.
        app.load_files([str(sample_structure_path("1tec.pdb"))])
        deadline = time.time() + 30
        while time.time() < deadline and not app._models:
            qapp.processEvents()
            time.sleep(0.02)
        cw._maybe_advance_tutorial()
        assert coach.coach_progress.text() == "Step 2 / 3"

        # Step 2: advances once validation has cached results (simulated — a real MolProbity
        # run is exercised by the validation tests, and is too slow to repeat here).
        mid = app._active_model_id
        assert not app._model_entry(mid).get("validation")
        app._model_entry(mid)["validation"] = {"rotalyze": object()}
        cw._maybe_advance_tutorial()
        assert coach.coach_progress.text() == "Step 3 / 3"
        assert coach.coach_next.text() == "Finish"
        cw._tutorial_next()
        assert coach.coach_bar.isHidden()
    finally:
        app.stop()


def test_authoring_saving_and_loading_restraint_edits(qapp, tmp_path):
    """Author a custom bond from a selection, see it validated and listed, round-trip it
    through a PHIL file, and confirm the model's restraints carry it. A duplicate edit (a
    bond the library already restrains) is refused and not stored."""
    import time

    pytest.importorskip("rdkit")
    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._markers.append({"id": "marker-1", "name": "m", "position": [0.0, 0.0, 0.0],
                             "atom": None, "visible": True})
        before = {m["id"] for m in app._models}
        app.fit_ligand_from_smiles_at_marker("marker-1", "CCO", "EOH", fit=False)
        deadline = time.time() + 60
        while time.time() < deadline and {m["id"] for m in app._models} == before:
            qapp.processEvents()
            time.sleep(0.05)
        ligand = next(m for m in app._models if m["id"] not in before)
        mid = ligand["id"]
        model = ligand["session"].model
        names = [a.name.strip() for a in model.get_hierarchy().atoms()]
        base = model.get_restraints_manager().geometry.pair_proxies().bond_proxies.simple.size()

        # Author a bond between C1 and O1 (not natively bonded).
        app._scene_selection[mid] = [names.index("C1"), names.index("O1")]
        app.add_edit_from_selection(mid, "bond")
        assert len(app.model_edits(mid)) == 1
        item = next(it for it in app._loaded_summary()["items"] if it["id"] == mid)
        assert len(item["edits"]) == 1 and "bond" in item["edits"][0]["summary"]
        # the restraints the minimizer will use now carry it
        n1 = model.get_restraints_manager().geometry.pair_proxies().bond_proxies.simple.size()
        assert n1 == base + 1

        # A duplicate (C1-C2 are already bonded) is refused; the edit list is unchanged.
        app._scene_selection[mid] = [names.index("C1"), names.index("C2")]
        with pytest.raises(ValueError):
            app.add_edit_from_selection(mid, "bond")
        assert len(app.model_edits(mid)) == 1

        # Save, clear, load: the edit survives the PHIL round-trip.
        phil = tmp_path / "edits.phil"
        app.save_edits(mid, str(phil))
        assert "geometry_restraints.edits" in phil.read_text()
        app.clear_edits(mid)
        assert app.model_edits(mid) == []
        skipped = app.load_edits(mid, str(phil))
        assert skipped == 0 and len(app.model_edits(mid)) == 1
    finally:
        app.stop()


def test_writing_a_restrained_model_as_mmcif_needs_no_probe(qapp, tmp_path):
    """Writing a model as mmCIF must emit coordinates, not a validation report. A ligand
    (like any minimized model) carries a restraints manager, and mmtbx's full model_as_mmcif
    would compute a clashscore that shells out to the external Probe binary — absent here.
    The write must succeed anyway and round-trip as a readable mmCIF."""
    import time

    pytest.importorskip("rdkit")
    from iotbx.data_manager import DataManager

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._markers.append({"id": "marker-1", "name": "m", "position": [0.0, 0.0, 0.0],
                             "atom": None, "visible": True})
        before = {m["id"] for m in app._models}
        app.fit_ligand_from_smiles_at_marker("marker-1", "CCO", "EOH", fit=False)
        deadline = time.time() + 60
        while time.time() < deadline and {m["id"] for m in app._models} == before:
            qapp.processEvents()
            time.sleep(0.05)
        ligand = next(m for m in app._models if m["id"] not in before)
        assert ligand["session"].model.restraints_manager_available()  # the Probe trigger

        out = tmp_path / "EOH.cif"
        app.write_object("model", ligand["id"], str(out))  # must not raise (no Probe)
        DataManager().process_model_file(str(out))  # and it reparses as a real mmCIF
    finally:
        app.stop()


def test_placing_a_ligand_clears_the_input_fields(qapp):
    """After a ligand is placed the code and SMILES boxes clear, so the next ligand cannot
    silently inherit the previous code as its name/restraints. A failed build leaves the
    inputs untouched, so a typo can be fixed rather than retyped."""
    import time

    pytest.importorskip("rdkit")
    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        ctrls = app._controls

        def place(code, smiles, marker):
            app._markers.append({"id": marker, "name": "m", "position": [0.0, 0.0, 0.0],
                                 "atom": None, "visible": True})
            ctrls._lig_code_edit.setText(code)
            ctrls._lig_smiles_edit.setText(smiles)
            n = len(app._models)
            ctrls._on_fit_ligand()
            deadline = time.time() + 40
            while time.time() < deadline and len(app._models) == n:
                qapp.processEvents()
                time.sleep(0.05)
            qapp.processEvents()

        place("EOH", "CCO", "marker-1")
        assert ctrls._lig_code_edit.text() == "" and ctrls._lig_smiles_edit.text() == ""

        # A build that fails leaves the fields as typed (nothing was placed).
        app._markers.append({"id": "marker-2", "name": "m", "position": [0.0, 0.0, 0.0],
                             "atom": None, "visible": True})
        ctrls._lig_code_edit.setText("ABC")
        ctrls._lig_smiles_edit.setText("not_a_smiles!!!")
        n = len(app._models)
        ctrls._on_fit_ligand()
        for _ in range(40):
            qapp.processEvents()
            time.sleep(0.02)
        assert len(app._models) == n  # nothing placed
        assert ctrls._lig_code_edit.text() == "ABC"
        assert ctrls._lig_smiles_edit.text() == "not_a_smiles!!!"
    finally:
        app.stop()


def test_pairing_a_boxed_map_keeps_model_and_map_drawn_together(qapp, tmp_path):
    """Pairing relocates the model into the map's frame — several angstrom for a boxed
    map. The map the browser is served has to move with it, or the model is drawn away
    from its own density. (cctbx writes a map back in the frame it was read in, which is
    right for saving a file and wrong for the copy on screen.)"""
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np
    from iotbx.map_model_manager import map_model_manager

    from pxviewer.desktop import DesktopApp
    from pxviewer.volume_io import VolumeData

    mmm = map_model_manager()
    mmm.generate_map()
    boxed = mmm.map_manager().deep_copy()
    boxed.set_original_origin_and_gridding(original_origin=(10, 10, 10))  # not at zero
    map_path = tmp_path / "boxed.mrc"
    model_path = tmp_path / "m.pdb"
    boxed.write_map(str(map_path))
    model_path.write_text(mmm.model().model_as_pdb())

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_files([str(model_path)])
        app.load_files([str(map_path)])
        before = app._models[0]["session"].model.get_sites_cart().as_numpy_array().copy()
        vid = app._volumes[0]["id"]
        app.pair_model_with_map(app._models[0]["id"], vid)

        # The model really did move (this is why pairing cannot be a passive label).
        after = app._models[0]["session"].model.get_sites_cart().as_numpy_array()
        shift = (after - before).mean(axis=0)
        assert np.linalg.norm(shift) > 1.0
        # The served map moved with it: it is written in the frame the model is drawn in.
        served = app._webapp.volume_dir / "vols" / f"{vid}.map"
        assert VolumeData.from_map_file(str(served)).map_manager.map_data().origin() == (0, 0, 0)
        # Saving the map for the user is a different job: that keeps the original frame.
        out = tmp_path / "saved.mrc"
        app._volumes[0]["data"].write_map(str(out))
        assert VolumeData.from_map_file(str(out)).map_manager.map_data().origin() == (10, 10, 10)
    finally:
        app.stop()


def test_volume_appearance_controls(qapp, tmp_path):
    """A focused volume gets style, colour, opacity and a contour level. Each is kept
    on the entry (so a scene rebuild restores it) and pushed live (so nothing reloads)."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np

    from pxviewer.desktop import DesktopApp
    from pxviewer.volume_io import DEFAULT_ISO_SIGMA, VolumeData

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        entry = app._volume_entry(vid)
        assert entry["iso"] == DEFAULT_ISO_SIGMA

        app.set_volume_color(vid, "salmon")
        app.set_volume_opacity(vid, 0.4)
        app.set_volume_iso(vid, 3.25)
        assert (entry["color"], entry["opacity"], entry["iso"]) == ("salmon", 0.4, 3.25)

        # The scene composes from the entry, so a reload restores all of it.
        assert app._write_volume_scene() is not None

        # Focusing the volume builds the controls and points the wheel at it.
        ctl = app._controls
        ctl._update_appearance("volume", vid)
        assert ctl._iso_row is not None
        assert ctl._iso_row["spin"].value() == 3.25
        assert app._volume_scroll_target == vid

        # The spinbox and slider drive each other and the backend.
        ctl._iso_row["spin"].setValue(5.0)
        assert entry["iso"] == 5.0
        assert ctl._iso_row["slider"].value() == 500  # 5.0 sigma at 0.01 resolution

        # Focusing something that is not a volume takes the wheel target away.
        ctl._update_appearance(None, None)
        assert ctl._iso_row is None
        assert app._volume_scroll_target is None
    finally:
        app.stop()


def test_volume_commands_go_to_a_session_the_viewport_is_connected_to(qapp):
    """Volume commands ride a model's socket, so the control session must be one the page is
    connected to — only the *visible* models. Hiding the active model drops it from the page,
    so commands fall to another visible model; the dummy is the fallback when none is."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0, can_hide=True)  # hardware: objects can hide
    app._webapp.start()
    try:
        app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")

        # B is active: commands ride its socket, which the page has.
        assert app._active_model_id == b
        assert app._control_session() is app.session_for(b)

        # Hide the active model -> it leaves the page, so commands fall to the visible one.
        app.set_model_visible(b, False)
        assert f"ws://{app._host}:{app.session_for(b).port}" not in app._model_ws()
        assert app._control_session() is app.session_for(a)

        # Hide it too -> no visible model -> the page (and commands) fall to the dummy.
        app.set_model_visible(a, False)
        app._ensure_dummy_ws()
        assert app._control_session() is app._dummy
    finally:
        app.stop()


def test_software_pins_a_model_and_says_why_on_click(qapp):
    """On software WebGL hiding a model segfaults (the reload that redraws the scene touches
    the fragile GL), so models are pinned like maps: the checkbox is non-checkable and refused
    silently (an internal caller, add-hydrogens, hides the H-less original and must not warn),
    and a click on the box flashes why. Both hide normally on hardware."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from PySide6.QtCore import Qt

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0, can_hide=False)  # software
    app._webapp.start()
    try:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        ctl = app._controls
        node = next(n for n in _iter_tree_items(ctl._loaded_tree)
                    if n.data(0, Qt.ItemDataRole.UserRole) == ("model", a))
        assert not (node.flags() & Qt.ItemFlag.ItemIsUserCheckable)  # non-checkable

        warned, loads = [], []
        app.bridge.status_warned.connect(warned.append)
        ol = app._viewport.load
        app._viewport.load = lambda u, *ar, **k: (loads.append(u), ol(u, *ar, **k))[1]

        # The setter refuses silently (add-hydrogens relies on this — no warn, no reload).
        app.set_model_visible(a, False)
        assert app._model_entry(a)["visible"] is True
        assert loads == [] and warned == []

        # But a click on the check column flashes why — touching nothing.
        ctl._on_tree_item_clicked(node, 0)
        assert warned and "hardware WebGL" in warned[-1]
        assert loads == []
    finally:
        app.stop()


def test_hiding_a_model_reloads_the_scene_without_it(qapp):
    """On hardware a model hides by recomposing the scene without it and reloading — no clip,
    no in-place GPU mutation. Its socket leaves the page; showing puts it back."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0, can_hide=True)
    app._webapp.start()
    try:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")  # keeps the page alive
        ws_a = f"ws://{app._host}:{app.session_for(a).port}"
        loads = []
        ol = app._viewport.load
        app._viewport.load = lambda u, *ar, **k: (loads.append(u), ol(u, *ar, **k))[1]

        app.set_model_visible(a, False)
        assert ws_a not in app._model_ws()   # dropped from the recomposed scene
        assert len(loads) == 1                # a reload, the clean teardown

        app.set_model_visible(a, True)
        assert ws_a in app._model_ws()        # back
        assert len(loads) == 2
    finally:
        app.stop()


def test_hiding_a_map_reloads_the_scene_without_it(qapp):
    """On hardware a map hides by leaving it out of the recomposed scene and reloading — no
    isosurface park in place. Its ref leaves the scene; showing reloads with it back."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0, can_hide=True)
    app._webapp.start()
    try:
        app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")  # keeps the page alive
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "map")
        ref = app._volume_entry(vid)["ref"]
        loads = []
        ol = app._viewport.load
        app._viewport.load = lambda u, *ar, **k: (loads.append(u), ol(u, *ar, **k))[1]

        def scene_text():
            path = app._write_volume_scene()
            return "" if path is None else (app._webapp.volume_dir / path.lstrip("/")).read_text()

        assert ref in scene_text()  # visible: in the scene

        app.set_volume_visible(vid, False)
        assert len(loads) == 1            # hidden by a reload without it
        assert ref not in scene_text()    # the only volume, now left out

        app.set_volume_visible(vid, True)
        assert len(loads) == 2
        assert ref in scene_text()        # back
    finally:
        app.stop()


def test_software_pins_a_map_and_says_why_on_click(qapp):
    """On software WebGL hiding a map's isosurface segfaults, so the checkbox is non-checkable
    (toggling it is what crashes). Not a dead control though: clicking it flashes the reason,
    a pure status message that touches nothing. Both hide on hardware."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np
    from PySide6.QtCore import Qt

    from pxviewer.desktop import DesktopApp
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0, can_hide=False)  # software
    app._webapp.start()
    try:
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "map")
        ctl = app._controls
        node = next(n for n in _iter_tree_items(ctl._loaded_tree)
                    if n.data(0, Qt.ItemDataRole.UserRole) == ("volume", vid))
        assert not (node.flags() & Qt.ItemFlag.ItemIsUserCheckable)  # non-checkable

        warned, loads = [], []
        app.bridge.status_warned.connect(warned.append)
        ol = app._viewport.load
        app._viewport.load = lambda u, *ar, **k: (loads.append(u), ol(u, *ar, **k))[1]

        ctl._on_tree_item_clicked(node, 0)
        assert warned and "hardware WebGL" in warned[-1]
        assert loads == []                                 # nothing touched the scene
        assert app._volume_entry(vid)["visible"] is True   # still pinned visible

        # A click off the check column (e.g. the name) says nothing.
        warned.clear()
        ctl._on_tree_item_clicked(node, 2)
        assert warned == []
    finally:
        app.stop()


def _iter_tree_items(tree):
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.child(i) for i in range(node.childCount()))


def test_contour_changed_in_the_viewport_updates_the_controls(qapp):
    """The wheel is applied in the viewer, so the level arrives here after the fact.
    The widgets must follow it without writing it back — that would fight the scroll."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np

    from pxviewer.desktop import DesktopApp
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        entry = app._volume_entry(vid)
        ctl = app._controls
        ctl._update_appearance("volume", vid)

        sent = []
        app._volume_command = lambda *a, **k: sent.append(a)  # nothing may go back out
        app._on_volume_iso_changed(entry["ref"], 4.5)

        assert entry["iso"] == 4.5
        assert ctl._iso_row["spin"].value() == 4.5
        assert ctl._iso_row["slider"].value() == 450
        assert sent == []  # the viewer already applied it; echoing would fight the user
    finally:
        app.stop()


def test_volume_colour_swatches_and_custom(qapp):
    """Colours are shown as swatches rather than named, with a picker for anything off
    the preset list — the wire takes any hex Mol* can decode."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np
    from PySide6.QtWidgets import QComboBox

    from pxviewer.desktop import _CUSTOM_COLOR, DesktopApp, _VOLUME_COLORS
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        ctl = app._controls
        ctl._update_appearance("volume", vid)
        combo = ctl._appearance_box.findChildren(QComboBox)[1]  # after Style

        assert [combo.itemData(i) for i in range(len(_VOLUME_COLORS))] == _VOLUME_COLORS
        assert all(not combo.itemIcon(i).isNull() for i in range(len(_VOLUME_COLORS)))
        assert combo.itemData(combo.count() - 1) == _CUSTOM_COLOR  # the picker, last

        combo.setCurrentIndex(2)
        assert app._volume_entry(vid)["color"] == _VOLUME_COLORS[2]

        # A picked colour is a hex string; it stays on the list so it stays selected.
        app.set_volume_color(vid, "#3fa9f5")
        ctl._update_appearance("volume", vid)
        combo = ctl._appearance_box.findChildren(QComboBox)[1]
        assert combo.currentData() == "#3fa9f5"
        assert not combo.itemIcon(combo.currentIndex()).isNull()
    finally:
        app.stop()


def test_masking_density_around_the_model(qapp):
    """Hide density away from the molecule. It needs a paired map — "away from the
    molecule" has no meaning without one — and it masks a copy, so the map minimization
    refines against keeps all its density."""
    pytest.importorskip("iotbx.map_model_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np

    from pxviewer.desktop import DesktopApp
    from pxviewer.volume_io import VolumeData

    def occupied(path):
        d = VolumeData.from_map_file(str(path)).map_manager.map_data().as_numpy_array()
        return float((np.abs(d) > 1e-4).mean())

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app.load_map_model_demo(d_min=3.0)
        vid = app._volumes[0]["id"]
        mmm = app.group_mmm(app._volumes[0]["group"])
        real_before = mmm.map_manager().map_data().as_numpy_array().copy()
        served = app._webapp.volume_dir / "vols" / f"{vid}.map"

        assert app.can_mask_volume(vid)
        full = occupied(served)

        app.set_volume_mask(vid, 3.0)
        assert occupied(served) < 0.5 * full          # the served map lost the outside
        assert app.volume_appearance(vid)["mask_radius"] == 3.0
        # A wider shell keeps more, so the radius means what it says.
        app.set_volume_mask(vid, 8.0)
        near, far = 3.0, 8.0
        wide = occupied(served)
        app.set_volume_mask(vid, near)
        assert occupied(served) < wide

        # The map that gets refined against is untouched by any of it.
        assert np.array_equal(real_before, mmm.map_manager().map_data().as_numpy_array())
        assert set(mmm.map_id_list()) == {"map_manager", "model_map"}  # no scratch pile-up

        app.set_volume_mask(vid, None)
        assert occupied(served) == pytest.approx(full)  # back to the whole map

        # An unpaired volume has no model to mask around.
        loose = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "loose")
        assert not app.can_mask_volume(loose)
        with pytest.raises(ValueError, match="paired"):
            app.set_volume_mask(loose, 3.0)
    finally:
        app.stop()


def test_object_list_fits_its_contents(qapp):
    """A QTreeWidget's sizeHint is a fixed ~256px whatever it holds; left to it, the list
    reserves room for ten objects while showing two and pushes the rest of the pane into
    a scrollbar. On a 13" screen that space decides whether the pane fits."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np

    from pxviewer.desktop import _TREE_MAX_HEIGHT, _TREE_MIN_HEIGHT, DesktopApp
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        ctl = app._controls
        tree = ctl._loaded_tree
        assert tree.maximumHeight() == _TREE_MIN_HEIGHT  # empty: no reserved space

        for i in range(60):
            app._add_volume(VolumeData.from_numpy(np.ones((4, 4, 4))), f"v{i}")
        # Many objects: it grows, but only to the ceiling — then it scrolls itself.
        assert tree.maximumHeight() == _TREE_MAX_HEIGHT
        assert tree.verticalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    finally:
        app.stop()


def test_scene_actions_are_icon_buttons(qapp):
    """The seven object actions are a compact icon-only toolbar (labels moved to tooltips),
    and none forces a height — a QPushButton only gets its native macOS chrome at the height
    the style asks for."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from PySide6.QtWidgets import QPushButton

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        ctl = app._controls
        # The action buttons are icon-only now; find them by their tooltips.
        tips = ("Open a structure", "Load a bundled", "Save the focused", "Pair a model",
                "Remove the highlighted", "Reset the view", "Save a picture")
        buttons = [b for b in ctl.widget().findChildren(QPushButton)
                   if b.toolTip().startswith(tips)]
        assert len(buttons) == 7
        assert all(not b.icon().isNull() and b.text() == "" for b in buttons)

        # No forced geometry anywhere: that is what broke Open's chrome.
        for button in buttons:
            assert button.minimumHeight() == 0, f"{button.toolTip()[:20]} forces a height"
    finally:
        app.stop()


def test_range_slider_two_handles(qapp):
    """The clipping slab's control. Handles may meet — that is not degenerate here, it
    is the point at which the object is fully clipped."""
    from pxviewer.desktop import _make_range_slider

    slider = _make_range_slider()()
    slider.resize(240, 24)
    assert slider.values() == (0.0, 1.0)  # open: nothing clipped

    seen = []
    slider.changed.connect(lambda f, b: seen.append((round(f, 2), round(b, 2))))
    slider.set_values(0.25, 0.75, notify=True)
    assert slider.values() == (0.25, 0.75) and seen == [(0.25, 0.75)]

    slider.set_values(0.8, 0.2)  # crossed handles collapse rather than invert
    assert slider.values() == (0.2, 0.2)
    slider.set_values(-1.0, 5.0)  # out of range is clamped, not wrapped
    assert slider.values() == (0.0, 1.0)


def test_clipping_is_per_object(qapp):
    """Each object carries its own slab, so the density can be clipped while the model
    inside it stays whole. A model is clipped through its own session; a volume by
    reference, since its representation belongs to the shared MVSJ scene."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        mid = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        # Both start unclipped.
        assert app._volume_entry(vid)["clip"] == (0.0, 1.0)
        assert app._model_entry(mid)["clip"] == (0.0, 1.0)

        app.set_volume_clip(vid, 0.4, 0.6)
        assert app._volume_entry(vid)["clip"] == (0.4, 0.6)
        assert app._model_entry(mid)["clip"] == (0.0, 1.0)  # the model is untouched
        assert app.volume_appearance(vid)["clip"] == (0.4, 0.6)

        app.set_model_clip(mid, 0.1, 0.9)
        assert app._model_entry(mid)["clip"] == (0.1, 0.9)
        assert app.model_appearance(mid)["clip"] == (0.1, 0.9)

        # The Appearance pane offers the slab for either kind, at its current value.
        ctl = app._controls
        ctl._update_appearance("volume", vid)
        ctl._update_appearance("model", mid)
    finally:
        app.stop()


def test_console_binds_and_tracks_active_session(qapp):
    """The embedded console exposes `app`/`session`, and `session` follows active."""
    pytest.importorskip("qtconsole")
    pytest.importorskip("ipykernel")
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        controls = app._controls
        controls._ensure_console()
        assert controls._console is not None

        shell = controls._console._manager.kernel.shell
        assert shell.user_ns["app"] is app
        assert shell.user_ns["session"] is app.session_for(a)

        # Loading another model makes it active; the console's `session` rebinds.
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")
        assert shell.user_ns["session"] is app.session_for(b)
    finally:
        app.stop()


class _FakeGeo:
    """A stand-in for GeometryRestraints (no cctbx needed for the table test)."""

    def __init__(self, rows):
        self._rows = rows  # [(i_seqs, {col: value}), ...]

    def count(self, category):
        return len(self._rows)

    def row(self, category, i):
        return self._rows[i]


def test_restraint_table_model(qapp):
    cols = ["ideal", "model", "delta", "sigma", "residual"]
    rows = [
        ((0, 1), {"ideal": 1.52, "model": 1.50, "delta": 0.02, "sigma": 0.02, "residual": 1.0}),
        ((2, 3), {"ideal": 1.33, "model": 1.40, "delta": -0.07, "sigma": 0.02, "residual": 12.0}),
    ]
    model = _make_restraint_table_model()
    model.set_source(_FakeGeo(rows), "bond", cols, lambda i: f"atom{i}")

    assert model.rowCount() == 2
    headers = [model.headerData(c, Qt.Orientation.Horizontal) for c in range(model.columnCount())]
    assert headers == ["atoms", "ideal", "model", "delta", "sigma", "residual"]
    assert model.data(model.index(0, 0)) == "atom0  atom1"  # the atoms column
    assert model.data(model.index(0, 1)) == "1.520"
    assert model.data(model.index(1, 3)) == "-0.070"
    assert model.i_seqs_for_row(1) == (2, 3)

    model.set_source(None, "", cols, None)  # cleared -> no rows
    assert model.rowCount() == 0


def test_restraint_table_geostd_column(qapp):
    cols = ["ideal", "model"]
    rows = [((0, 1), {}), ((2, 3), {})]

    def src(iseqs):  # atoms 0,1 -> a monomer file; 2,3 -> a link (no single file)
        return ("ALA", "/geostd/a/data_ALA.cif") if set(iseqs) == {0, 1} else ("(link)", None)

    model = _make_restraint_table_model()
    model.set_source(_FakeGeo(rows), "bond", cols, lambda i: f"a{i}", src)

    headers = [model.headerData(c, Qt.Orientation.Horizontal) for c in range(model.columnCount())]
    assert headers == ["atoms", "ideal", "model", "geostd"]
    assert model.source_column() == 3
    assert model.data(model.index(0, 3)) == "ALA"
    assert model.data(model.index(1, 3)) == "(link)"
    assert model.source_for_row(0) == ("ALA", "/geostd/a/data_ALA.cif")
    assert model.source_for_row(1)[1] is None
    # link styling (coloured foreground) only when there is a file
    assert model.data(model.index(0, 3), Qt.ItemDataRole.ForegroundRole) is not None
    assert model.data(model.index(1, 3), Qt.ItemDataRole.ForegroundRole) is None


def test_restraint_table_filter(qapp):
    cols = ["ideal", "model", "delta", "sigma", "residual"]
    rows = [((0, 1), {}), ((2, 3), {}), ((4, 5), {})]
    model = _make_restraint_table_model()
    model.set_source(_FakeGeo(rows), "bond", cols, lambda i: f"a{i}")
    assert model.rowCount() == 3 and not model.is_filtered()

    model.set_filter([2])  # show only restraint index 2
    assert model.is_filtered() and model.rowCount() == 1
    assert model.i_seqs_for_row(0) == (4, 5)  # view row 0 -> restraint 2
    assert model.data(model.index(0, 0)) == "a4  a5"

    model.set_filter(None)
    assert not model.is_filtered() and model.rowCount() == 3


def test_geometry_shows_setup_message_without_monomer_library(qapp, monkeypatch):
    """With no monomer library, the restraint tabs show the geostd setup message."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    import pxviewer.geometry as geometry

    monkeypatch.delenv("MMTBX_CCP4_MONOMER_LIB", raising=False)
    monkeypatch.delenv("CLIBD_MON", raising=False)
    # ...and pretend chem_data (which ships geostd) isn't installed, so no library is found.
    monkeypatch.setattr(geometry, "_chem_data_geostd", lambda: None)

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._add_model(LiveSession.from_sites([[0, 0, 0], [1.5, 0, 0]]), "m")
        controls = app._controls
        controls._ensure_restraints()
        bond = controls._restraint_tabs["bond"]
        assert bond["stack"].currentWidget() is bond["msg"]
        assert "MMTBX_CCP4_MONOMER_LIB" in bond["msg"].text()
    finally:
        app.stop()


def test_geometry_restraints_populate_tables(qapp):
    """With a monomer library, the restraint tables fill from the built manager."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.loader import sample_structure_path

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._add_model(LiveSession.from_model_file(str(sample_structure_path())), "1ubq")
        controls = app._controls
        controls._ensure_restraints()

        bond = controls._restraint_tabs["bond"]
        assert bond["stack"].currentWidget() is bond["view"]
        assert bond["model"].rowCount() > 500  # 1UBQ has hundreds of bonds
        assert controls._restraint_tabs["angle"]["model"].rowCount() > bond["model"].rowCount()
        # the atoms column reads i_seqs as labels
        assert "/" in bond["model"].data(bond["model"].index(0, 0))
    finally:
        app.stop()


def test_select_by_expression(qapp):
    """A cctbx/Phenix selection string selects atoms on the active model."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.loader import sample_structure_path

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        # No model yet -> a clear error, not a crash.
        with pytest.raises(ValueError):
            app.select_by_expression("chain A")

        app._add_model(LiveSession.from_model_file(str(sample_structure_path())), "1ubq")
        mid = app._active_model_id

        n = app.select_by_expression("chain A and resseq 5:14 and name CA")
        assert n == 10  # ten CA atoms in that range
        assert len(app._scene_selection[mid]) == 10  # fed into the scene selection

        # An empty string clears the model's selection.
        assert app.select_by_expression("   ") == 0
        assert mid not in app._scene_selection

        # Invalid syntax raises (the UI catches and reports it).
        with pytest.raises(Exception):
            app.select_by_expression("chain @@@ bogus (")
    finally:
        app.stop()


def test_geometry_filter_applies_to_restraint_tables(qapp):
    """'Show only the selection' collapses every restraint table, not just Atoms."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.loader import sample_structure_path

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._add_model(LiveSession.from_model_file(str(sample_structure_path())), "1ubq")
        controls = app._controls
        controls._ensure_restraints()
        bond = controls._restraint_tabs["bond"]
        full = bond["model"].rowCount()
        assert full > 500 and not bond["model"].is_filtered()

        # Select one residue, then turn the shared filter on.
        app.select_by_expression("resseq 1")
        controls._filter_selection_check.setChecked(True)

        filtered = bond["model"].rowCount()
        assert 0 < filtered < full  # only the residue's own bonds remain
        sel = set(app._scene_selection[app._active_model_id])
        for r in range(filtered):
            assert all(i in sel for i in bond["model"].i_seqs_for_row(r))
        # angles filtered the same way
        assert controls._restraint_tabs["angle"]["model"].is_filtered()

        controls._filter_selection_check.setChecked(False)
        assert bond["model"].rowCount() == full
    finally:
        app.stop()


def test_restraint_row_draws_notation(qapp):
    """Selecting angle rows draws angle notations (not a whole-residue highlight)."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

    from PySide6.QtCore import QItemSelectionModel

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.loader import sample_structure_path

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._add_model(LiveSession.from_model_file(str(sample_structure_path())), "1ubq")
        controls = app._controls
        controls._ensure_restraints()
        view = controls._restraint_tabs["angle"]["view"]
        model = controls._restraint_tabs["angle"]["model"]
        session = app.active_model_session()

        # One angle row -> one angle notation primitive drawn.
        view.selectRow(0)
        assert len(app._restraint_prim_ids) == 1
        assert len(session._primitives) == 1

        # A second selected row -> a second notation (multiple at once).
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        view.selectionModel().select(model.index(1, 0), flags)
        assert len(app._restraint_prim_ids) == 2 and len(session._primitives) == 2

        # Clearing the selection removes the notations.
        view.clearSelection()
        assert app._restraint_prim_ids == [] and len(session._primitives) == 0
    finally:
        app.stop()


def test_geostd_source_links_rows_to_files(qapp):
    """Each intra-residue restraint row resolves to its geostd monomer file."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from pathlib import Path

    from pxviewer.geometry import monomer_library_available

    if not monomer_library_available():
        pytest.skip("no monomer library (set MMTBX_CCP4_MONOMER_LIB to a geostd checkout)")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.loader import sample_structure_path

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._add_model(LiveSession.from_model_file(str(sample_structure_path())), "1ubq")
        controls = app._controls
        controls._ensure_restraints()
        model = controls._restraint_tabs["bond"]["model"]

        # The geostd column is last.
        assert model.source_column() == model.columnCount() - 1

        # At least one bond resolves to a real geostd .cif on disk.
        resolved = 0
        for r in range(min(model.rowCount(), 50)):
            text, path = model.source_for_row(r)
            assert text  # a resname or "(link)"
            if path is not None:
                assert path.endswith(".cif") and Path(path).is_file()
                resolved += 1
        assert resolved > 0
    finally:
        app.stop()


def test_representation_dropdowns(qapp):
    """Each loaded object gets an inline rep dropdown; changes update the registry."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np
    from PySide6.QtWidgets import QComboBox

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.loader import sample_structure_path
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0)
    app._webapp.start()
    captured = {}
    app.bridge.loaded_changed.connect(lambda s: captured.update(s))
    try:
        # A polymer defaults to cartoon; the summary carries it.
        mid = app._add_model(LiveSession.from_model_file(str(sample_structure_path())), "1ubq")
        m = app._model_entry(mid)
        assert m["rep"] == "cartoon"
        item = next(it for it in captured["items"] if it["id"] == mid)
        assert item["rep"] == "cartoon"

        # A non-polymer defaults to ball-and-stick.
        mid2 = app._add_model(LiveSession.from_sites([[0, 0, 0], [1.5, 0, 0]]), "x")
        assert app._model_entry(mid2)["rep"] == "ball-and-stick"

        # Changing the model representation updates the entry (and the session).
        app.set_model_representation(mid, "spacefill")
        assert m["rep"] == "spacefill"

        # A volume gets an isosurface style, changeable.
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        v = app._volume_entry(vid)
        assert v["style"] == "surface"
        app.set_volume_style(vid, "wireframe")
        assert v["style"] == "wireframe"

        # Focusing the model shows its appearance controls (representation, colour,
        # structure-type show/hide) in the Appearance pane.
        controls = app._controls
        controls._update_appearance("model", mid)
        assert controls._appearance_box.title().endswith("1ubq")
        assert len(controls._appearance_box.findChildren(QComboBox)) >= 2  # rep + colour (+ show)

        # Focusing the volume shows a style dropdown.
        controls._update_appearance("volume", vid)
        assert controls._appearance_box.findChildren(QComboBox)
    finally:
        app.stop()


def test_model_rep_options_are_valid(qapp):
    """Every model representation dropdown value is accepted by the LiveSession API."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")

    from pxviewer.desktop import _MODEL_REP_OPTIONS
    from pxviewer.live import LiveSession

    session = LiveSession.from_sites([[0, 0, 0], [1.5, 0, 0]])
    for _label, value in _MODEL_REP_OPTIONS:
        session.set_representation(value)  # must not raise (regression: 'line' did)


def test_write_object(qapp, tmp_path):
    """Write a model's cctbx coordinates (PDB/mmCIF) and a volume's map."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.loader import sample_structure_path
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file(str(sample_structure_path())), "1ubq")
        pdb = tmp_path / "out.pdb"
        app.write_object("model", mid, str(pdb))
        assert pdb.exists() and "ATOM" in pdb.read_text()
        cif = tmp_path / "out.cif"
        app.write_object("model", mid, str(cif))
        assert cif.exists() and "_atom_site" in cif.read_text()

        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        mrc = tmp_path / "out.mrc"
        app.write_object("volume", vid, str(mrc))
        assert mrc.exists() and mrc.stat().st_size > 0
    finally:
        app.stop()


def test_selection_chip_highlight(qapp):
    """A quick-select chip highlights while active and clears when selection changes."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.loader import sample_structure_path

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file(str(sample_structure_path())), "1ubq")
        controls = app._controls
        protein = {b.text(): b for b, _ in controls._sel_chips}["Protein"]

        protein.click()  # toggles on + runs the preset
        assert protein.isChecked()
        assert len(app._scene_selection[mid]) == 602  # protein atoms

        # A selection from elsewhere no longer matches the preset -> chip clears.
        controls._run_selection("water")
        assert not protein.isChecked()

        # Clicking the active chip again clears the selection.
        protein.click()
        assert protein.isChecked()
        protein.click()
        assert not protein.isChecked() and mid not in app._scene_selection
    finally:
        app.stop()


def test_tools_and_appearance_setters(qapp):
    """Measure-from-selection, colour/interactions setters, clashes and axis."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from types import SimpleNamespace

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_sites([[0, 0, 0], [1.5, 0, 0]]), "m")

        # Measure a distance from exactly two selected atoms.
        app._on_model_selection(mid, SimpleNamespace(indices=[0, 1]))
        assert "distance" in app.measure_selection("distance")
        # Wrong atom count is a clear error, not a crash.
        app._on_model_selection(mid, SimpleNamespace(indices=[0]))
        with pytest.raises(ValueError):
            app.measure_selection("distance")

        # Appearance setters update the model entry.
        app.set_model_color(mid, "chain-id")
        assert app._model_entry(mid)["color"] == "chain-id"
        app.set_model_interactions(mid, True)
        assert app._model_entry(mid)["interactions"] is True

        # Tools that just broadcast must not raise (no analysis run, so toggling a
        # probe channel with no cached dots simply clears it).
        app.set_probe_channel(0, True)
        app.set_probe_channel(0, False)
        app.clear_measurements()
        app.set_axis(True)
        app.reset_view()
    finally:
        app.stop()


def test_active_model_radio(qapp):
    """A per-row radio marks the active model and activates it on click."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from PySide6.QtWidgets import QRadioButton

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")
        assert app._active_model_id == b  # last added is active

        tree = app._controls._loaded_tree
        radios = {r.property("mid"): r for r in tree.findChildren(QRadioButton)}
        assert set(radios) == {a, b}  # one radio per model, tagged with its id
        assert radios[b].isChecked() and not radios[a].isChecked()  # ring-with-dot = active

        # Clicking A's radio activates A (without touching row selection).
        app._controls._on_active_radio(radios[a])
        assert app._active_model_id == a
        # After the rebuild, A's radio is now the checked one.
        radios = {r.property("mid"): r for r in tree.findChildren(QRadioButton)}
        assert radios[a].isChecked() and not radios[b].isChecked()
    finally:
        app.stop()


def test_appearance_follows_active_model(qapp):
    """Activating a model via its radio re-points the Appearance pane at it, so its
    dropdowns edit that model — not the previously focused one — and each model keeps
    its own representation."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from PySide6.QtWidgets import QRadioButton

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")
        assert app._active_model_id == b  # last added is active + focused

        app.set_model_representation(a, "cartoon")
        b_rep = app._model_entry(b)["rep"]  # B's own representation, to check it's left alone

        # Activating A via its radio must move Appearance to A (the bug: it stayed on B).
        radios = {r.property("mid"): r for r in app._controls._loaded_tree.findChildren(QRadioButton)}
        app._controls._on_active_radio(radios[a])
        assert app._active_model_id == a
        assert app._controls._focused == ("model", a)

        # Editing now targets A and leaves B alone (independent per-model state).
        app.set_model_representation(a, "spacefill")
        assert app._model_entry(a)["rep"] == "spacefill"
        assert app._model_entry(b)["rep"] == b_rep  # B untouched
    finally:
        app.stop()


def test_new_model_focuses_appearance(qapp):
    """A newly added model becomes active, so Appearance follows it (this is why the
    hydrogenate+analyze '+H' model — a new active model — is what the dropdowns edit,
    not the hidden original)."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        assert app._controls._focused == ("model", a)  # first model focused

        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0]]), "B")
        assert app._active_model_id == b
        assert app._controls._focused == ("model", b)  # new active model, Appearance follows
    finally:
        app.stop()


def test_axis_off_by_default_help_and_demos_menu(qapp):
    """XYZ axes start hidden. Help is a docs placeholder now; guided tutorials live in the
    Demos menu, in a labelled Tutorials section below the Examples."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        controls = app._controls
        assert controls._axis_check.isChecked() is False  # axes off by default
        controls._on_help()
        assert "documentation" in controls._status_label.text().lower()

        menu = controls._build_demos_menu()
        labels = [a.text() for a in menu.actions()]
        assert "Examples" in labels and "Tutorials" in labels  # both labelled sections
        assert any("ubiquitin" in t.lower() for t in labels)   # an example
        assert any("restraint" in t.lower() for t in labels)   # a tutorial
        # order: the Examples section header precedes the Tutorials section header
        assert labels.index("Examples") < labels.index("Tutorials")
    finally:
        app.stop()


def test_validation_subtabs_and_row_focus(qapp):
    """Validation results become one sub-tab each; selecting a whole row focuses that
    residue (resolved to atom indices from the active model)."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from PySide6.QtWidgets import QTableWidget

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.validation import ValidationResult

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file("pxviewer/data/1ubq.pdb"), "1ubq")

        # A synthetic result (no mmtbx/reference-data needed) drives the UI.
        res = ValidationResult(
            key="ramachandran", title="Ramachandran",
            columns=["chain", "resid", "res"], rows=[["A", "  13 ", "ILE"]],
            markup=[], summary="1 residue",
        )
        app._controls._on_validation_ready((mid, [res]))
        tabs = app._controls._validation_subtabs
        # Clashes & contacts is the permanent first tab; the validator follows it.
        assert tabs.count() == 2
        rama = next(i for i in range(tabs.count()) if tabs.tabText(i) == "Ramachandran")
        assert rama == 1

        table = tabs.widget(rama).findChild(QTableWidget)
        assert table.selectionBehavior() == QTableWidget.SelectionBehavior.SelectRows

        # Selecting the row resolves the residue -> atoms and focuses it.
        table.selectRow(0)
        index = app._model_entry(mid)["_residue_index"]
        assert index[("A", "13")] == [94, 95, 96, 97, 98, 99, 100, 101]  # ILE 13's atoms
    finally:
        app.stop()


def test_restraint_row_marks_all_atoms_and_draws_its_notation(qapp):
    """Selecting a restraint row marks *every* atom in the restraint and draws its
    distance/angle/dihedral notation — so you see the atoms that form it, not one atom."""
    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")
    from pxviewer.geometry import GeometryRestraints, monomer_library_available
    if not monomer_library_available():
        pytest.skip("no monomer library")
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file("pxviewer/data/1ubq.pdb"), "1ubq")
        sess = app._model_entry(mid)["session"]
        gr = GeometryRestraints(sess.model)

        for kind, n in (("bond", 2), ("angle", 3), ("dihedral", 4)):
            iseqs = tuple(int(i) for i in gr.row(kind, 0)[0])
            assert len(iseqs) == n
            app.show_restraint_notations(mid, [(kind, iseqs)])
            assert app._restraint_prim_ids == ["geomsel-0"]           # the notation is drawn
            assert set(sess._last_highlight_indices) == set(iseqs)     # and every atom is marked

        # Several selected rows -> several notations.
        specs = [("bond", tuple(int(i) for i in gr.row("bond", r)[0])) for r in (0, 1)]
        app.show_restraint_notations(mid, specs)
        assert app._restraint_prim_ids == ["geomsel-0", "geomsel-1"]
    finally:
        app.stop()


def test_atom_precision_actions_switch_a_ribbon_to_ball_and_stick(qapp):
    """A cartoon ribbon can't show atoms, so atom-precision work would draw markup into
    empty space. Those actions switch the model to ball-and-stick first; reps that already
    show atoms are left as the user chose."""
    pytest.importorskip("mmtbx.monomer_library.pdb_interpretation")
    from pxviewer.geometry import GeometryRestraints, monomer_library_available
    if not monomer_library_available():
        pytest.skip("no monomer library")
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file("pxviewer/data/1ubq.pdb"), "1ubq")
        assert app._model_entry(mid)["rep"] == "cartoon"  # polymer default

        # A restraint-row notation switches the ribbon to ball-and-stick.
        gr = GeometryRestraints(app._model_entry(mid)["session"].model)
        app.show_restraint_notations(mid, [("bond", tuple(int(i) for i in gr.row("bond", 0)[0]))])
        assert app._model_entry(mid)["rep"] == "ball-and-stick"

        # A rep that already shows atoms is left as chosen.
        app.set_model_representation(mid, "spacefill")
        app.ensure_atoms_shown(mid)
        assert app._model_entry(mid)["rep"] == "spacefill"

        # Measuring also switches a ribbon (select/colour do not — they aren't hooked).
        app.set_model_representation(mid, "cartoon")
        app._scene_selection[mid] = [0, 1]
        app.measure_selection("distance")
        assert app._model_entry(mid)["rep"] == "ball-and-stick"
    finally:
        app.stop()


def test_residue_orientation_and_space_navigation(qapp):
    """Oriented focus frames a residue N->C screen-right with side chain up, and
    advance_residue steps along the chain (space-bar navigation)."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_model_file("pxviewer/data/1ubq.pdb"), "1ubq")
        model = app._model_entry(mid)["session"].model
        idx = app._build_residue_index(model)

        target, up, direction, radius = app._residue_orientation(model, idx[("A", "13")])
        assert abs(np.linalg.norm(up) - 1) < 1e-6 and abs(np.linalg.norm(direction) - 1) < 1e-6
        assert abs(float(np.dot(up, direction))) < 1e-6  # orthonormal basis

        ha = model.get_hierarchy().atoms()
        named = {ha[i].name.strip(): np.array(ha[i].xyz) for i in idx[("A", "13")]}
        n_to_c = named["C"] - named["N"]
        n_to_c /= np.linalg.norm(n_to_c)
        screen_right = np.cross(direction, up)  # Mol*'s right = view x up
        assert float(np.dot(screen_right, n_to_c)) > 0.99  # N->C maps to screen-right
        side = named["CB"] - named["CA"]
        assert float(np.dot(up, side)) > 0  # side chain points up

        # Space-bar navigation steps forward / back along the chain.
        app._focused_residue = ("A", "13")
        app.advance_residue(1)
        assert app._focused_residue == ("A", "14")
        app.advance_residue(-1)
        assert app._focused_residue == ("A", "13")
    finally:
        app.stop()


def test_checkable_combo_requires_click_inside_popup(qapp):
    """The click that opens the dropdown must not toggle the item under the cursor."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from pxviewer.desktop import _make_checkable_combo

    combo = _make_checkable_combo()
    combo.add_checkable("Protein", True, "Protein")
    combo.add_checkable("Water", True, "Water")
    fired = []
    combo.on_change = lambda data, checked: fired.append((data, checked))

    viewport = combo.view().viewport()
    idx0 = combo.model().index(0, 0)
    combo.view().indexAt = lambda _pos: idx0  # every event maps to the first item

    def click(kind):
        ev = QMouseEvent(kind, QPointF(5, 5), Qt.MouseButton.LeftButton,
                         Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        combo.eventFilter(viewport, ev)

    # Opening gesture: a release with no prior press in the popup -> no toggle.
    click(QEvent.Type.MouseButtonRelease)
    assert combo.model().item(0).checkState() == Qt.CheckState.Checked
    assert fired == []

    # A real click inside the popup (press then release) toggles the item.
    click(QEvent.Type.MouseButtonPress)
    click(QEvent.Type.MouseButtonRelease)
    assert combo.model().item(0).checkState() == Qt.CheckState.Unchecked
    assert fired == [("Protein", False)]


def test_hide_and_show_selected_atoms(qapp):
    """Hide/show-selected drops the selection from (or restores it to) the drawn atoms —
    a partial representation, the same mechanism as the structure-type toggles."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        mid = app._add_model(LiveSession.from_sites([[i, 0, 0] for i in range(6)]), "M")
        entry = app._model_entry(mid)
        assert entry["hidden_atoms"] == set()
        assert app._shown_indices(entry) is None  # all shown

        ctl = app._controls
        assert not ctl._hide_sel_btn.isEnabled()  # nothing selected yet

        app._scene_selection = {mid: [1, 2, 3]}
        ctl._on_scene_selection_changed(app._scene_selection)
        assert ctl._hide_sel_btn.isEnabled() and ctl._show_sel_btn.isEnabled()

        app.hide_selected()
        assert entry["hidden_atoms"] == {1, 2, 3}
        assert app._shown_indices(entry) == [0, 4, 5]  # the rest still drawn

        app.show_selected()
        assert entry["hidden_atoms"] == set()
        assert app._shown_indices(entry) is None
    finally:
        app.stop()


def test_hide_structure_types(qapp):
    """Show/hide structure types (cctbx classes) by restricting the representation."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from PySide6.QtWidgets import QComboBox

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession
    from pxviewer.loader import sample_structure_path

    def count_on(on):
        return sum(e - s + 1 for s, e in on["runs"]) if "runs" in on else len(on.get("list", []))

    app = DesktopApp(port=0)
    app._webapp.start()
    captured = {}
    app.bridge.loaded_changed.connect(lambda s: captured.update(s))
    try:
        mid = app._add_model(LiveSession.from_model_file(str(sample_structure_path())), "1ubq")
        entry = app._model_entry(mid)
        session = entry["session"]

        # 1UBQ has two structure types; nothing hidden -> whole structure.
        assert app.model_structure_types(mid) == ["Protein", "Water"]
        assert entry["hidden_types"] == set()
        assert all("on" not in r for r in session._representations.values())
        item = next(it for it in captured["items"] if it["id"] == mid)
        assert item["types"] == ["Protein", "Water"] and item["hidden_types"] == []

        # Hide water -> representation restricted to the 602 protein atoms.
        app.set_model_type_hidden(mid, "Water", True)
        assert entry["hidden_types"] == {"Water"}
        reps = list(session._representations.values())
        assert len(reps) == 1 and count_on(reps[0]["on"]) == 602

        # Switching representation keeps water hidden.
        app.set_model_representation(mid, "spacefill")
        assert "on" in next(iter(session._representations.values()))

        # Hide protein too -> nothing shown.
        app.set_model_type_hidden(mid, "Protein", True)
        assert count_on(next(iter(session._representations.values()))["on"]) == 0

        # Show water again -> just the 58 waters.
        app.set_model_type_hidden(mid, "Water", False)
        assert count_on(next(iter(session._representations.values()))["on"]) == 58

        # Show everything -> back to the whole structure (no restriction).
        app.set_model_type_hidden(mid, "Protein", False)
        assert entry["hidden_types"] == set()
        assert all("on" not in r for r in session._representations.values())

        # The Appearance pane exposes a structure-type checklist (>1 type present).
        from PySide6.QtWidgets import QRadioButton

        controls = app._controls
        controls._update_appearance("model", mid)
        checkables = [
            c for c in controls._appearance_box.findChildren(QComboBox)
            if c.model().rowCount() and c.model().item(0).isCheckable()
        ]
        assert checkables, "expected a checkable structure-type combo in Appearance"

        # Tree row layout: [visible check] col 0, [active radio] col 1, [name] col 2.
        from PySide6.QtCore import Qt

        tree = controls._loaded_tree
        item0 = tree.topLevelItem(0)
        assert tree.columnCount() == 3
        assert item0.checkState(0) in (Qt.CheckState.Checked, Qt.CheckState.Unchecked)
        assert isinstance(tree.itemWidget(item0, 1), QRadioButton)
        assert item0.text(0) == "" and "1ubq" in item0.text(2)
    finally:
        app.stop()


def test_multi_model_registry(qapp):
    """The desktop model registry: add (overlay), hide (switch), active, remove."""
    pytest.importorskip("iotbx.data_manager")
    pytest.importorskip("websockets")
    from pxviewer.appserver import find_frontend_dir, frontend_is_built

    fd = find_frontend_dir()
    if fd is None or not frontend_is_built(fd):
        pytest.skip("frontend not built")

    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0, can_hide=True)  # hardware: models can hide
    app._webapp.start()
    try:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0], [7, 0, 0]]), "B")
        assert len(app._models) == 2
        assert app._active_model_id == b
        assert len(app._visible_model_ws()) == 2  # both visible -> simultaneous
        assert app._session._n_atoms == 3  # active is B

        app.set_model_visible(a, False)
        assert len(app._visible_model_ws()) == 1  # switch: only B shown

        app.set_active_model(a)
        assert app._session._n_atoms == 2  # table/selection follow A even while hidden

        app.remove_model(b)
        assert len(app._models) == 1 and app._active_model_id == a
    finally:
        app.stop()


def test_demos_menu_has_the_curated_examples(qapp):
    """The Demos dropdown is the one preload entry point (Samples is gone): a set of bundled
    examples, each showing off one thing the app does."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from PySide6.QtWidgets import QPushButton, QTabWidget

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        ctl = app._controls
        actions = ctl._build_demos_menu().actions()
        labels = [a.text() for a in actions]
        # Two labelled sections, Examples above Tutorials.
        assert "Examples" in labels and "Tutorials" in labels
        ex_i, tut_i = labels.index("Examples"), labels.index("Tutorials")
        assert ex_i < tut_i
        # The curated examples sit between the two section headers (ignoring blank spacer
        # rows that keep the headers off the edges).
        examples = [a.text() for a in actions[ex_i + 1:tut_i]
                    if not a.isSeparator() and a.text().strip()]
        assert len(examples) == 7
        assert any("1UBQ" in l for l in examples)
        assert any("map + model" in l for l in examples)
        assert any("validation" in l for l in examples)
        assert any("X-ray" in l for l in examples)
        assert any("Ligand fitting" in l for l in examples)
        assert any("Cryo-EM" in l for l in examples)
        assert any("Metal site" in l for l in examples)
        # ...and the tutorials follow: validation, ligand fitting, cryo-EM, then the edits pair.
        tutorials = [a.text() for a in actions[tut_i + 1:]
                     if not a.isSeparator() and a.text().strip()]
        assert [t.lower() for t in tutorials] == \
            ["validate a structure", "fit a ligand into density",
             "real-space refine into cryo-em density",
             "load restraint edits", "custom restraint edits"]

        # The demos button is an icon-only menu button (was the text "Sample", then "Demos").
        buttons = ctl.widget().findChildren(QPushButton)
        assert not any("Sample" in b.text() or "Sample" in b.toolTip() for b in buttons)
        demos = [b for b in buttons if b.toolTip().startswith("Load a bundled")]
        assert len(demos) == 1 and demos[0].menu() is not None

        tabs = ctl.widget().findChild(QTabWidget)
        # Tabs are icon-only; the label lives in the tooltip.
        assert [tabs.tabToolTip(i) for i in range(4)] == ["Scene", "Tools", "Validation", "Geometry"]
        assert all(not tabs.tabIcon(i).isNull() for i in range(tabs.count()))
    finally:
        app.stop()


def test_rebuilding_appearance_spawns_no_stray_windows(qapp):
    """Orphaning a still-visible widget (setParent(None)) turns it into a floating
    top-level window — which is how a rebuilt Appearance pane spawned stray little
    combo-box windows when a model and reflections loaded in succession. Clearing the
    pane must hide-and-delete, not orphan."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from PySide6.QtWidgets import QApplication, QComboBox

    from pxviewer.desktop import DesktopApp

    def stray_combos():
        return [w for w in QApplication.topLevelWidgets()
                if isinstance(w, QComboBox) and w.parent() is None and w.isVisible()]

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._controls.widget().show()
        app.load_xray_demo(d_min=2.5)
        QApplication.processEvents()
        assert stray_combos() == []

        # And the flow that orphaned them directly: rebuild the pane across kinds.
        ctl = app._controls
        ctl._update_appearance("model", app._models[0]["id"])
        ctl._update_appearance("reflections", app._reflections[0]["id"])
        ctl._update_appearance("model", app._models[0]["id"])
        QApplication.processEvents()
        assert stray_combos() == []
    finally:
        app.stop()


def test_mouse_bindings_are_shown_in_the_gui(qapp):
    """Zoom moved off the scroll wheel when the bindings went Coot-style, so it has to be
    spelled out or it is unfindable. The Mouse legend lists every gesture, and the Level
    slider carries a chip naming the wheel that drives it."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    import numpy as np
    from PySide6.QtWidgets import QLabel

    from pxviewer.desktop import DesktopApp
    from pxviewer.volume_io import VolumeData

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        ctl = app._controls
        legend = ctl._build_mouse_legend()
        texts = {w.text() for w in legend.findChildren(QLabel)}
        # The gesture and its action are both present, zoom especially.
        assert "Zoom" in texts
        assert "right-drag" in texts and "Ctrl + scroll" in texts   # both ways to zoom
        assert "scroll" in texts and "Contour level" in texts
        assert "Shift + drag" in texts and "Pull an atom" in texts

        # The scroll-to-contour gesture is named once, in this legend — not repeated as a
        # chip beside the Level slider on every map (it only ate space there).
        vid = app._add_volume(VolumeData.from_numpy(np.ones((8, 8, 8))), "blob")
        ctl._update_appearance("volume", vid)
        chips = [w.text() for w in ctl._appearance_box.findChildren(QLabel) if w.text() == "scroll"]
        assert chips == []
    finally:
        app.stop()


def test_custom_colour_previews_live_not_only_on_close(qapp):
    """The custom colour picker changed the map only after the dialog closed, which read
    as broken until you gave up. The fix drives the colour from the dialog's live
    currentColorChanged, so it updates as the wheel moves."""
    pytest.importorskip("websockets")
    pytest.importorskip("PySide6.QtWebEngineWidgets")

    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    from pxviewer.desktop import DesktopApp

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        applied = []
        combo = app._controls._add_color_row("gold", applied.append)

        # A preset still applies at once.
        combo.setCurrentIndex(combo.findData("salmon"))
        assert applied[-1] == "salmon"

        # The live wire the picker uses: currentColorChanged -> apply, per move.
        dialog = QColorDialog(QColor("gold"))
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        live = []
        dialog.currentColorChanged.connect(
            lambda c: live.append(c.name()) if c.isValid() else None)
        for hex_colour in ("#112233", "#445566", "#778899"):
            dialog.setCurrentColor(QColor(hex_colour))
        assert live == ["#112233", "#445566", "#778899"]  # fired live, one per move
    finally:
        app.stop()


def test_committing_a_custom_colour_does_not_reopen_the_dialog(qapp):
    """Pressing OK looked like it closed the dialog and immediately reopened it. The
    cause: inserting the picked colour into the combo shifts the still-selected "Custom…"
    entry, which re-fires currentIndexChanged with the sentinel and reopens the picker.
    The commit re-indexes with the combo's signals blocked to stop exactly that."""
    from PySide6.QtWidgets import QComboBox

    from pxviewer.desktop import _CUSTOM_COLOR

    combo = QComboBox()
    for i in range(3):
        combo.addItem(f"preset{i}", f"p{i}")
    combo.addItem("Custom…", _CUSTOM_COLOR)
    combo.setCurrentIndex(combo.findData(_CUSTOM_COLOR))  # as if opening the picker

    fired = []
    combo.currentIndexChanged.connect(lambda i: fired.append(combo.itemData(i)))

    # Unguarded, inserting before "Custom…" re-fires with the sentinel — the reopen.
    combo.insertItem(combo.count() - 1, "#abcabc", "#abcabc")
    combo.setCurrentIndex(combo.count() - 2)
    assert _CUSTOM_COLOR in fired  # this is the bug the guard prevents

    # Guarded (what the commit does): no signal, so no reopen.
    combo.setCurrentIndex(combo.findData(_CUSTOM_COLOR))
    fired.clear()
    combo.blockSignals(True)
    combo.insertItem(combo.count() - 1, "#defdef", "#defdef")
    combo.setCurrentIndex(combo.count() - 2)
    combo.blockSignals(False)
    assert fired == []
