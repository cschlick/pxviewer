"""Tests for the desktop atoms-table model (no QWebEngine needed)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pxviewer.data import AtomArrays  # noqa: E402
from pxviewer.desktop import _make_atom_table_model, _runs  # noqa: E402


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

        app.set_volume_visible(vid, False)
        assert app._write_volume_scene() is None  # nothing visible -> no scene
        app.remove_volume(vid)
        assert not app._volumes

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


def test_map_model_demo_loads_bundled_lysozyme_as_group(qapp):
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
        assert app._models[0]["session"]._n_atoms == 1079  # lysozyme
        # Model + map compose the viewport together.
        assert len(app._visible_model_ws()) == 1 and app._write_volume_scene() is not None
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

        # The console tab hides the Display/Selection controls so it fills the pane.
        from PySide6.QtWidgets import QTabWidget

        tabs = controls._window.findChild(QTabWidget)
        tabs.setCurrentIndex(controls._console_tab_index)
        assert controls._bottom_controls.isHidden()
        tabs.setCurrentIndex(0)  # File
        assert not controls._bottom_controls.isHidden()
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

    app = DesktopApp(port=0)
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
