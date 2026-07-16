"""Tests for the desktop atoms-table model (no QWebEngine needed)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
    monkeypatch.delenv("MMTBX_CCP4_MONOMER_LIB", raising=False)
    monkeypatch.delenv("CLIBD_MON", raising=False)

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

        # Tools that just broadcast must not raise.
        app.show_clashes()
        app.clear_clashes()
        app.clear_measurements()
        app.set_axis(True)
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
