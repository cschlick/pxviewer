"""The desktop's tables: atoms, and the geometry restraints.

Both are Qt table models over data the app already holds, so what matters is the mapping
between the two -- which columns appear, how a value is formatted, and what a view row
means once a filter is on. A filtered table whose row-to-atom mapping is wrong reads as
plausible data about the wrong atoms, which is the failure worth pinning.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import os
import sys

from pxviewer.regression.tst_utils import (
    closing_modals, data_path, dispose, have, process_events, qt_application,
    shipped_defaults, skip, tmp_dir)

if not have("PySide6.QtWebEngineWidgets", "websockets", "iotbx.data_manager"):
    skip("PySide6 QtWebEngine / websockets / iotbx.data_manager not available")

qt_application()

from PySide6.QtCore import Qt                    # noqa: E402

from pxviewer.desktop import (                   # noqa: E402
    DesktopApp, _make_atom_table_model, _make_restraint_table_model, _runs)
from pxviewer.live import LiveSession            # noqa: E402

#: Three atoms of one residue, with distinct B-factors so a column can be told apart from
#: its neighbours. Written and read back rather than hand-built: the table reads
#: ``AtomArrays``, and building those from a file is the path the app actually takes.
THREE_ATOMS = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 30.00           C
END
"""


@contextlib.contextmanager
def three_atom_session(attributes=None):
    """A real ``LiveSession`` over :data:`THREE_ATOMS`, with optional per-atom values."""
    with tmp_dir() as directory:
        path = os.path.join(directory, "three.pdb")
        with open(path, "w") as handle:
            handle.write(THREE_ATOMS)
        session = LiveSession.from_model_file(path)
        for name, values in (attributes or {}).items():
            session.set_attribute(name, values)
        yield session


@contextlib.contextmanager
def desktop():
    app = DesktopApp(port=0)
    app._webapp.start()
    with closing_modals():
        try:
            yield app
        finally:
            dispose(app)


def headers_of(model):
    return [model.headerData(c, Qt.Orientation.Horizontal)
            for c in range(model.columnCount())]


def monomer_library():
    from pxviewer.geometry import monomer_library_available

    return monomer_library_available()


def ubiquitin_desktop():
    """An app with 1UBQ loaded and its restraint tables built."""
    app = DesktopApp(port=0)
    app._webapp.start()
    app._add_model(LiveSession.from_model_file(data_path("1ubq.pdb")), "1ubq")
    app._controls._ensure_restraints()
    return app


# -- index runs ---------------------------------------------------------------


def exercise_runs_collapses_contiguous_indices():
    """Selections go on the wire as ranges, so scattered picks must collapse."""
    assert list(_runs([3, 1, 2, 2, 5])) == [(1, 3), (5, 5)]
    assert list(_runs([])) == []


# -- the atoms table ----------------------------------------------------------


def exercise_atom_table_columns_and_values():
    with three_atom_session({"score": [0.1, 0.2, 0.3]}) as session:
        model = _make_atom_table_model()
        model.set_session(session)

        assert model.rowCount() == 3
        headers = headers_of(model)
        assert headers[:6] == ["#", "element", "name", "resname", "chain", "resseq"]
        assert {"x", "y", "z", "B", "occ", "score"} <= set(headers)

        assert model.data(model.index(1, 0)) == "1"          # the "#" index column
        assert model.data(model.index(0, headers.index("element"))) == "N"
        assert model.data(model.index(1, headers.index("B"))) == "20.000"
        # A per-atom attribute becomes a column of its own.
        assert model.data(model.index(2, headers.index("score"))) == "0.300"


def exercise_atom_table_renders_a_missing_value_blank():
    """nan means "not computed here" -- Q-score leaves it on every hydrogen. Formatting
    it as a number would present a missing value as a real one."""
    with three_atom_session({"partial": [1.0, float("nan"), 3.0]}) as session:
        model = _make_atom_table_model()
        model.set_session(session)
        column = headers_of(model).index("partial")

        assert model.data(model.index(0, column)) == "1.000"
        assert model.data(model.index(1, column)) == ""


def exercise_atom_table_is_empty_without_a_session():
    model = _make_atom_table_model()
    model.set_session(None)
    assert model.rowCount() == 0
    assert model.columnCount() == 0


def exercise_atom_table_filter_to_selection():
    """Show-only-selected restricts the visible rows, and every row-to-atom mapping has
    to follow it -- a stale mapping shows one atom's numbers under another's label."""
    with three_atom_session({"score": [0.1, 0.2, 0.3]}) as session:
        model = _make_atom_table_model()
        model.set_session(session)
        headers = headers_of(model)

        assert not model.is_filtered()
        assert model.rowCount() == 3

        model.set_filter([2, 0])              # unordered and partial: sorted and deduped
        assert model.is_filtered()
        assert model.rowCount() == 2
        # The "#" column keeps showing the real atom index, not the view row.
        assert model.data(model.index(0, 0)) == "0"
        assert model.data(model.index(1, 0)) == "2"
        assert model.data(model.index(1, headers.index("B"))) == "30.000"
        assert model.row_atom(1) == 2
        assert model.atom_row(2) == 1
        assert model.atom_row(1) == -1        # atom 1 is filtered out

        model.set_filter(None)
        assert not model.is_filtered()
        assert model.rowCount() == 3
        assert model.row_atom(1) == 1
        assert model.atom_row(1) == 1


# -- the scene selection behind the table -------------------------------------


def exercise_scene_selection_is_the_union_over_models():
    """Each model reports its own picks; the desktop unions them into one selection."""
    from types import SimpleNamespace

    with desktop() as app:
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0], [7, 0, 0]]), "B")

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


def exercise_the_table_follows_the_active_model_but_can_be_pinned():
    """The dropdown tracks the active model until the user chooses otherwise, and the
    filter checkbox collapses the table to the picked atoms."""
    from types import SimpleNamespace

    from pxviewer.appserver import find_frontend_dir, frontend_is_built

    frontend = find_frontend_dir()
    if frontend is None or not frontend_is_built(frontend):
        print("    (skipped: frontend not built)")
        return

    with desktop() as app:
        controls = app._controls
        a = app._add_model(LiveSession.from_sites([[0, 0, 0], [1, 0, 0]]), "A")
        b = app._add_model(LiveSession.from_sites([[5, 0, 0], [6, 0, 0], [7, 0, 0]]), "B")

        # Both models listed, following the active one (B, added last).
        assert controls._table_model_combo.count() == 2
        assert controls._table_model_id == b
        assert controls._atom_model.rowCount() == 3          # B's atoms

        # A pick in B shows up as selected rows.
        app._on_model_selection(b, SimpleNamespace(indices=[0, 2]))
        assert controls._table_selection_indices() == [0, 2]
        selected = set(i.row() for i in controls._atom_view.selectionModel().selectedRows())
        assert selected == {0, 2}

        controls._filter_selection_check.setChecked(True)
        assert controls._atom_model.is_filtered()
        assert controls._atom_model.rowCount() == 2
        controls._filter_selection_check.setChecked(False)
        assert not controls._atom_model.is_filtered()

        # Pinning to A keeps the table on A even though B is active.
        controls._table_model_combo.setCurrentIndex(0)
        assert controls._table_pinned
        assert controls._table_model_id == a
        assert controls._atom_model.rowCount() == 2          # A's atoms

        # Choosing the active model again resumes auto-follow.
        controls._table_model_combo.setCurrentIndex(1)
        assert not controls._table_pinned
        assert controls._table_model_id == b


# -- the restraint table ------------------------------------------------------


class Row_source(object):
    """A restraint source of known rows.

    A real ``GeometryRestraints`` is used further down, where the point is that the
    tables fill from a built manager. Here the point is formatting -- that 1.52 renders
    as "1.520" and -0.07 keeps its sign -- which needs values chosen, not discovered.
    """

    def __init__(self, rows):
        self._rows = rows                    # [(i_seqs, {column: value}), ...]

    def count(self, category):
        return len(self._rows)

    def row(self, category, i):
        return self._rows[i]


def exercise_restraint_table_columns_and_formatting():
    columns = ["ideal", "model", "delta", "sigma", "residual"]
    rows = [
        ((0, 1), {"ideal": 1.52, "model": 1.50, "delta": 0.02,
                  "sigma": 0.02, "residual": 1.0}),
        ((2, 3), {"ideal": 1.33, "model": 1.40, "delta": -0.07,
                  "sigma": 0.02, "residual": 12.0}),
    ]
    model = _make_restraint_table_model()
    model.set_source(Row_source(rows), "bond", columns, lambda i: "atom%d" % i)

    assert model.rowCount() == 2
    assert headers_of(model) == ["atoms", "ideal", "model", "delta", "sigma", "residual"]
    assert model.data(model.index(0, 0)) == "atom0  atom1"
    assert model.data(model.index(0, 1)) == "1.520"
    assert model.data(model.index(1, 3)) == "-0.070"
    assert model.i_seqs_for_row(1) == (2, 3)

    model.set_source(None, "", columns, None)
    assert model.rowCount() == 0


def exercise_restraint_table_geostd_column():
    """The geostd column names where a restraint came from, and links only when there is
    a single file to link to -- a link restraint spans two monomers and has none."""
    columns = ["ideal", "model"]
    rows = [((0, 1), {}), ((2, 3), {})]

    def source(i_seqs):
        if set(i_seqs) == {0, 1}:
            return ("ALA", "/geostd/a/data_ALA.cif")
        return ("(link)", None)

    model = _make_restraint_table_model()
    model.set_source(Row_source(rows), "bond", columns, lambda i: "a%d" % i, source)

    assert headers_of(model) == ["atoms", "ideal", "model", "geostd"]
    assert model.source_column() == 3
    assert model.data(model.index(0, 3)) == "ALA"
    assert model.data(model.index(1, 3)) == "(link)"
    assert model.source_for_row(0) == ("ALA", "/geostd/a/data_ALA.cif")
    assert model.source_for_row(1)[1] is None
    # Styled as a link only where there is a file behind it.
    assert model.data(model.index(0, 3), Qt.ItemDataRole.ForegroundRole) is not None
    assert model.data(model.index(1, 3), Qt.ItemDataRole.ForegroundRole) is None


def exercise_restraint_table_filter():
    columns = ["ideal", "model", "delta", "sigma", "residual"]
    rows = [((0, 1), {}), ((2, 3), {}), ((4, 5), {})]
    model = _make_restraint_table_model()
    model.set_source(Row_source(rows), "bond", columns, lambda i: "a%d" % i)

    assert model.rowCount() == 3
    assert not model.is_filtered()

    model.set_filter([2])
    assert model.is_filtered()
    assert model.rowCount() == 1
    assert model.i_seqs_for_row(0) == (4, 5)        # view row 0 is restraint 2
    assert model.data(model.index(0, 0)) == "a4  a5"

    model.set_filter(None)
    assert not model.is_filtered()
    assert model.rowCount() == 3


# -- the restraint tables against a real model --------------------------------


def exercise_geometry_restraints_populate_the_tables():
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    app = ubiquitin_desktop()
    try:
        bond = app._controls._restraint_tabs["bond"]
        assert bond["stack"].currentWidget() is bond["view"]
        assert bond["model"].rowCount() > 500              # 1UBQ has hundreds of bonds
        angle = app._controls._restraint_tabs["angle"]["model"]
        assert angle.rowCount() > bond["model"].rowCount()
        # The atoms column reads i_seqs back as labels.
        assert "/" in bond["model"].data(bond["model"].index(0, 0))
    finally:
        dispose(app)


def zn_site_desktop(with_edits=True):
    """An app with the Zn site loaded, its user-supplied edits applied, and the restraint
    tables built -- the one shipped combination that has more than one restraint origin."""
    app = DesktopApp(port=0)
    app._webapp.start()
    app._add_model(LiveSession.from_model_file(data_path("zn_site.pdb")), "zn_site")
    if with_edits:
        app.load_edits(app._models[0]["id"], data_path("zn_site_edits.phil"))
    app._controls._ensure_restraints()
    return app


def exercise_the_origin_filter_lists_the_origins_the_model_has():
    """Numbered, named and counted. cctbx defines over a hundred origins; only the ones
    with restraints behind them are offered, or nearly every entry would filter to zero."""
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    app = zn_site_desktop()
    try:
        combo = app._controls._origin_filter
        labels = [combo.itemText(i) for i in range(combo.count())]
        assert labels[0] == "All"
        assert combo.itemData(0) is None

        # The user's own restraint, named as such and reachable by its cctbx origin id.
        edits_label = next(t for t in labels if "user-defined" in t)
        assert edits_label.startswith("4:")
        assert combo.itemData(labels.index(edits_label)) == 4

        # A model whose restraints all come from the monomer library has nothing to
        # choose between, so the control is disabled rather than offering one option.
        assert combo.isEnabled()
    finally:
        dispose(app)


def exercise_filtering_by_origin_isolates_the_user_supplied_restraint():
    """The point of the whole control: one hand-written bond among dozens from the
    library, found without reading every row."""
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    app = zn_site_desktop()
    try:
        controls = app._controls
        bond = controls._restraint_tabs["bond"]["model"]
        full = bond.rowCount()

        combo = controls._origin_filter
        combo.setCurrentIndex(combo.findData(4))          # 'edits'
        assert bond.rowCount() == 1
        assert bond.rowCount() < full

        # And it really is the edit: 2.1 A, the distance the PHIL asked for.
        row = bond.i_seqs_for_row(0)
        assert len(row) == 2

        combo.setCurrentIndex(combo.findData(None))       # back to All
        assert bond.rowCount() == full
    finally:
        dispose(app)


def exercise_origin_and_selection_filters_intersect():
    """Two independent reasons to hide a restraint, so asking for both means both --
    "the edits I made, within what I have selected" is the question worth asking."""
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    app = zn_site_desktop()
    try:
        controls = app._controls
        bond = controls._restraint_tabs["bond"]["model"]

        combo = controls._origin_filter
        combo.setCurrentIndex(combo.findData(4))
        assert bond.rowCount() == 1                   # the edit, origin-filtered only

        # Selecting elsewhere removes it: origin alone is not enough to keep a row.
        app.select_by_expression("name ND1")          # the histidines, not the Zn site
        controls._filter_selection_check.setChecked(True)
        assert bond.rowCount() == 0

        # Selecting the chain the edit lives on brings it back, still origin-filtered --
        # so neither filter is overriding the other.
        app.select_by_expression("chain S")           # the Zn and its water
        assert bond.rowCount() == 1

        controls._filter_selection_check.setChecked(False)
        assert bond.rowCount() == 1                   # origin filter alone, unchanged
    finally:
        dispose(app)


def exercise_a_model_without_user_edits_still_offers_its_origins():
    """The Zn site has metal-coordination restraints even with no PHIL loaded, so the
    dropdown is populated but 'edits' is absent -- it lists what is there, not a fixed
    menu."""
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    app = zn_site_desktop(with_edits=False)
    try:
        combo = app._controls._origin_filter
        labels = [combo.itemText(i) for i in range(combo.count())]
        assert not any("user-defined" in t for t in labels)
        assert combo.findData(4) < 0
    finally:
        dispose(app)


def exercise_a_link_authored_from_two_atoms_reaches_the_table_and_a_file():
    """The whole loop the feature exists for: pick two atoms, add the bond, see it in the
    Bonds table, write it out as a PHIL that phenix can read.

    The table half is the part that silently did not work. The restraint was real --
    minimize honoured it and it was in the saved file -- but ``_geo_cache`` is keyed by
    model id and held a wrapper around the *previous* restraints manager, so the table
    went on showing the restraints from before the edit. Nothing said so; the row simply
    was not there.
    """
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return
    import os

    app = zn_site_desktop(with_edits=False)
    try:
        controls = app._controls
        bond = controls._restraint_tabs["bond"]["model"]
        before = bond.rowCount()
        mid = app._models[0]["id"]

        # The Zn and the water oxygen it coordinates -- the bond cctbx does not add.
        zinc, water_o = 30, 31
        app._scene_selection[mid] = [zinc, water_o]
        app.add_edit_from_selection(mid, "bond")
        process_events()

        assert bond.rowCount() == before + 1
        assert len(app.model_edits(mid)) == 1

        # It is there as a *user* restraint, not lost among the library's.
        combo = controls._origin_filter
        assert combo.findData(4) >= 0
        combo.setCurrentIndex(combo.findData(4))
        assert bond.rowCount() == 1
        assert set(bond.i_seqs_for_row(0)) == {zinc, water_o}
        combo.setCurrentIndex(combo.findData(None))

        # And it writes out as a file phenix reads and this reads back.
        with tmp_dir() as directory:
            path = os.path.join(directory, "link.phil")
            app.save_edits(mid, path)
            text = open(path).read()
            assert "geometry_restraints" in text
            assert "name ZN" in text and "distance_ideal" in text

            app.clear_edits(mid)
            process_events()
            assert bond.rowCount() == before        # the row goes when the edit does

            assert app.load_edits(mid, path) == 1
            process_events()
            assert bond.rowCount() == before + 1    # and comes back with it
    finally:
        dispose(app)


def exercise_removing_an_authored_link_takes_it_out_of_the_table():
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    app = zn_site_desktop(with_edits=True)
    try:
        controls = app._controls
        bond = controls._restraint_tabs["bond"]["model"]
        with_edit = bond.rowCount()
        mid = app._models[0]["id"]

        app.remove_edit(mid, 0)
        process_events()
        assert bond.rowCount() == with_edit - 1
        assert app.model_edits(mid) == []
        assert controls._origin_filter.findData(4) < 0   # no user restraints left
    finally:
        dispose(app)


def exercise_the_shared_filter_applies_to_every_restraint_table():
    """"Show only the selection" collapses the restraint tables too, not just Atoms."""
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    app = ubiquitin_desktop()
    try:
        controls = app._controls
        bond = controls._restraint_tabs["bond"]
        full = bond["model"].rowCount()
        assert full > 500
        assert not bond["model"].is_filtered()

        app.select_by_expression("resseq 1")
        controls._filter_selection_check.setChecked(True)

        filtered = bond["model"].rowCount()
        assert 0 < filtered < full                  # only that residue's own bonds remain
        selected = set(app._scene_selection[app._active_model_id])
        for r in range(filtered):
            assert all(i in selected for i in bond["model"].i_seqs_for_row(r))
        assert controls._restraint_tabs["angle"]["model"].is_filtered()

        controls._filter_selection_check.setChecked(False)
        assert bond["model"].rowCount() == full
    finally:
        dispose(app)


def exercise_every_geostd_row_resolves_to_a_file_on_disk():
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return

    app = ubiquitin_desktop()
    try:
        model = app._controls._restraint_tabs["bond"]["model"]
        assert model.source_column() == model.columnCount() - 1

        resolved = 0
        for r in range(min(model.rowCount(), 50)):
            text, path = model.source_for_row(r)
            assert text                             # a resname, or "(link)"
            if path is not None:
                assert path.endswith(".cif")
                assert os.path.isfile(path)
                resolved += 1
        assert resolved > 0
    finally:
        dispose(app)


# -- selecting a restraint row ------------------------------------------------


def exercise_selecting_restraint_rows_draws_their_notations():
    """Selecting angle rows draws angle notations rather than a whole-residue
    highlight, and several selected rows draw several notations."""
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return
    from PySide6.QtCore import QItemSelectionModel

    app = ubiquitin_desktop()
    try:
        controls = app._controls
        view = controls._restraint_tabs["angle"]["view"]
        model = controls._restraint_tabs["angle"]["model"]
        session = app.active_model_session()

        view.selectRow(0)
        assert len(app._restraint_prim_ids) == 1
        assert len(session._primitives) == 1

        flags = (QItemSelectionModel.SelectionFlag.Select
                 | QItemSelectionModel.SelectionFlag.Rows)
        view.selectionModel().select(model.index(1, 0), flags)
        assert len(app._restraint_prim_ids) == 2
        assert len(session._primitives) == 2

        view.clearSelection()
        assert app._restraint_prim_ids == []
        assert len(session._primitives) == 0
    finally:
        dispose(app)


def exercise_a_restraint_row_marks_every_atom_in_the_restraint():
    """You should see the atoms that form the restraint, not just one of them -- a bond
    marks two, an angle three, a dihedral four."""
    if not monomer_library():
        print("    (skipped: no monomer library)")
        return
    from pxviewer.geometry import GeometryRestraints

    with desktop() as app:
        mid = app._add_model(
            LiveSession.from_model_file(data_path("1ubq.pdb")), "1ubq")
        session = app._model_entry(mid)["session"]
        restraints = GeometryRestraints(session.model)

        for kind, n in (("bond", 2), ("angle", 3), ("dihedral", 4)):
            i_seqs = tuple(int(i) for i in restraints.row(kind, 0)[0])
            assert len(i_seqs) == n
            app.show_restraint_notations(mid, [(kind, i_seqs)])
            assert app._restraint_prim_ids == ["geomsel-0"]           # drawn
            assert set(session._last_highlight_indices) == set(i_seqs)  # all marked

        specs = [("bond", tuple(int(i) for i in restraints.row("bond", r)[0]))
                 for r in (0, 1)]
        app.show_restraint_notations(mid, specs)
        assert app._restraint_prim_ids == ["geomsel-0", "geomsel-1"]


# -- selecting by expression --------------------------------------------------


def exercise_select_by_expression():
    """A cctbx/Phenix selection string selects atoms on the active model."""
    from libtbx.test_utils import raises

    with desktop() as app:
        with raises(ValueError):
            app.select_by_expression("chain A")          # no model: a clear error

        app._add_model(LiveSession.from_model_file(data_path("1ubq.pdb")), "1ubq")
        mid = app._active_model_id

        assert app.select_by_expression("chain A and resseq 5:14 and name CA") == 10
        assert len(app._scene_selection[mid]) == 10      # fed into the scene selection

        assert app.select_by_expression("   ") == 0      # empty clears
        assert mid not in app._scene_selection

        with raises(Exception):
            app.select_by_expression("chain @@@ bogus (")


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
