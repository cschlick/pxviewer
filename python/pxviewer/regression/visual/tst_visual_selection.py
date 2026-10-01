"""Visual pass 5 -- selection, oriented focus, clipping, table stepping.

The automatable core of VISUAL_TESTS.md's Pass 5, plus the camera-radius and
pixel-diff assertions that pin the clip lift/re-apply fix: a lifted clip frees
the depth slab (radius returns to radiusMax), and re-applying restores the same
view (radius back to the selection's depth extent, pixels back to the clipped
frame).
"""

import traceback

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from pxviewer.regression.visual.harness import *


def run() -> None:
    app = make_app()
    try:
        load_bundled(app, "1ubq.pdb")
        pump(until=lambda: bool(app._models), timeout=20)
        settle(3.0)
        mid = app._models[0]["id"]
        controls = app._controls
        entry = app._model_entry(mid)

        # -- typed selection with focus+clip -------------------------------
        check("focus+clip+context checked by default",
              controls._focus_on_select.isChecked()
              and controls._clip_on_select.isChecked()
              and controls._context_on_select.isChecked())
        controls._select_expr.setText("resseq 29")
        controls._on_select_expression()
        pump(until=lambda: bool(app._scene_selection.get(mid)), timeout=15)
        settle(3.0)
        clipped = shot_viewport(app, "v5-1-resseq29-clipped")
        clip_radius = (cam_state(app) or {}).get("radius")
        check("selection label", controls._selection_label.text() != "None",
              controls._selection_label.text())
        check("auto clip standing", entry.get("_auto_clip") is True)
        check("depth slab parked on the selection",
              clip_radius is not None and clip_radius < 20.0,
              "radius=%s" % clip_radius)

        # -- clip checkbox lifts and re-applies ----------------------------
        controls._clip_on_select.setChecked(False)
        settle(2.0)
        lifted = shot_viewport(app, "v5-2-clip-lifted")
        lift_radius = (cam_state(app) or {}).get("radius")
        check("unchecking lifts the sphere", "_auto_clip" not in entry)
        check("lifting frees the depth slab too",
              lift_radius is not None and clip_radius is not None
              and lift_radius > clip_radius * 2,
              "radius %s -> %s" % (clip_radius, lift_radius))
        controls._clip_on_select.setChecked(True)
        settle(2.0)
        reapplied = shot_viewport(app, "v5-2-clip-reapplied")
        re_radius = (cam_state(app) or {}).get("radius")
        check("re-checking re-clips the standing selection",
              entry.get("_auto_clip") is True)
        check("re-apply restores the depth slab",
              re_radius is not None and clip_radius is not None
              and abs(re_radius - clip_radius) < 0.05,
              "radius %s -> %s" % (lift_radius, re_radius))
        if clipped and lifted and reapplied:
            check("clipped and lifted views differ",
                  img_diff(clipped, lifted) > 1.0,
                  "%.2f%%" % img_diff(clipped, lifted))
            check("re-applied matches the first clip",
                  img_diff(clipped, reapplied) < 2.0,
                  "%.2f%%" % img_diff(clipped, reapplied))

        # -- focus unchecked: selection but camera stays --------------------
        before = cam_state(app)
        controls._focus_on_select.setChecked(False)
        controls._select_expr.setText("resseq 40")
        controls._on_select_expression()
        pump(until=lambda: len(app._scene_selection.get(mid, [])) > 0,
             timeout=15)
        settle(2.0)
        after = cam_state(app)
        shot_viewport(app, "v5-3-nofocus")
        check("camera unmoved with focus off",
              before and after
              and before.get("target") == after.get("target")
              and before.get("radius") == after.get("radius"),
              "%s -> %s" % (before and before.get("target"),
                            after and after.get("target")))
        controls._focus_on_select.setChecked(True)

        # -- helix selection orients along the axis -------------------------
        controls._select_expr.setText("resseq 20:35")
        controls._on_select_expression()
        settle(3.0)
        shot_viewport(app, "v5-4-helix")
        note("expect: helix fills frame along its long axis, not skewed")

        # -- space residue navigation ----------------------------------------
        app.clear_selection()
        settle(1.0)
        app._viewport.widget().setFocus()
        pump()
        QTest.keyClick(app._main, Qt.Key_Space)   # real key, not the API
        settle(2.5)
        r1 = app._focused_residue
        shot_viewport(app, "v5-5-space-residue-1")
        QTest.keyClick(app._main, Qt.Key_Space)
        QTest.keyClick(app._main, Qt.Key_Space)
        settle(2.5)
        shot_viewport(app, "v5-5-space-residue-3")
        check("space walks residues",
              app._focused_residue is not None and app._focused_residue != r1,
              str(app._focused_residue))
        check("residue steps clip", entry.get("_auto_clip") is True)
        QTest.keyClick(app._main, Qt.Key_Space, Qt.ShiftModifier)
        settle(1.5)

        # -- table stepping: Components -------------------------------------
        controls._tabs.setCurrentIndex(controls._tab_labels.index("Geometry"))
        pump(); settle(0.5)
        controls._geo_subtabs.setCurrentIndex(0)   # Components
        settle(1.0)
        controls_window(app, "v5-5a-components-tab")
        cview = controls._component_view
        cview.setCurrentIndex(cview.model().index(5, 0))
        settle(1.5)   # debounce + selection echo
        check("component row stays selected after echo",
              [i.row() for i in cview.selectionModel().selectedRows()] == [5],
              str([i.row() for i in cview.selectionModel().selectedRows()]))
        shot_viewport(app, "v5-5a-component-row5")
        cview.setFocus(); pump()
        QTest.keyClick(cview, Qt.Key_Space)
        settle(1.5)
        check("space steps the components table",
              cview.currentIndex().row() == 6,
              str(cview.currentIndex().row()))

        # -- Atoms table ----------------------------------------------------
        controls._geo_subtabs.setCurrentIndex(1)   # Atoms
        settle(0.5)
        aview = controls._atom_view
        aview.setCurrentIndex(aview.model().index(10, 0))
        settle(1.5)
        check("atom row stays selected",
              [i.row() for i in aview.selectionModel().selectedRows()] == [10],
              str([i.row() for i in aview.selectionModel().selectedRows()]))
        shot_viewport(app, "v5-5a-atom-row10")

        # -- restraint rows are selections ----------------------------------
        controls._geo_subtabs.setCurrentIndex(2)   # first restraint subtab
        pump(until=lambda: controls._restraint_tabs.get("bond") is not None
             and controls._restraint_tabs["bond"]["model"].rowCount() > 0,
             timeout=30)
        settle(1.0)
        bview = controls._restraint_tabs["bond"]["view"]
        controls_window(app, "v5-5a-bonds-tab")
        bview.setCurrentIndex(bview.model().index(0, 0))
        settle(2.0)   # selectionChanged -> notation push -> WS echo -> filter pass
        rows = [i.row() for i in bview.selectionModel().selectedRows()]
        check("bond row STAYS selected after the echo", rows == [0], str(rows))
        shot_viewport(app, "v5-5a-bond-row0")
        check("bond row is a scene selection",
              len(app._scene_selection.get(mid, [])) == 2,
              str(app._scene_selection.get(mid)))
        check("sphere re-centered on the bond", entry.get("_auto_clip") is True)
        bview.setFocus(); pump()
        QTest.keyClick(bview, Qt.Key_Space)
        settle(2.0)
        check("space steps bonds", bview.currentIndex().row() == 1,
              str(bview.currentIndex().row()))
        check("second bond row stays selected",
              [i.row() for i in bview.selectionModel().selectedRows()] == [1])
        QTest.keyClick(bview, Qt.Key_Space)
        QTest.keyClick(bview, Qt.Key_Space)
        settle(2.0)
        check("keeps stepping", bview.currentIndex().row() == 3,
              str(bview.currentIndex().row()))

        # clip off while stepping bonds
        controls._clip_on_select.setChecked(False)
        QTest.keyClick(bview, Qt.Key_Space)
        settle(2.0)
        check("bond step with clip off lifts the sphere",
              "_auto_clip" not in entry)
        shot_viewport(app, "v5-5a-bond-noclip")
        controls._clip_on_select.setChecked(True)
        settle(1.5)

        # sticky target: back on the Scene tab, space still steps the bond table
        controls._tabs.setCurrentIndex(0)
        app._viewport.widget().setFocus()
        pump(); settle(0.5)
        before_row = bview.currentIndex().row()
        app.step_next(1)
        settle(1.5)
        check("space keeps the restraint table (sticky)",
              bview.currentIndex().row() == before_row + 1,
              "%s -> %s" % (before_row, bview.currentIndex().row()))

        controls_window(app, "v5-6-final-scene")
    except Exception:
        check("selection pass ran clean", False, traceback.format_exc(limit=3))
    finish(app)


run()
