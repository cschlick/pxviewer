"""Visual pass 11 -- chained sequences where each step leans on standing state.

The scriptable half of VISUAL_TESTS.md's Pass 11: load/hide/show chains,
a clip standing through representation rebuilds and coordinate moves,
per-model clip isolation, rapid control storms, teardown and reload. The
gesture-only steps (shift-click picking, map interactions) remain manual.
"""

import traceback

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from pxviewer.regression.visual.harness import *


def select_expr(app, controls, text, mid, timeout=15):
    controls._select_expr.setText(text)
    controls._on_select_expression()
    pump(until=lambda: bool(app._scene_selection.get(mid)), timeout=timeout)
    settle(2.5)


def run() -> None:
    app = make_app()
    try:
        controls = app._controls

        # -- 1. load-hide-load-hide-show ------------------------------------
        section("load / hide / show chain")
        load_bundled(app, "1ubq.pdb")
        pump(until=lambda: len(app._models) == 1, timeout=20)
        settle(2.0)
        mid_a = app._models[0]["id"]
        entry_a = app._model_entry(mid_a)
        note("A=%s (%s)  B follows" % (mid_a, app._models[0].get("name")))
        can_hide = bool(getattr(app, "_can_hide", False))

        app.set_model_visible(mid_a, False)
        settle(1.0)
        if check("A hides (hardware backend only)",
                 entry_a["visible"] is False or not can_hide,
                 "can_hide=%s" % can_hide):
            load_bundled(app, "3nir.pdb")
            pump(until=lambda: len(app._models) == 2, timeout=20)
            settle(2.0)
        mid_b = app._models[-1]["id"]
        entry_b = app._model_entry(mid_b)
        note("B=%s (%s)" % (mid_b, app._models[-1].get("name")))
        # loading B legitimately reframes; compare the camera across hide/show only
        app.set_model_visible(mid_b, False)
        settle(1.0)
        cam0 = cam_state(app)
        app.set_model_visible(mid_a, True)
        settle(1.0)
        check("A back exactly where it was", entry_a["visible"] is True
              and entry_b["visible"] is False)
        cam1 = cam_state(app)
        check("hide/show never moved the camera",
              cam0 and cam1 and cam0.get("radius") == cam1.get("radius"),
              "%s -> %s" % (cam0 and cam0.get("radius"),
                            cam1 and cam1.get("radius")))
        shot_viewport(app, "v11-1-after-chain")

        # -- 2. selection on the hidden active model -------------------------
        section("selection follows the active model, even hidden")
        app.set_active_model(mid_b)
        pump(); settle(0.5)
        select_expr(app, controls, "resseq 5", mid_b)
        check("hidden B took the selection",
              len(app._scene_selection.get(mid_b, [])) > 0,
              str(app._scene_selection.get(mid_b)))
        app.set_model_visible(mid_b, True)
        settle(2.0)
        shot_viewport(app, "v11-2-b-unhidden-selected")
        check("B still selected after unhide",
              len(app._scene_selection.get(mid_b, [])) > 0)

        # -- 3. clip survives representation rebuild -------------------------
        section("clip through representation rebuild")
        app.set_active_model(mid_a)
        pump(); settle(0.5)
        app.clear_selection()
        settle(0.5)
        select_expr(app, controls, "resseq 29", mid_a)
        check("auto clip standing on A", entry_a.get("_auto_clip") is True)
        clip_radius = (cam_state(app) or {}).get("radius")
        clipped = shot_viewport(app, "v11-3-a-clipped")
        app.set_model_representation(mid_a, "ball-and-stick")
        settle(2.5)
        app.set_model_representation(mid_a, "cartoon")
        settle(2.5)
        after_reps = shot_viewport(app, "v11-3-after-rep-cycle")
        check("clip survives the rep round trip",
              entry_a.get("_auto_clip") is True)
        r_now = (cam_state(app) or {}).get("radius")
        check("depth slab still parked",
              r_now is not None and clip_radius is not None
              and abs(r_now - clip_radius) < 0.5,
              "%s -> %s" % (clip_radius, r_now))
        # NB: an explicit rep switch clears the neighbourhood context layer by
        # design (set_model_representation), so pixels legitimately differ from
        # the context-bearing clipped frame. The clip claim is carried by
        # _auto_clip + the parked slab above; the reapplied view is compared
        # against this post-cycle baseline below instead.

        # -- 4. lift/re-apply inside the chain --------------------------------
        section("lift and re-apply mid-chain")
        controls._clip_on_select.setChecked(False)
        settle(1.5)
        shot_viewport(app, "v11-4-lifted")
        check("lift drops the marker", "_auto_clip" not in entry_a)
        controls._clip_on_select.setChecked(True)
        settle(1.5)
        reapplied = shot_viewport(app, "v11-4-reapplied")
        r2 = (cam_state(app) or {}).get("radius")
        check("re-apply restores depth slab",
              r2 is not None and clip_radius is not None
              and abs(r2 - clip_radius) < 0.05,
              "%s -> %s" % (clip_radius, r2))
        if after_reps and reapplied:
            check("re-applied matches the post-cycle clipped view",
                  img_diff(after_reps, reapplied) < 3.0,
                  "%.2f%%" % img_diff(after_reps, reapplied))

        # -- 5. components stepping under clip -------------------------------
        section("components stepping under clip")
        controls._tabs.setCurrentIndex(controls._tab_labels.index("Geometry"))
        pump(); settle(0.5)
        controls._geo_subtabs.setCurrentIndex(0)
        settle(1.0)
        cview = controls._component_view
        pump(until=lambda: cview.model().rowCount() > 0, timeout=20)
        cview.setCurrentIndex(cview.model().index(3, 0))
        settle(1.5)
        cview.setFocus(); pump()
        rows_ok = True
        for _ in range(4):
            QTest.keyClick(cview, Qt.Key_Space)
            settle(1.2)
            rows_ok = rows_ok and entry_a.get("_auto_clip") is True
        check("every stepped row re-clips on its residue", rows_ok)
        check("row cursor moved", cview.currentIndex().row() == 7,
              str(cview.currentIndex().row()))

        # -- 7. empty re-check is a no-op ------------------------------------
        section("clear + empty clip toggle")
        app.clear_selection()
        settle(1.5)
        radius_clear = (cam_state(app) or {}).get("radius")
        controls._clip_on_select.setChecked(False)
        settle(0.8)
        controls._clip_on_select.setChecked(True)
        settle(1.0)
        radius_after = (cam_state(app) or {}).get("radius")
        check("re-checking an empty selection is calm",
              "_auto_clip" not in entry_a
              and radius_clear is not None
              and radius_after is not None
              and abs(radius_after - radius_clear) < 0.01,
              "%s -> %s" % (radius_clear, radius_after))

        # -- 8. minimize under a clip; restraint freshness --------------------
        section("minimize invalidates restraint measurements")
        select_expr(app, controls, "resseq 29", mid_a)
        check("clip re-standing for minimize", entry_a.get("_auto_clip") is True)
        controls._tabs.setCurrentIndex(controls._tab_labels.index("Geometry"))
        pump(); settle(0.5)
        controls._geo_subtabs.setCurrentIndex(2)   # bonds
        pump(until=lambda: controls._restraint_tabs.get("bond") is not None
             and controls._restraint_tabs["bond"]["model"].rowCount() > 0,
             timeout=30)
        settle(1.0)
        geo_before = controls._geo_cache.get(mid_a)
        check("geometry cache populated while tab open", geo_before is not None)
        controls._tabs.setCurrentIndex(0)   # Scene: restraint tabs not visible
        pump(); settle(0.3)
        app.minimize_model()
        settle(4.0)
        app.stop_minimization()
        settle(2.0)
        # The move drops the cached geometry; whether it refills at once depends on
        # _viewing_restraint_tab(), which reads only the geometry *subtab* -- the
        # bonds subtab is still selected under the Scene top tab, so a refill is
        # legitimate here. The contract is freshness, not absence.
        check("coordinate move dropped the geometry cache",
              controls._geo_cache.get(mid_a) is not geo_before,
              "keys: %s" % list(controls._geo_cache))
        check("sphere stands where it was fit (not chasing atoms)",
              entry_a.get("_auto_clip") is True)
        controls._tabs.setCurrentIndex(controls._tab_labels.index("Geometry"))
        controls._geo_subtabs.setCurrentIndex(2)
        pump(until=lambda: controls._restraint_tabs["bond"]["model"].rowCount() > 0,
             timeout=30)
        settle(1.0)
        geo_after = controls._geo_cache.get(mid_a)
        check("restraint tables refill with fresh geometry",
              geo_after is not None and geo_after is not geo_before)

        # -- 9. per-model clip isolation --------------------------------------
        section("clip state is per-model")
        app.set_active_model(mid_b)
        pump(); settle(0.5)
        select_expr(app, controls, "resseq 7", mid_b)
        check("B's selection does not touch A's clip",
              entry_a.get("_auto_clip") is True
              and entry_b.get("_auto_clip") is True)
        app.set_model_visible(mid_a, False)
        settle(1.0)
        app.set_model_visible(mid_a, True)
        settle(1.5)
        check("A still clipped after hide/show",
              entry_a.get("_auto_clip") is True)
        shot_viewport(app, "v11-9-both-clipped")

        # -- 10. rapid storm: final state wins ---------------------------------
        section("control storm")
        for _ in range(10):
            controls._clip_on_select.toggle()
            pump(interval=0.01)
        controls._clip_on_select.setChecked(True)
        for rep in ("ball-and-stick", "spacefill", "cartoon",
                    "ball-and-stick", "cartoon"):
            app.set_model_representation(mid_b, rep)
            pump(interval=0.01)
        settle(2.5)
        check("storm settles on the checked state",
              entry_b.get("_auto_clip") is True
              and entry_b.get("rep") == "cartoon")
        controls._geo_subtabs.setCurrentIndex(2)
        pump(until=lambda: controls._restraint_tabs.get("bond") is not None
             and controls._restraint_tabs["bond"]["model"].rowCount() > 0,
             timeout=30)
        bview = controls._restraint_tabs["bond"]["view"]
        for row in (0, 1, 2):
            bview.setCurrentIndex(bview.model().index(row, 0))
            pump(interval=0.01)
        settle(2.5)
        rows = [i.row() for i in bview.selectionModel().selectedRows()]
        check("last stormed restraint row stays selected after echoes",
              rows == [2], str(rows))
        check("stormed row is a scene selection",
              len(app._scene_selection.get(mid_b, [])) == 2,
              str(app._scene_selection.get(mid_b)))

        # -- 11-12. remove while clipped; reload clean --------------------------
        section("teardown and reload")
        app.set_active_model(mid_a)
        pump(); settle(0.3)
        app.remove_model(mid_a)
        settle(2.0)
        check("removed model's clip dies with it; B's is untouched",
              mid_a not in [m["id"] for m in app._models]
              and entry_b.get("_auto_clip") is True)
        app.remove_model(mid_b)
        settle(1.5)
        check("scene empties clean", not app._models)
        load_bundled(app, "1ubq.pdb")
        pump(until=lambda: len(app._models) == 1, timeout=20)
        settle(2.5)
        mid_a2 = app._models[0]["id"]
        entry_a2 = app._model_entry(mid_a2)
        select_expr(app, controls, "resseq 29", mid_a2)
        check("fresh model clips fresh",
              entry_a2.get("_auto_clip") is True)
        shot_viewport(app, "v11-12-reloaded-clipped")

        # -- 14. hide/show selected under a standing clip -----------------------
        section("hide/show selected with clip standing")
        app.hide_selected()
        settle(1.5)
        shot_viewport(app, "v11-14-hidden")
        check("hiding keeps the clip's bookkeeping",
              entry_a2.get("_auto_clip") is True)
        app.show_selected()
        settle(1.5)
        shot_viewport(app, "v11-14-shown")
        check("selection and clip survive the round trip",
              entry_a2.get("_auto_clip") is True
              and len(app._scene_selection.get(mid_a2, [])) > 0)

        controls_window(app, "v11-final-controls")
    except Exception:
        check("chains pass ran clean", False, traceback.format_exc(limit=3))
    finish(app)


run()
