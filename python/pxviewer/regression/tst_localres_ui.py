"""The object list and appearance pane around a pinned local-resolution map.

A resolution map is an odd object: it is pinned under the map it colours, and it is never
drawn -- it is a colour source, not a surface. Both of those broke the panel around it in
ways that looked like the feature was broken rather than the widgets:

* nesting it under its full map pushed its checkbox past the right edge of the visibility
  column, leaving a few pixels of it clickable and the rest cut off;
* the appearance pane offered it Style, Colour, Opacity, Level and Clipping, all of which
  moved when dragged and changed nothing on screen, because nothing renders this object.

Both are checked here against a real window, since both are questions about geometry and
widgets that only a laid-out tree can answer.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

from pxviewer.regression.tst_utils import have, process_events, skip, tmp_dir

if not have("PySide6.QtWebEngineWidgets", "numpy"):
    skip("PySide6 QtWebEngine not available")

import numpy as np                                          # noqa: E402
from PySide6.QtWidgets import (QComboBox, QSlider,          # noqa: E402
                               QTreeWidgetItemIterator)

from pxviewer import write_volume                           # noqa: E402
from pxviewer.desktop import DesktopApp                     # noqa: E402
from pxviewer.regression.tst_utils import dispose           # noqa: E402
from pxviewer.volume_io import VolumeData                   # noqa: E402


def _small_maps(work):
    """A map and a plausible resolution field on the same grid."""
    full_path = os.path.join(work, "full.mrc")
    res_path = os.path.join(work, "res.mrc")
    rng = np.random.default_rng(0)
    write_volume((rng.random((16, 16, 16)) * 10).astype(np.float32), full_path,
                 voxel_size=(2.0, 2.0, 2.0), origin=(0.0, 0.0, 0.0))
    z, y, x = np.mgrid[0:16, 0:16, 0:16]
    field = (4.0 + 0.6 * np.sqrt((x - 8) ** 2 + (y - 8) ** 2 + (z - 8) ** 2)).astype(np.float32)
    write_volume(field, res_path, voxel_size=(2.0, 2.0, 2.0), origin=(0.0, 0.0, 0.0))
    return full_path, res_path


class pinned_resolution_map:
    """A window holding a map with a resolution map pinned under it."""

    def __enter__(self):
        self._tmp = tmp_dir()
        work = self._tmp.__enter__()
        full_path, res_path = _small_maps(work)
        self.app = DesktopApp(port=0)
        self.app._webapp.start()
        # Hiding is disabled on software WebGL, which also strips the checkboxes this file
        # is about. Assert the hardware behaviour, which is what a real machine gets.
        self.app._can_hide = True
        self.app.load_file(full_path)
        process_events()
        self.full_vid = self.app._volumes[0]["id"]
        self.app._pin_resolution_map(self.full_vid, VolumeData.from_map_file(res_path),
                                     color=True)
        process_events()
        self.res_vid = self.app._volumes[1]["id"]
        return self

    def __exit__(self, *exc):
        try:
            dispose(self.app)
        finally:
            self._tmp.__exit__(*exc)
        return False


def _rows(tree):
    """(depth, node) for every row, in view order."""
    out = []
    iterator = QTreeWidgetItemIterator(tree)
    while iterator.value():
        node, depth = iterator.value(), 0
        parent = node.parent()
        while parent is not None:
            depth += 1
            parent = parent.parent()
        out.append((depth, node))
        iterator += 1
    return out


def exercise_a_nested_rows_checkbox_is_not_clipped():
    """Column 0 must hold a whole checkbox at the deepest row, not just at the root.

    Qt draws the tree's indentation inside column 0 while sizing ResizeToContents from the
    items' contents alone, so every level of nesting takes an indent's worth of room away
    from the checkbox. How much is left is style-dependent, and that is the point: on the
    machine this was reported from, the pinned resolution map's checkbox was a few pixels
    of sliver that was nearly impossible to click.

    Note this is a guard, not a reproduction. On a Qt style whose column 0 is generous the
    nested row still had 29 px for a 19 px indicator and nothing looked wrong, so this
    assertion passes with or without the fix *here*; it fails wherever the column is tight
    enough for the indent to eat the checkbox, which is the case being defended against.
    The fix adds a full indent per level of depth, so the margin no longer depends on how
    generous the style happens to be.
    """
    with pinned_resolution_map() as fixture:
        tree = fixture.app._controls._loaded_tree
        rows = _rows(tree)
        depths = [d for d, _ in rows]
        assert max(depths) >= 1, "nothing nested, so this proves nothing: %r" % (depths,)

        # visualRect gives the width left *after* a row's indentation -- the space the
        # checkbox actually gets drawn into.
        from PySide6.QtWidgets import QStyle

        indicator = tree.style().pixelMetric(QStyle.PixelMetric.PM_IndicatorWidth, None, tree)
        assert indicator > 0, "no checkbox metric to test against"
        for depth, node in rows:
            content = tree.visualRect(tree.indexFromItem(node, 0)).width()
            assert content >= indicator, (
                "a row at depth %d gets %d px of column 0 for a %d px checkbox "
                "(column %d wide, indent %d)"
                % (depth, content, indicator, tree.columnWidth(0), tree.indentation()))


def exercise_the_resolution_map_offers_no_controls_that_do_nothing():
    """It is never drawn, so surface controls on it are levers connected to nothing.

    The reported symptom was "changing the level of the local resolution doesn't change
    anything on the screen" -- which was true of every control in the pane.
    """
    with pinned_resolution_map() as fixture:
        controls = fixture.app._controls

        controls._update_appearance("volume", fixture.res_vid, force=True)
        process_events()
        box = controls._appearance_box
        assert not box.findChildren(QComboBox), "resolution map still offers Style/Colour"
        assert not box.findChildren(QSlider), "resolution map still offers Opacity/Level"

        controls._update_appearance("volume", fixture.full_vid, force=True)
        process_events()
        box = controls._appearance_box
        assert box.findChildren(QComboBox), "the full map lost its own controls"
        assert box.findChildren(QSlider), "the full map lost its level/opacity"


def exercise_the_resolution_map_says_what_it_is_and_points_somewhere_useful():
    from PySide6.QtWidgets import QLabel, QPushButton

    with pinned_resolution_map() as fixture:
        controls = fixture.app._controls
        controls._update_appearance("volume", fixture.res_vid, force=True)
        process_events()
        box = controls._appearance_box

        blurb = " ".join(w.text() for w in box.findChildren(QLabel) if w.text())
        assert "not drawn" in blurb, blurb
        assert "Colour by local resolution" in blurb, blurb

        buttons = [b for b in box.findChildren(QPushButton) if b.text().startswith("Select")]
        assert buttons, "no way to get to the map whose controls do apply"
        buttons[0].click()
        process_events()
        current = controls._loaded_tree.currentItem()
        from PySide6.QtCore import Qt
        assert current is not None
        assert current.data(0, Qt.ItemDataRole.UserRole) == ("volume", fixture.full_vid)


def exercise_the_resolution_map_is_not_in_the_drawn_scene():
    """The blob, at its root: the resolution map's own isosurface must never be drawn.

    It was in the MVSJ scene marked hidden, with the hide re-broadcast around each
    reload -- but a reload connects a new client, and until visibility was replayed the
    new client drew everything in the scene. A smooth field contoured at its midpoint is
    a giant featureless spheroid on top of the data. Keeping the map out of the scene
    makes "never drawn" structural rather than a race.
    """
    with pinned_resolution_map() as fixture:
        app = fixture.app
        scene_path = app._write_volume_scene()
        assert scene_path, "no scene written despite a loaded map"
        scene = (app._webapp.volume_dir / scene_path.lstrip("/")).read_text()

        full = app._volume_entry(fixture.full_vid)
        res = app._volume_entry(fixture.res_vid)
        assert full["map_url"] in scene, "the full map fell out of the scene"
        assert res["map_url"] not in scene, "the resolution map is still drawn"
        assert res["ref"] not in scene


def exercise_the_resolution_map_has_no_visibility_checkbox():
    """It is never drawn, so show/hide on it is a lever connected to nothing -- and its
    checkbox was the clipped sliver users tried to click. visible=None is how the tree
    already says "nothing drawable" for reflections; a resolution map reports the same."""
    from PySide6.QtCore import Qt

    with pinned_resolution_map() as fixture:
        items = fixture.app._emitted_items()
        by_id = {i["id"]: i for i in items}
        assert by_id[fixture.res_vid]["visible"] is None
        assert by_id[fixture.full_vid]["visible"] is not None

        tree = fixture.app._controls._loaded_tree
        for depth, node in _rows(tree):
            kind_id = node.data(0, Qt.ItemDataRole.UserRole)
            checkable = bool(node.flags() & Qt.ItemFlag.ItemIsUserCheckable)
            if kind_id == ("volume", fixture.res_vid):
                assert not checkable, "the resolution map still offers show/hide"
            elif kind_id == ("volume", fixture.full_vid):
                assert checkable, "the full map lost its show/hide"


def exercise_a_level_change_takes_the_cheap_path():
    """With colour-by-resolution on, the Level slider must not re-stream the grids.

    The browser retained both grids with the first payload, so a level change is one
    float (set_localres_iso) and a client-side re-contour -- the cost a plain map pays.
    The full re-encode (show_localres_grid, ~128 MB for a 256^3 pair) is reserved for
    changes that alter a grid. This was the reported "terribly slow to change contour":
    every slider tick was paying the full re-encode and re-send.
    """
    class RecordingSession:
        def __init__(self):
            self.levels = []
            self.grids = 0

        def set_localres_iso(self, value):
            self.levels.append(float(value))

        def show_localres_grid(self, payload):
            self.grids += 1

        def clear_localres_grid(self):
            pass

    with pinned_resolution_map() as fixture:
        app = fixture.app
        stub = RecordingSession()
        app._control_session = lambda: stub

        app.set_volume_iso(fixture.full_vid, 2.0)

        assert stub.grids == 0, "a level change re-streamed the full grids"
        assert len(stub.levels) == 1, "no light level message was sent: %r" % (stub.levels,)
        # The wire level is absolute on the map's own scale: mean + sigma * std.
        surface = app._display_map_data(app._volume_entry(fixture.full_vid))
        stats = surface.stats()
        expected = stats["mean"] + 2.0 * stats["std"]
        assert abs(stub.levels[0] - expected) < 1e-6, (
            "wire level %.6f, expected %.6f" % (stub.levels[0], expected))


def exercise_the_downsample_choice_is_explicit_and_defaults_to_4x():
    """The coloured surface's display resolution is a user setting, not an adaptive one.

    x4 by default -- 64^3 on a typical box, which re-levels instantly -- shown as a
    Downsample dropdown on the full map's pane. What is drawn is what was asked for:
    the settle-time rebuild at a different resolution than the drag preview visibly
    changed the surface, which read as "it recalculates and looks bad".
    """
    class RecordingSession:
        def __init__(self):
            self.calls = []

        def set_localres_downsample(self, factor):
            self.calls.append(("factor", int(factor)))

        def show_localres_grid(self, payload):
            self.calls.append(("grids", None))

        def set_localres_iso(self, value):
            self.calls.append(("level", float(value)))

        def clear_localres_grid(self):
            pass

    with pinned_resolution_map() as fixture:
        app = fixture.app
        full = app._volume_entry(fixture.full_vid)
        assert full.get("localres_downsample") == 4, "pinning did not default to 4x"

        stub = RecordingSession()
        app._control_session = lambda: stub
        app._push_localres(full)
        kinds = [k for k, _ in stub.calls]
        assert kinds.index("factor") < kinds.index("grids"), (
            "the factor must be sent before the grids: %r" % (stub.calls,))

        app.set_localres_downsample(fixture.full_vid, 2)
        assert full["localres_downsample"] == 2
        assert stub.calls[-1] == ("factor", 2)

        # The pane offers it, current value shown, beside the colour switch.
        controls = app._controls
        controls._update_appearance("volume", fixture.full_vid, force=True)
        process_events()
        combos = controls._appearance_box.findChildren(QComboBox)
        values = {c.currentText() for c in combos}
        assert "2×" in values, "no Downsample dropdown showing the current factor: %r" % (values,)


def exercise_busy_holds_until_the_viewport_confirms_the_drawing():
    """The indicator must span "payload streamed" to "surface on screen".

    The reported dead air: model visible at ~5 s, coloured map usable at ~20 s, and the
    busy oscillator gone after ~2 s -- because the worker's busy ended when it returned,
    and the payload's send was treated as done. Now the push opens a hold that only the
    viewport's localres-shown ack releases, and the tutorial's "ready" predicate follows
    the ack rather than the pinning.
    """
    class SilentSession:
        def set_localres_downsample(self, factor):
            pass

        def show_localres_grid(self, payload):
            pass

        def clear_localres_grid(self):
            pass

    with pinned_resolution_map() as fixture:
        app = fixture.app
        full = app._volume_entry(fixture.full_vid)
        app._control_session = lambda: SilentSession()

        app._push_localres(full)
        assert app._busy_labels and "Drawing local resolution" in app._busy_labels, (
            "no busy hold while the viewport builds: %r" % (app._busy_labels,))
        assert not full.get("localres_drawn"), "drawn before any acknowledgement"

        from pxviewer.tutorial import _resolution_ready

        class CW:  # what the coach hands a done-predicate
            _desktop = app
        assert not _resolution_ready(CW()), "tutorial ready before the surface exists"

        app.bridge.localres_shown.emit()
        process_events()
        assert "Drawing local resolution" not in (app._busy_labels or []), (
            "the ack did not release the hold")
        assert full.get("localres_drawn") is True
        assert _resolution_ready(CW()), "tutorial not ready after the ack"


def exercise_the_colour_range_is_stable_until_the_user_moves_it():
    """The mapping never drifts on its own -- that is the figure-making contract.

    The domain used to be recomputed from the whole resolution map on every push, and
    the full-map percentiles spend most of the ramp on solvent-adjacent voxels no
    realistic contour shows: the visible particle sat entirely in the blue end. It is
    now stored state -- initialised once at pin time, resent verbatim on every push,
    changed only by the user's own set/fit/reset.
    """
    class RecordingSession:
        def __init__(self):
            self.domains = []

        def set_localres_downsample(self, factor):
            pass

        def set_localres_domain(self, lo, hi):
            self.domains.append((round(lo, 4), round(hi, 4)))

        def show_localres_grid(self, payload):
            pass

        def clear_localres_grid(self):
            pass

    with pinned_resolution_map() as fixture:
        app = fixture.app
        full = app._volume_entry(fixture.full_vid)
        first = full.get("localres_domain")
        assert first is not None, "pinning did not initialise a colour range"

        stub = RecordingSession()
        app._control_session = lambda: stub
        app._push_localres(full)
        app._push_localres(full)
        assert full["localres_domain"] == first, "a push moved the colour range"

        # An explicit change sticks, reaches the session, and rejects a crossed range.
        app.set_localres_domain(fixture.full_vid, 4.2, 7.0)
        assert full["localres_domain"] == (4.2, 7.0)
        assert stub.domains[-1] == (4.2, 7.0)
        app.set_localres_domain(fixture.full_vid, 9.0, 3.0)
        assert full["localres_domain"] == (4.2, 7.0), "a crossed range was accepted"

        # Fit spans what the current contour shows, matching an independent computation.
        surface = app._display_map_data(full)
        level = app._absolute_iso(full, surface)
        res = app._volume_entry(fixture.res_vid)
        inside = res["data"].array[surface.array >= level]
        inside = inside[np.isfinite(inside)]
        inside = inside[inside != 0.0]
        assert inside.size >= 100, "fixture leaves too little visible to fit to"
        app.fit_localres_domain(fixture.full_vid)
        lo, hi = full["localres_domain"]
        assert abs(lo - np.percentile(inside, 2)) < 0.011, (lo, np.percentile(inside, 2))
        assert abs(hi - np.percentile(inside, 98)) < 0.011, (hi, np.percentile(inside, 98))

        # Reset restores the full-map default.
        app.reset_localres_domain(fixture.full_vid)
        lo, hi = full["localres_domain"]
        d_lo, d_hi = app._localres_domain(res["data"])
        assert abs(lo - d_lo) < 0.011 and abs(hi - d_hi) < 0.011

        # And the pane shows the numbers with the two buttons.
        from PySide6.QtWidgets import QDoubleSpinBox, QPushButton
        controls = app._controls
        controls._update_appearance("volume", fixture.full_vid, force=True)
        process_events()
        box = controls._appearance_box
        spins = [w for w in box.findChildren(QDoubleSpinBox)
                 if w.objectName().startswith("localres-domain-")]
        assert len(spins) == 2, "no colour-range spinboxes on the pane"
        assert {round(sp.value(), 2) for sp in spins} == {round(lo, 2), round(hi, 2)}
        texts = {b.text() for b in box.findChildren(QPushButton)}
        assert "Fit to surface" in texts and "Reset" in texts, texts

        # The overlay's controls live inside one checkable group whose title is the
        # on/off switch -- not interleaved among the map's own rows, which is how they
        # started and read as unrelated neighbours with Level stranded beneath them.
        from PySide6.QtWidgets import QGroupBox
        parent = spins[0].parent()
        while parent is not None and not isinstance(parent, QGroupBox):
            parent = parent.parent()
        assert isinstance(parent, QGroupBox), "the colour range escaped the group"
        assert parent.title() == "Colour by local resolution", parent.title()
        assert parent.isCheckable(), "the group's title is not the on/off switch"
        assert parent.isChecked() == bool(full.get("color_by_resolution"))


def exercise_the_computed_resolution_map_is_saved_and_reused():
    """The minute-long computation runs once; a re-run loads the saved map from disk.

    Offline, through the real fetch_and_compute_resolution path: the "fetched" files are
    pre-placed in the working directory and reuse_existing skips the network, which is
    exactly the tutorial's warm re-run. The computation itself is patched and counted --
    the cache decision is what is under test, not cctbx -- and the second run's map is
    compared against the first, so the disk round trip (write_map -> from_map_file) is
    proven to reproduce the grid rather than assumed to.

    The cache is trusted only when newer than both half-maps: a re-download is a
    deliberate refresh and must recompute (the mtime prong below).
    """
    import shutil
    import time

    from pxviewer import volume_io

    with tmp_dir() as work:
        full_path, res_path = _small_maps(work)
        # The files fetch_entry would have produced, for a made-up entry 9999.
        names = ["emd_9999.map", "emd_9999_half_map_1.map", "emd_9999_half_map_2.map"]
        for name in names:
            shutil.copy(full_path, os.path.join(work, name))

        computed = []
        real_compute = volume_io.local_resolution_from_half_maps

        def fake_compute(*args, **kwargs):
            computed.append(kwargs.get("d_min"))
            return VolumeData.from_map_file(res_path)

        volume_io.local_resolution_from_half_maps = fake_compute
        app = DesktopApp(port=0)
        app._webapp.start()
        try:
            def run_once():
                app.fetch_and_compute_resolution(
                    emdb_number="9999", reuse_existing=True, work_dir=work)
                deadline = time.time() + 60
                while time.time() < deadline and not any(
                        v.get("resolution_map") for v in app._volumes):
                    process_events()
                    time.sleep(0.02)
                process_events()
                res_vid = next(v["id"] for v in app._volumes if v.get("is_resolution"))
                return app._volume_entry(res_vid)["data"]

            first = run_once()
            assert len(computed) == 1, "the computation did not run on a cold start"
            saved = [n for n in os.listdir(work) if "local_resolution" in n]
            assert saved, "nothing was saved to the working directory: %r" % os.listdir(work)

            # A fresh run must load the saved map, not recompute -- and reproduce it.
            for v in list(app._volumes):
                app.remove_volume(v["id"])
            process_events()
            second = run_once()
            assert len(computed) == 1, "a warm re-run recomputed instead of loading the cache"
            assert first.array.shape == second.array.shape
            assert np.allclose(first.array, second.array), "the disk round trip changed the map"
            assert np.allclose(first.origin, second.origin), "the disk round trip moved the map"

            # Refreshed inputs invalidate: make a half-map newer than the cache.
            cache_file = os.path.join(work, saved[0])
            future = time.time() + 60
            os.utime(os.path.join(work, "emd_9999_half_map_1.map"), (future, future))
            assert not DesktopApp._cache_is_fresh(
                cache_file, os.path.join(work, "emd_9999_half_map_1.map"),
                os.path.join(work, "emd_9999_half_map_2.map")), (
                "a cache older than its half-maps was still trusted")
        finally:
            volume_io.local_resolution_from_half_maps = real_compute
            dispose(app)


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
