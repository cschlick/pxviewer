"""Hotspot wiring through the desktop shell and the live session.

Needs PySide6 + QtWebEngine and websockets, so ``run_tests.py`` keeps it in the gated list;
a headless cctbx build skips the whole script. The pure-computation half is in
tst_hotspots.py and tst_concern.py.
"""

from __future__ import absolute_import, division, print_function

import struct
import sys
import time

from libtbx.test_utils import approx_equal

from pxviewer.regression.tst_concern import write_manifest
from pxviewer.regression.tst_utils import (
    data_path, dispose, have, process_events, qt_application, shipped_defaults, skip,
    tmp_dir)

if not have("mmtbx", "numpy", "websockets", "PySide6.QtWebEngineWidgets"):
    skip("needs mmtbx, websockets and PySide6 QtWebEngine")

import numpy as np                                    # noqa: E402

from pxviewer import hotspots                         # noqa: E402

MODEL = data_path("1tec.pdb")
_qapp = []


def qapp():
    if not _qapp:
        _qapp.append(qt_application())
    return _qapp[0]


def _wait_for(signal_box, seconds=300):
    """Pump the Qt loop until a background score lands."""
    deadline = time.time() + seconds
    while time.time() < deadline and not signal_box:
        process_events()
        time.sleep(0.05)
    assert signal_box, "hotspots never landed"


def _app_with_model():
    from pxviewer.desktop import DesktopApp
    from pxviewer.live import LiveSession

    app = DesktopApp(port=0)
    app._webapp.start()
    mid = app._add_model(LiveSession.from_model_file(MODEL), "1tec")
    return app, mid


# -- the live session's own state ---------------------------------------------


def exercise_streaming_a_cloud_sets_the_replay_payload():
    """The cloud rides its own wire tag and is replayed to late viewers, so it survives a
    viewport reload the way the difference map does."""
    from pxviewer.live import LiveSession, _TAG_HOTSPOT_VOLUME

    session = LiveSession.from_model_file(MODEL)
    assert session._last_hotspot_volume is None
    session.show_hotspot_volume(b"\x00\x01\x02\x03")
    assert session._last_hotspot_volume[:4] == _TAG_HOTSPOT_VOLUME.to_bytes(4, "little")
    assert session._last_hotspot_volume[4:] == b"\x00\x01\x02\x03"
    session.clear_hotspot_volume()
    assert session._last_hotspot_volume is None


def exercise_the_opacity_knee_is_remembered_for_late_viewers():
    """The knee is a lightweight control message (no grid re-stream), so it is remembered and
    re-sent on connect -- a reload keeps the slider where the user left it."""
    from pxviewer.live import LiveSession

    session = LiveSession.from_model_file(MODEL)
    session.show_hotspot_volume(b"\x00")
    assert session._hotspot_knee is None          # a fresh cloud is at its default knee
    session.set_hotspot_opacity(0.6)
    assert approx_equal(session._hotspot_knee, 0.6)
    session.show_hotspot_volume(b"\x01")          # a new cloud resets it
    assert session._hotspot_knee is None
    session.set_hotspot_opacity(0.4)
    session.clear_hotspot_volume()                # clearing drops it
    assert session._hotspot_knee is None


# -- importing concern through the app ----------------------------------------


def exercise_a_manifest_imports_without_running_analysis():
    """A manifest imports every field, shows combined, and streams concern unrescaled."""
    with tmp_dir() as work:
        manifest = write_manifest(work)
        app, mid = _app_with_model()
        try:
            entry = app._model_entry(mid)
            assert entry.get("hotspots") is None

            app.open_hotspot_volume(manifest, mid)

            assert entry.get("hotspots") is None
            imported = entry["concern"]
            assert str(imported.source) == manifest
            assert entry["concern_metric"] == "combined"       # combined leads
            assert set(imported.fields) == set(["combined", "clash"])
            assert entry["hotspot_cloud"] is True

            payload = entry["session"]._last_hotspot_volume
            # Bounded concern goes out unrescaled: the knee is the contract's yellow anchor,
            # and the values are NOT divided by the legacy severity cap.
            assert approx_equal(struct.unpack_from("<f", payload, 4)[0], 0.5)
            # NXSTART/map_data origin (2,3,4) at anisotropic spacing lands in Cartesian space.
            assert approx_equal(struct.unpack_from("<fff", payload, 24), (2.0, 4.5, 8.0))
            assert approx_equal(
                np.frombuffer(payload, dtype="<f4", offset=72).max(), 0.6)
            # The viewer is told the contract rather than left to infer a ramp from the knee.
            assert entry["session"]._hotspot_anchors == {
                "yellow": 0.5, "orange": 0.75, "red": 1.0}

            app.set_hotspot_field_metric("clash", mid)
            assert entry["concern_metric"] == "clash"
            assert entry["session"]._last_hotspot_volume is not None
        finally:
            dispose(app)


def exercise_importing_concern_drops_a_computed_score():
    """The two scales must never be on screen together.

    A computed score already clears an import; this pins the other direction, which is what
    let a severity table -- where a favored Ramachandran residue still scores ~0.4 -- sit
    beside a concern map that correctly reads 0 at that residue.
    """
    from pxviewer.desktop import _HOTSPOT_COLOR

    with tmp_dir() as work:
        manifest = write_manifest(work, metrics=("combined",))
        app, mid = _app_with_model()
        try:
            entry = app._model_entry(mid)
            # Stand in for a finished Find hotspots run: a score, and a coloured model.
            entry["hotspots"] = object()
            entry["hotspot_palette"] = ["#000000"]
            entry["color"] = _HOTSPOT_COLOR
            entry["attribute"] = {"name": _HOTSPOT_COLOR, "values": np.zeros(3)}

            app.open_hotspot_volume(manifest, mid)

            assert entry.get("hotspots") is None          # the severity score is gone
            assert entry.get("hotspot_palette") is None
            assert entry.get("attribute") is None         # and so is severity colouring
            assert entry["color"] != _HOTSPOT_COLOR
            assert entry["concern_metric"] == "combined"
        finally:
            dispose(app)


def exercise_percentile_never_gates_visibility():
    """A manifest with no percentile map imports and draws; every voxel of concern reaches
    the wire, unmasked."""
    with tmp_dir() as work:
        manifest = write_manifest(work, metrics=("rama",), percentile=False)
        app, mid = _app_with_model()
        try:
            app.open_hotspot_volume(manifest, mid)
            entry = app._model_entry(mid)

            assert entry["concern"].fields["rama"].percentile is None
            assert entry["hotspot_cloud"] is True
            payload = entry["session"]._last_hotspot_volume
            assert approx_equal(
                np.frombuffer(payload, dtype="<f4", offset=72).min(), 0.6)
        finally:
            dispose(app)


def exercise_the_table_reads_the_maps_not_a_second_scale():
    """Table values are sampled from the concern grids, so they are bounded to [0, 1] and
    cannot disagree with what the viewport draws."""
    with tmp_dir() as work:
        manifest = write_manifest(work, metrics=("combined", "rama"))
        app, mid = _app_with_model()
        try:
            emitted = []
            app.bridge.concern_ready.connect(emitted.append)
            app.open_hotspot_volume(manifest, mid)

            assert emitted, "importing concern fields should publish a table"
            _mid, summary, columns, rows = emitted[-1]
            assert columns == ["chain", "resid", "res", "combined concern", "rama concern"]
            assert "concern" in summary
            # Only atoms inside the (tiny) test grid sample nonzero, but every value that
            # appears is bounded concern, never a severity above 1.
            for row in rows:
                for cell in row[3:]:
                    if cell:
                        assert 0.0 <= float(cell) <= 1.0

            # Raising the threshold past the field drops the rows that just became
            # invisible, rather than leaving them listed with no density beside them.
            app.set_hotspot_threshold(mid, 0.9)
            assert emitted[-1][3] == []
        finally:
            dispose(app)


def exercise_the_table_says_it_ranks_neighbourhoods():
    """The generator splats each observation with a ~2 A Gaussian, wider than the ~3.8 A
    between adjacent CA atoms, so a residue beside a hotspot genuinely sits in its density.
    That is not recoverable here, so the table must present itself as a neighbourhood ranking
    rather than invent a de-blurring rule and claim per-residue attribution."""
    from pxviewer import concern as concern_mod

    with tmp_dir() as work:
        manifest = write_manifest(work, metrics=("combined",))
        app, mid = _app_with_model()
        try:
            emitted = []
            app.bridge.concern_ready.connect(emitted.append)
            app.open_hotspot_volume(manifest, mid)
            _mid, summary, _columns, _rows = emitted[-1]
            assert concern_mod.TABLE_CAVEAT in summary
            assert "rank neighborhoods, not residues" in summary
        finally:
            dispose(app)


# -- the shared threshold -----------------------------------------------------


def exercise_the_knee_is_given_in_severity_and_sent_normalized():
    """The slider is in severity units (1.0 = the cut); the wire speaks the grid's [0,1]
    scale, so the desktop divides by the cap. A no-op unless a cloud is actually showing."""
    app, mid = _app_with_model()
    try:
        entry = app._model_entry(mid)

        app.set_hotspot_opacity(mid, 1.5)            # no cloud showing -> ignored
        assert entry["session"]._hotspot_knee is None

        entry["hotspot_cloud"] = True                # pretend a cloud is up
        app.set_hotspot_opacity(mid, 1.5)
        assert approx_equal(entry["session"]._hotspot_knee,
                            1.5 / hotspots.SEVERITY_CAP)
    finally:
        dispose(app)


def exercise_the_threshold_updates_a_contour_level_and_colour():
    """Contour uses the same absolute slider as the cloud and follows the hotspot palette.

    Driven through a real volume rather than by watching which methods get called: what
    matters is the level and colour the volume ends up carrying, not the route taken there.
    """
    from pxviewer.volume_io import VolumeData

    app, mid = _app_with_model()
    try:
        data = VolumeData.from_numpy(
            np.zeros((4, 4, 4), dtype=np.float32), spacing=1.0, name="field")
        vid = app._add_volume(data, "field", iso=hotspots.FIELD_ISO, iso_kind="absolute")
        app._model_entry(mid)["hotspot_volume"] = vid

        app.set_hotspot_threshold(mid, 1.5)

        volume = app._volume_entry(vid)
        assert approx_equal(volume["iso"], 1.5)          # the absolute level, not sigma
        assert volume["color"] == hotspots.severity_color(1.5)
        assert volume["color"] == hotspots.WARM[1]
    finally:
        dispose(app)


def exercise_an_absolute_level_is_converted_for_the_sigma_wire():
    """The live volume_iso command speaks sigma, which is right for maps. A severity field
    contours on absolute values because its levels are calibrated, so its level must be
    converted or a live change would land somewhere else from a scene rebuild."""
    from pxviewer.desktop import DesktopApp

    stats = {"mean": 0.02, "std": 0.25}
    absolute = {"iso_kind": "absolute",
                "data": type("D", (), {"stats": lambda self: stats})()}
    assert approx_equal(DesktopApp._iso_for_wire(absolute, 1.0), 3.92)  # (1.0-0.02)/0.25
    # A map is untouched: its level already is sigma.
    assert DesktopApp._iso_for_wire({"iso_kind": "relative"}, 3.0) == 3.0
    assert DesktopApp._iso_for_wire({}, 3.0) == 3.0


# -- the desktop wiring -------------------------------------------------------


def exercise_the_cloud_and_contour_are_mutually_exclusive():
    """One 3-D field per model. The cloud streams on the session; the contour is an MVS-scene
    volume. Switching between them, or turning the field off, must leave neither behind."""
    app, mid = _app_with_model()
    try:
        got = []
        app.bridge.hotspots_ready.connect(got.append)
        app.compute_hotspots(mid)
        _wait_for(got)
        entry = app._model_entry(mid)

        app.show_hotspot_field(mid, on=True, style="cloud")
        assert entry.get("hotspot_cloud") is True
        assert entry.get("hotspot_volume") is None        # no MVS contour
        assert not app._volumes
        payload = entry["session"]._last_hotspot_volume
        assert payload is not None
        # The cloud streams the quality preset's grid: its step vector (offset 36 past the
        # tag, cutFrac, stepsPerCell, dims and origin) is the voxel size.
        low_spacing = hotspots.CLOUD_QUALITY["low"][0]
        assert approx_equal(struct.unpack_from("<f", payload, 36)[0], low_spacing)

        # Bumping quality restreams a finer grid at more steps -- same cloud, cleaner render.
        app.set_cloud_quality("high")
        payload = entry["session"]._last_hotspot_volume
        hi_spacing, hi_steps = hotspots.CLOUD_QUALITY["high"]
        assert approx_equal(struct.unpack_from("<f", payload, 36)[0], hi_spacing)
        assert approx_equal(struct.unpack_from("<f", payload, 8)[0], hi_steps)
        assert struct.unpack_from("<f", payload, 36)[0] < low_spacing

        app.show_hotspot_field(mid, on=True, style="contour")
        assert entry.get("hotspot_cloud") is None         # the cloud was torn down
        assert entry["session"]._last_hotspot_volume is None
        assert entry.get("hotspot_volume") is not None    # and a contour drawn
        assert len(app._volumes) == 1

        app.show_hotspot_field(mid, on=False)
        assert entry.get("hotspot_cloud") is None and entry.get("hotspot_volume") is None
        assert not app._volumes
    finally:
        dispose(app)


def exercise_the_contour_is_added_once_and_removed_on_toggle():
    """One shell per model: recomputing or re-showing must replace it, never stack a second
    surface on the first."""
    app, mid = _app_with_model()
    try:
        got = []
        app.bridge.hotspots_ready.connect(got.append)

        # Refuses politely before there is anything to contour.
        said = []
        app.bridge.status_changed.connect(said.append)
        app.show_hotspot_field(mid, on=True, style="contour")
        assert not app._volumes and any("no hotspots yet" in s for s in said)

        app.compute_hotspots(mid)
        _wait_for(got)

        app.show_hotspot_field(mid, on=True, style="contour")
        assert len(app._volumes) == 1
        volume = app._volumes[0]
        assert volume["iso_kind"] == "absolute"      # calibrated level, not sigma
        assert volume["iso"] == hotspots.FIELD_ISO
        assert volume["color"] == hotspots.severity_color(hotspots.FIELD_ISO)
        assert volume["opacity"] < 1.0               # the model stays readable through it

        app.show_hotspot_field(mid, on=True, style="contour")   # re-show replaces
        assert len(app._volumes) == 1
        app.show_hotspot_field(mid, on=False)                   # and off removes
        assert not app._volumes
        assert app._model_entry(mid).get("hotspot_volume") is None
    finally:
        dispose(app)


def exercise_computing_colours_through_the_attribute_path():
    """The score reaches the viewport as a named per-atom attribute on a fixed severity
    domain, not as a Mol* theme and not stretched to this structure's own range."""
    from pxviewer.desktop import _HOTSPOT_COLOR

    app, mid = _app_with_model()
    try:
        got = []
        app.bridge.hotspots_ready.connect(got.append)
        app.compute_hotspots(mid)
        _wait_for(got)

        entry = app._model_entry(mid)
        assert entry["color"] == _HOTSPOT_COLOR
        spec = list(entry["session"]._representations.values())[0]
        assert spec["color"] == "attribute"
        assert spec["attribute"]["name"] == _HOTSPOT_COLOR
        assert list(spec["attribute"]["domain"]) == list(hotspots.DOMAIN)
    finally:
        dispose(app)


def exercise_choosing_hydrogens_drops_a_stale_score():
    """Turning hydrogens on or off changes what a clash *is*, so a score computed under the
    old setting no longer describes the checkbox -- it is dropped rather than left silently
    disagreeing. (No recompute here; that is the user's next click.)"""
    app, mid = _app_with_model()
    try:
        entry = app._model_entry(mid)
        entry["hotspots"] = object()            # stand in for a finished score
        assert app._hotspot_hydrogens is False  # fast pass by default

        app.set_hotspot_hydrogens(True)
        assert app._hotspot_hydrogens is True
        assert entry.get("hotspots") is None    # the stale score was dropped
    finally:
        dispose(app)


def exercise_validation_staleness_tracks_model_movement():
    """Once a model is validated, moving any atom must flip it to stale, and re-validating at
    the new coordinates must clear it."""
    app, mid = _app_with_model()
    try:
        entry = app._model_entry(mid)
        seen = []
        app.bridge.validation_stale_changed.connect(seen.append)

        app._refresh_validation_staleness()     # never validated: nothing to be stale against
        assert seen[-1] is False

        app._mark_validated(entry)              # fingerprint the current coordinates
        app._refresh_validation_staleness()
        assert seen[-1] is False

        model = entry["session"].model          # move an atom
        sites = model.get_sites_cart()
        moved = sites.deep_copy()
        moved[0] = (moved[0][0] + 0.5, moved[0][1], moved[0][2])
        model.set_sites_cart(moved)
        app._refresh_validation_staleness()
        assert seen[-1] is True

        app._mark_validated(entry)              # re-validate at the moved coordinates
        app._refresh_validation_staleness()
        assert seen[-1] is False
    finally:
        dispose(app)


def run():
    qapp()
    # Every exercise builds a DesktopApp, which reads its defaults from QSettings -- so
    # the whole file runs against a fresh install's preferences, not the user's.
    with shipped_defaults():
        for name, fn in sorted(globals().items()):
            if name.startswith("exercise"):
                print("  %s" % name)
                sys.stdout.flush()
                fn()
    print("OK")


if __name__ == "__main__":
    run()
