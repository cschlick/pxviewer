"""The GUI-agnostic AppCore: semantics without widgets or a display.

This is the architecture test for the desktop/headless split — the whole
point of ``AppCore`` is that importing it, constructing it and exercising its
semantic surface never touches QtWidgets or a windowing system. A regression
here means GUI code has crept back into the core.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

from pxviewer.regression.tst_utils import have, skip

if not have("PySide6"):
    skip("PySide6 not available")


def exercise_importing_the_core_never_touches_widgets():
    """``pxviewer.appcore`` must be importable with only QtCore — the module
    where the core lives cannot know widgets exist."""
    import pxviewer.appcore  # noqa: F401

    # sys.modules is the ledger: importing the core may pull QtCore (bridge
    # signals need it) but never QtWidgets/QtWebEngine.
    for forbidden in ("PySide6.QtWidgets", "PySide6.QtWebEngineWidgets",
                      "PySide6.QtWebEngineCore"):
        assert forbidden not in sys.modules, forbidden


def exercise_core_constructs_on_a_bare_qcoreapplication():
    """No QApplication, no platform plugin: the core needs only the QObject
    machinery (bridge signals, the app-thread invoker)."""
    from PySide6.QtCore import QCoreApplication

    assert QCoreApplication.instance() is None, (
        "a QApplication already exists — this exercise must run before any "
        "widget test in the same process")
    from pxviewer.appcore import AppCore

    app = AppCore(port=0)
    try:
        assert app._app is not None  # the core made its own QCoreApplication
        # …and it is NOT a QApplication — no widgets anywhere in the core.
        assert type(app._app).__name__ == "QCoreApplication"
        assert type(app._viewport).__name__ == "_NullViewport"
        assert type(app._controls).__name__ == "_NullControls"
        assert app._headless is True
    finally:
        app.stop()


def exercise_stub_viewports_and_controls_are_honest_noops():
    """The few semantic call sites that notify the GUI must no-op cleanly —
    a headless run has nobody to notify but must not raise."""
    from pxviewer.appcore import AppCore

    app = AppCore(port=0)
    try:
        app._viewport.load("http://127.0.0.1:1/index.html?ws=x")  # nowhere
        app._viewport.close()
        app._controls._update_appearance("model", "model-1")
        app._controls._coordinates_moved("model-1")
        assert app._controls._focused is None  # matches no (kind, id) pair
    finally:
        app.stop()


def exercise_load_select_describe_without_a_viewer():
    """The semantic loop an agent drives: load, select, describe — the page
    never connects and nothing is missing."""
    from pxviewer.appcore import AppCore
    from pxviewer.loader import sample_structure_path

    app = AppCore(port=0)
    try:
        app._webapp.start()
        app.load_file(sample_structure_path())
        assert len(app._models) == 1
        n = app.select_by_expression("chain A and resseq 1:5",
                                   focus=False, clip=False)
        description = app.selection_description()
        assert "5 residues" in description or "43 atoms" in description, description
        assert app.job_state() == {"busy": [], "errors": []}
    finally:
        app.stop()


def exercise_oversized_maps_open_decimated():
    """A grid too heavy for marching cubes opens already decimated — the
    quality knob's automatic position — and ``set_volume_downsample`` moves
    it by hand. A rewritten display map must regenerate the decimation:
    drawing a stale pre-mask decimated copy is exactly the divergence the
    cache invalidation exists to prevent."""
    if not have("iotbx.map_manager"):
        return  # no cctbx: nothing to decimate anyway
    import numpy as np

    from pxviewer.appcore import AppCore
    from pxviewer.volume_io import VolumeData

    app = AppCore(port=0)
    try:
        app._webapp.start()
        app._DOWNSAMPLE_VOXEL_CAP = 10_000          # a 24^3 grid counts as huge
        data = VolumeData.from_numpy(np.zeros((24, 24, 24), dtype=np.float32),
                                     name="big")
        vid = app._add_volume(data, "big")
        entry = app._volume_entry(vid)
        assert entry["localres_downsample"] == 2, entry.get("localres_downsample")
        vols = app._webapp.volume_dir / "vols"
        dec = vols / f"{vid}.d2.map"
        assert dec.is_file(), "no decimated display copy written"
        scene = (app._webapp.volume_dir
                 / app._write_volume_scene().lstrip("/")).read_text()
        assert f"{vid}.d2.map" in scene, "the scene still draws the full grid"

        small = VolumeData.from_numpy(np.zeros((10, 10, 10), dtype=np.float32),
                                      name="small")
        vid2 = app._add_volume(small, "small")
        assert int(app._volume_entry(vid2).get("localres_downsample") or 1) == 1

        # The display map rewritten (mask, frame shift): the coarse copy must be
        # re-derived, not left drawing the pre-rewrite density.
        data2 = VolumeData.from_numpy(np.ones((24, 24, 24), dtype=np.float32),
                                      name="big")
        app._write_display_map(vid, data2)
        coarse = VolumeData.from_map_file(str(dec))
        assert float(coarse.array.mean()) > 0.9, "decimated copy kept stale density"

        # The knob moves by hand, under the canonical name.
        app.set_volume_downsample(vid, 3)
        assert (vols / f"{vid}.d3.map").is_file()
        scene = (app._webapp.volume_dir
                 / app._write_volume_scene().lstrip("/")).read_text()
        assert f"{vid}.d3.map" in scene
        app.set_volume_downsample(vid, 1)
        scene = (app._webapp.volume_dir
                 / app._write_volume_scene().lstrip("/")).read_text()
        assert ".d3.map" not in scene and f"{vid}.map" in scene
    finally:
        app.stop()


def exercise_invoke_on_gui_marshals_onto_the_app_thread():
    """The dispatcher the kernel/MCP marshal through is the same one the
    desktop uses — it must deliver a callable on this thread and return it."""
    import threading

    from pxviewer.appcore import AppCore

    app = AppCore(port=0)
    try:
        me = threading.current_thread()

        def probe():
            return threading.current_thread() is me, 21 * 2

        # Emitted on the object's own thread: auto-connection calls through
        # directly even with no event loop, so this answers.
        fut = app.invoke_on_gui(probe)
        same_thread, answer = fut.result(timeout=10)
        assert same_thread and answer == 42
    finally:
        app.stop()


def run():
    for name, fn in sorted(globals().items()):
        if name.startswith("exercise"):
            print("  %s" % name)
            sys.stdout.flush()
            fn()
    print("OK")


if __name__ == "__main__":
    run()
