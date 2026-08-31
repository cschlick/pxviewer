"""Drive the real desktop app and save screenshots -- eyes for development.

    QT_QPA_PLATFORM=cocoa libtbx.python scripts/dev_snap.py MODEL [SELECTION] [OUT.png]

Loads MODEL (a path, or a bundled sample name like 3nir.pdb), optionally applies a
typed SELECTION (driving the same focus/orient/clip pipeline the Selection box uses),
and writes the viewer's own screenshot to OUT.png (default: snap.png).

The environment matters, and each requirement below was found the hard way:

* ``QT_QPA_PLATFORM=cocoa`` -- the test helpers default to Qt's ``offscreen`` platform
  on any machine without $DISPLAY, which includes every Mac (that variable is X11's).
  Offscreen windows are never really shown, the web page believes it is 0x0, WebGL
  never initialises, and screenshots fail with "empty textures are not allowed".
* ``gpu.configure`` must run before QApplication exists, as run_desktop does, or the
  WebEngine backend is undecided.
* Show ``app._main`` -- the app is one QMainWindow; the viewport window object is a
  reparented child and showing it directly does nothing.
* The screenshot is Mol*'s own (``session.screenshot()``): real renderer pixels. A
  QWidget.grab() of the main window also works once the window is truly shown, and
  captures the whole UI including the controls pane.

This is also the calibration instrument for the orient framing (FOV_CALIBRATION in
frontend/src/live.ts): orient at a known distance, screenshot, measure the fraction of
the frame the selection spans.
"""
import sys
import time

from pxviewer import gpu as gpu_backend

gpu_backend.configure("hardware")

from pxviewer.regression.tst_utils import data_path, dispose, process_events  # noqa: E402
from pxviewer.desktop import DesktopApp  # noqa: E402


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "3nir.pdb"
    selection = sys.argv[2] if len(sys.argv) > 2 else ""
    out = sys.argv[3] if len(sys.argv) > 3 else "snap.png"
    if "/" not in model:
        bundled = data_path(model)
        model = str(bundled) if bundled else model

    app = DesktopApp(port=0)
    app._webapp.start()
    try:
        app._main.resize(1700, 900)
        app._main.show()
        for _ in range(20):
            process_events(); time.sleep(0.1)
        app.load_file(model)
        mid = app._active_model_id
        app.set_model_representation(mid, "ball-and-stick")
        for _ in range(60):
            process_events(); time.sleep(0.1)
        if selection:
            app.select_by_expression(selection)
            for _ in range(30):
                process_events(); time.sleep(0.1)
        png = app._model_entry(mid)["session"].screenshot(timeout=30)
        if not png:
            print("no screenshot came back", file=sys.stderr)
            return 1
        with open(out, "wb") as handle:
            handle.write(png)
        print(f"wrote {out} ({len(png)} bytes)")
        return 0
    finally:
        dispose(app)


if __name__ == "__main__":
    sys.exit(main())
